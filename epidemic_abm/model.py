"""Agent-based SIRD model with symptom-driven transmission, recovery and mortality.

Each infected agent follows the scenario's symptom timeline by day of illness. Its
transmission rate, recovery rate and daily health loss come from the symptom-network
engine, modulated by the agent's immunity class and health reserve. Contacts come from
a small-world network plus daily cafeteria tables, and severe cases compete for
hospital beds.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import pandas as pd

from .scenarios import Scenario
from .symptoms import SYMPTOMS, DayParameters, daily_parameters, parameter_table

IMMUNITY_CLASSES = ("strong", "normal", "weak")
INITIAL_HEALTH = {"strong": (601, 1000), "normal": (301, 600), "weak": (100, 300)}
INFECTION_MULTIPLIER = {"strong": 0.8, "normal": 1.0, "weak": 1.4}   # sigma_i
RECOVERY_MULTIPLIER = {"strong": 1.5, "normal": 1.0, "weak": 0.6}    # M_immunity
HEALTH_LOSS_MODIFIER = {"strong": 0.7, "normal": 1.0, "weak": 1.3}
# Health-reserve bonus on recovery: (health score must exceed, multiplier), checked in order.
HEALTH_BONUS = {
    "strong": ((700, 4.0), (500, 2.5), (300, 1.5), (-math.inf, 0.8)),
    "normal": ((450, 3.0), (300, 2.0), (150, 1.0), (-math.inf, 0.6)),
    "weak": ((200, 2.0), (120, 1.2), (80, 0.8), (-math.inf, 0.4)),
}
# Daily health loss that treatment offsets, per care setting.
TREATMENT_RELIEF = {"home": 4.0, "general": 12.0, "pandemic": 18.0, "icu": 25.0}
# Health loss per unit of total severity (raw + synergy zeta). 2.5 reproduces the report's
# single-symptom examples: cough (zeta 0.2) -> 0.5 and fever (zeta 0.3) -> 0.75 per day.
HEALTH_LOSS_PER_ZETA = 2.5
# Patients with pneumonia or high fever go to the ICU below these health scores.
ICU_HEALTH_THRESHOLD = {"strong": 500, "normal": 300, "weak": 120}

CRITICAL_SYMPTOMS = frozenset({"Coma", "Sepsis", "Loss of Consciousness"})
SEVERE_SYMPTOMS = frozenset({"Pneumonia", "High Fever"})
RECOVERY_THRESHOLD_RANGE = (1.5, 3.5)
MIN_RECOVERY_RATE = 0.01
MIN_ILLNESS_DAYS = 5
MAX_HEALTH = 1000
DAILY_HUNGER_INCREMENT = 0.5
HUNGER_THRESHOLD = 2.0
INITIAL_HUNGER = (0.0, 0.5, 1.0)
MAX_CONTACTS_BETA_MINUS = 3


@dataclass
class SimulationConfig:
    n_agents: int = 15_000
    k_neighbors: int = 8           # small-world neighbours per agent
    rewire_prob: float = 0.1
    initial_infected: float = 0.03
    n_tables: int = 500            # cafeteria tables
    max_seats: int = 6             # maximum seats per table
    hospital_beds: int = 300       # regular capacity (general ward + ICU)
    icu_share: float = 0.2
    pandemic_beds: int = 200       # pandemic hospital opened on the epidemic day (0 = never)
    surge_factor: float = 1.2      # hospital capacity during the epidemic phase
    death_rate: float = 0.02       # daily death probability scale for lethal combinations
    isolation_group_max: int = 3   # largest contact group after the epidemic day
    seed: int = 42


@dataclass(frozen=True)
class Stage:
    """Everything the model needs about one day of illness."""
    params: DayParameters
    beta_minus: bool
    critical: bool
    severe: bool
    lethal: bool        # two or more beta-minus symptoms at once
    health_loss: float  # before the immunity modifier and treatment relief

    @classmethod
    def from_params(cls, params: DayParameters) -> "Stage":
        symptoms = set(params.symptoms)
        n_beta_minus = sum(SYMPTOMS[s].beta_minus for s in symptoms)
        return cls(
            params=params,
            beta_minus=n_beta_minus > 0,
            critical=bool(symptoms & CRITICAL_SYMPTOMS),
            severe=bool(symptoms & SEVERE_SYMPTOMS),
            lethal=n_beta_minus >= 2,
            health_loss=HEALTH_LOSS_PER_ZETA * params.zeta_raw,
        )


class Agent:
    __slots__ = ("id", "immunity", "health", "initial_health", "status", "infection_day",
                 "outcome_day", "recovery_score", "recovery_threshold", "stage", "hunger",
                 "unit", "worst_unit", "ever_severe", "ever_critical")

    def __init__(self, agent_id: int, immunity: str, health: int, hunger: float):
        self.id = agent_id
        self.immunity = immunity
        self.health = float(health)
        self.initial_health = health
        self.status = "S"
        self.infection_day: int | None = None
        self.outcome_day: int | None = None
        self.recovery_score = 0.0
        self.recovery_threshold = 0.0
        self.stage: Stage | None = None
        self.hunger = hunger
        self.unit: str | None = None        # current hospital unit
        self.worst_unit: str | None = None  # highest level of care received
        self.ever_severe = False
        self.ever_critical = False

    @property
    def critical(self) -> bool:
        return self.status == "I" and self.stage.critical

    @property
    def present(self) -> bool:
        """In the community: alive, not hospitalized and able to have social contacts."""
        return self.status != "D" and self.unit is None and not self.critical


@dataclass
class SimulationResult:
    scenario: Scenario
    config: SimulationConfig
    parameters: pd.DataFrame
    daily: pd.DataFrame
    symptoms: pd.DataFrame
    agents: pd.DataFrame
    epidemic_day: int | None
    normalization_day: int | None


class Simulation:
    def __init__(self, scenario: Scenario, config: SimulationConfig | None = None):
        self.scenario = scenario
        self.config = config or SimulationConfig()
        cfg = self.config
        self.rng = random.Random(cfg.seed)

        self.days = daily_parameters(scenario.timeline, self.rng)
        self.stages = [Stage.from_params(p) for p in self.days]

        counts = [round(share * cfg.n_agents) for share in scenario.immunity_mix[:2]]
        counts.append(cfg.n_agents - sum(counts))
        classes = [c for c, n in zip(IMMUNITY_CLASSES, counts) for _ in range(n)]
        self.rng.shuffle(classes)
        self.agents = [
            Agent(i, imm, self.rng.randint(*INITIAL_HEALTH[imm]), self.rng.choice(INITIAL_HUNGER))
            for i, imm in enumerate(classes)
        ]

        graph = nx.watts_strogatz_graph(cfg.n_agents, cfg.k_neighbors, cfg.rewire_prob,
                                        seed=self.rng.randrange(2**32))
        self.base_adjacency = [sorted(graph[i]) for i in range(cfg.n_agents)]
        self.adjacency = self.base_adjacency

        icu = round(cfg.hospital_beds * cfg.icu_share)
        self.base_capacity = {"general": cfg.hospital_beds - icu, "icu": icu, "pandemic": 0}
        self.capacity = dict(self.base_capacity)
        self.epidemic_day: int | None = None
        self.normalization_day: int | None = None

    # ---------------------------------------------------------------- disease course
    def stage_for(self, illness_day: int) -> Stage:
        return self.stages[min(illness_day, len(self.stages) - 1)]

    def infect(self, agent: Agent, day: int) -> None:
        agent.status = "I"
        agent.infection_day = day
        agent.recovery_score = 0.0
        agent.recovery_threshold = self.rng.uniform(*RECOVERY_THRESHOLD_RANGE)
        self.set_stage(agent, self.stages[0])

    @staticmethod
    def set_stage(agent: Agent, stage: Stage) -> None:
        agent.stage = stage
        agent.ever_severe |= stage.beta_minus
        agent.ever_critical |= stage.critical

    def progress(self, agent: Agent, day: int) -> None:
        illness_day = day - agent.infection_day
        stage = self.stage_for(illness_day)
        self.set_stage(agent, stage)

        relief = TREATMENT_RELIEF[agent.unit or "home"]
        loss = stage.health_loss * HEALTH_LOSS_MODIFIER[agent.immunity] - relief
        agent.health = max(0.0, agent.health - max(0.0, loss))

        p_death = 1.0 if agent.health <= 0 else (
            self.config.death_rate * (1 - agent.health / MAX_HEALTH) if stage.lethal else 0.0)
        if p_death > 0 and self.rng.random() < p_death:
            agent.status, agent.outcome_day = "D", day
            return

        gamma = stage.params.gamma_sir
        agent.recovery_score += gamma
        p_recovery = (1 - math.exp(-max(gamma, MIN_RECOVERY_RATE))) * RECOVERY_MULTIPLIER[agent.immunity]
        p_recovery *= health_bonus(agent.immunity, agent.health)
        if (illness_day >= MIN_ILLNESS_DAYS and agent.recovery_score >= agent.recovery_threshold
                and self.rng.random() < p_recovery):
            agent.status, agent.outcome_day = "R", day

    # ---------------------------------------------------------------- hospitals
    def update_hospitals(self, day: int) -> dict[str, int]:
        occupancy = Counter(a.unit for a in self.agents if a.unit)
        discharges = 0
        for a in self.agents:
            if a.unit and a.status in ("R", "D"):
                occupancy[a.unit] -= 1
                a.unit = None
                discharges += 1

        # Critical patients move from the general or pandemic wards to the ICU when a bed frees up.
        for a in self.agents:
            if a.unit in ("general", "pandemic") and a.critical and occupancy["icu"] < self.capacity["icu"]:
                occupancy[a.unit] -= 1
                occupancy["icu"] += 1
                a.unit = "icu"
                a.worst_unit = "icu"

        # Triage: critical patients first, then the lowest health score.
        requests = [a for a in self.agents if a.status == "I" and a.unit is None and a.stage.beta_minus]
        requests.sort(key=lambda a: (not a.stage.critical, a.health, a.id))
        admissions = turned_away = 0
        for a in requests:
            needs_icu = a.stage.critical or (a.stage.severe and a.health < ICU_HEALTH_THRESHOLD[a.immunity])
            preference = ("icu", "pandemic", "general") if needs_icu else ("general", "pandemic")
            unit = next((u for u in preference if occupancy[u] < self.capacity[u]), None)
            if unit is None:
                turned_away += 1
                continue
            occupancy[unit] += 1
            a.unit = unit
            if a.worst_unit != "icu":
                a.worst_unit = unit
            admissions += 1

        # Epidemic day: hospitals turn patients away while discharges lag behind demand.
        if self.epidemic_day is None and turned_away > 0 and discharges < len(requests):
            self.epidemic_day = day
            self.capacity = {
                "general": round(self.base_capacity["general"] * self.config.surge_factor),
                "icu": round(self.base_capacity["icu"] * self.config.surge_factor),
                "pandemic": self.config.pandemic_beds,
            }
            self.adjacency = self.isolation_groups()
        # Normalization day: no one is turned away and discharges outpace admissions again.
        elif (self.epidemic_day is not None and self.normalization_day is None and day > self.epidemic_day
              and turned_away == 0 and discharges > admissions):
            self.normalization_day = day
            self.adjacency = self.base_adjacency

        return {"admissions": admissions, "discharges": discharges, "turned_away": turned_away,
                **{f"hospital_{u}": occupancy[u] for u in ("general", "icu", "pandemic")}}

    def isolation_groups(self) -> list[list[int]]:
        """Contact network of the epidemic phase: small isolated cliques."""
        ids = list(range(self.config.n_agents))
        self.rng.shuffle(ids)
        adjacency: list[list[int]] = [[] for _ in ids]
        i = 0
        while i < len(ids):
            group = ids[i:i + self.rng.randint(1, self.config.isolation_group_max)]
            for u in group:
                adjacency[u] = sorted(v for v in group if v != u)
            i += len(group)
        return adjacency

    # ---------------------------------------------------------------- cafeteria
    def cafeteria_tables(self) -> list[list[int]]:
        applicants = []
        for a in self.agents:
            if a.status == "D":
                continue
            a.hunger += DAILY_HUNGER_INCREMENT
            if not a.present:
                continue
            if a.hunger >= HUNGER_THRESHOLD or self.rng.random() < a.hunger / HUNGER_THRESHOLD:
                applicants.append(a)
        self.rng.shuffle(applicants)

        tables, i = [], 0
        for _ in range(self.config.n_tables):
            if i >= len(applicants):
                break
            group = applicants[i:i + self.rng.randint(1, self.config.max_seats)]
            for a in group:
                a.hunger = 0.0
            tables.append([a.id for a in group])
            i += len(group)
        return tables

    # ---------------------------------------------------------------- transmission
    def daily_contacts(self, tables: Sequence[Sequence[int]]) -> list[list[int]]:
        present = [a.present for a in self.agents]
        contacts = [[v for v in self.adjacency[u] if present[v]] if present[u] else []
                    for u in range(self.config.n_agents)]
        for table in tables:
            for u in table:
                seen = set(contacts[u])
                contacts[u].extend(v for v in table if v != u and v not in seen)

        # Agents with beta-minus symptoms keep at most three social contacts.
        for a in self.agents:
            if a.status == "I" and present[a.id] and a.stage.beta_minus \
                    and len(contacts[a.id]) > MAX_CONTACTS_BETA_MINUS:
                keep = set(self.rng.sample(sorted(contacts[a.id]), MAX_CONTACTS_BETA_MINUS))
                for v in contacts[a.id]:
                    if v not in keep:
                        contacts[v].remove(a.id)
                contacts[a.id] = sorted(keep)
        return contacts

    def transmit(self, day: int, contacts: list[list[int]]) -> int:
        agents = self.agents
        infectious = [a.status == "I" and a.present for a in agents]
        # Share of each agent's contacts that are infectious: N_i / d_i.
        infected_share = [sum(infectious[w] for w in c) / len(c) if c else 0.0 for c in contacts]
        new_cases: set[int] = set()
        for a in agents:
            if not infectious[a.id] or a.stage.params.beta_sir <= 0:
                continue
            beta = a.stage.params.beta_sir
            for v in contacts[a.id]:
                target = agents[v]
                if target.status != "S" or v in new_cases:
                    continue
                pressure = beta * infected_share[v]
                p = min(1.0, (1 - math.exp(-pressure)) * INFECTION_MULTIPLIER[target.immunity])
                if self.rng.random() < p:
                    new_cases.add(v)
        for v in sorted(new_cases):
            self.infect(agents[v], day)
        return len(new_cases)

    # ---------------------------------------------------------------- main loop
    def run(self) -> SimulationResult:
        cfg = self.config
        for idx in self.rng.sample(range(cfg.n_agents), int(cfg.n_agents * cfg.initial_infected)):
            self.infect(self.agents[idx], 0)

        daily_rows, symptom_rows = [], []
        for day in range(self.scenario.n_days):
            for a in self.agents:
                if a.status == "I" and a.infection_day < day:
                    self.progress(a, day)
            hospital = self.update_hospitals(day)
            tables = self.cafeteria_tables()
            new_cases = self.transmit(day, self.daily_contacts(tables))

            infected = [a for a in self.agents if a.status == "I"]
            status = Counter(a.status for a in self.agents)
            mean_beta = sum(a.stage.params.beta_sir for a in infected) / len(infected) if infected else 0.0
            mean_gamma = sum(a.stage.params.gamma_sir for a in infected) / len(infected) if infected else 0.0
            daily_rows.append({
                "day": day, "S": status["S"], "I": status["I"], "R": status["R"], "D": status["D"],
                "new_infections": new_cases, "cafeteria_visitors": sum(len(t) for t in tables),
                **hospital,
                "hospital_capacity": sum(self.capacity.values()),
                "mean_beta": mean_beta, "mean_gamma": mean_gamma,
                "r0": mean_beta / mean_gamma if mean_gamma > 0 else float("nan"),
            })
            symptom_rows.append(Counter(s for a in infected for s in a.stage.params.symptoms))

        return SimulationResult(
            scenario=self.scenario,
            config=cfg,
            parameters=parameter_table(self.days),
            daily=pd.DataFrame(daily_rows),
            symptoms=pd.DataFrame(symptom_rows).fillna(0).astype(int).sort_index(axis=1),
            agents=self.agent_table(),
            epidemic_day=self.epidemic_day,
            normalization_day=self.normalization_day,
        )

    def agent_table(self) -> pd.DataFrame:
        outcome = {"S": "never infected", "I": "infected", "R": "recovered", "D": "dead"}
        return pd.DataFrame([{
            "id": a.id,
            "immunity": a.immunity,
            "initial_health": a.initial_health,
            "infection_day": a.infection_day,
            "outcome": outcome[a.status],
            "outcome_day": a.outcome_day,
            "hospitalized": a.worst_unit is not None,
            "worst_unit": a.worst_unit,
            "severe": a.ever_severe,
            "critical": a.ever_critical,
        } for a in self.agents])


def health_bonus(immunity: str, health: float) -> float:
    for threshold, bonus in HEALTH_BONUS[immunity]:
        if health > threshold:
            return bonus
    return HEALTH_BONUS[immunity][-1][1]


def run_scenario(scenario: Scenario, config: SimulationConfig | None = None) -> SimulationResult:
    return Simulation(scenario, config).run()
