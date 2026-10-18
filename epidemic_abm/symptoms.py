"""Symptom catalogue and the symptom-network parameter engine.

For the set of symptoms an agent shows on a given day of illness, the engine builds a
symptom graph and derives the bounded transmission rate (beta_SIR) and the daily
recovery rate (gamma_SIR) that drive the agent-based model.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import networkx as nx
import pandas as pd

ALPHA = 0.4               # weight of the same-system share in network scaling
THETA = 0.3               # synergy coefficient for co-occurring symptoms
LAMBDA_SYSTEM = 0.7       # weight of physiological-system proximity in severity synergy
LAMBDA_NETWORK = 0.3      # weight of symptom-graph distance in severity synergy
K_SIGMOID = 4.0           # steepness of the severity and recovery sigmoids
DELTA_SENSITIVITY = 0.2   # network sensitivity used for zeta_max
BETA_MINUS_RANGE = (0.001, 0.005)
DEFAULT_SYSTEM_WEIGHT = 0.2


@dataclass(frozen=True)
class Symptom:
    name: str
    system: str
    beta_raw: float | None  # None for beta-minus symptoms, 0.0 for transmission-neutral ones
    beta_minus: bool        # suppresses transmission through isolation and bed rest
    zeta: float             # severity / recovery difficulty


SYMPTOMS: dict[str, Symptom] = {s.name: s for s in [
    Symptom("Sneezing", "respiratory", 0.84, False, 0.20),
    Symptom("Cough", "respiratory", 0.68, False, 0.20),
    Symptom("Runny Nose", "respiratory", 0.41, False, 0.20),
    Symptom("Diarrhea", "excretory", 0.23, False, 0.30),
    Symptom("Vomiting", "digestive", 0.25, False, 0.45),
    Symptom("Skin Lesions", "skin", 0.25, False, 0.38),
    Symptom("Fever", "immune", 0.0, False, 0.30),
    Symptom("Muscle Pain", "muscle", 0.0, False, 0.35),
    Symptom("Nausea", "digestive", 0.0, False, 0.38),
    Symptom("Sweating", "immune", 0.0, False, 0.32),
    Symptom("High Fever", "immune", None, True, 0.40),
    Symptom("Pneumonia", "respiratory", None, True, 0.66),
    Symptom("Sepsis", "immune", None, True, 0.76),
    Symptom("Loss of Consciousness", "nervous", None, True, 0.79),
    Symptom("Coma", "nervous", None, True, 0.85),
]}

# Proximity between physiological systems; symptoms of the same system get 1.0.
SYSTEM_WEIGHTS: dict[frozenset[str], float] = {
    frozenset({"respiratory", "excretory"}): 0.2,
    frozenset({"respiratory", "immune"}): 0.6,
    frozenset({"respiratory", "nervous"}): 0.8,
    frozenset({"excretory", "immune"}): 0.6,
    frozenset({"immune", "nervous"}): 0.6,
    frozenset({"muscle", "immune"}): 0.2,
    frozenset({"digestive", "immune"}): 0.2,
    frozenset({"skin", "immune"}): 0.2,
}


def system_weight(system_a: str, system_b: str) -> float:
    if system_a == system_b:
        return 1.0
    return SYSTEM_WEIGHTS.get(frozenset({system_a, system_b}), DEFAULT_SYSTEM_WEIGHT)


@dataclass(frozen=True)
class DayParameters:
    symptoms: tuple[str, ...]
    beta_combined: float
    beta_sir: float
    zeta_raw: float
    zeta_net: float
    gamma_sir: float
    beta_scaled: Mapping[str, float] = field(default_factory=dict)

    @property
    def beta_minus(self) -> bool:
        return any(SYMPTOMS[s].beta_minus for s in self.symptoms)


def validate(symptoms: Sequence[str]) -> tuple[str, ...]:
    unknown = [s for s in symptoms if s not in SYMPTOMS]
    if unknown:
        raise ValueError(f"unknown symptoms: {unknown}; known: {sorted(SYMPTOMS)}")
    return tuple(symptoms)


def symptom_graph(symptoms: Sequence[str]) -> nx.Graph:
    """Nodes are the active symptoms; edges join beta-defined symptoms of the same system."""
    graph = nx.Graph()
    for name in symptoms:
        s = SYMPTOMS[name]
        graph.add_node(name, system=s.system, beta_raw=s.beta_raw, zeta=s.zeta)
    for i, a in enumerate(symptoms):
        for b in symptoms[i + 1:]:
            sa, sb = SYMPTOMS[a], SYMPTOMS[b]
            if sa.beta_raw is not None and sb.beta_raw is not None and sa.system == sb.system:
                graph.add_edge(a, b)
    return graph


def day_parameters(symptoms: Sequence[str], rng: random.Random) -> DayParameters:
    """Transmission and recovery parameters for one day's symptom set."""
    symptoms = validate(symptoms)
    graph = symptom_graph(symptoms)
    nodes = list(graph.nodes)

    # Network-weighted scaling of each beta-defined symptom.
    beta_defined_per_system: dict[str, int] = {}
    for n in nodes:
        if graph.nodes[n]["beta_raw"] is not None:
            system = graph.nodes[n]["system"]
            beta_defined_per_system[system] = beta_defined_per_system.get(system, 0) + 1
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees and max(degrees.values()) > 0 else 1

    beta_scaled: dict[str, float] = {}
    for n in nodes:
        beta_raw = graph.nodes[n]["beta_raw"]
        if beta_raw is None:
            beta_scaled[n] = 0.0
            continue
        system = graph.nodes[n]["system"]
        degree = degrees[n]
        system_share = degree / beta_defined_per_system[system]
        beta_scaled[n] = round(beta_raw * (degree / max_degree + ALPHA * system_share), 4)

    # Synergy scaling and the bounded transmission rate; beta-minus symptoms override it.
    beta_minus = any(SYMPTOMS[s].beta_minus for s in symptoms)
    if beta_minus:
        beta_combined = 0.0
        beta_sir = round(rng.uniform(*BETA_MINUS_RANGE), 4)
    else:
        valid = [SYMPTOMS[s].beta_raw for s in symptoms if SYMPTOMS[s].beta_raw is not None]
        if len(valid) == 1:
            beta_combined = valid[0]
        else:
            n = len(beta_scaled)
            beta_combined = sum(beta_scaled.values()) * (1 + THETA * (n - 1) / n) if n else 0.0
        beta_sir = beta_combined / (1 + beta_combined) if beta_combined > 0 else 0.0

    if not nodes:
        return DayParameters(symptoms, 0.0, 0.0, 0.0, 0.0, 0.95, {})

    # Severity: raw zeta plus pairwise synergy weighted by system proximity and graph distance.
    zeta = {n: graph.nodes[n]["zeta"] for n in nodes}
    zeta_raw = sum(zeta.values())
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            w_sys = system_weight(graph.nodes[a]["system"], graph.nodes[b]["system"])
            distance = nx.shortest_path_length(graph, a, b) if nx.has_path(graph, a, b) else max_degree
            w_net = 1 - distance / max_degree
            weight = LAMBDA_SYSTEM * w_sys + LAMBDA_NETWORK * w_net
            zeta_raw += weight * max(zeta[a], zeta[b])

    zeta_mean = sum(zeta.values()) / len(zeta)
    zeta_max = len(zeta) * zeta_mean * (1 + DELTA_SENSITIVITY)
    if zeta_mean < 0.03:
        zeta_crit = 0.15
    elif zeta_mean < 0.08:
        zeta_crit = 0.2
    elif zeta_mean < 0.2:
        zeta_crit = 0.3
    else:
        zeta_crit = 0.4
    zeta_net = zeta_max / (1 + math.exp(-K_SIGMOID * (zeta_raw - zeta_crit)))
    gamma = 1 / (1 + math.exp(K_SIGMOID * (zeta_net - zeta_crit)))

    # Without beta-minus symptoms the recovery rate keeps a severity-dependent floor.
    if not beta_minus:
        if zeta_net < 0.5:
            gamma = max(0.5, gamma)
        elif zeta_net < 1.2:
            gamma = max(0.4, gamma)
        elif zeta_net < 2.0:
            gamma = max(0.3, gamma)

    return DayParameters(
        symptoms=symptoms,
        beta_combined=round(beta_combined, 4),
        beta_sir=round(beta_sir, 4),
        zeta_raw=round(zeta_raw, 4),
        zeta_net=round(zeta_net, 4),
        gamma_sir=round(gamma, 4),
        beta_scaled=beta_scaled,
    )


def daily_parameters(timeline: Mapping[int, Sequence[str]], rng: random.Random) -> list[DayParameters]:
    """Parameters for every day of illness; each symptom set carries forward until the next change."""
    if 0 not in timeline:
        raise ValueError("the timeline must define the symptoms of day 0")
    days: list[DayParameters] = []
    current: Sequence[str] = ()
    for day in range(max(timeline) + 1):
        current = timeline.get(day, current)
        days.append(day_parameters(current, rng))
    return days


def parameter_table(days: Sequence[DayParameters]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "day": d,
            "symptoms": ", ".join(p.symptoms),
            "beta_combined": p.beta_combined,
            "beta_sir": p.beta_sir,
            "zeta_raw": p.zeta_raw,
            "zeta_net": p.zeta_net,
            "gamma_sir": p.gamma_sir,
        }
        for d, p in enumerate(days)
    ])
