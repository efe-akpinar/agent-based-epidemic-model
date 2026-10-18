"""Survival analysis and scenario summaries."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .model import IMMUNITY_CLASSES, SimulationResult


def kaplan_meier(durations, events, z: float = 1.96) -> pd.DataFrame:
    """Kaplan-Meier survival curve with log(-log) Greenwood confidence bands."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    rows = [{"time": 0.0, "survival": 1.0, "lower": 1.0, "upper": 1.0}]
    survival, greenwood = 1.0, 0.0
    for t in np.unique(durations[events]):
        at_risk = np.sum(durations >= t)
        deaths = np.sum((durations == t) & events)
        survival *= 1 - deaths / at_risk
        if at_risk > deaths:
            greenwood += deaths / (at_risk * (at_risk - deaths))
        if 0 < survival < 1:
            c = z * math.sqrt(greenwood) / abs(math.log(survival))
            lower, upper = survival ** math.exp(c), survival ** math.exp(-c)
        else:
            lower = upper = survival
        rows.append({"time": float(t), "survival": survival, "lower": lower, "upper": upper})
    if len(durations) and durations.max() > rows[-1]["time"]:
        rows.append({**rows[-1], "time": float(durations.max())})
    return pd.DataFrame(rows)


def logrank_test(durations_a, events_a, durations_b, events_b) -> tuple[float, float]:
    """Two-group log-rank test; returns (chi-square statistic, p-value)."""
    da, ea = np.asarray(durations_a, float), np.asarray(events_a, bool)
    db, eb = np.asarray(durations_b, float), np.asarray(events_b, bool)
    observed_minus_expected, variance = 0.0, 0.0
    for t in np.unique(np.concatenate([da[ea], db[eb]])):
        n_a, n_b = np.sum(da >= t), np.sum(db >= t)
        o_a = np.sum((da == t) & ea)
        o = o_a + np.sum((db == t) & eb)
        n = n_a + n_b
        if n < 2:
            continue
        observed_minus_expected += o_a - o * n_a / n
        variance += o * (n_a / n) * (n_b / n) * (n - o) / (n - 1)
    if variance == 0:
        return 0.0, 1.0
    statistic = observed_minus_expected ** 2 / variance
    return statistic, math.erfc(math.sqrt(statistic / 2))


def survival_frame(result: SimulationResult) -> pd.DataFrame:
    """One row per infected agent: days from infection to death, or censoring at the end of the run."""
    agents = result.agents[result.agents["infection_day"].notna()].copy()
    last_day = result.scenario.n_days - 1
    agents["event"] = agents["outcome"] == "dead"
    end = agents["outcome_day"].where(agents["event"], last_day)
    agents["duration"] = end - agents["infection_day"]
    agents["care"] = np.where(agents["hospitalized"], "hospital", "home")
    agents["severity"] = np.where(agents["severe"], "severe", "mild to moderate")
    agents["unit"] = agents["worst_unit"].map({"icu": "ICU", "general": "general ward", "pandemic": "general ward"})
    return agents


def reproduction_by_cohort(result: SimulationResult) -> pd.DataFrame:
    """Measured case reproduction number: mean secondary infections of the agents infected each day."""
    infected = result.agents[result.agents["infection_day"].notna()]
    cohorts = infected.groupby("infection_day")["secondary_infections"].agg(["size", "mean"])
    return cohorts.rename(columns={"size": "cases", "mean": "r_case"}).reset_index()


def early_reproduction_number(result: SimulationResult, susceptible_share: float = 0.9) -> float | None:
    """Mean secondary infections of agents infected while most of the population was still
    susceptible, an empirical estimate of R0 that avoids the depletion of susceptibles."""
    daily = result.daily
    n = result.config.n_agents
    early_days = daily.loc[daily["S"] + daily["new_infections"] >= susceptible_share * n, "day"]
    infected = result.agents[result.agents["infection_day"].isin(early_days)]
    return float(infected["secondary_infections"].mean()) if len(infected) else None


def survival_at_end(frame: pd.DataFrame) -> float:
    return float(kaplan_meier(frame["duration"], frame["event"])["survival"].iloc[-1])


def summarize(result: SimulationResult) -> dict:
    daily, agents = result.daily, result.agents
    infected = agents[agents["infection_day"].notna()]
    recovered = infected[infected["outcome"] == "recovered"]
    recovery_days = recovered["outcome_day"] - recovered["infection_day"]
    peak = daily.loc[daily["I"].idxmax()]
    # R0 = mean beta / mean gamma is only informative while the outbreak is spreading;
    # once beta-minus symptoms dominate, both rates approach zero and the ratio is noise.
    early = daily[(daily["day"] < 10) & daily["r0"].notna()]
    r0_peak = early.loc[early["r0"].idxmax()] if len(early) else None
    hospital_load = daily[["hospital_general", "hospital_icu", "hospital_pandemic"]].sum(axis=1)

    summary = {
        "scenario": result.scenario.name,
        "agents": result.config.n_agents,
        "days": result.scenario.n_days,
        "seed": result.config.seed,
        "ever_infected": int(len(infected)),
        "ever_severe": int(infected["severe"].sum()),
        "peak_severe": int(daily["severe_now"].max()),
        "attack_rate": round(len(infected) / len(agents), 4),
        "peak_infected": int(peak["I"]),
        "peak_day": int(peak["day"]),
        "final": {k: int(daily[k].iloc[-1]) for k in ("S", "I", "R", "D")},
        "case_fatality": round(float((infected["outcome"] == "dead").mean()), 4),
        "r0_start": _round(_finite(daily["r0"].iloc[0]), 3),
        "r0_peak": None if r0_peak is None else round(float(r0_peak["r0"]), 3),
        "r0_peak_day": None if r0_peak is None else int(r0_peak["day"]),
        "r0_below_one_day": next((int(d) for d, r in zip(daily["day"], daily["r0"]) if r < 1), None),
        "r_seed_cases": round(float(infected.loc[infected["route"] == "seed", "secondary_infections"].mean()), 3),
        "r_early": _round(early_reproduction_number(result), 3),
        "infections_by_route": {k: int(v) for k, v in infected["route"].value_counts().items()},
        "epidemic_day": result.epidemic_day,
        "normalization_day": result.normalization_day,
        "peak_hospital_occupancy": int(hospital_load.max()),
        "max_turned_away": int(daily["turned_away"].max()),
        "recovery_days_by_immunity": {
            c: _describe(recovery_days[recovered["immunity"] == c]) for c in IMMUNITY_CLASSES
        },
        "infected_by_immunity": {c: int((infected["immunity"] == c).sum()) for c in IMMUNITY_CLASSES},
        "severe_by_immunity": {c: int(((infected["immunity"] == c) & infected["severe"]).sum())
                               for c in IMMUNITY_CLASSES},
        "deaths_by_immunity": {
            c: int(((infected["immunity"] == c) & (infected["outcome"] == "dead")).sum()) for c in IMMUNITY_CLASSES
        },
    }

    frame = survival_frame(result)
    if frame["event"].any():
        groups = {"all": frame}
        for column in ("care", "severity", "unit"):
            for value, group in frame.groupby(column):
                groups[value] = group
        summary["survival_at_end"] = {k: round(survival_at_end(g), 4) for k, g in groups.items()}
        summary["group_sizes"] = {k: int(len(g)) for k, g in groups.items()}
        for name, column, a, b in [("hospital_vs_home", "care", "hospital", "home"),
                                   ("severe_vs_mild", "severity", "severe", "mild to moderate")]:
            ga, gb = frame[frame[column] == a], frame[frame[column] == b]
            if len(ga) and len(gb):
                statistic, p = logrank_test(ga["duration"], ga["event"], gb["duration"], gb["event"])
                summary[f"logrank_{name}"] = {"chi2": round(statistic, 2), "p_value": p}
    return summary


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _describe(values: pd.Series) -> dict | None:
    if values.empty:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"n": int(len(values)), "min": int(values.min()), "q1": float(q1), "median": float(median),
            "q3": float(q3), "max": int(values.max())}
