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
    early = daily[daily["day"] < 10]
    r0_peak = early.loc[early["r0"].idxmax()]
    hospital_load = daily[["hospital_general", "hospital_icu", "hospital_pandemic"]].sum(axis=1)

    summary = {
        "scenario": result.scenario.name,
        "agents": result.config.n_agents,
        "days": result.scenario.n_days,
        "seed": result.config.seed,
        "ever_infected": int(len(infected)),
        "attack_rate": round(len(infected) / len(agents), 4),
        "peak_infected": int(peak["I"]),
        "peak_day": int(peak["day"]),
        "final": {k: int(daily[k].iloc[-1]) for k in ("S", "I", "R", "D")},
        "case_fatality": round(float((infected["outcome"] == "dead").mean()), 4),
        "r0_start": round(float(daily["r0"].iloc[0]), 3),
        "r0_peak": round(float(r0_peak["r0"]), 3),
        "r0_peak_day": int(r0_peak["day"]),
        "r0_below_one_day": next((int(d) for d, r in zip(daily["day"], daily["r0"]) if r < 1), None),
        "epidemic_day": result.epidemic_day,
        "normalization_day": result.normalization_day,
        "peak_hospital_occupancy": int(hospital_load.max()),
        "max_turned_away": int(daily["turned_away"].max()),
        "recovery_days_by_immunity": {
            c: _describe(recovery_days[recovered["immunity"] == c]) for c in IMMUNITY_CLASSES
        },
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


def _describe(values: pd.Series) -> dict | None:
    if values.empty:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"n": int(len(values)), "min": int(values.min()), "q1": float(q1), "median": float(median),
            "q3": float(q3), "max": int(values.max())}
