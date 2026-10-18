"""Planning experiments: interventions in the calibrated boarding-school model and hospital
capacity in the lower respiratory scenario."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .calibration import MAJOR_OUTBREAK, SCHOOL_SIZE, job
from .ensemble import interval, run_many
from .model import SimulationConfig
from .scenarios import LRTI, Scenario

SCHOOL_INTERVENTIONS = {
    "No intervention": {},
    "Dining hall closed": {"n_tables": 0},
    "Dining hall at half capacity": {"n_tables": 48},
    "Tables of at most 4": {"max_seats": 4},
    "Hygiene and masks (−30% transmission)": {"beta_factor": 0.7},
    "Isolate at first symptoms (50% a day)": {"symptom_isolation": 0.5},
    "Half capacity + isolation": {"n_tables": 48, "symptom_isolation": 0.5},
}


def school_interventions(posterior: pd.DataFrame, seeds_per_draw: int = 4, seed: int = 99,
                         interventions=SCHOOL_INTERVENTIONS, workers: int | None = None) -> pd.DataFrame:
    """Every intervention is simulated for every accepted posterior draw (``Calibration.accepted``),
    so the comparison carries the calibration uncertainty. Runs that fizzle out count (an
    intervention can make a major outbreak less likely). Outcomes are bimodal (fizzle or major
    outbreak), so the expected burden is summarized by the mean, alongside the probability of a
    major outbreak and the median size of one."""
    jobs, labels = [], []
    draws = posterior.to_dict("records")
    for name, change in interventions.items():
        change = dict(change)
        factor = change.pop("beta_factor", 1.0)
        for i, params in enumerate(draws):
            for k in range(seeds_per_draw):
                scenario, config = job(params, seed=seed + i * seeds_per_draw + k)
                jobs.append((scenario, replace(config, beta_scale=config.beta_scale * factor, **change)))
                labels.append(name)
    summaries = [s for _, s in run_many(jobs, workers)]
    rows = pd.DataFrame({"intervention": labels,
                         "boys_ill": [s["ever_severe"] for s in summaries],
                         "peak_in_bed": [s["peak_severe"] for s in summaries],
                         "major_outbreak": [s["ever_severe"] >= MAJOR_OUTBREAK for s in summaries]})
    table = []
    for name in interventions:
        group = rows[rows["intervention"] == name]
        major = group[group["major_outbreak"]]
        ill = interval(group["boys_ill"])
        table.append({"intervention": name,
                      "boys_ill_mean": float(group["boys_ill"].mean()),
                      "boys_ill_low": ill["low"], "boys_ill_high": ill["high"],
                      "major_outbreak_probability": float(group["major_outbreak"].mean()),
                      "boys_ill_if_major_median": float(major["boys_ill"].median()) if len(major) else 0.0,
                      "peak_in_bed_mean": float(group["peak_in_bed"].mean()),
                      "runs": len(group)})
    return pd.DataFrame(table)


POLICIES = {
    "Isolation when hospitals are full (report)": {"isolation_response": True},
    "No isolation phase": {"isolation_response": False},
}


def hospital_capacity_sweep(beds=(75, 150, 300, 600, 1200), n_runs: int = 20, scenario: Scenario = LRTI,
                            base: SimulationConfig | None = None, policies=POLICIES,
                            workers: int | None = None) -> pd.DataFrame:
    """Deaths, unmet demand and attack rate by regular bed capacity (no pandemic hospital), under
    each response policy. In the report's design the isolation phase starts when hospitals are
    full, so bed capacity also decides when isolation begins."""
    base = base or SimulationConfig()
    jobs, labels = [], []
    for policy, change in policies.items():
        for b in beds:
            for i in range(n_runs):
                jobs.append((scenario, replace(base, hospital_beds=b, pandemic_beds=0, seed=1 + i, **change)))
                labels.append((policy, b))
    summaries = [s for _, s in run_many(jobs, workers)]
    rows = pd.DataFrame({"policy": [p for p, _ in labels], "beds": [b for _, b in labels],
                         "deaths": [s["final"]["D"] for s in summaries],
                         "max_turned_away": [s["max_turned_away"] for s in summaries],
                         "attack_rate": [s["attack_rate"] for s in summaries],
                         "epidemic_day": [s["epidemic_day"] for s in summaries]})
    table = []
    for (policy, b), group in rows.groupby(["policy", "beds"], sort=False):
        deaths = interval(group["deaths"])
        table.append({"policy": policy, "beds": b, "beds_per_1000": 1000 * b / base.n_agents,
                      "deaths_median": deaths["median"], "deaths_low": deaths["low"], "deaths_high": deaths["high"],
                      "max_turned_away_median": interval(group["max_turned_away"])["median"],
                      "attack_rate_median": interval(group["attack_rate"])["median"],
                      "epidemic_day_median": interval(group["epidemic_day"].astype(float))["median"]})
    return pd.DataFrame(table)
