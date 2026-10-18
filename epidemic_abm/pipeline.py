"""The analysis steps behind the README and notebooks. Each step saves its figures and a JSON
summary and returns the objects it computed."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .analysis import summarize
from .calibration import BOYS_ILL, PRIORS, calibrate, fit_sir
from .data import load_boarding_school
from .ensemble import interval, run_ensemble
from .experiments import hospital_capacity_sweep, school_interventions
from .model import SimulationConfig, run_scenario
from .scenarios import LRTI, SCENARIOS
from .sensitivity import morris_screening

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def write_json(path: Path, payload) -> Path:
    def default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if np.isnan(value) else float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        raise TypeError(type(value))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=default) + "\n")
    return path


# ------------------------------------------------------------------ 1. report scenarios
def report_scenarios(config: SimulationConfig | None = None, keys=tuple(SCENARIOS), figures: Path = FIGURES,
                     results: Path = RESULTS) -> dict:
    config = config or SimulationConfig()
    runs = {}
    for key in keys:
        result = run_scenario(SCENARIOS[key], config)
        runs[key] = result
        plots.plot_sird(result, figures / f"{key}_sird.png")
        plots.plot_symptoms(result, figures / f"{key}_symptoms.png")
        if key == "urti":
            plots.plot_r0(result, figures / f"{key}_r0.png")
            plots.plot_reproduction(result, figures / f"{key}_reproduction.png")
        if result.daily["admissions"].sum() > 0:
            plots.plot_hospital(result, figures / f"{key}_hospital.png")
        if (result.agents["outcome"] == "recovered").any():
            plots.plot_recovery_times(result, figures / f"{key}_recovery_times.png")
        if (result.agents["outcome"] == "dead").any():
            plots.plot_survival(result, figures / f"{key}_survival.png")
        summary = summarize(result)
        summary["config"] = asdict(result.config)
        summary["parameters"] = result.parameters.to_dict(orient="records")
        write_json(results / f"{key}_summary.json", summary)
    if len(runs) > 1:
        plots.plot_outcomes(list(runs.values()), figures / "scenario_outcomes.png")
    return runs


# ------------------------------------------------------------------ 2. uncertainty
ENSEMBLE_METRICS = {
    "attack_rate": lambda s: s["attack_rate"],
    "peak_infected": lambda s: s["peak_infected"],
    "peak_day": lambda s: s["peak_day"],
    "deaths": lambda s: s["final"]["D"],
    "case_fatality": lambda s: s["case_fatality"],
    "r_early": lambda s: np.nan if s["r_early"] is None else s["r_early"],
    "r_report_peak": lambda s: np.nan if s["r0_peak"] is None else s["r0_peak"],
}


def ensembles(n_runs: int = 50, config: SimulationConfig | None = None, figures: Path = FIGURES,
              results: Path = RESULTS) -> dict:
    config = config or SimulationConfig()
    runs = {key: run_ensemble(s, config, n_runs=n_runs) for key, s in SCENARIOS.items()}
    runs["lrti_stochastic"] = run_ensemble(LRTI, replace(config, progression="stochastic"), n_runs=n_runs)
    plots.plot_ensembles([runs[k] for k in SCENARIOS], figures / "ensemble_sird.png")
    plots.plot_progression(runs["lrti"], runs["lrti_stochastic"], figures / "lrti_progression.png")
    payload = {key: {"runs": n_runs, **{m: interval(ens.metric(f)) for m, f in ENSEMBLE_METRICS.items()}}
               for key, ens in runs.items()}
    write_json(results / "ensembles.json", payload)
    return runs


def sensitivity(trajectories: int = 20, figures: Path = FIGURES, results: Path = RESULTS):
    indices, design = morris_screening(LRTI, trajectories=trajectories)
    plots.plot_morris(indices, figures / "lrti_sensitivity.png")
    write_json(results / "sensitivity_lrti.json", {"runs": len(design), "trajectories": trajectories,
                                                   "indices": indices.to_dict(orient="records")})
    return indices, design


# ------------------------------------------------------------------ 3. calibration
def calibration(n_draws: int = 5000, seeds_per_draw: int = 3, figures: Path = FIGURES, results: Path = RESULTS):
    observed = load_boarding_school()
    sir = fit_sir(observed)
    cal = calibrate(observed, n_draws=n_draws, seeds_per_draw=seeds_per_draw)
    plots.plot_calibration(cal, sir, figures / "calibration_fit.png")
    plots.plot_posterior(cal, figures / "calibration_posterior.png")
    write_json(results / "calibration.json", {
        "observed_boys_ill": BOYS_ILL,
        "draws": len(cal.draws),
        "accepted": len(cal.accepted),
        "priors": {k: list(v) for k, v in PRIORS.items()},
        "posterior": cal.posterior().to_dict(orient="index"),
        "fit": cal.fit_metrics(),
        "sir_baseline": sir,
        "predictive_bands": {
            column: [{"date": d, **dict(zip(("low", "median", "high"), row))}
                     for d, row in zip(observed["date"].dt.date.astype(str), cal.bands(column).to_numpy().tolist())]
            for column in ("in_bed", "convalescent")
        },
        "observed": observed.assign(date=observed["date"].dt.date.astype(str)).to_dict(orient="records"),
        "accepted_draws": cal.accepted.to_dict(orient="records"),
    })
    return cal, sir


def load_posterior(results: Path = RESULTS) -> pd.DataFrame:
    return pd.DataFrame(json.loads((results / "calibration.json").read_text())["accepted_draws"])


# ------------------------------------------------------------------ 4. experiments
def experiments(posterior: pd.DataFrame | None = None, capacity_runs: int = 20, figures: Path = FIGURES,
                results: Path = RESULTS):
    posterior = load_posterior(results) if posterior is None else posterior
    interventions = school_interventions(posterior)
    capacity = hospital_capacity_sweep(n_runs=capacity_runs)
    plots.plot_interventions(interventions, figures / "school_interventions.png")
    plots.plot_capacity(capacity, figures / "lrti_hospital_capacity.png")
    write_json(results / "experiments.json", {"school_interventions": interventions.to_dict(orient="records"),
                                              "hospital_capacity": capacity.to_dict(orient="records")})
    return interventions, capacity
