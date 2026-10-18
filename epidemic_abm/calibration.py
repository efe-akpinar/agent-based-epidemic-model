"""Calibrate the model to the 1978 boarding-school influenza outbreak.

The model is fitted to the daily number of boys confined to bed only. The convalescent
curve and the number of boys who fell ill (512 of 763) are held out as checks.

Influenza course (days of illness): an optional latent period, a prodrome with cough and
sneezing (infectious), a bed stage with high fever, muscle pain and cough (a beta-minus
profile, so the boy is confined to the sick bay), then convalescence with muscle pain.
Only a share of infections (``bed_probability``) progresses to the bed stage; the other
boys stay in the mild stage until they recover. Stage durations vary between boys
(stochastic progression).

Every run starts from one index case. When an outbreak takes off after a single
introduction is highly random, so each run is aligned on its take-off, the first day with
at least ``TAKE_OFF`` boys in bed (24 January in the data), before it is compared with the
observations. The deterministic SIR baseline gets the same freedom through its fitted
start time. The take-off day of the simulations is reported separately as a check against
the documented infection of the index case by 10 January 1978.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .ensemble import QUANTILES, run_many
from .model import SimulationConfig
from .scenarios import Scenario

INDEX_INFECTION = pd.Timestamp("1978-01-10")
SCHOOL_SIZE = 763
BOYS_ILL = 512
N_DAYS = 60
TAKE_OFF = 20          # boys in bed that mark the take-off used for alignment
MAJOR_OUTBREAK = 50    # boys ever in bed; smaller runs are outbreaks that fizzled out

SCHOOL_CONFIG = SimulationConfig(
    n_agents=SCHOOL_SIZE,
    initial_cases=1,
    n_tables=95,                # dining hall: ~8 boys per table on average
    max_seats=12,
    hospital_beds=SCHOOL_SIZE,  # every boy with a high fever is confined to bed in the sick bay
    icu_share=0.0,
    pandemic_beds=0,
)

# Uniform priors: (low, high) for continuous parameters, a list of values for integers.
PRIORS = {
    "beta_scale": (0.3, 12.0),
    "latent_days": [0, 1, 2, 3],
    "prodromal_days": [1, 2, 3],
    "bed_days": [2, 3, 4, 5, 6, 7],
    "bed_probability": (0.3, 1.0),
}


def school_scenario(latent_days: int, prodromal_days: int, bed_days: int, n_days: int = N_DAYS) -> Scenario:
    return Scenario(
        key="school",
        name="Boarding-school influenza, 1978",
        timeline={
            0: ("Cough", "Sneezing"),
            prodromal_days: ("High Fever", "Muscle Pain", "Cough"),
            prodromal_days + bed_days: ("Muscle Pain",),
        },
        immunity_mix=(0.30, 0.60, 0.10),
        n_days=n_days,
        latent_days=latent_days,
    )


def sample_prior(rng: np.random.Generator) -> dict:
    draw = {}
    for name, prior in PRIORS.items():
        draw[name] = float(rng.uniform(*prior)) if isinstance(prior, tuple) else int(rng.choice(prior))
    return draw


def job(params: dict, seed: int, base: SimulationConfig = SCHOOL_CONFIG):
    scenario = school_scenario(params["latent_days"], params["prodromal_days"], params["bed_days"])
    p = params["bed_probability"]
    return scenario, replace(base, beta_scale=params["beta_scale"], severe_progression=(p, p, p),
                             progression="stochastic", seed=seed)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def observation_days(observed: pd.DataFrame) -> np.ndarray:
    """Days since the index infection (10 January 1978) of each observation."""
    return (observed["date"] - INDEX_INFECTION).dt.days.to_numpy()


def take_off_day(series) -> int | None:
    hits = np.flatnonzero(np.asarray(series) >= TAKE_OFF)
    return int(hits[0]) if len(hits) else None


def aligned(daily: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame | None, int | None]:
    """The run's in-bed and convalescent counts on the observation dates, after shifting the
    run so that its take-off falls on the observed take-off day."""
    days = observation_days(observed)
    observed_take_off = days[take_off_day(observed["in_bed"])]
    simulated_take_off = take_off_day(daily["severe_now"])
    if simulated_take_off is None:
        return None, None
    shifted = days + (simulated_take_off - observed_take_off)
    if shifted.max() >= len(daily) or shifted.min() < 0:
        return None, simulated_take_off
    rows = daily.set_index("day").loc[shifted, ["severe_now", "post_severe"]].to_numpy()
    return pd.DataFrame({"day": days, "in_bed": rows[:, 0], "convalescent": rows[:, 1]}), simulated_take_off


@dataclass
class Calibration:
    draws: pd.DataFrame        # every prior draw with its distance to the data
    accepted: pd.DataFrame     # the closest draws: the approximate posterior
    predictive: pd.DataFrame   # aligned posterior predictive runs (major outbreaks), with a "run" column
    predictive_ill: np.ndarray  # boys who fell ill (were ever in bed) in each major-outbreak run
    predictive_r_early: np.ndarray
    take_off_days: np.ndarray  # simulated take-off day (days after the index infection)
    fizzled_share: float       # share of predictive runs that never became a major outbreak
    observed: pd.DataFrame

    def bands(self, column: str, quantiles=QUANTILES) -> pd.DataFrame:
        return self.predictive.groupby("day")[column].quantile(list(quantiles)).unstack()

    def posterior(self) -> pd.DataFrame:
        table = self.accepted[list(PRIORS)].quantile(list(QUANTILES)).T
        return table.rename(columns=dict(zip(QUANTILES, ("low", "median", "high"))))

    def fit_metrics(self) -> dict:
        def check(column, observed):
            b = self.bands(column)
            return (rmse(b[0.5], observed),
                    float(np.mean((b[QUANTILES[0]].to_numpy() <= observed) & (observed <= b[QUANTILES[-1]].to_numpy()))))

        in_bed_rmse, in_bed_cov = check("in_bed", self.observed["in_bed"].to_numpy())
        conv_rmse, conv_cov = check("convalescent", self.observed["convalescent"].to_numpy())
        q = lambda values: dict(zip(("low", "median", "high"), np.nanquantile(values, QUANTILES).round(3).tolist()))
        return {
            "in_bed_rmse": round(in_bed_rmse, 2),
            "in_bed_coverage_90": round(in_bed_cov, 3),
            "convalescent_rmse": round(conv_rmse, 2),
            "convalescent_coverage_90": round(conv_cov, 3),
            "boys_ill": q(self.predictive_ill),
            "r_early": q(self.predictive_r_early),
            "take_off_day": q(self.take_off_days),
            "observed_take_off_day": int(observation_days(self.observed)[take_off_day(self.observed["in_bed"])]),
            "fizzled_share": round(self.fizzled_share, 3),
        }


def calibrate(observed: pd.DataFrame, n_draws: int = 5000, seeds_per_draw: int = 3, accept: float = 0.02,
              predictive_runs: int = 4, seed: int = 2024, workers: int | None = None) -> Calibration:
    """Approximate Bayesian computation (rejection).

    Sample the priors and simulate each draw with several seeds. The distance of a draw is the
    mean RMSE between its aligned in-bed curve and the data (runs that never take off count as
    an empty curve), so draws are accepted for fitting typically rather than by luck. The
    closest draws form the posterior and are re-simulated with fresh seeds for the posterior
    predictive, conditioned (as observed) on a major outbreak."""
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame([sample_prior(rng) for _ in range(n_draws)])
    records = draws.to_dict("records")
    outputs = run_many([job(row, seed=seed + i * seeds_per_draw + k)
                        for i, row in enumerate(records) for k in range(seeds_per_draw)], workers)
    empty = np.zeros(len(observed))
    distances = []
    for daily, _ in outputs:
        curve, _ = aligned(daily, observed)
        distances.append(rmse(empty if curve is None else curve["in_bed"], observed["in_bed"]))
    draws["distance"] = np.array(distances).reshape(n_draws, seeds_per_draw).mean(axis=1)
    accepted = draws.nsmallest(max(1, int(round(accept * n_draws))), "distance").reset_index(drop=True)

    predictive_jobs = [job(row, seed=10_000_000 + seed + i * predictive_runs + k)
                       for i, row in enumerate(accepted.to_dict("records")) for k in range(predictive_runs)]
    curves, ill, r_early, take_offs = [], [], [], []
    predictive = run_many(predictive_jobs, workers)
    for daily, summary in predictive:
        curve, start = aligned(daily, observed)
        if summary["ever_severe"] < MAJOR_OUTBREAK or curve is None:
            continue
        curves.append(curve.assign(run=len(curves)))
        ill.append(summary["ever_severe"])
        r_early.append(np.nan if summary["r_early"] is None else summary["r_early"])
        take_offs.append(start)
    return Calibration(
        draws=draws,
        accepted=accepted,
        predictive=pd.concat(curves, ignore_index=True),
        predictive_ill=np.array(ill, dtype=float),
        predictive_r_early=np.array(r_early, dtype=float),
        take_off_days=np.array(take_offs, dtype=float),
        fizzled_share=1 - len(curves) / len(predictive),
        observed=observed,
    )


# ---------------------------------------------------------------------- SIR baseline
def sir_curve(beta: float, gamma: float, t0: float, days: np.ndarray, n: int = SCHOOL_SIZE) -> np.ndarray:
    """Infected count of a deterministic SIR model started with one case at time t0."""
    def rhs(_, y):
        s, i = y
        infection = beta * s * i / n
        return [-infection, infection - gamma * i]

    days = np.asarray(days, dtype=float)
    out = np.zeros(len(days))
    mask = days >= t0
    if mask.any():
        sol = solve_ivp(rhs, (t0, float(days.max())), [n - 1, 1], t_eval=days[mask], rtol=1e-8, atol=1e-8)
        out[mask] = sol.y[1]
    return out


def fit_sir(observed: pd.DataFrame) -> dict:
    """Least-squares fit of beta, gamma and the start time t0 to the in-bed counts."""
    days = observation_days(observed).astype(float)
    target = observed["in_bed"].to_numpy(float)

    def loss(theta):
        return rmse(sir_curve(np.exp(theta[0]), np.exp(theta[1]), theta[2], days), target)

    best = None
    for start in [(0.5, -0.7, 5.0), (0.3, -1.0, 8.0), (0.7, -0.5, 10.0)]:
        res = minimize(loss, np.array(start), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    beta, gamma, t0 = float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.x[2])
    return {"beta": beta, "gamma": gamma, "t0": t0, "r0": beta / gamma, "in_bed_rmse": float(best.fun),
            "predicted_ever_infected": sir_final_size(beta / gamma) * SCHOOL_SIZE}


def sir_final_size(r0: float) -> float:
    """Share ever infected in a deterministic SIR epidemic: solves z = 1 - exp(-R0 z)."""
    z = 0.5
    for _ in range(200):
        z = 1 - np.exp(-r0 * z)
    return float(z)
