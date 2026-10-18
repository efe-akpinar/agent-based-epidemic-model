"""Run many seeds of a scenario in parallel and summarize the spread of outcomes."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .analysis import summarize
from .model import SimulationConfig, run_scenario
from .scenarios import Scenario

QUANTILES = (0.05, 0.5, 0.95)


@dataclass
class Ensemble:
    scenario: Scenario
    config: SimulationConfig
    daily: pd.DataFrame          # all runs stacked, with a "seed" column
    summaries: list[dict]

    def bands(self, columns: Sequence[str], quantiles: Sequence[float] = QUANTILES) -> pd.DataFrame:
        """Per-day quantiles of each column across runs (columns: (name, quantile))."""
        return self.daily.groupby("day")[list(columns)].quantile(list(quantiles)).unstack()

    def metric(self, getter: Callable[[dict], float]) -> np.ndarray:
        return np.array([getter(s) for s in self.summaries], dtype=float)


def _run(args: tuple[Scenario, SimulationConfig]) -> tuple[pd.DataFrame, dict]:
    scenario, config = args
    result = run_scenario(scenario, config)
    return result.daily.assign(seed=config.seed), summarize(result)


def run_many(jobs: Iterable[tuple[Scenario, SimulationConfig]], workers: int | None = None) -> list[tuple[pd.DataFrame, dict]]:
    jobs = list(jobs)
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers <= 1:
        return [_run(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


def run_ensemble(scenario: Scenario, config: SimulationConfig | None = None, n_runs: int = 50,
                 first_seed: int = 1, workers: int | None = None) -> Ensemble:
    config = config or SimulationConfig()
    outputs = run_many([(scenario, replace(config, seed=first_seed + i)) for i in range(n_runs)], workers)
    return Ensemble(scenario, config, pd.concat([d for d, _ in outputs], ignore_index=True), [s for _, s in outputs])


def interval(values: Sequence[float], quantiles: Sequence[float] = QUANTILES) -> dict:
    """Median and 90% interval of a metric across runs."""
    low, median, high = np.nanquantile(np.asarray(values, dtype=float), quantiles)
    return {"median": float(median), "low": float(low), "high": float(high)}
