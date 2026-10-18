from dataclasses import replace

import pandas as pd

from epidemic_abm import URTI, SimulationConfig
from epidemic_abm.ensemble import interval, run_many
from epidemic_abm.sensitivity import FACTORS


def test_parallel_runs_match_serial_runs():
    config = SimulationConfig(n_agents=400, n_tables=15)
    jobs = [(replace(URTI, n_days=10), replace(config, seed=s)) for s in (1, 2)]
    serial, parallel = run_many(jobs, workers=1), run_many(jobs, workers=2)
    for (a, _), (b, _) in zip(serial, parallel):
        pd.testing.assert_frame_equal(a, b)


def test_interval_returns_median_and_90_percent_bounds():
    values = list(range(101))
    assert interval(values) == {"median": 50.0, "low": 5.0, "high": 95.0}


def test_sensitivity_factors_map_to_valid_settings():
    for factor in FACTORS:
        for x in (factor.low, (factor.low + factor.high) / 2, factor.high):
            value = factor.value(x)
            if factor.kind == "even":
                assert value % 2 == 0
            replace(SimulationConfig(), **{factor.name: value})
