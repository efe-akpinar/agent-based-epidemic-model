from dataclasses import replace

import pandas as pd
import pytest

from epidemic_abm import LRTI, MEV1, URTI, SimulationConfig, run_scenario, summarize

SMALL = SimulationConfig(n_agents=1_500, n_tables=50, hospital_beds=30, pandemic_beds=20, seed=3)


@pytest.fixture(scope="module")
def lrti():
    return run_scenario(replace(LRTI, n_days=60), SMALL)


def test_population_is_conserved(lrti):
    totals = lrti.daily[["S", "I", "R", "D"]].sum(axis=1)
    assert (totals == SMALL.n_agents).all()


def test_hospital_never_exceeds_capacity(lrti):
    occupancy = lrti.daily[["hospital_general", "hospital_icu", "hospital_pandemic"]].sum(axis=1)
    assert (occupancy <= lrti.daily["hospital_capacity"]).all()


def test_dead_and_recovered_agents_have_outcome_days(lrti):
    finished = lrti.agents[lrti.agents["outcome"].isin(["dead", "recovered"])]
    assert finished["outcome_day"].notna().all()
    assert (finished["outcome_day"] >= finished["infection_day"]).all()


def test_runs_are_reproducible_with_a_seed():
    config = replace(SMALL, n_agents=600)
    first = run_scenario(replace(URTI, n_days=15), config)
    second = run_scenario(replace(URTI, n_days=15), config)
    pd.testing.assert_frame_equal(first.daily, second.daily)


def test_mild_scenario_has_no_deaths_and_minimum_illness_length():
    result = run_scenario(URTI, replace(SMALL, n_agents=1_000))
    recovered = result.agents[result.agents["outcome"] == "recovered"]
    assert result.daily["D"].iloc[-1] == 0
    assert ((recovered["outcome_day"] - recovered["infection_day"]) >= 5).all()


def test_high_lethality_scenario_allows_no_recovery():
    # Cumulative gamma never reaches the recovery threshold within the run.
    result = run_scenario(replace(MEV1, n_days=60), SMALL)
    assert result.daily["R"].iloc[-1] == 0
    assert result.daily["D"].iloc[-1] > 0
    assert summarize(result)["case_fatality"] > 0
