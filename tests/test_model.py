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


def test_infection_routes_and_chains_are_consistent(lrti):
    daily, agents = lrti.daily, lrti.agents
    assert (daily["new_network"] + daily["new_cafeteria"] == daily["new_infections"]).all()
    infected = agents[agents["infection_day"].notna()]
    assert agents["secondary_infections"].sum() == (infected["route"] != "seed").sum()
    assert set(infected["route"]) <= {"seed", "network", "cafeteria"}


def test_latent_period_delays_transmission_and_symptoms():
    result = run_scenario(replace(URTI, n_days=6, latent_days=2), replace(SMALL, n_agents=600))
    first = result.daily.iloc[0]
    assert first["E"] == first["I"] > 0
    assert result.daily.loc[result.daily["day"] < 2, "new_infections"].sum() == 0


def test_stochastic_progression_makes_severe_disease_depend_on_immunity():
    config = replace(SMALL, n_agents=3_000, progression="stochastic", severe_progression=(0.1, 0.5, 0.95))
    agents = run_scenario(replace(LRTI, n_days=30), config).agents
    infected = agents[agents["infection_day"].notna()]
    severe_share = infected.groupby("immunity")["severe"].mean()
    assert severe_share["strong"] < severe_share["normal"] < severe_share["weak"]


def test_symptom_isolation_reduces_spread():
    config = replace(SMALL, n_agents=1_000)
    baseline = run_scenario(URTI, config).daily
    isolating = run_scenario(URTI, replace(config, symptom_isolation=0.8)).daily
    assert isolating["isolated"].max() > 0
    assert isolating["S"].iloc[-1] > baseline["S"].iloc[-1]


def test_unknown_progression_mode_is_rejected():
    with pytest.raises(ValueError):
        run_scenario(URTI, replace(SMALL, progression="random"))
