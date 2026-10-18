"""Interactive explorer for the symptom-based epidemic model.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from epidemic_abm import SCENARIOS, SYMPTOMS, Scenario, SimulationConfig, run_scenario, summarize
from epidemic_abm import plots
from epidemic_abm.calibration import SCHOOL_CONFIG, school_scenario

CALIBRATION = Path(__file__).resolve().parent / "results" / "calibration.json"
SCHOOL = "Boarding-school influenza, 1978 (calibrated)"
CUSTOM = "Custom symptom timeline"
PRESETS = [s.name for s in SCENARIOS.values()] + [SCHOOL, CUSTOM]


def school_parameters() -> dict:
    """Posterior medians of the calibration, or typical values if it has not been run."""
    try:
        posterior = json.loads(CALIBRATION.read_text())["posterior"]
        return {k: v["median"] for k, v in posterior.items()}
    except (OSError, KeyError, ValueError):
        return {"beta_scale": 8.0, "latent_days": 1, "prodromal_days": 2, "bed_days": 4, "bed_probability": 0.5}


@st.cache_data(show_spinner="Simulating…", max_entries=32)
def simulate(scenario: Scenario, config: SimulationConfig):
    result = run_scenario(scenario, config)
    return result, summarize(result)


def custom_timeline() -> dict[int, tuple[str, ...]]:
    st.subheader("Symptom timeline")
    st.caption("Each row starts on a day of illness and lasts until the next row. Leave a row empty to skip it.")
    defaults = [(0, ["Cough", "Sneezing"]), (3, ["Cough", "Fever"]), (6, ["Pneumonia", "High Fever"]), (12, [])]
    timeline = {}
    for i, (day, symptoms) in enumerate(defaults):
        left, right = st.columns([1, 5])
        start = left.number_input("From day", 0, 60, day, key=f"day_{i}", disabled=i == 0)
        chosen = right.multiselect("Symptoms", list(SYMPTOMS), symptoms, key=f"symptoms_{i}")
        if chosen:
            timeline[int(start)] = tuple(chosen)
    return timeline


def show(fig) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


st.set_page_config(page_title="Symptom-based epidemic model", layout="wide")
st.title("Symptom-based agent-based epidemic model")
st.caption("Each infected agent's transmission, recovery and health loss are recomputed every day from its symptoms. "
           "Change the scenario and settings in the sidebar; results update automatically.")

with st.sidebar:
    preset = st.selectbox("Scenario", PRESETS)
    is_school = preset == SCHOOL
    preset_days = {s.name: s.n_days for s in SCENARIOS.values()} | {SCHOOL: 40, CUSTOM: 60}
    agents = st.slider("Agents", 300, 15_000, 763 if is_school else 3_000, step=1 if is_school else 100,
                       key=f"agents_{preset}")
    n_days = st.slider("Days", 10, 200, preset_days[preset], key=f"days_{preset}")
    seed = int(st.number_input("Random seed", 0, 100_000, 42))

    with st.expander("Contacts", expanded=False):
        k = st.slider("Network neighbours (k)", 2, 16, 8, step=2)
        rewire = st.slider("Network rewiring probability", 0.0, 0.5, 0.1, 0.01)
        tables_per_1000 = st.slider("Cafeteria tables per 1,000 agents", 0, 150, 125 if is_school else 33)
        seats = st.slider("Maximum seats per table", 1, 12, 12 if is_school else 6)
    with st.expander("Hospitals", expanded=False):
        beds_per_1000 = st.slider("Regular beds per 1,000 agents", 0, 1000, 1000 if is_school else 20)
        icu_share = st.slider("ICU share of beds", 0.0, 0.5, 0.0 if is_school else 0.2, 0.05)
        pandemic_per_1000 = st.slider("Pandemic-hospital beds per 1,000 agents", 0, 100, 0 if is_school else 13)
        isolation = st.checkbox("Isolation phase when hospitals are full", value=True)
    with st.expander("Disease course and interventions", expanded=False):
        progression = st.radio("Progression", ["fixed", "stochastic"], index=1 if is_school else 0, horizontal=True,
                               help="Fixed: every agent follows the timeline (as in the report). Stochastic: stage "
                                    "durations vary and only some agents reach the severe stage.")
        death_rate = st.slider("Death-risk scale", 0.0, 0.1, 0.02, 0.005)
        beta_factor = st.slider("Transmission multiplier (e.g. masks < 1)", 0.1, 2.0, 1.0, 0.05)
        symptom_isolation = st.slider("Daily chance a symptomatic agent self-isolates", 0.0, 1.0, 0.0, 0.05)

if preset == CUSTOM:
    timeline = custom_timeline()
    if 0 not in timeline:
        st.warning("The first row (day 0) needs at least one symptom.")
        st.stop()
    scenario = Scenario("custom", "Custom scenario", timeline, (0.3, 0.6, 0.1), n_days)
    base, severe = SimulationConfig(), (0.3, 0.6, 0.9)
elif is_school:
    params = school_parameters()
    scenario = school_scenario(int(params["latent_days"]), int(params["prodromal_days"]), int(params["bed_days"]),
                               n_days=n_days)
    base = replace(SCHOOL_CONFIG, beta_scale=params["beta_scale"])
    severe = (params["bed_probability"],) * 3
else:
    scenario = replace(next(s for s in SCENARIOS.values() if s.name == preset), n_days=n_days)
    base, severe = SimulationConfig(), (0.3, 0.6, 0.9)

config = replace(
    base, n_agents=agents, seed=seed, k_neighbors=k, rewire_prob=rewire,
    n_tables=round(tables_per_1000 * agents / 1000), max_seats=seats,
    hospital_beds=round(beds_per_1000 * agents / 1000), icu_share=icu_share,
    pandemic_beds=round(pandemic_per_1000 * agents / 1000), isolation_response=isolation,
    progression=progression, severe_progression=severe if progression == "stochastic" else base.severe_progression,
    death_rate=death_rate, beta_scale=base.beta_scale * beta_factor, symptom_isolation=symptom_isolation,
    initial_cases=1 if is_school else None,
)

result, summary = simulate(scenario, config)

cols = st.columns(5)
cols[0].metric("Ever infected", f"{summary['attack_rate']:.0%}", f"{summary['ever_infected']:,} agents", delta_color="off")
cols[1].metric("Peak infected", f"{summary['peak_infected']:,}", f"day {summary['peak_day']}", delta_color="off")
cols[2].metric("Deaths", f"{summary['final']['D']:,}", f"{summary['case_fatality']:.1%} of infected", delta_color="off")
cols[3].metric("Measured early R", "–" if summary["r_early"] is None else f"{summary['r_early']:.2f}")
cols[4].metric("Epidemic day", "none" if summary["epidemic_day"] is None else f"day {summary['epidemic_day']}")

tabs = st.tabs(["Epidemic curve", "Hospitals", "Symptoms", "Reproduction", "Survival", "Parameters"])
with tabs[0]:
    show(plots.plot_sird(result))
with tabs[1]:
    if result.daily["admissions"].sum() > 0:
        show(plots.plot_hospital(result))
    else:
        st.info("No agent needed a hospital bed in this run.")
with tabs[2]:
    show(plots.plot_symptoms(result))
with tabs[3]:
    show(plots.plot_reproduction(result, days=min(40, n_days)))
with tabs[4]:
    if (result.agents["outcome"] == "dead").any():
        show(plots.plot_survival(result))
    else:
        st.info("No deaths in this run.")
with tabs[5]:
    st.caption("Parameters of each day of illness, from the symptom-network engine.")
    st.dataframe(result.parameters, hide_index=True, width="stretch")
    st.download_button("Download daily results (CSV)", result.daily.to_csv(index=False), "daily_results.csv",
                       mime="text/csv")
