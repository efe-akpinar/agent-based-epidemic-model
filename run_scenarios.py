"""Run the report's outbreak scenarios and save figures and summaries.

Usage:
    python run_scenarios.py                    # all three scenarios, 15,000 agents, seed 42
    python run_scenarios.py --scenario urti --agents 5000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from epidemic_abm import SCENARIOS, SimulationConfig, run_scenario, summarize
from epidemic_abm import plots


def save_outputs(result, figures: Path, results: Path) -> dict:
    key = result.scenario.key
    plots.plot_sird(result, figures / f"{key}_sird.png")
    plots.plot_symptoms(result, figures / f"{key}_symptoms.png")
    if key == "urti":
        plots.plot_r0(result, figures / f"{key}_r0.png")
    if result.daily["hospital_capacity"].gt(0).any() and result.daily["admissions"].sum() > 0:
        plots.plot_hospital(result, figures / f"{key}_hospital.png")
    if (result.agents["outcome"] == "recovered").any():
        plots.plot_recovery_times(result, figures / f"{key}_recovery_times.png")
    if (result.agents["outcome"] == "dead").any():
        plots.plot_survival(result, figures / f"{key}_survival.png")

    summary = summarize(result)
    summary["config"] = asdict(result.config)
    summary["parameters"] = result.parameters.to_dict(orient="records")
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{key}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--agents", type=int, default=SimulationConfig.n_agents)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    finished = []
    for key in keys:
        start = time.perf_counter()
        result = run_scenario(SCENARIOS[key], SimulationConfig(n_agents=args.agents, seed=args.seed))
        summary = save_outputs(result, args.figures, args.results)
        finished.append(result)
        final = summary["final"]
        print(f"{result.scenario.name}: peak {summary['peak_infected']:,} infected on day {summary['peak_day']}, "
              f"final S/I/R/D = {final['S']:,}/{final['I']:,}/{final['R']:,}/{final['D']:,} "
              f"({time.perf_counter() - start:.1f}s)")
    if len(finished) > 1:
        plots.plot_outcomes(finished, args.figures / "scenario_outcomes.png")


if __name__ == "__main__":
    main()
