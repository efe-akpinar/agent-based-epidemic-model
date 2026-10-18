"""Reproduce every figure and result in the repository.

Usage:
    python run_analyses.py all            # everything (about 20 minutes on 8 cores)
    python run_analyses.py scenarios      # the report's three scenarios (seconds)
    python run_analyses.py ensembles      # 50 seeds per scenario + stochastic progression
    python run_analyses.py sensitivity    # Morris screening of the lower respiratory scenario
    python run_analyses.py calibration    # ABC fit to the 1978 boarding-school outbreak
    python run_analyses.py experiments    # interventions and hospital capacity (needs calibration)
    python run_analyses.py scenarios --agents 5000 --seed 7
"""
from __future__ import annotations

import argparse
import time

from epidemic_abm import SimulationConfig, pipeline

STEPS = ("scenarios", "ensembles", "sensitivity", "calibration", "experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--agents", type=int, default=SimulationConfig.n_agents, help="scenarios step only")
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed, help="scenarios step only")
    args = parser.parse_args()

    for step in STEPS if args.step == "all" else (args.step,):
        start = time.perf_counter()
        if step == "scenarios":
            runs = pipeline.report_scenarios(SimulationConfig(n_agents=args.agents, seed=args.seed))
            for result in runs.values():
                final = "/".join(f"{int(result.daily[k].iloc[-1]):,}" for k in "SIRD")
                print(f"  {result.scenario.name}: final S/I/R/D = {final}")
        elif step == "ensembles":
            pipeline.ensembles()
        elif step == "sensitivity":
            pipeline.sensitivity()
        elif step == "calibration":
            cal, sir = pipeline.calibration()
            print(f"  in-bed RMSE {cal.fit_metrics()['in_bed_rmse']} (SIR baseline {sir['in_bed_rmse']:.2f})")
        elif step == "experiments":
            pipeline.experiments()
        print(f"{step}: done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
