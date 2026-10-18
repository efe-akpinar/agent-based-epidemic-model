"""Global sensitivity screening (Morris elementary effects) of the settings the report leaves open."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from .ensemble import run_many
from .model import SimulationConfig
from .scenarios import Scenario


@dataclass(frozen=True)
class Factor:
    name: str
    low: float
    high: float
    kind: str = "float"   # "float", "int" or "even"
    label: str = ""

    def value(self, x: float):
        if self.kind == "int":
            return int(round(x))
        if self.kind == "even":
            return int(2 * round(x / 2))
        return float(x)


FACTORS = [
    Factor("death_rate", 0.005, 0.05, label="Death-risk scale"),
    Factor("health_loss_per_zeta", 1.0, 5.0, label="Health loss per unit severity"),
    Factor("min_illness_days", 3, 7, "int", label="Minimum illness days"),
    Factor("beta_scale", 0.5, 1.5, label="Transmission scale"),
    Factor("initial_infected", 0.005, 0.05, label="Initially infected share"),
    Factor("k_neighbors", 4, 12, "even", label="Network neighbours (k)"),
    Factor("rewire_prob", 0.01, 0.3, label="Network rewiring probability"),
    Factor("n_tables", 100, 1000, "int", label="Cafeteria tables"),
    Factor("max_seats", 2, 10, "int", label="Seats per table"),
    Factor("hospital_beds", 100, 1000, "int", label="Regular hospital beds"),
    Factor("pandemic_beds", 0, 600, "int", label="Pandemic hospital beds"),
    Factor("surge_factor", 1.0, 1.5, label="Epidemic-phase capacity surge"),
    Factor("isolation_group_max", 1, 5, "int", label="Isolation group size"),
]

OUTPUTS = {
    "attack_rate": ("Attack rate", lambda s: s["attack_rate"]),
    "deaths": ("Deaths", lambda s: s["final"]["D"]),
    "peak_infected": ("Peak infected", lambda s: s["peak_infected"]),
}


def problem(factors=FACTORS) -> dict:
    return {"num_vars": len(factors), "names": [f.name for f in factors],
            "bounds": [[f.low, f.high] for f in factors]}


def morris_screening(scenario: Scenario, base: SimulationConfig | None = None, trajectories: int = 20,
                     levels: int = 4, seed: int = 7, factors=FACTORS, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (indices, design): Morris mu*, sigma and confidence per factor and output, and
    the sampled design with its outputs. All runs share one random seed (common random
    numbers), so differences between runs come from the factors rather than chance."""
    base = base or SimulationConfig()
    spec = problem(factors)
    X = morris_sample.sample(spec, N=trajectories, num_levels=levels, seed=seed)
    configs = [replace(base, **{f.name: f.value(x) for f, x in zip(factors, row)}) for row in X]
    summaries = [s for _, s in run_many([(scenario, c) for c in configs], workers)]

    design = pd.DataFrame(X, columns=spec["names"])
    rows = []
    for key, (label, getter) in OUTPUTS.items():
        y = np.array([getter(s) for s in summaries], dtype=float)
        design[key] = y
        indices = morris_analyze.analyze(spec, X, y, num_levels=levels, conf_level=0.95, seed=seed)
        for f, mu_star, conf, sigma, mu in zip(factors, indices["mu_star"], indices["mu_star_conf"],
                                               indices["sigma"], indices["mu"]):
            rows.append({"output": key, "output_label": label, "factor": f.name, "factor_label": f.label,
                         "mu_star": mu_star, "mu_star_conf": conf, "sigma": sigma, "mu": mu})
    indices = pd.DataFrame(rows)
    # mu* relative to the largest effect for that output, for comparison across outputs.
    indices["relative"] = indices["mu_star"] / indices.groupby("output")["mu_star"].transform("max")
    return indices, design
