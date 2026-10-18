"""The three outbreak scenarios studied in the report."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    timeline: Mapping[int, tuple[str, ...]]     # day of illness -> symptoms from that day on
    immunity_mix: tuple[float, float, float]     # share of strong / normal / weak immunity
    n_days: int


URTI = Scenario(
    key="urti",
    name="Upper respiratory tract infection",
    timeline={
        0: ("Runny Nose", "Sneezing", "Muscle Pain"),
        3: ("Cough", "Runny Nose", "Fever", "Muscle Pain"),
        5: ("Muscle Pain",),
    },
    immunity_mix=(0.30, 0.60, 0.10),
    n_days=30,
)

LRTI = Scenario(
    key="lrti",
    name="Lower respiratory tract infection",
    timeline={
        0: ("Cough", "Sneezing", "Fever"),
        3: ("Sneezing", "Cough"),
        6: ("Pneumonia", "Cough", "Sneezing", "Muscle Pain", "High Fever"),
    },
    immunity_mix=(0.40, 0.50, 0.10),
    n_days=200,
)

MEV1 = Scenario(
    key="mev1",
    name="MeV-1-like high-lethality infection",
    timeline={
        0: ("Cough", "Fever", "Runny Nose"),
        2: ("High Fever", "Muscle Pain"),
        3: ("Loss of Consciousness", "Fever", "Muscle Pain"),
        4: ("Coma", "Sepsis"),
    },
    immunity_mix=(0.30, 0.60, 0.10),  # not stated in the report; same mix as the URTI scenario
    n_days=150,
)

SCENARIOS: dict[str, Scenario] = {s.key: s for s in (URTI, LRTI, MEV1)}
