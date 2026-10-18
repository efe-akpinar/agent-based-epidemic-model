"""Download and load the 1978 English boarding-school influenza outbreak.

Source: the ``influenza_england_1978_school`` dataset of the R package ``outbreaks``
(MIT licence), transcribed from Anonymous (1978), "Influenza in a boarding school",
British Medical Journal 1:587. 763 boys, 512 of whom fell ill; daily counts of boys
confined to bed and convalescing from 22 January to 4 February 1978. The data are
downloaded on first use and cached locally; they are not redistributed with this code.
"""
from __future__ import annotations

import hashlib
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

BOARDING_SCHOOL_URL = (
    "https://raw.githubusercontent.com/reconverse/outbreaks/7661de09e74b5f872e22ecb1f812973874ee1113/"
    "data/influenza_england_1978_school.RData"
)
BOARDING_SCHOOL_SHA256 = "395f577bb82f1d93e3a43cfbf6a23330316e80b26bba4b61e7f15fd1cca1f7b2"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"


def download_boarding_school(cache_dir: Path | str = DEFAULT_CACHE) -> Path:
    path = Path(cache_dir) / "influenza_england_1978_school.RData"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BOARDING_SCHOOL_URL, timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != BOARDING_SCHOOL_SHA256:
            raise ValueError("downloaded file does not match the expected checksum")
        path.write_bytes(payload)
    return path


def load_boarding_school(cache_dir: Path | str = DEFAULT_CACHE) -> pd.DataFrame:
    """Columns: date, in_bed, convalescent (14 daily rows)."""
    import rdata

    path = download_boarding_school(cache_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # rdata returns R's Date class as plain day numbers
        frame = rdata.read_rda(str(path))["influenza_england_1978_school"]
    return pd.DataFrame({
        "date": pd.to_datetime(frame["date"].astype(int), unit="D"),
        "in_bed": frame["in_bed"].astype(int).to_numpy(),
        "convalescent": frame["convalescent"].astype(int).to_numpy(),
    })
