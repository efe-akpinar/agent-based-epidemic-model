import numpy as np
import pandas as pd
import pytest

from epidemic_abm.calibration import (INDEX_INFECTION, TAKE_OFF, aligned, fit_sir, sir_curve, sir_final_size,
                                      take_off_day)


def synthetic_observations(values):
    dates = INDEX_INFECTION + pd.to_timedelta(np.arange(12, 12 + len(values)), unit="D")
    return pd.DataFrame({"date": dates, "in_bed": np.asarray(values), "convalescent": np.zeros(len(values))})


def test_sir_fit_recovers_known_parameters():
    days = np.arange(12, 26, dtype=float)
    observed = synthetic_observations(sir_curve(1.7, 0.45, 11.0, days))
    fit = fit_sir(observed)
    assert fit["beta"] == pytest.approx(1.7, rel=0.02)
    assert fit["gamma"] == pytest.approx(0.45, rel=0.02)
    assert fit["in_bed_rmse"] < 1.0


def test_sir_final_size_solves_the_final_size_equation():
    z = sir_final_size(2.0)
    assert z == pytest.approx(1 - np.exp(-2.0 * z))
    assert 0.79 < z < 0.80


def test_runs_are_aligned_on_take_off():
    observed = synthetic_observations([1, 5, TAKE_OFF, 40, 30, 10])
    curve = np.zeros(40)
    curve[20:26] = [1, 5, TAKE_OFF, 40, 30, 10]          # same shape, 6 days later than observed
    daily = pd.DataFrame({"day": np.arange(40), "severe_now": curve, "post_severe": curve / 2})
    shifted, start = aligned(daily, observed)
    assert start == take_off_day(curve) == 22
    assert shifted["in_bed"].tolist() == observed["in_bed"].tolist()


def test_runs_that_never_take_off_are_rejected():
    observed = synthetic_observations([1, 5, TAKE_OFF, 40])
    daily = pd.DataFrame({"day": np.arange(30), "severe_now": np.ones(30), "post_severe": np.zeros(30)})
    assert aligned(daily, observed) == (None, None)


def test_boarding_school_data_loads():
    from epidemic_abm.data import load_boarding_school

    try:
        data = load_boarding_school()
    except OSError as error:  # no network and no cached copy
        pytest.skip(f"dataset unavailable: {error}")
    assert len(data) == 14
    assert data["date"].iloc[0] == pd.Timestamp("1978-01-22")
    assert data["in_bed"].max() == 298
