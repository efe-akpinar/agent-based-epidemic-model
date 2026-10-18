import pytest

from epidemic_abm import kaplan_meier, logrank_test


def test_kaplan_meier_matches_hand_calculation():
    # Deaths at t=2 (1 of 5 at risk) and t=4 (1 of 3 at risk); censored at 3, 5, 5.
    km = kaplan_meier([2, 3, 4, 5, 5], [1, 0, 1, 0, 0])
    survival = dict(zip(km["time"], km["survival"]))
    assert survival[2.0] == pytest.approx(4 / 5)
    assert survival[4.0] == pytest.approx(4 / 5 * 2 / 3)
    assert km["time"].iloc[-1] == 5.0
    assert (km["lower"] <= km["survival"]).all() and (km["survival"] <= km["upper"]).all()


def test_kaplan_meier_without_events_stays_at_one():
    km = kaplan_meier([3, 7, 9], [0, 0, 0])
    assert (km["survival"] == 1.0).all()


def test_logrank_identical_groups_gives_no_evidence():
    durations, events = [1, 2, 3, 4, 5], [1, 1, 0, 1, 0]
    statistic, p = logrank_test(durations, events, durations, events)
    assert statistic == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_logrank_detects_clearly_different_groups():
    early = ([1, 1, 2, 2, 3, 3, 4, 4], [1] * 8)
    late = ([20, 21, 22, 23, 24, 25, 26, 27], [1] * 8)
    _, p = logrank_test(*early, *late)
    assert p < 0.001
