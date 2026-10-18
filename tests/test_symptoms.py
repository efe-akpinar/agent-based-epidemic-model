import random

import pytest

from epidemic_abm import LRTI, MEV1, URTI, day_parameters, daily_parameters

# Tables 2-4 of the report: (beta_combined, beta_SIR, zeta_raw, zeta_net, gamma_SIR) per symptom set.
# beta_SIR is None where a beta-minus symptom makes it a random draw.
REPORT_TABLES = {
    "urti": [
        (1.8000, 0.6429, 0.988, 0.8218, 0.4000),
        (1.8000, 0.6429, 0.988, 0.8218, 0.4000),
        (1.8000, 0.6429, 0.988, 0.8218, 0.4000),
        (1.6023, 0.6157, 1.589, 1.2493, 0.3000),
        (1.6023, 0.6157, 1.589, 1.2493, 0.3000),
        (0.0000, 0.0000, 0.350, 0.1891, 0.6992),
    ],
    "lrti": [
        (2.1888, 0.6864, 1.0920, 0.7904, 0.4000),
        (2.1888, 0.6864, 1.0920, 0.7904, 0.4000),
        (2.1888, 0.6864, 1.0920, 0.7904, 0.4000),
        (2.0976, 0.6772, 0.5400, 0.3055, 0.5934),
        (2.0976, 0.6772, 0.5400, 0.3055, 0.5934),
        (2.0976, 0.6772, 0.5400, 0.3055, 0.5934),
        (0.0000, None, 3.7336, 2.1720, 0.0008),
    ],
    "mev1": [
        (1.5696, 0.6108, 1.0920, 0.7904, 0.4000),
        (1.5696, 0.6108, 1.0920, 0.7904, 0.4000),
        (0.0000, None, 0.8060, 0.7518, 0.1967),
        (0.0000, None, 1.9314, 1.7242, 0.0050),
        (0.0000, None, 1.9670, 1.9283, 0.0022),
    ],
}


@pytest.mark.parametrize("scenario", [URTI, LRTI, MEV1], ids=lambda s: s.key)
def test_engine_reproduces_report_tables(scenario):
    days = daily_parameters(scenario.timeline, random.Random(0))
    expected_rows = REPORT_TABLES[scenario.key]
    assert len(days) == len(expected_rows)
    for day, (params, expected) in enumerate(zip(days, expected_rows)):
        beta_combined, beta_sir, zeta_raw, zeta_net, gamma = expected
        assert params.beta_combined == pytest.approx(beta_combined, abs=1e-4), day
        assert params.zeta_raw == pytest.approx(zeta_raw, abs=1e-4), day
        assert params.zeta_net == pytest.approx(zeta_net, abs=1e-4), day
        assert params.gamma_sir == pytest.approx(gamma, abs=1e-4), day
        if beta_sir is None:
            assert 0.001 <= params.beta_sir <= 0.005
        else:
            assert params.beta_sir == pytest.approx(beta_sir, abs=1e-4), day


def test_beta_minus_symptom_overrides_transmission():
    mild = day_parameters(["Cough", "Sneezing"], random.Random(0))
    severe = day_parameters(["Cough", "Sneezing", "Pneumonia"], random.Random(0))
    assert mild.beta_sir > 0.6
    assert severe.beta_combined == 0 and 0.001 <= severe.beta_sir <= 0.005


def test_unknown_symptom_is_rejected():
    with pytest.raises(ValueError):
        day_parameters(["Headache"], random.Random(0))


def test_timeline_must_start_on_day_zero():
    with pytest.raises(ValueError):
        daily_parameters({2: ("Cough",)}, random.Random(0))
