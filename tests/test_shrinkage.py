from __future__ import annotations

import numpy as np
import pytest

from feature_composition import shrinkage


def test_tau2_is_zero_when_effects_agree_within_noise():
    effects = np.array([0.020, 0.021, 0.019, 0.020])
    ses = np.array([0.005, 0.005, 0.005, 0.005])
    assert shrinkage.tau2_dl(effects, ses) == 0.0
    assert shrinkage.tau2_pm(effects, ses) == 0.0
    assert shrinkage.tau2_reml(effects, ses) == 0.0


def test_tau2_estimators_agree_on_clear_heterogeneity():
    effects = np.array([0.01, 0.05, 0.02, 0.06, 0.03])
    ses = np.full(5, 0.003)
    t_dl, t_pm, t_reml = (
        shrinkage.tau2_dl(effects, ses),
        shrinkage.tau2_pm(effects, ses),
        shrinkage.tau2_reml(effects, ses),
    )
    assert t_dl > 0 and t_pm > 0 and t_reml > 0
    # with equal standard errors DL and PM coincide exactly
    assert t_dl == pytest.approx(t_pm, rel=1e-6)
    assert t_reml == pytest.approx(t_dl, rel=0.5)


def test_dl_shrink_collapses_to_pooled_mean_when_tau2_is_zero():
    effects = np.array([0.020, 0.021, 0.019])
    ses = np.array([0.01, 0.01, 0.01])
    shrunk, tau2 = shrinkage.dl_shrink(effects, ses)
    assert tau2 == 0.0
    np.testing.assert_allclose(shrunk, effects.mean())


def test_adaptive_weights_equal_in_magnitude_when_homogeneous():
    bic = np.array([0.02, -0.021, 0.019])
    bse = np.array([0.01, 0.01, 0.01])
    w, tau2 = shrinkage.adaptive_block_weights(bic, bse)
    assert tau2 == 0.0
    np.testing.assert_allclose(np.abs(w), np.abs(w[0]))
    np.testing.assert_array_equal(np.sign(w), np.sign(bic))
    np.testing.assert_array_equal(shrinkage.equal_block_weights(bic), [1, -1, 1])


def test_hartung_knapp_interval_contains_pooled_mean():
    effects = np.array([0.01, 0.03, 0.02])
    ses = np.array([0.005, 0.005, 0.005])
    mu, lo, hi = shrinkage.hartung_knapp_interval(effects, ses, tau2=0.0)
    assert lo < mu < hi
    _, lo1, hi1 = shrinkage.hartung_knapp_interval(effects[:1], ses[:1], 0.0)
    assert np.isnan(lo1) and np.isnan(hi1)
