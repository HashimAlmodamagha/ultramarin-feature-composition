from __future__ import annotations

import numpy as np
import pytest

from feature_composition import metrics


def test_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert metrics.pearson(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])
    assert np.isnan(metrics.pearson(np.ones(5), b[:5]))


def test_rank_by_day_ranks_within_each_day():
    v = np.array([3.0, 1.0, 2.0, 10.0, 5.0])
    day_starts = np.array([0, 3, 5])
    np.testing.assert_array_equal(metrics.rank_by_day(v, day_starts), [3, 1, 2, 2, 1])


def test_daily_rank_ic_is_spearman_per_day(panel):
    sig = panel.X[:, 0]
    ic = panel.daily_ic(sig)
    assert ic.shape == (panel.n_days,)
    from scipy.stats import spearmanr

    a, b = panel.day_starts[0], panel.day_starts[1]
    assert ic[0] == pytest.approx(spearmanr(sig[a:b], panel.y[a:b]).statistic)


def test_newey_west_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.1, 800)
    x[1:] += 0.5 * x[:-1]
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 25})
    mean, t = metrics.newey_west(x, 25)
    assert mean == pytest.approx(res.params[0])
    assert t == pytest.approx(res.tvalues[0], rel=1e-9)
    assert metrics.hac_se(x, 25) == pytest.approx(res.bse[0], rel=1e-9)


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=2000)
    x = np.convolve(e, np.ones(10) / 10, mode="same")  # strongly autocorrelated
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    _, hac_t = metrics.newey_west(x)
    assert abs(hac_t) < abs(naive_t)


def test_half_life_interpolates_and_handles_no_crossing():
    assert metrics.half_life(np.array([1.0, 0.75, 0.5, 0.25])) == pytest.approx(2.0)
    assert metrics.half_life(np.array([1.0, 0.8, 0.4])) == pytest.approx(1.75)
    assert np.isnan(metrics.half_life(np.array([1.0, 0.9, 0.8])))


def test_decay_and_autocorr_curves_have_expected_anchors(panel):
    sig = panel.X[:, 0]
    d = metrics.decay_curve(panel.dates, panel.ids, sig, panel.y, range(0, 4))
    ac = metrics.autocorr_curve(panel.dates, panel.ids, sig, range(1, 4))
    assert d.shape == (4,) and ac.shape == (4,)
    assert d[0] == pytest.approx(np.nanmean(panel.daily_ic(sig)), abs=1e-9)
    assert ac[0] == 1.0
    assert np.all(np.diff(ac) <= 1e-9)  # persistent signal: autocorrelation decays with lag


def test_weekly_ic_series_averages_the_two_legs(panel):
    sig = panel.X[:, 0]
    wk = metrics.weekly_ic_series(panel.dates, panel.ids, sig, panel.y)
    d = metrics.decay_curve(panel.dates, panel.ids, sig, panel.y, (0, 5))
    # the lag-5 leg drops the first five days, so the weekly mean is close to but not
    # identical with the average of the two curve values
    assert wk.shape[0] == panel.n_days - 5
    assert np.nanmean(wk) == pytest.approx(0.5 * (d[0] + d[1]), abs=5e-3)


def test_summarize_ic_and_portfolio_ir(panel):
    ic = panel.daily_ic(panel.X[:, 0])
    s = metrics.summarize_ic(ic)
    assert set(s) == {"mean_ic", "ic_ir", "nw_t", "hit_rate", "n_days"}
    assert 0 <= s["hit_rate"] <= 1
    ir = metrics.portfolio_ir(panel.X[:, 0], panel.y, panel.day_starts)
    assert np.isfinite(ir)
