from __future__ import annotations

import numpy as np
import pytest

from feature_composition import cluster_methods, linear
from feature_composition.panel import Panel


@pytest.fixture(scope="module")
def aligned(frame) -> Panel:
    return Panel(frame, reflect=["f2", "f6"], name="aligned")


def test_pls_one_component_is_ic_weighting_and_full_pls_is_ols(aligned):
    Z, _, rows = aligned.fit(aligned.all_days)
    y = aligned.y
    w1 = linear.pls_coef(Z, rows, y, 1)
    direction = Z[rows].T @ (y[rows] - y[rows].mean())
    cos = w1 @ direction / (np.linalg.norm(w1) * np.linalg.norm(direction))
    assert cos == pytest.approx(1.0, abs=1e-9)
    wp = linear.pls_coef(Z, rows, y, Z.shape[1])
    ols = np.linalg.lstsq(Z[rows], y[rows] - y[rows].mean(), rcond=None)[0]
    np.testing.assert_allclose(wp, ols, atol=1e-8)


def test_lasso_path_is_monotone_in_sparsity(aligned):
    Z, rows = linear._standardize(aligned, aligned.all_days)
    nnz = [
        int((np.abs(linear.coordinate_descent_lasso(Z[rows], aligned.y[rows], f)) > 0).sum())
        for f in (1.0, 0.5, 0.1, 0.01)
    ]
    assert nnz[0] == 0
    assert nnz == sorted(nnz)


def test_every_linear_method_returns_a_full_length_signal(aligned):
    train = next(aligned.folds())[1]
    for name, fn in linear.LINEAR_METHODS.items():
        sig = fn(aligned, train)
        assert sig.shape == (aligned.n_rows,), name
        assert np.isfinite(sig).all(), name


def test_dedup_with_singletons_matches_ic_weighting_direction(aligned):
    """At rho -> 1 every feature is its own block: equal signed weights on standardized
    features, i.e. the sign-aligned average of z-scores."""
    train = aligned.all_days
    sig = cluster_methods.cluster_signal(aligned, train, rho=0.999, across="eq")
    Z, ic, _ = aligned.fit(train)
    ref = Z @ (np.where(ic >= 0, 1.0, -1.0) / np.sqrt(len(ic)))
    np.testing.assert_allclose(sig, ref, atol=1e-9)


def test_dedup_beats_the_benchmark_when_duplication_bias_binds(aligned):
    """The fixture re-implements one idea four times and another once; the plain average
    is that first idea four times over, dedup counts each idea once."""
    from feature_composition.cv import cross_validate

    table, _ = cross_validate(
        aligned,
        {"naive": linear.naive_averaged, "dedup": cluster_methods.cluster_eq},
    )
    ic = dict(zip(table["method"], table["cv_mean_ic"], strict=True))
    assert ic["dedup"] > ic["naive"]


def test_cluster_eb_reduces_to_cluster_eq_when_tau2_is_zero(aligned):
    log: list[dict[str, float]] = []
    train = aligned.all_days
    eb = cluster_methods.cluster_eb(aligned, train, log=log)
    eq = cluster_methods.cluster_eq(aligned, train)
    if log[0]["tau2"] == 0.0:
        np.testing.assert_allclose(eb, eq, atol=1e-9)
    else:
        assert np.corrcoef(eb, eq)[0, 1] > 0.9


def test_two_level_pooling_signals(aligned):
    train = aligned.all_days
    sig, n_tilted = cluster_methods.eb_within(aligned, train)
    assert sig.shape == (aligned.n_rows,) and n_tilted >= 0
    flat = cluster_methods.eb_flat(aligned, train)
    assert np.isfinite(flat).all()


def test_tuned_cluster_logs_threshold_and_block_count(aligned):
    log: list[tuple[float, int]] = []
    train = next(aligned.folds())[1]
    cluster_methods.cluster_tuned(aligned, train, "eq", log=log)
    rho, k = log[0]
    assert rho in cluster_methods.RHO_GRID and 1 <= k <= len(aligned.cols)


def test_pls_sweep_shape(aligned):
    sweep = linear.pls_sweep(aligned)
    assert len(sweep["k"]) == min(8, len(aligned.cols))
    assert len(sweep["mean"]) == len(sweep["k"])
