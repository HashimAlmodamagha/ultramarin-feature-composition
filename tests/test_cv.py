from __future__ import annotations

import numpy as np

from feature_composition.cv import cross_validate, inner_split, paired_against, purged_kfold
from feature_composition.linear import ic_weight, naive_averaged


def test_purged_folds_are_contiguous_disjoint_and_buffered():
    n_days, horizon = 300, 21
    seen = np.zeros(n_days, dtype=int)
    for f, train, test in purged_kfold(n_days, 5, horizon, horizon):
        assert 0 <= f < 5
        assert not np.any(train & test)
        idx = np.flatnonzero(test)
        assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))  # one contiguous block
        seen[test] += 1
        # every training day is at least `horizon` days before or `embargo` days after the block
        before = np.flatnonzero(train & (np.arange(n_days) < idx[0]))
        after = np.flatnonzero(train & (np.arange(n_days) > idx[-1]))
        if before.size:
            assert idx[0] - before.max() > horizon
        if after.size:
            assert after.min() - idx[-1] > horizon
    assert np.all(seen == 1)  # every day is tested exactly once


def test_inner_split_is_chronological_and_purged():
    n_days, horizon = 200, 21
    train = np.ones(n_days, dtype=bool)
    itr, iv = inner_split(train, n_days, horizon, 0.75)
    assert not np.any(itr & iv)
    assert np.flatnonzero(itr).max() < np.flatnonzero(iv).min()
    assert np.flatnonzero(iv).min() - np.flatnonzero(itr).max() >= horizon


def test_cross_validate_and_paired_tests(panel):
    table, oof = cross_validate(panel, {"naive": naive_averaged, "icw": ic_weight})
    assert set(table["method"]) == {"naive", "icw"}
    assert table["cv_mean_ic"].is_sorted(descending=True)
    # every day sits in exactly one test block, so the stitched series is complete
    assert np.isfinite(oof["naive"]).all() and oof["naive"].shape == (panel.n_days,)
    paired = paired_against(oof, "naive")
    assert paired.height == 1 and paired["method"][0] == "icw"
