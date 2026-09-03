from __future__ import annotations

import numpy as np
import pytest

from feature_composition import multiplicity
from feature_composition.cli import main
from feature_composition.synthetic import make_panel_frames


def test_holm_and_bh_on_a_textbook_example():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(multiplicity.holm(p), [0.04, 0.09, 0.09, 0.20])
    np.testing.assert_allclose(
        multiplicity.benjamini_hochberg(p), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20]
    )


def test_westfall_young_maxt_calibrates_under_the_null():
    rng = np.random.default_rng(3)
    diffs = [rng.normal(size=400) for _ in range(6)]
    t_obs = np.array([multiplicity.metrics.newey_west(d)[1] for d in diffs])
    adj, bar = multiplicity.westfall_young_maxt(diffs, t_obs, block_len=20, n_boot=200, seed=1)
    assert adj.shape == (6,) and np.all((adj >= 0) & (adj <= 1))
    assert 2.0 < bar < 5.0
    table, _ = multiplicity.multiplicity_table(
        [("c1", f"m{i}", d) for i, d in enumerate(diffs)], block_len=20, n_boot=100
    )
    assert table.columns == ["class", "method", "diff", "t", "p", "p_holm", "p_bh", "p_wy_maxT"]
    assert table["p"].is_sorted()
    assert np.all(table["p_holm"].to_numpy() >= table["p"].to_numpy())


def test_synthetic_frames_have_the_ultramarin_schema():
    feats, tgts = make_panel_frames(n_days=30, n_names=20, block_sizes=(2, 1), seed=0)
    assert feats.columns == ["date", "identifier", "f1", "f2", "f3"]
    assert tgts.columns == ["date", "identifier", "target"]
    assert feats["date"].dtype.is_integer()
    assert feats.height == 30 * 20
    assert feats["f1"].min() >= 0.0 and feats["f1"].max() <= 1.0
    # target is de-meaned per day
    assert (
        abs(tgts.group_by("date").agg(m=__import__("polars").col("target").mean())["m"].max())
        < 1e-9
    )


def test_cli_demo_runs(capsys):
    assert main(["demo", "--n-days", "120", "--n-names", "60"]) == 0
    out = capsys.readouterr().out
    assert "selected composite" in out and "benchmark" in out


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
