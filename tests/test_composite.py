from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from feature_composition.composite import DedupComposite
from feature_composition.diagnostics import diagnose
from feature_composition.holdout import rolling_mean_ic, score_signal
from feature_composition.panel import Panel


@pytest.fixture(scope="module")
def split(frame):
    cut = int(frame["date"].min()) + 110
    train = Panel(frame.filter(pl.col("date") < cut), reflect=["f2", "f6"], name="train")
    test = Panel(frame.filter(pl.col("date") >= cut + 21), reflect=["f2", "f6"], name="test")
    return train, test


def test_fit_transform_and_feature_weights(split):
    train, test = split
    comp = DedupComposite().fit(train)
    assert comp.diagnostics["n_blocks"] == 3
    sig = comp.transform_panel(test)
    assert sig.shape == (test.n_rows,)
    # transform is the frozen linear map Z @ feature_weights
    Z = (test.X - comp.mu) / comp.sd
    np.testing.assert_allclose(sig, Z @ comp.feature_weights, atol=1e-12)
    assert {c for members in comp.blocks().values() for c in members} == set(train.cols)
    assert "3 ideas" in comp.describe()


def test_json_round_trip_reproduces_the_signal(split, tmp_path):
    train, test = split
    comp = DedupComposite().fit(train)
    path = tmp_path / "c.json"
    comp.save(path)
    again = DedupComposite.load(path)
    np.testing.assert_allclose(again.transform_panel(test), comp.transform_panel(test))
    assert json.loads(path.read_text())["rho"] == 0.7


def test_transform_refuses_mismatched_panels(split, frame):
    train, _ = split
    comp = DedupComposite().fit(train)
    unreflected = Panel(frame, name="raw")
    with pytest.raises(ValueError, match="reflections"):
        comp.transform_panel(unreflected)
    with pytest.raises(ValueError, match="features"):
        comp.transform(train.X[:, :3])
    with pytest.raises(RuntimeError):
        DedupComposite().transform(train.X)


def test_adaptive_weights_option_and_diagnostics(split):
    train, _ = split
    eb = DedupComposite(block_weights="eb").fit(train)
    eq = DedupComposite(block_weights="eq").fit(train)
    assert eb.diagnostics["n_blocks"] == eq.diagnostics["n_blocks"]
    for key in ("tau2_dl", "tau2_pm", "tau2_reml"):
        assert eb.diagnostics[key] >= 0


def test_selected_composite_beats_benchmark_out_of_sample(split):
    train, test = split
    comp = DedupComposite().fit(train)
    bench = (test.X * comp.signs).mean(1)
    m_sel = score_signal(test, comp.transform_panel(test))
    m_ben = score_signal(test, bench)
    assert m_sel["mean_ic"] > m_ben["mean_ic"]
    assert m_sel["n_days"] == test.n_days


def test_diagnose_report_card(split):
    train, _ = split
    card = diagnose(train)
    assert card.n_blocks == 3
    assert card.reversed_variants == []
    text = card.render()
    assert "PLS-k out-of-fold IC" in text and "reading:" in text
    assert 1 <= card.pls_best_k <= 8


def test_rolling_mean_ic():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = rolling_mean_ic(x, window=2)
    assert np.isnan(out[0])
    assert out[1] == 1.5 and out[2] == 2.0 and out[3] == 4.0 and out[4] == 4.5
