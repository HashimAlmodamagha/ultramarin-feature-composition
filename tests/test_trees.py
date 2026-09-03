from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from feature_composition import trees  # noqa: E402
from feature_composition.composite import DedupComposite  # noqa: E402
from feature_composition.gate import gate_decision  # noqa: E402
from feature_composition.panel import Panel  # noqa: E402


@pytest.fixture(scope="module")
def aligned(frame) -> Panel:
    return Panel(frame, reflect=["f2", "f6"], name="aligned")


def test_stump_shapes_reconstruct_the_model_exactly(aligned):
    """On the tabulation grid the additive curves reproduce the booster's own predictions.

    Between grid points the curves are linearly interpolated (a deliberate smoothing of the
    step functions, and what the study evaluated), so exactness is checked on the grid.
    """
    X, y = aligned.X[:4000], aligned.y[:4000]
    shapes = trees.fit_stump_shapes(X, y, n_stumps=60)
    assert shapes.shape == (X.shape[1], len(trees.SHAPE_GRID))
    bst = xgb.train(
        dict(trees.XGB_PARAMS, max_depth=1), xgb.DMatrix(X, label=y), num_boost_round=60
    )
    import json

    base = float(
        json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]")
    )
    rng = np.random.default_rng(0)
    Xg = trees.SHAPE_GRID[rng.integers(0, len(trees.SHAPE_GRID), size=X.shape)]
    pred = bst.predict(xgb.DMatrix(Xg))
    recon = base + trees.apply_shapes(Xg, shapes).sum(1)
    assert np.abs(pred - recon).max() < 1e-5


def test_shape_response_split_sums_to_one():
    shapes = np.cumsum(np.random.default_rng(0).normal(size=(3, 400)), axis=1)
    lo, mid, hi = trees.shape_response_split(shapes)
    assert lo + mid + hi == pytest.approx(1.0)


def test_shaped_composite_fits_transforms_and_serializes(aligned, tmp_path):
    comp = DedupComposite(shapes=True, n_stumps=40).fit(aligned)
    sig = comp.transform_panel(aligned)
    assert sig.shape == (aligned.n_rows,)
    comp.save(tmp_path / "shaped.json")
    again = DedupComposite.load(tmp_path / "shaped.json")
    np.testing.assert_allclose(again.transform_panel(aligned), sig)
    v2 = DedupComposite(shapes=True, align_shaped_members=True, n_stumps=40).fit(aligned)
    assert v2.shaped_signs is not None and set(np.unique(v2.shaped_signs)) <= {-1.0, 1.0}


def test_xgb_signal_and_quantile_bucket(aligned):
    train = next(aligned.folds())[1]
    log: list[int] = []
    sig = trees.xgb_signal(aligned, train, log=log)
    assert sig.shape == (aligned.n_rows,) and log[0] in trees.XGB_ROUNDS
    b = trees.quantile_bucket(aligned, sig, 5)
    assert set(np.unique(b)) <= set(range(5))


def test_gate_decision_reports_both_conditions(aligned):
    d = gate_decision(aligned, n_stumps=30)
    assert isinstance(d.adopt_shapes, bool)
    assert d.adopt_shapes == (d.pass_a and d.pass_b)
    assert len(d.per_fold_shaped) == aligned.n_folds
    assert "accuracy non-inferiority" in str(d)
