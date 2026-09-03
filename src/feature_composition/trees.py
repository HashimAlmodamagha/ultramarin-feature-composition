"""Gradient-boosted trees and the per-feature response curves they contain.

A depth-3 boosted tree matched the linear methods on accuracy but aged better: nearly twice
the half-life at lower turnover. The ablation showed that depth-1 *stumps*, which cannot
represent interactions, reproduce the full tree's profile, so the edge is a set of learned
additive per-feature transforms. A stump ensemble is exactly an additive model, and the
production version is one lookup table per feature: ``x_j -> g_j(x_j)``.

``xgboost`` is an optional dependency (``pip install "ultramarin-feature-composition[trees]"``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from feature_composition.panel import Panel

#: Tree hyper-parameters; ``max_depth=1`` turns the same fit into the stump model.
XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 1000,
    "reg_lambda": 5.0,
    "seed": 0,
}
XGB_ROUNDS = (25, 50, 100, 150, 200, 300, 400, 600)
#: Tree count of the canonical stump fit. Deliberately NOT tuned: the gate that decides
#: whether shapes are used at all, not the fit, protects classes where shapes do not help.
N_STUMPS = 600
#: Rank grid on which the response curves are tabulated (features are ranks in [0, 1]).
SHAPE_GRID: NDArray[np.float64] = np.linspace(0.001, 0.999, 400, dtype=np.float64)


def _xgb() -> Any:
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "xgboost is required for the tree methods: "
            'pip install "ultramarin-feature-composition[trees]"'
        ) from exc
    return xgboost


def xgb_signal(
    panel: Panel,
    train_days: NDArray[np.bool_],
    X: NDArray[np.float64] | None = None,
    params: dict[str, Any] | None = None,
    log: list[int] | None = None,
) -> NDArray[np.float64]:
    """Boosted-tree composite with the tree count chosen on the inner validation slice.

    Trees use the raw rank features (scale-invariant). Optional ``X`` and ``params``
    overrides serve the mechanism probes (stumps, one-dimensional input).
    """
    xgb = _xgb()
    Xu = panel.X if X is None else X
    prm = dict(XGB_PARAMS if params is None else params)
    dall = xgb.DMatrix(Xu)

    def fit(rows: NDArray[np.bool_], n_trees: int) -> Any:
        return xgb.train(prm, xgb.DMatrix(Xu[rows], label=panel.y[rows]), num_boost_round=n_trees)

    itr, iv = panel.inner_split(train_days)
    bst = fit(panel.rows_of(itr), max(XGB_ROUNDS))
    n = max(
        XGB_ROUNDS,
        key=lambda nt: panel.mean_ic_on(bst.predict(dall, iteration_range=(0, nt)), iv),
    )
    if log is not None:
        log.append(n)
    return np.asarray(fit(panel.rows_of(train_days), n).predict(dall), dtype=float)


def quantile_bucket(panel: Panel, signal: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Per-day quantile bucketing into ``k`` bins: pure coarsening, no learning (a control)."""
    out = np.empty_like(signal)
    for d in range(panel.n_days):
        a, b = panel.day_starts[d], panel.day_starts[d + 1]
        ranks = signal[a:b].argsort().argsort()
        out[a:b] = np.floor(ranks * k / (b - a))
    return out


def fit_stump_shapes(
    X: NDArray[np.float64], y: NDArray[np.float64], n_stumps: int = N_STUMPS
) -> NDArray[np.float64]:
    """Depth-1 boosted fit -> exact per-feature response curves ``g_j`` on ``SHAPE_GRID``.

    Every stump splits one feature at one threshold and adds two leaf values, so summing the
    stumps per feature gives the additive model exactly (the tree dump is read at full float
    precision). The returned array is ``(n_features, len(SHAPE_GRID))``.
    """
    xgb = _xgb()
    bst = xgb.train(
        dict(XGB_PARAMS, max_depth=1), xgb.DMatrix(X, label=y), num_boost_round=n_stumps
    )
    shapes = np.zeros((X.shape[1], len(SHAPE_GRID)))
    for tree in bst.get_dump(dump_format="json"):
        t = json.loads(tree)
        if "children" not in t:
            continue
        j, thr = int(t["split"][1:]), float(t["split_condition"])
        kids = {c["nodeid"]: c["leaf"] for c in t["children"]}
        shapes[j] += np.where(thr > SHAPE_GRID, kids[1], kids[2])
    return shapes


def apply_shapes(X: NDArray[np.float64], shapes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pass every variant through its response curve: ``x~_j = g_j(x_j)`` by interpolation."""
    return np.column_stack([np.interp(X[:, j], SHAPE_GRID, shapes[j]) for j in range(X.shape[1])])


def shape_response_split(shapes: NDArray[np.float64]) -> tuple[float, float, float]:
    """Where the response lives: share of total curve movement in the low/mid/high rank third."""
    move = np.abs(np.diff(shapes, axis=1))
    n = move.shape[1]
    lo = move[:, : n // 3].sum()
    mid = move[:, n // 3 : 2 * n // 3].sum()
    hi = move[:, 2 * n // 3 :].sum()
    tot = lo + mid + hi
    return float(lo / tot), float(mid / tot), float(hi / tot)
