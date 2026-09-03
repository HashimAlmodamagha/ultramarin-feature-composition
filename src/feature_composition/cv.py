"""Purged chronological cross-validation.

Labels are ~21-day overlapping forward returns and both features and ICs are serially
correlated, so random row splits leak near-identical events across train and test. Folds are
therefore contiguous blocks of trading days, and a horizon-long buffer around each test block
is purged from training (no training label's forward window reaches into the test period) and
embargoed after it (no test label reaches into the post-block training days). This is the
purged k-fold of Lopez de Prado.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from feature_composition import metrics

if TYPE_CHECKING:
    from numpy.typing import NDArray


def purged_kfold(
    n_days: int, n_folds: int = 5, horizon: int = 21, embargo: int = 21
) -> Iterator[tuple[int, NDArray[np.bool_], NDArray[np.bool_]]]:
    """Yield ``(fold, train_days, test_days)`` boolean day masks for purged k-fold CV."""
    bounds = np.linspace(0, n_days, n_folds + 1).astype(int)
    for f in range(n_folds):
        a, b = bounds[f], bounds[f + 1]
        test = np.zeros(n_days, dtype=bool)
        test[a:b] = True
        train = ~test
        train[max(0, a - horizon) : a] = False
        train[b : min(n_days, b + embargo)] = False
        yield f, train, test


def inner_split(
    train_days: NDArray[np.bool_], n_days: int, horizon: int = 21, frac: float = 0.75
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Last ``1 - frac`` of the training days become a purged inner validation slice.

    Every tuned hyper-parameter in the study (ridge lambda, PLS components, tree count, merge
    threshold) is chosen on this slice and never on the outer test block.
    """
    days = np.flatnonzero(train_days)
    cut = days[int(frac * len(days))]
    idx = np.arange(n_days)
    inner_train = train_days & (idx <= cut)
    inner_val = train_days & (idx > cut)
    inner_train[max(0, cut - horizon + 1) : cut + 1] = False
    return inner_train, inner_val


def cross_validate(
    panel: Any, methods: Mapping[str, Callable[..., NDArray[np.float64]]]
) -> tuple[pl.DataFrame, dict[str, NDArray[np.float64]]]:
    """Run several methods through a panel's purged folds.

    Returns a table (method, cv_mean_ic, cv_ic_std, worst_fold) sorted by mean IC and the
    stitched out-of-fold daily IC series per method, which feed the paired tests.
    """
    rows, oof = [], {}
    for name, fn in methods.items():
        folds, oof[name] = panel.oof_run(fn)
        rows.append(
            {
                "method": name,
                "cv_mean_ic": float(np.mean(folds)),
                "cv_ic_std": float(np.std(folds, ddof=1)) if len(folds) > 1 else float("nan"),
                "worst_fold": float(np.min(folds)),
            }
        )
    return pl.DataFrame(rows).sort("cv_mean_ic", descending=True), oof


def paired_against(
    oof: Mapping[str, NDArray[np.float64]], anchor: str, lags: int = metrics.NW_LAGS
) -> pl.DataFrame:
    """Paired Newey-West tests of every out-of-fold series against one anchor method."""
    rows = []
    for name, series in oof.items():
        if name == anchor:
            continue
        diff, t = metrics.paired_newey_west(series, oof[anchor], lags)
        rows.append({"method": name, "vs": anchor, "ic_diff": diff, "nw_t": t})
    return pl.DataFrame(rows)
