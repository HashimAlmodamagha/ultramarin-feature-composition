"""A feature-class panel: the numpy core, purged folds and per-feature ICs behind every method.

One ``Panel`` is one feature class (a cluster of correlated variants of the same idea) joined
to its target. It owns the leak-free ingredients that every composition method needs and that
the study computed exactly once per class: the day index, the target ranks, the per-feature
daily IC matrix, the purged chronological folds and the inner validation split used for
tuning.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import rankdata

from feature_composition import metrics
from feature_composition.cv import inner_split, purged_kfold

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: The label span of the target in trading days, inferred from its autocorrelation triangle.
#: It sets the purge/embargo window and the Newey-West lag length.
HORIZON = 21
N_FOLDS = 5

Method = Callable[["Panel", "NDArray[np.bool_]"], "NDArray[np.float64]"]


class Panel:
    """A complete-case, day-sorted panel of one feature class plus its target.

    Parameters
    ----------
    frame:
        Columns ``date`` (integer trading-day index), ``identifier`` (name), the feature
        columns and ``target``. Rows with a missing target or any missing feature are dropped.
    feature_cols:
        Feature columns; defaults to every column other than ``date``/``identifier``/``target``.
    reflect:
        Feature columns to reflect as ``x -> 1 - x`` (percentile-rank sign flip) before
        anything else is computed. The study locks class 1's orientation this way.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        feature_cols: list[str] | None = None,
        *,
        reflect: Iterable[str] = (),
        horizon: int = HORIZON,
        n_folds: int = N_FOLDS,
        name: str = "panel",
    ) -> None:
        cols = feature_cols or [
            c for c in frame.columns if c not in ("date", "identifier", "target")
        ]
        n0 = frame.height
        frame = frame.filter(pl.col("target").is_not_null())
        frame = frame.filter(~pl.any_horizontal([pl.col(c).is_null() for c in cols]))
        frame = frame.sort(["date", "identifier"])
        flips = [c for c in reflect if c in cols]
        if flips:
            frame = frame.with_columns([(1.0 - pl.col(c)).alias(c) for c in flips])

        self.name = name
        self.frame = frame
        self.cols: list[str] = list(cols)
        self.flipped: list[str] = flips
        self.n_dropped = n0 - frame.height
        self.horizon = horizon
        self.n_folds = n_folds

        self.X: NDArray[np.float64] = frame.select(cols).to_numpy().astype(float)
        self.y: NDArray[np.float64] = frame["target"].to_numpy().astype(float)
        self.dates: NDArray[np.int64] = frame["date"].to_numpy()
        self.ids: NDArray[np.str_] = frame["identifier"].to_numpy()
        self.n_rows = len(self.y)
        _, starts = np.unique(self.dates, return_index=True)
        self.day_starts: NDArray[np.int64] = np.append(starts, self.n_rows)
        self.n_days = len(self.day_starts) - 1
        self.all_days: NDArray[np.bool_] = np.ones(self.n_days, dtype=bool)
        self.y_rank: NDArray[np.float64] = metrics.rank_by_day(self.y, self.day_starts)
        self._feature_ic: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_parquet(
        cls,
        features_path: str | Path,
        target_path: str | Path,
        *,
        reflect: Iterable[str] = (),
        name: str | None = None,
        horizon: int = HORIZON,
        n_folds: int = N_FOLDS,
    ) -> Panel:
        """Join a feature parquet and a target parquet on ``(date, identifier)``."""
        feats = pl.read_parquet(features_path)
        tgts = pl.read_parquet(target_path)
        frame = feats.join(tgts, on=["date", "identifier"], how="inner")
        return cls(
            frame,
            reflect=reflect,
            name=name or Path(features_path).stem,
            horizon=horizon,
            n_folds=n_folds,
        )

    @classmethod
    def load_class(
        cls,
        data_dir: str | Path,
        feature_class: int,
        split: str = "in_sample",
        *,
        reflect: Iterable[str] = (),
        horizon: int = HORIZON,
        n_folds: int = N_FOLDS,
    ) -> Panel:
        """Load ``feature_class_<n>_<split>.parquet`` and its target file from ``data_dir``."""
        data_dir = Path(data_dir)
        stem = f"feature_class_{feature_class}"
        return cls.from_parquet(
            data_dir / f"{stem}_{split}.parquet",
            data_dir / f"{stem}_target_{split}.parquet",
            reflect=reflect,
            name=f"{stem} ({split})",
            horizon=horizon,
            n_folds=n_folds,
        )

    def subset(self, feature_idx: Iterable[int]) -> Panel:
        """A view of the panel restricted to some feature columns (for the drop-menu tests).

        Shares the day index and target ranks; recomputes nothing but the feature slices.
        """
        idx = np.asarray(list(feature_idx), dtype=int)
        sub = object.__new__(Panel)
        sub.__dict__.update(self.__dict__)
        sub.cols = [self.cols[i] for i in idx]
        sub.X = self.X[:, idx]
        sub._feature_ic = None if self._feature_ic is None else self._feature_ic[:, idx]
        return sub

    # -------------------------------------------------------------------- core ingredients
    @property
    def feature_ic(self) -> NDArray[np.float64]:
        """Per-feature daily rank IC, ``(n_days, n_features)``, computed once and cached.

        Ranks are invariant to any per-fold standardization, so a training fold's feature-IC
        vector is just a masked mean over these rows.
        """
        if self._feature_ic is None:
            fic = np.empty((self.n_days, len(self.cols)))
            for j in range(len(self.cols)):
                col = self.X[:, j]
                for k in range(self.n_days):
                    a, b = self.day_starts[k], self.day_starts[k + 1]
                    fic[k, j] = metrics.pearson(rankdata(col[a:b]), self.y_rank[a:b])
            self._feature_ic = fic
        return self._feature_ic

    def rows_of(self, day_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Row mask covering every row of the selected days."""
        out = np.zeros(self.n_rows, dtype=bool)
        for k in np.flatnonzero(day_mask):
            out[self.day_starts[k] : self.day_starts[k + 1]] = True
        return out

    def daily_ic(
        self, signal: NDArray[np.float64], days: Iterable[int] | None = None
    ) -> NDArray[np.float64]:
        """Daily rank IC of a signal on the requested day indices (default: all days)."""
        return metrics.daily_rank_ic(signal, self.y_rank, self.day_starts, days)

    def mean_ic_on(self, signal: NDArray[np.float64], day_mask: NDArray[np.bool_]) -> float:
        """Mean daily rank IC over a day mask; ``-inf`` for a degenerate (constant) signal."""
        ic = self.daily_ic(signal, np.flatnonzero(day_mask))
        ic = ic[~np.isnan(ic)]
        return float(ic.mean()) if ic.size else float("-inf")

    def fit(
        self, train_days: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Leak-free per-fold ingredients: ``(Z, ic, train_rows)``.

        ``Z`` is the full panel standardized with train-only moments, ``ic`` the train-day mean
        of each feature's daily IC and ``train_rows`` the row mask of the training days.
        """
        rows = self.rows_of(train_days)
        Z = np.asarray((self.X - self.X[rows].mean(0)) / self.X[rows].std(0), dtype=np.float64)
        ic = np.asarray(np.nanmean(self.feature_ic[train_days], axis=0), dtype=np.float64)
        return Z, ic, rows

    def correlation(self, rows: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
        """Feature correlation matrix over the given rows (default: all rows)."""
        return np.asarray(
            np.corrcoef(self.X if rows is None else self.X[rows], rowvar=False), dtype=np.float64
        )

    # ------------------------------------------------------------------- cross-validation
    def folds(self) -> Iterator[tuple[int, NDArray[np.bool_], NDArray[np.bool_]]]:
        """Purged, embargoed chronological k-fold over trading days: ``(fold, train, test)``."""
        return purged_kfold(self.n_days, self.n_folds, self.horizon, self.horizon)

    def inner_split(
        self, train_days: NDArray[np.bool_], frac: float = 0.75
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Chronological inner train/validation split of a training mask, purged at the cut."""
        return inner_split(train_days, self.n_days, self.horizon, frac)

    def oof_run(self, method: Method) -> tuple[list[float], NDArray[np.float64]]:
        """Run a method through the purged folds.

        Returns the per-fold mean out-of-fold IC and the stitched out-of-fold daily IC series
        (NaN on the purged days that belong to no test fold).
        """
        oof = np.full(self.n_days, np.nan)
        per_fold = []
        for _, train, test in self.folds():
            signal = method(self, train)
            days = np.flatnonzero(test)
            oof[days] = self.daily_ic(signal, days)
            per_fold.append(float(np.nanmean(oof[days])))
        return per_fold, oof

    # -------------------------------------------------------------------- tradeability
    def tradeability(
        self, signal: NDArray[np.float64], decay_max: int = 21, ac_max: int = 10
    ) -> dict[str, object]:
        """Half-life, 5-day retention, turnover and the underlying curves of a signal."""
        return metrics.tradeability(self.dates, self.ids, signal, self.y, decay_max, ac_max)

    def weekly_ic(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Daily series of the weekly-book criterion (mean of lag-0 and lag-5 rank IC)."""
        return metrics.weekly_ic_series(self.dates, self.ids, signal, self.y)

    def weekly_score(
        self,
        signal: NDArray[np.float64],
        day_mask: NDArray[np.bool_],
        lags: tuple[int, ...] = (0, 5),
    ) -> float:
        """Weekly-book criterion on the selected days: mean over lags of the mean lagged rank IC."""
        udates = np.unique(self.dates)
        sel = pl.Series("date", udates[np.flatnonzero(day_mask)])
        base = pl.DataFrame({"date": self.dates, "identifier": self.ids, "s": signal})
        tgt = pl.DataFrame({"date": self.dates, "identifier": self.ids, "t": self.y})
        out = []
        for lag in lags:
            m = (
                base.with_columns((pl.col("date") + lag).alias("date"))
                .join(tgt, on=["date", "identifier"], how="inner")
                .filter(pl.col("date").is_in(sel.implode()))
                .with_columns(
                    sr=pl.col("s").rank().over("date"), tr=pl.col("t").rank().over("date")
                )
                .group_by("date")
                .agg(pl.corr("sr", "tr").alias("ic"))
            )
            out.append(metrics.series_mean(m["ic"]))
        return float(np.mean(out))

    def __repr__(self) -> str:
        return (
            f"Panel({self.name}: {self.n_rows:,} rows, {self.n_days:,} days, "
            f"{len(self.cols)} features, flipped={self.flipped})"
        )
