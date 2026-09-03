"""The evaluation harness: daily rank IC, Newey-West significance, decay and turnover.

Every number in the study is produced by the functions in this module, and nothing here
depends on any composition method. The measuring apparatus was fixed before the first
comparison ran and is reused unchanged by the cross-validation, the stress tests and the
blind hold-out.

Conventions
-----------
* A *panel* is sorted by ``(date, identifier)`` so that each trading day is one contiguous
  block of rows; ``day_starts`` holds the row offsets of those blocks (length ``n_days + 1``).
* A *signal* is a float vector with one entry per panel row.
* A *day mask* is a boolean vector with one entry per trading day.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import rankdata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

#: Bartlett-kernel lag length used for every Newey-West statistic in the study. The target is
#: a ~21-day overlapping forward return, so 25 lags cover the induced serial correlation.
NW_LAGS = 25


def series_mean(s: pl.Series) -> float:
    """Mean of a polars series as a float (nulls ignored, NaN if empty)."""
    v = s.mean()
    return float("nan") if v is None else float(v)  # type: ignore[arg-type]


def pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Pearson correlation of two equal-length vectors; NaN if either is constant."""
    a = a - a.mean()
    b = b - b.mean()
    da, db = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
    return float((a @ b) / (da * db)) if da > 0 and db > 0 else float("nan")


def rank_by_day(values: NDArray[np.float64], day_starts: NDArray[np.int64]) -> NDArray[np.float64]:
    """Cross-sectional (within-day) average ranks of a row vector."""
    out = np.empty(len(values), dtype=float)
    for k in range(len(day_starts) - 1):
        a, b = day_starts[k], day_starts[k + 1]
        out[a:b] = rankdata(values[a:b])
    return out


def daily_rank_ic(
    signal: NDArray[np.float64],
    target_rank: NDArray[np.float64],
    day_starts: NDArray[np.int64],
    days: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """Spearman IC of ``signal`` against the (pre-ranked) target, one value per requested day.

    ``days`` defaults to every trading day in the panel.
    """
    idx = np.arange(len(day_starts) - 1) if days is None else np.asarray(list(days), dtype=int)
    out = np.empty(len(idx))
    for i, k in enumerate(idx):
        a, b = day_starts[k], day_starts[k + 1]
        out[i] = pearson(rankdata(signal[a:b]), target_rank[a:b])
    return out


def hac_variance(x: NDArray[np.float64], lags: int = NW_LAGS) -> float:
    """Newey-West (Bartlett kernel) long-run variance of a series, without small-sample scaling."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    d = x - x.mean()
    v = float(d @ d) / n
    for j in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - j / (lags + 1.0)) * float(d[:-j] @ d[j:]) / n
    return max(v, 1e-18)


def newey_west(x: NDArray[np.float64], lags: int = NW_LAGS) -> tuple[float, float]:
    """HAC mean and t-statistic of a serially correlated series (mean, t).

    Numerically identical to ``statsmodels`` OLS on a constant with ``cov_type="HAC"`` and
    ``maxlags=lags``; the dependency is not needed.
    """
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    mean = float(x.mean())
    se = float(np.sqrt(hac_variance(x, lags) / len(x)))
    return mean, mean / se


def hac_se(x: NDArray[np.float64], lags: int = NW_LAGS) -> float:
    """Newey-West standard error of the mean of a series."""
    x = x[~np.isnan(x)]
    return float(np.sqrt(hac_variance(x, lags) / len(x))) if len(x) else float("nan")


def paired_newey_west(
    a: NDArray[np.float64], b: NDArray[np.float64], lags: int = NW_LAGS
) -> tuple[float, float]:
    """Paired HAC test of two daily-IC series on the same days: mean and t of ``a - b``."""
    d = a - b
    return newey_west(d[~np.isnan(d)], lags)


def summarize_ic(ic: NDArray[np.float64]) -> dict[str, float]:
    """The IC scorecard: mean, per-day information ratio, Newey-West t, hit rate, day count."""
    mean, t = newey_west(ic)
    ic = ic[~np.isnan(ic)]
    return {
        "mean_ic": mean,
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan"),
        "nw_t": t,
        "hit_rate": float((ic > 0).mean()),
        "n_days": float(len(ic)),
    }


def half_life(curve: NDArray[np.float64]) -> float:
    """Lag at which a decay curve first crosses half its lag-0 value (linear interpolation).

    NaN when the curve never drops that far inside the window.
    """
    half = 0.5 * curve[0]
    for i in range(1, len(curve)):
        if curve[i] <= half:
            return float((i - 1) + (curve[i - 1] - half) / (curve[i - 1] - curve[i]))
    return float("nan")


def _frame(
    dates: NDArray[np.int64], ids: NDArray[np.str_], name: str, values: NDArray[np.float64]
) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, "identifier": ids, name: values})


def decay_curve(
    dates: NDArray[np.int64],
    ids: NDArray[np.str_],
    signal: NDArray[np.float64],
    target: NDArray[np.float64],
    lags: Iterable[int] = range(0, 22),
) -> NDArray[np.float64]:
    """Signal decay: mean rank IC of ``signal_{t-l}`` against ``target_t``, aligned by name.

    Dates are integer trading-day indices, so lagging is a plain integer shift.
    """
    sig = _frame(dates, ids, "s", signal)
    tgt = _frame(dates, ids, "target", target)
    out = []
    for lag in lags:
        m = (
            sig.with_columns((pl.col("date") + lag).alias("date"))
            .join(tgt, on=["date", "identifier"], how="inner")
            .with_columns(
                pl.col("s").rank().over("date").alias("sr"),
                pl.col("target").rank().over("date").alias("tr"),
            )
        )
        out.append(series_mean(m.group_by("date").agg(pl.corr("sr", "tr").alias("ic"))["ic"]))
    return np.asarray(out, dtype=float)


def autocorr_curve(
    dates: NDArray[np.int64],
    ids: NDArray[np.str_],
    signal: NDArray[np.float64],
    lags: Iterable[int] = range(1, 11),
) -> NDArray[np.float64]:
    """Factor autocorrelation ``E[corr(F_{t-l}, F_t)]``; lag 0 is 1 by definition.

    Turnover at lag ``l`` is ``1 - autocorr[l]``.
    """
    sig = _frame(dates, ids, "s", signal)
    out = [1.0]
    for lag in lags:
        m = (
            sig.with_columns((pl.col("date") + lag).alias("date"))
            .rename({"s": "s_lag"})
            .join(sig, on=["date", "identifier"], how="inner")
        )
        out.append(series_mean(m.group_by("date").agg(pl.corr("s_lag", "s").alias("ac"))["ac"]))
    return np.asarray(out, dtype=float)


def tradeability(
    dates: NDArray[np.int64],
    ids: NDArray[np.str_],
    signal: NDArray[np.float64],
    target: NDArray[np.float64],
    decay_max: int = 21,
    ac_max: int = 10,
) -> dict[str, object]:
    """Half-life, 5-day IC retention and 1/5-day turnover of one signal, plus the raw curves."""
    d = decay_curve(dates, ids, signal, target, range(0, decay_max + 1))
    ac = autocorr_curve(dates, ids, signal, range(1, ac_max + 1))
    return {
        "half_life": half_life(d),
        "ret_5d": float(d[5] / d[0]),
        "turn_l1": float(1 - ac[1]),
        "turn_l5": float(1 - ac[5]),
        "decay": d,
        "autocorr": ac,
    }


def weekly_ic_series(
    dates: NDArray[np.int64],
    ids: NDArray[np.str_],
    signal: NDArray[np.float64],
    target: NDArray[np.float64],
    lags: tuple[int, ...] = (0, 5),
) -> NDArray[np.float64]:
    """Daily series of the weekly-book criterion: the average of the fresh and week-old rank IC.

    A weekly rebalance trades a signal that is zero to five days stale, so pricing the level
    and the retention together is what such a book actually earns.
    """
    sig = _frame(dates, ids, "s", signal)
    tgt = _frame(dates, ids, "target", target)
    legs = []
    for lag in lags:
        m = (
            sig.with_columns((pl.col("date") + lag).alias("date"))
            .join(tgt, on=["date", "identifier"], how="inner")
            .with_columns(
                pl.col("s").rank().over("date").alias("sr"),
                pl.col("target").rank().over("date").alias("tr"),
            )
            .group_by("date")
            .agg(pl.corr("sr", "tr").alias(f"ic{lag}"))
            .sort("date")
        )
        legs.append(m)
    joined = legs[0]
    for leg in legs[1:]:
        joined = joined.join(leg, on="date", how="inner")
    cols = [f"ic{lag}" for lag in lags]
    return joined.select(pl.mean_horizontal(cols).alias("wk"))["wk"].to_numpy().astype(float)


def portfolio_ir(
    signal: NDArray[np.float64],
    target: NDArray[np.float64],
    day_starts: NDArray[np.int64],
    periods_per_year: int = 252,
) -> float:
    """Annualized IR of a dollar-neutral, unit-gross, rank-weighted long-short portfolio."""
    rets = []
    for k in range(len(day_starts) - 1):
        a, b = day_starts[k], day_starts[k + 1]
        z = rankdata(signal[a:b]).astype(float)
        z -= z.mean()
        gross = float(np.abs(z).sum())
        if gross > 0:
            rets.append(float((z / gross) @ target[a:b]))
    r = np.asarray(rets)
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
