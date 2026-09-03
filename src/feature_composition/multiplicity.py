"""Family-wise multiplicity corrections for a family of paired method-versus-benchmark tests.

The study ran 56 paired comparisons against the benchmark. Seven cleared |t| > 2 nominally;
after Holm, Benjamini-Hochberg or the Westfall-Young max-T bootstrap, zero survived,
including the study's own headline win. The only results with any post-correction standing
were two *negative* ones. This module is that accounting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import norm

from feature_composition import metrics

if TYPE_CHECKING:
    from numpy.typing import NDArray


def holm(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values (controls the false discovery rate)."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return adj


def _hac_t_columns(D: NDArray[np.float64], lags: int = metrics.NW_LAGS) -> NDArray[np.float64]:
    n = D.shape[0]
    Dc = D - D.mean(0)
    v = (Dc * Dc).sum(0) / n
    for j in range(1, lags + 1):
        v = v + 2.0 * (1.0 - j / (lags + 1.0)) * (Dc[:-j] * Dc[j:]).sum(0) / n
    return D.mean(0) / np.sqrt(np.maximum(v, 1e-18) / n)


def westfall_young_maxt(
    diffs: Sequence[NDArray[np.float64]],
    t_obs: NDArray[np.float64],
    block_len: int = 50,
    n_boot: int = 4000,
    seed: int = 7,
    lags: int = metrics.NW_LAGS,
) -> tuple[NDArray[np.float64], float]:
    """Westfall-Young max-T adjusted p-values by a circular block bootstrap.

    ``diffs`` are the daily paired-difference series (NaN where a day is not scored); series
    of equal length are resampled with the same day draws so their dependence is preserved.
    The null is enforced by centering. Returns the adjusted p-values and the 95% max-|t| bar.
    """
    rng = np.random.default_rng(seed)
    groups: dict[int, list[NDArray[np.float64]]] = {}
    for d in diffs:
        centered = np.where(np.isnan(d), 0.0, d - np.nanmean(d))
        groups.setdefault(len(d), []).append(centered)
    max_t = np.zeros(n_boot)
    for n, members in groups.items():
        D = np.column_stack(members)
        n_blocks = int(np.ceil(n / block_len))
        for b in range(n_boot):
            starts = rng.integers(0, n, n_blocks)
            idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n] % n
            max_t[b] = max(max_t[b], float(np.abs(_hac_t_columns(D[idx], lags)).max()))
    adj = np.array([(max_t >= tv).mean() for tv in np.abs(t_obs)])
    return adj, float(np.quantile(max_t, 0.95))


def multiplicity_table(
    tests: Sequence[tuple[str, str, NDArray[np.float64]]],
    block_len: int = 50,
    n_boot: int = 4000,
    seed: int = 7,
) -> tuple[pl.DataFrame, float]:
    """Nominal and corrected p-values for a family of ``(class, method, daily_diff)`` tests.

    Returns the table sorted by nominal p together with the max-T 95% bar on |t|.
    """
    rows, diffs = [], []
    for cls, method, d in tests:
        mean, t = metrics.newey_west(d[~np.isnan(d)])
        rows.append(
            {
                "class": cls,
                "method": method,
                "diff": mean,
                "t": t,
                "p": float(2 * (1 - norm.cdf(abs(t)))),
            }
        )
        diffs.append(d)
    p = np.array([r["p"] for r in rows])
    t_obs = np.array([r["t"] for r in rows])
    wy, bar = westfall_young_maxt(diffs, t_obs, block_len, n_boot, seed)
    table = pl.DataFrame(rows).with_columns(
        pl.Series("p_holm", holm(p)),
        pl.Series("p_bh", benjamini_hochberg(p)),
        pl.Series("p_wy_maxT", wy),
    )
    return table.sort("p"), bar
