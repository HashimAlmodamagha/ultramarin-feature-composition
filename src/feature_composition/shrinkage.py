"""Random-effects pooling of block ICs, borrowed from clinical meta-analysis.

Each redundancy block is a *study* reporting an effect (its training-window mean IC) with a
standard error (the Newey-West s.e. of that mean). The between-study variance tau^2 decides
how much the block means are shrunk toward the pooled mean: tau^2 = 0 gives equal weights,
tau^2 > 0 a tilt toward the better blocks. The shrinkage strength is *estimated*, not tuned.

The study's finding: the detector is robust (DerSimonian-Laird, Paule-Mandel and REML agree
on the tau^2 map, fold by fold), but monetizing a positive tau^2 as differential weights lost
on the blind hold-out. The selected composite therefore keeps tau^2 as a diagnostic and acts
with equal weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


def tau2_dl(effects: NDArray[np.float64], ses: NDArray[np.float64]) -> float:
    """DerSimonian-Laird moment estimator of the between-study variance."""
    w = 1.0 / ses**2
    mu = float((w * effects).sum() / w.sum())
    Q = float((w * (effects - mu) ** 2).sum())
    C = float(w.sum() - (w**2).sum() / w.sum())
    k = len(effects)
    return max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0


def tau2_pm(effects: NDArray[np.float64], ses: NDArray[np.float64], itmax: int = 200) -> float:
    """Paule-Mandel estimator: solve ``Q(tau^2) = k - 1`` by bisection."""
    k = len(effects)

    def q_stat(t2: float) -> float:
        w = 1.0 / (ses**2 + t2)
        mu = (w * effects).sum() / w.sum()
        return float((w * (effects - mu) ** 2).sum())

    if k < 2 or q_stat(0.0) <= k - 1:
        return 0.0
    lo, hi = 0.0, float(np.var(effects, ddof=1) * 10 + (ses.max() ** 2) * 10 + 1e-8)
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q_stat(mid) > k - 1 else (lo, mid)
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def tau2_reml(effects: NDArray[np.float64], ses: NDArray[np.float64], itmax: int = 500) -> float:
    """Restricted maximum-likelihood estimator by fixed-point iteration."""
    if len(effects) < 2:
        return 0.0
    t2 = max(1e-10, float(np.var(effects, ddof=1) - (ses**2).mean()))
    for _ in range(itmax):
        w = 1.0 / (ses**2 + t2)
        mu = (w * effects).sum() / w.sum()
        num = (w**2 * ((effects - mu) ** 2 - ses**2)).sum() + (w**2).sum() / w.sum() * t2
        t2n = max(0.0, float(num / (w**2).sum()))
        if abs(t2n - t2) < 1e-12:
            return t2n
        t2 = t2n
    return t2


def dl_shrink(
    effects: NDArray[np.float64], ses: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Random-effects shrinkage of effects toward their pooled mean: ``(shrunk, tau2)``.

    The shrinkage factor of effect ``i`` is ``tau2 / (tau2 + se_i^2)``; with tau2 = 0 every
    effect collapses to the pooled mean, i.e. equal weights.
    """
    tau2 = tau2_dl(effects, ses)
    w_re = 1.0 / (ses**2 + tau2)
    mu_re = float((w_re * effects).sum() / w_re.sum())
    return mu_re + (tau2 / (tau2 + ses**2)) * (effects - mu_re), tau2


def shrink_with_tau2(
    effects: NDArray[np.float64], ses: NDArray[np.float64], tau2: float
) -> NDArray[np.float64]:
    """Random-effects shrinkage at a given tau^2 (to compare estimators on equal footing)."""
    w = 1.0 / (ses**2 + tau2)
    mu = float((w * effects).sum() / w.sum())
    return mu + (tau2 / (tau2 + ses**2)) * (effects - mu)


def hartung_knapp_interval(
    effects: NDArray[np.float64], ses: NDArray[np.float64], tau2: float, level: float = 0.95
) -> tuple[float, float, float]:
    """Hartung-Knapp interval for the pooled mean: ``(pooled, lo, hi)``; NaNs when ``k < 2``."""
    k = len(effects)
    w = 1.0 / (ses**2 + tau2)
    mu = float((w * effects).sum() / w.sum())
    if k < 2:
        return mu, float("nan"), float("nan")
    se_hk = float(np.sqrt(((w * (effects - mu) ** 2).sum() / (k - 1)) / w.sum()))
    tc = float(stats.t.ppf(0.5 + level / 2, k - 1))
    return mu, mu - tc * se_hk, mu + tc * se_hk


def equal_block_weights(block_ic: NDArray[np.float64]) -> NDArray[np.float64]:
    """Equal block weights, oriented by the sign of each block's training IC."""
    return np.where(block_ic >= 0, 1.0, -1.0) if len(block_ic) > 1 else np.ones(1)


def adaptive_block_weights(
    block_ic: NDArray[np.float64], block_se: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Random-effects (DerSimonian-Laird) block weights: ``(weights, tau2)``.

    Signs are removed before pooling and restored after, so orientation stays the one
    supervised bit and only magnitudes are shrunk.
    """
    s = np.where(block_ic >= 0, 1.0, -1.0)
    shrunk, tau2 = dl_shrink(s * block_ic, block_se)
    return s * shrunk, tau2
