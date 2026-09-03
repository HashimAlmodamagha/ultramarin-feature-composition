"""Cluster-then-weight composites: count each idea once, then decide the block weights.

The variants are grouped into redundancy blocks (average linkage on ``1 - |corr|``, cut at
``rho``), each block collapses to the sign-aligned average of its members, and the blocks are
combined with equal weights (``eq``), by their training IC (``ic``), by a GLS solve across
blocks (``gls``) or by random-effects shrinkage (``eb``). Equal weights at the frozen
``rho = 0.7`` is the study's selected composite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from feature_composition import metrics
from feature_composition.clustering import RECIPE_RHO, block_matrix, cluster_labels
from feature_composition.shrinkage import adaptive_block_weights, dl_shrink, equal_block_weights

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from feature_composition.panel import Panel

Across = Literal["eq", "ic", "gls", "eb"]
RHO_GRID = (0.5, 0.6, 0.7, 0.8)


def _unit(w: NDArray[Any]) -> NDArray[np.float64]:
    w = np.asarray(w, dtype=np.float64)
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


def block_ingredients(
    panel: Panel, train_days: NDArray[np.bool_], rho: float = RECIPE_RHO
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Everything a block weighting needs, all estimated on the training days.

    Returns ``(B, block_ic, block_se, W, labels, Sigma_blocks)``: the block signals over the
    whole panel, each block's training mean IC and Newey-West s.e., the feature-to-block
    matrix, the labels and the training correlation of the block signals.
    """
    Z, ic, rows = panel.fit(train_days)
    Sigma = panel.correlation(rows)
    labels = cluster_labels(Sigma, rho)
    W = block_matrix(labels, np.where(ic >= 0, 1.0, -1.0))
    B = Z @ W
    days = np.flatnonzero(train_days)
    bic, bse = [], []
    for j in range(B.shape[1]):
        series = panel.daily_ic(B[:, j], days)
        bic.append(float(np.nanmean(series)))
        bse.append(metrics.hac_se(series))
    Sigma_b = np.asarray(
        np.corrcoef(B[rows], rowvar=False) if B.shape[1] > 1 else np.ones((1, 1)), dtype=np.float64
    )
    return (
        B,
        np.asarray(bic, dtype=np.float64),
        np.asarray(bse, dtype=np.float64),
        W,
        labels,
        Sigma_b,
    )


def block_weights(
    across: Across,
    block_ic: NDArray[np.float64],
    block_se: NDArray[np.float64],
    Sigma_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unit-norm block weights for one of the four across-block rules."""
    w: NDArray[Any]
    if across == "eq":
        w = equal_block_weights(block_ic)
    elif across == "ic":
        w = block_ic
    elif across == "gls":
        w = np.linalg.solve(Sigma_b + 1e-3 * np.eye(len(block_ic)), block_ic)
    elif across == "eb":
        w, _ = adaptive_block_weights(block_ic, block_se)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown across-block rule {across!r}")
    return _unit(w)


def cluster_signal(
    panel: Panel, train_days: NDArray[np.bool_], rho: float = RECIPE_RHO, across: Across = "eq"
) -> NDArray[np.float64]:
    """Cluster-then-weight composite at a fixed merge threshold."""
    B, bic, bse, _, _, Sigma_b = block_ingredients(panel, train_days, rho)
    return B @ block_weights(across, bic, bse, Sigma_b)


def cluster_tuned(
    panel: Panel,
    train_days: NDArray[np.bool_],
    across: Across = "eq",
    log: list[tuple[float, int]] | None = None,
) -> NDArray[np.float64]:
    """Cluster-then-weight with ``rho`` chosen on the inner validation slice.

    Leak-free tuning of the threshold never beat the fixed 0.7 in the study; this is the
    method that established that.
    """
    itr, iv = panel.inner_split(train_days)
    rho = max(RHO_GRID, key=lambda r: panel.mean_ic_on(cluster_signal(panel, itr, r, across), iv))
    if log is not None:
        labels = cluster_labels(panel.correlation(panel.rows_of(train_days)), rho)
        log.append((rho, len(np.unique(labels))))
    return cluster_signal(panel, train_days, rho, across)


def cluster_eq(
    panel: Panel, train_days: NDArray[np.bool_], rho: float = RECIPE_RHO
) -> NDArray[np.float64]:
    """Dedup at ``rho`` + equal block weights: the selected composite's linear arm."""
    return cluster_signal(panel, train_days, rho, "eq")


def cluster_eb(
    panel: Panel,
    train_days: NDArray[np.bool_],
    rho: float = RECIPE_RHO,
    log: list[dict[str, float]] | None = None,
) -> NDArray[np.float64]:
    """Dedup at ``rho`` + DerSimonian-Laird block weights (the adaptive tilt)."""
    B, bic, bse, _, labels, _ = block_ingredients(panel, train_days, rho)
    w, tau2 = adaptive_block_weights(bic, bse)
    if log is not None:
        log.append({"n_blocks": float(len(np.unique(labels))), "tau2": tau2})
    return B @ _unit(w)


# ------------------------------------------------- two-level (feature-within-block) pooling
def within_block_shrink(
    ic_f: NDArray[np.float64], ses: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Shrink each member's IC toward its block mean by within-block DL.

    Returns ``(g, signs, n_tilted)``.
    """
    s = np.where(ic_f >= 0, 1.0, -1.0)
    e = s * ic_f
    g = e.copy()
    n_tilted = 0
    for b in np.unique(labels):
        m = labels == b
        if m.sum() > 1:
            shrunk, tau2 = dl_shrink(e[m], ses[m])
            g[m] = shrunk
            n_tilted += int(tau2 > 0)
    return g, s, n_tilted


def eb_within(
    panel: Panel, train_days: NDArray[np.bool_], rho: float = RECIPE_RHO
) -> tuple[NDArray[np.float64], int]:
    """Dedup-normalized partial pooling: block mass from the DL block weights, spread within
    each block by the within-shrunk member ICs. Within-tau2 = 0 reproduces ``cluster_eb``.

    Returns the signal and the number of blocks where the within-shrinkage actually tilted.
    """
    Z, _, _ = panel.fit(train_days)
    _, bic, bse, _, labels, _ = block_ingredients(panel, train_days, rho)
    wb, _ = adaptive_block_weights(bic, bse)
    F = panel.feature_ic[train_days]
    ic_f = np.nanmean(F, axis=0)
    ses = np.array([metrics.hac_se(F[:, j]) for j in range(F.shape[1])])
    g, s, n_tilted = within_block_shrink(ic_f, ses, labels)
    gpos = np.maximum(g, 0.0)
    w = np.zeros(len(ic_f))
    for j, b in enumerate(np.unique(labels)):
        m = labels == b
        tot = gpos[m].sum()
        u = gpos[m] / tot if tot > 0 else np.ones(m.sum()) / m.sum()
        w[m] = wb[j] * s[m] * u
    return Z @ _unit(w), n_tilted


def eb_flat(
    panel: Panel, train_days: NDArray[np.bool_], rho: float = RECIPE_RHO
) -> NDArray[np.float64]:
    """The un-normalized version: weight features by their within-shrunk ICs directly, so a
    block's mass grows with its member count (duplicate-publication bias on purpose)."""
    Z, _, rows = panel.fit(train_days)
    labels = cluster_labels(panel.correlation(rows), rho)
    F = panel.feature_ic[train_days]
    ic_f = np.nanmean(F, axis=0)
    ses = np.array([metrics.hac_se(F[:, j]) for j in range(F.shape[1])])
    g, s, _ = within_block_shrink(ic_f, ses, labels)
    return Z @ _unit(s * g)


CLUSTER_METHODS = {
    "cluster_eq": cluster_eq,
    "cluster_eb": cluster_eb,
    "cluster_eq_tuned": lambda p, tm: cluster_tuned(p, tm, "eq"),
    "cluster_ic_tuned": lambda p, tm: cluster_tuned(p, tm, "ic"),
    "cluster_gls_tuned": lambda p, tm: cluster_tuned(p, tm, "gls"),
}
