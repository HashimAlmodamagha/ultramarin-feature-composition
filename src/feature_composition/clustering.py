"""Orientation and redundancy structure of a cluster of feature variants.

Two things hide in a correlation matrix and only one of them is fixable. An *orientation*
artifact (variants that measure the same idea with opposite sign conventions) is removed by
reflecting the offending variants. *Redundancy* (several implementations of the same idea) is
what the study's deduplication step counts once: average-linkage clustering on the distance
``1 - |corr|`` cut at a fixed merge threshold ``rho``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: The frozen merge threshold: variants sharing at least |corr| = 0.7 (half their variance)
#: are one idea. Leak-free tuning was offered twice in the study and never beat it.
RECIPE_RHO = 0.7


def pc1_sign_align(corr: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unsupervised orientation: flip each variant so that it loads non-negatively on PC1.

    Returns ``(aligned_corr, flip)`` with ``flip`` in {-1, +1}. Reflecting a percentile-rank
    feature is ``x -> 1 - x``, which sends ``corr -> -corr``, so on the correlation matrix the
    flip is the outer product of the sign vector. The global sign is arbitrary and the
    majority orientation is kept.
    """
    _, evecs = np.linalg.eigh(corr)
    flip = np.sign(evecs[:, -1])
    flip[flip == 0] = 1.0
    if flip.mean() < 0:
        flip = -flip
    return corr * np.outer(flip, flip), flip


def orientation_flips(corr: NDArray[np.float64], cols: list[str]) -> list[str]:
    """Names of the variants the PC1 alignment flags as reversed."""
    _, flip = pc1_sign_align(corr)
    return [c for c, f in zip(cols, flip, strict=True) if f < 0]


def _condensed(dist: NDArray[np.float64]) -> NDArray[np.float64]:
    d = dist.copy()
    np.fill_diagonal(d, 0.0)
    d = 0.5 * (d + d.T)
    return squareform(d, checks=False)


def cluster_labels(
    corr: NDArray[np.float64], rho: float = RECIPE_RHO, method: str = "average"
) -> NDArray[np.int64]:
    """Redundancy blocks: hierarchical clustering on ``1 - |corr|`` cut at distance ``1 - rho``.

    Labels are 1-based integers, one per feature. ``rho -> 1`` gives one feature per block
    (the benchmark's implicit partition); ``rho -> 0`` collapses the cluster to one block.
    """
    d = 1.0 - np.abs(corr)
    lk = linkage(_condensed(d), method=method)
    return fcluster(lk, 1.0 - rho, criterion="distance")


def merge_tree(corr: NDArray[np.float64], method: str = "average") -> NDArray[np.float64]:
    """The scipy linkage matrix on ``1 - |corr|`` (for dendrograms and cut analysis)."""
    return linkage(_condensed(1.0 - np.abs(corr)), method=method)


def clustered_order(corr: NDArray[np.float64]) -> NDArray[np.int64]:
    """Leaf order of an average-linkage tree on ``1 - corr`` (for readable heatmaps)."""
    return leaves_list(linkage(_condensed(1.0 - corr), method="average"))


def block_matrix(labels: NDArray[np.int64], signs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Feature-to-block collapse matrix ``W`` (features x blocks).

    Each block column holds sign-aligned equal weights ``sign_j / |block|`` on its members, so
    ``Z @ W`` is the matrix of block signals: each idea counted once, however many times it was
    re-implemented.
    """
    blocks = np.unique(labels)
    W = np.zeros((len(labels), len(blocks)))
    for j, b in enumerate(blocks):
        m = labels == b
        W[m, j] = signs[m] / m.sum()
    return W


# --------------------------------------------------------------- alternative partitions
# Used by the stress tests: every alternative is forced to the SAME block count k as the anchor
# tree, so the experiment isolates how you group from how many groups you make.


def hier_labels_k(corr: NDArray[np.float64], k: int, method: str = "average") -> NDArray[np.int64]:
    """Hierarchical partition of ``1 - |corr|`` into exactly ``k`` blocks (any scipy linkage)."""
    lk = linkage(_condensed(1.0 - np.abs(corr)), method=method)
    return fcluster(lk, k, criterion="maxclust")


def ic_profile_labels(feature_ic: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Cluster on WHEN features predict: distance ``1 - |corr|`` of the daily IC series."""
    ok = ~np.isnan(feature_ic).any(axis=1)
    C = np.corrcoef(feature_ic[ok].T)
    lk = linkage(_condensed(1.0 - np.abs(C)), method="average")
    return fcluster(lk, k, criterion="maxclust")


def divisive_labels(corr: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Top-down spectral bisection into ``k`` blocks (the alternative to bottom-up merging).

    Recursively splits the most heterogeneous block by the sign of the second eigenvector of
    its ``|corr|`` sub-matrix; a degenerate split peels off the least-connected member.
    """
    A = np.abs(corr)
    clusters: list[NDArray[np.intp]] = [np.arange(len(A))]
    while len(clusters) < k:
        cand = [i for i, c in enumerate(clusters) if len(c) > 1]
        if not cand:
            break
        diam = [1.0 - A[np.ix_(clusters[i], clusters[i])].min() for i in cand]
        ci = cand[int(np.argmax(diam))]
        c = clusters.pop(ci)
        sub = A[np.ix_(c, c)]
        _, evecs = np.linalg.eigh(sub)
        v = evecs[:, -2]
        left, right = c[v < 0], c[v >= 0]
        if len(left) == 0 or len(right) == 0:
            j = int(np.argmin(sub.mean(0)))
            left, right = c[[j]], np.delete(c, j)
        clusters.extend([left, right])
    labels = np.empty(len(A), dtype=int)
    for i, c in enumerate(clusters):
        labels[c] = i + 1
    return labels


def random_labels(n_features: int, k: int, seed: int) -> NDArray[np.int64]:
    """Placebo partition: shuffled labels at matched block count."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_features)
    labels = np.empty(n_features, dtype=int)
    for i, chunk in enumerate(np.array_split(order, k)):
        labels[chunk] = i + 1
    return labels


def pc1_share(corr: NDArray[np.float64]) -> float:
    """Share of feature variance on the first principal component (how one-factor a cluster is)."""
    ev = np.linalg.eigvalsh(corr)
    return float(ev[-1] / ev.sum())


def eigen_shares(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Descending eigenvalue shares of the correlation matrix (the variance spectrum)."""
    ev = np.linalg.eigvalsh(corr)[::-1]
    return ev / ev.sum()
