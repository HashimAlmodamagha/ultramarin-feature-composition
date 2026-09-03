"""Linear composition methods: the benchmark and everything that tied with it.

Every method has the signature ``method(panel, train_days) -> signal`` where ``train_days`` is
a boolean day mask. Parameters are estimated on the training days only; the returned signal
covers the whole panel so that the caller can score any day block. Tuned methods pick their
hyper-parameter on the panel's purged inner validation slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from feature_composition import metrics
from feature_composition.shrinkage import dl_shrink

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from feature_composition.panel import Panel

RIDGE_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
UNIQUENESS_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LASSO_FRACTIONS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
PLS_MAX_COMPONENTS = 8
SHRINK_GRID = tuple((g, lam) for g in (0.0, 0.5, 1.0, 2.0) for lam in (0.1, 0.5, 1.0, 4.0, 16.0))


def _unit(w: NDArray[Any]) -> NDArray[np.float64]:
    w = np.asarray(w, dtype=np.float64)
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


# ------------------------------------------------------------------------- baselines
def naive_averaged(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The benchmark: flip each variant by the sign of its training IC, average the ranks."""
    _, ic, _ = panel.fit(train_days)
    return (panel.X * np.where(ic >= 0, 1.0, -1.0)).mean(1)


def ic_weight(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Weights proportional to each variant's training-window mean rank IC."""
    Z, ic, _ = panel.fit(train_days)
    return Z @ _unit(ic)


def pca_pc1(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """First principal component of the training correlation, oriented target-free.

    PCA maximizes variance, not correlation with the target: it rewards the most-repeated
    definition. Retired in-sample and confirmed out of sample as the study's worst method on
    the one genuinely multi-factor class.
    """
    Z, ic, rows = panel.fit(train_days)
    v = np.asarray(np.linalg.eigh(np.cov(Z[rows], rowvar=False))[1][:, -1], dtype=np.float64)
    s = Z @ v
    ref = (panel.X * np.where(ic >= 0, 1.0, -1.0)).mean(1)
    return -s if metrics.pearson(s[rows], ref[rows]) < 0 else s


def uniqueness_reg(
    panel: Panel, train_days: NDArray[np.bool_], lam: float = 1.0
) -> NDArray[np.float64]:
    """The 'uniqueness regression': ``w ∝ (Σ + λI)^-1 ic`` (GLS with a ridge prior).

    ``λ -> 0`` is the full redundancy-aware GLS solve; ``λ -> ∞`` recovers IC weighting.
    """
    Z, ic, rows = panel.fit(train_days)
    Sigma = panel.correlation(rows)
    w = np.linalg.solve(Sigma + lam * np.eye(len(ic)), ic)
    return Z @ _unit(w)


def uniqueness_tuned(
    panel: Panel, train_days: NDArray[np.bool_], log: list[float] | None = None
) -> NDArray[np.float64]:
    """Uniqueness regression with λ chosen on the inner validation slice."""
    itr, iv = panel.inner_split(train_days)
    lam = max(
        UNIQUENESS_LAMBDAS, key=lambda lm: panel.mean_ic_on(uniqueness_reg(panel, itr, lm), iv)
    )
    if log is not None:
        log.append(lam)
    return uniqueness_reg(panel, train_days, lam)


def eb_weight(
    panel: Panel, train_days: NDArray[np.bool_], log: list[dict[str, float]] | None = None
) -> NDArray[np.float64]:
    """Random-effects IC weighting at the feature level: no tuned hyper-parameters anywhere."""
    Z, _, _ = panel.fit(train_days)
    F = panel.feature_ic[train_days]
    ic_f = np.nanmean(F, axis=0)
    ses = np.array([metrics.hac_se(F[:, j]) for j in range(F.shape[1])])
    s = np.where(ic_f >= 0, 1.0, -1.0)
    shrunk, tau2 = dl_shrink(s * ic_f, ses)
    if log is not None:
        log.append({"tau2": tau2})
    return Z @ _unit(s * shrunk)


def shrink_weights(
    ic: NDArray[np.float64], Sigma: NDArray[np.float64], gamma: float, lam: float
) -> NDArray[np.float64]:
    """Diversification penalty: shrink toward the equal-weight portfolio, not toward zero.

    ``max_w  w'ic - (γ/2) w'Σw - (λ/2)||w - w_eq||²`` has the closed form
    ``w = (γΣ + λI)^-1 (ic + λ w_eq)``.
    """
    p = len(ic)
    w_eq = np.where(ic >= 0, 1.0, -1.0) / p
    scale = float(np.abs(ic).mean())
    w = np.linalg.solve(gamma * Sigma + lam * np.eye(p), ic / scale + lam * w_eq)
    return _unit(w)


def shrink_to_naive(
    panel: Panel, train_days: NDArray[np.bool_], log: list[tuple[float, float]] | None = None
) -> NDArray[np.float64]:
    """Diversification-penalty weights with ``(γ, λ)`` tuned on the inner slice."""
    Z, ic, rows = panel.fit(train_days)
    Sigma = panel.correlation(rows)
    itr, iv = panel.inner_split(train_days)
    Zi, ici, rowsi = panel.fit(itr)
    Sigi = panel.correlation(rowsi)
    g, lam = max(
        SHRINK_GRID, key=lambda gl: panel.mean_ic_on(Zi @ shrink_weights(ici, Sigi, *gl), iv)
    )
    if log is not None:
        log.append((g, lam))
    return Z @ shrink_weights(ic, Sigma, g, lam)


# ------------------------------------------------------------ regression benchmarks
def _standardize(
    panel: Panel, train_days: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    rows = panel.rows_of(train_days)
    return (panel.X - panel.X[rows].mean(0)) / panel.X[rows].std(0), rows


def ols(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Pooled least squares of the target on the standardized variants."""
    Z, rows = _standardize(panel, train_days)
    return Z @ np.linalg.lstsq(Z[rows], panel.y[rows], rcond=None)[0]


def _ridge_fit(
    Z: NDArray[np.float64], rows: NDArray[np.bool_], y: NDArray[np.float64], lam: float
) -> NDArray[np.float64]:
    n = rows.sum()
    return np.linalg.solve(
        Z[rows].T @ Z[rows] / n + lam * np.eye(Z.shape[1]), Z[rows].T @ y[rows] / n
    )


def ridge_tuned(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Ridge regression with λ chosen by inner-validation rank IC."""
    itr, iv = panel.inner_split(train_days)
    Zi, rowsi = _standardize(panel, itr)
    lam = max(
        RIDGE_LAMBDAS, key=lambda lm: panel.mean_ic_on(Zi @ _ridge_fit(Zi, rowsi, panel.y, lm), iv)
    )
    Z, rows = _standardize(panel, train_days)
    return Z @ _ridge_fit(Z, rows, panel.y, lam)


def coordinate_descent_lasso(
    Xc: NDArray[np.float64],
    yc: NDArray[np.float64],
    frac: float,
    n_iter: int = 300,
    tol: float = 1e-7,
) -> NDArray[np.float64]:
    """Gram-based coordinate descent for ``(1/2n)||y - Xw||² + λ||w||₁`` with ``λ = frac·λ_max``.

    ``λ_max = max_j |x_j'y| / n`` is the smallest penalty at which every coefficient is zero, so
    the sparsity range auto-calibrates to any target scale.
    """
    n, p = Xc.shape
    g = Xc.T @ yc / n
    lam = frac * float(np.abs(g).max())
    G = Xc.T @ Xc / n
    d = np.diag(G).copy()
    w = np.zeros(p)
    for _ in range(n_iter):
        delta = 0.0
        for j in range(p):
            rho = g[j] - G[j] @ w + d[j] * w[j]
            wj = np.sign(rho) * max(abs(rho) - lam, 0.0) / d[j]
            delta = max(delta, abs(wj - w[j]))
            w[j] = wj
        if delta < tol:
            break
    return w


def lasso_tuned(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """L1-penalized regression on the raw target, penalty fraction tuned on the inner slice."""
    itr, iv = panel.inner_split(train_days)
    Zi, rowsi = _standardize(panel, itr)
    fr = max(
        LASSO_FRACTIONS,
        key=lambda f: panel.mean_ic_on(
            Zi @ coordinate_descent_lasso(Zi[rowsi], panel.y[rowsi], f), iv
        ),
    )
    Z, rows = _standardize(panel, train_days)
    return Z @ coordinate_descent_lasso(Z[rows], panel.y[rows], fr)


def lasso_uniqueness(panel: Panel, train_days: NDArray[np.bool_]) -> NDArray[np.float64]:
    """L1 selection aligned to rank IC: the lasso fitted to the unit-variance rank target."""

    def rank_target(rows: NDArray[np.bool_]) -> NDArray[np.float64]:
        t = panel.y_rank[rows]
        return (t - t.mean()) / t.std()

    itr, iv = panel.inner_split(train_days)
    Zi, rowsi = _standardize(panel, itr)
    fr = max(
        LASSO_FRACTIONS,
        key=lambda f: panel.mean_ic_on(
            Zi @ coordinate_descent_lasso(Zi[rowsi], rank_target(rowsi), f), iv
        ),
    )
    Z, rows = _standardize(panel, train_days)
    return Z @ coordinate_descent_lasso(Z[rows], rank_target(rows), fr)


# ------------------------------------------------------------------------------ PLS
def pls_coef(
    Z: NDArray[np.float64], rows: NDArray[np.bool_], y: NDArray[np.float64], k: int
) -> NDArray[np.float64]:
    """NIPALS PLS1 with ``k`` components, returned as one regression coefficient vector.

    ``k = 1`` is proportional to ``X'y`` (IC weighting up to standardization); ``k = p``
    reproduces OLS.
    """
    Xd = Z[rows].astype(float).copy()
    yd = (y[rows] - y[rows].mean()).astype(float)
    W, P, Q = [], [], []
    for _ in range(k):
        w = Xd.T @ yd
        nw = float(np.linalg.norm(w))
        if nw == 0:
            break
        w = w / nw
        t = Xd @ w
        tt = float(t @ t)
        p_ = (Xd.T @ t) / tt
        q = float(yd @ t) / tt
        Xd = Xd - np.outer(t, p_)
        yd = yd - q * t
        W.append(w)
        P.append(p_)
        Q.append(q)
    Wm, Pm, Qv = np.column_stack(W), np.column_stack(P), np.asarray(Q)
    return Wm @ np.linalg.lstsq(Pm.T @ Wm, Qv, rcond=None)[0]


def pls_components(panel: Panel, train_days: NDArray[np.bool_], k: int) -> NDArray[np.float64]:
    """PLS with a fixed number of components."""
    Z, _, rows = panel.fit(train_days)
    return Z @ pls_coef(Z, rows, panel.y, k)


def pls_grid(panel: Panel) -> list[int]:
    """Candidate component counts: ``1..min(8, n_features)``."""
    return list(range(1, min(PLS_MAX_COMPONENTS, len(panel.cols)) + 1))


def pls_tuned(
    panel: Panel, train_days: NDArray[np.bool_], log: list[int] | None = None
) -> NDArray[np.float64]:
    """PLS with the component count chosen by inner-validation rank IC.

    The tuner picked ``k = 1`` in every fold on the study's near-one-factor classes: there is
    exactly one recoverable predictive direction. Where it chose more (class 3, ``k = 6``) the
    frozen fit halved the class's IC out of sample.
    """
    itr, iv = panel.inner_split(train_days)
    Zi, _, rowsi = panel.fit(itr)
    k = max(
        pls_grid(panel), key=lambda kk: panel.mean_ic_on(Zi @ pls_coef(Zi, rowsi, panel.y, kk), iv)
    )
    if log is not None:
        log.append(k)
    return pls_components(panel, train_days, k)


def pls_sweep(panel: Panel) -> dict[str, list[float]]:
    """Out-of-fold IC as PLS components are added: the *predictive* dimension of a cluster.

    Returns ``{"k": [...], "mean": [...], "std": [...]}`` over the purged folds.
    """
    ks = pls_grid(panel)
    per_k: dict[int, list[float]] = {k: [] for k in ks}
    for _, train, test in panel.folds():
        Z, _, rows = panel.fit(train)
        for k in ks:
            per_k[k].append(panel.mean_ic_on(Z @ pls_coef(Z, rows, panel.y, k), test))
    return {
        "k": [float(k) for k in ks],
        "mean": [float(np.mean(per_k[k])) for k in ks],
        "std": [float(np.std(per_k[k], ddof=1)) if len(per_k[k]) > 1 else float("nan") for k in ks],
    }


LINEAR_METHODS = {
    "naive_averaged": naive_averaged,
    "ic_weight": ic_weight,
    "pca_pc1": pca_pc1,
    "uniqueness_reg": uniqueness_reg,
    "uniqueness_tuned": uniqueness_tuned,
    "eb_weight": eb_weight,
    "shrink_to_naive": shrink_to_naive,
    "ols": ols,
    "ridge_tuned": ridge_tuned,
    "lasso_tuned": lasso_tuned,
    "pls_tuned": pls_tuned,
}
