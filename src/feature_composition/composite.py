"""The selected composite: count each idea once, weight ideas equally, freeze everything.

``DedupComposite`` is the deployable form of the study's methodology:

1. **Group the duplicates.** Average-linkage clustering on ``1 - |corr|`` of the training
   panel, merged above ``|corr| >= rho`` (frozen at 0.7).
2. **Average within blocks.** Flip negative-IC variants, then average each block's members.
3. **Weight the blocks equally, then freeze.** The random-effects tau^2 is reported as a
   diagnostic; acting on it lost on the blind test.
4. **Shapes only past the gate.** Optionally pass each variant through its learned stump
   response curve first (see :mod:`feature_composition.gate`); adopted on one of four classes.

Everything is estimated once on the training panel and applied unchanged afterwards. The
fitted state is plain arrays and serializes to JSON, so deployment needs neither this
package's estimation code nor a tree library: one clustering, one weight vector and, when
shapes are on, one lookup table per feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from feature_composition import metrics
from feature_composition.clustering import RECIPE_RHO, block_matrix, cluster_labels
from feature_composition.shrinkage import (
    adaptive_block_weights,
    equal_block_weights,
    tau2_dl,
    tau2_pm,
    tau2_reml,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from feature_composition.panel import Panel

BlockWeights = Literal["eq", "eb"]


def _nanmean0(x: NDArray[np.float64]) -> float:
    """Mean ignoring NaN; 0.0 when nothing is left (a constant signal has no IC)."""
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else 0.0


def _unit(w: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


@dataclass
class DedupComposite:
    """Dedup-and-equal-weight composite with an optional learned-shape front end.

    Parameters
    ----------
    rho:
        Merge threshold on ``|corr|``. Frozen at 0.7 in the study.
    block_weights:
        ``"eq"`` (selected) or ``"eb"`` (DerSimonian-Laird tilt, kept for comparison).
    shapes:
        Whether variants are passed through learned stump response curves before the block
        step. ``False`` for the plain composite; ``True`` for the shaped arm. Use
        :func:`feature_composition.gate.gate_decision` to decide this from evidence.
    align_shaped_members:
        Sign-align shaped members by their own training IC before block averaging (the
        study's "v2"; a statistical wash versus v1 on the hold-out).
    n_stumps:
        Boosting rounds of the depth-1 fit behind the shapes.
    """

    rho: float = RECIPE_RHO
    block_weights: BlockWeights = "eq"
    shapes: bool = False
    align_shaped_members: bool = False
    n_stumps: int = 600

    # ---- fitted state (set by fit) ----
    cols: list[str] = field(default_factory=list, init=False)
    flipped: list[str] = field(default_factory=list, init=False)
    mu: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    sd: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    signs: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    labels: NDArray[np.int64] | None = field(default=None, init=False, repr=False)
    weights: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    block_ic: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    block_se: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    shape_table: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    shaped_signs: NDArray[np.float64] | None = field(default=None, init=False, repr=False)
    diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------------------ fit
    def fit(self, panel: Panel, train_days: NDArray[np.bool_] | None = None) -> DedupComposite:
        """Estimate every parameter on the training days of ``panel`` (default: all days)."""
        train = panel.all_days if train_days is None else train_days
        Z, ic, rows = panel.fit(train)
        days = np.flatnonzero(train)
        self.cols = list(panel.cols)
        self.flipped = list(panel.flipped)
        self.mu, self.sd = panel.X[rows].mean(0), panel.X[rows].std(0)
        self.signs = np.where(ic >= 0, 1.0, -1.0)
        self.labels = cluster_labels(panel.correlation(rows), self.rho)

        if self.shapes:
            from feature_composition.trees import apply_shapes, fit_stump_shapes

            self.shape_table = fit_stump_shapes(panel.X[rows], panel.y[rows], self.n_stumps)
            Xs = apply_shapes(panel.X, self.shape_table)
            if self.align_shaped_members:
                self.shaped_signs = np.array(
                    [
                        1.0 if _nanmean0(panel.daily_ic(Xs[:, j], days)) >= 0 else -1.0
                        for j in range(len(self.cols))
                    ]
                )
            else:
                self.shaped_signs = np.ones(len(self.cols))
            B = self._shaped_blocks(Xs)
        else:
            B = Z @ block_matrix(self.labels, self.signs)

        series = [panel.daily_ic(B[:, j], days) for j in range(B.shape[1])]
        bic = np.array([_nanmean0(s) for s in series])
        bse = np.array([metrics.hac_se(s) for s in series])
        bse = np.where(np.isnan(bse), np.inf, bse)  # a constant block carries no information
        self.block_ic, self.block_se = bic, bse
        if self.block_weights == "eq":
            w = equal_block_weights(bic)
        else:
            w, _ = adaptive_block_weights(bic, bse)
        self.weights = _unit(w)

        s = np.where(bic >= 0, 1.0, -1.0)
        eff = s * bic
        self.diagnostics = {
            "n_features": len(self.cols),
            "n_blocks": len(np.unique(self.labels)),
            "tau2_dl": tau2_dl(eff, bse),
            "tau2_pm": tau2_pm(eff, bse),
            "tau2_reml": tau2_reml(eff, bse),
        }
        return self

    def _shaped_blocks(self, Xs: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.labels is not None and self.shaped_signs is not None
        return np.column_stack(
            [
                (Xs[:, self.labels == b] * self.shaped_signs[self.labels == b]).mean(1)
                for b in np.unique(self.labels)
            ]
        )

    # ------------------------------------------------------------------------ transform
    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Composite signal for a feature matrix in the fitted column order.

        ``X`` must already carry the same orientation as the training panel (apply the same
        reflections); see :meth:`transform_frame` for the checked path.
        """
        if self.weights is None or self.labels is None or self.signs is None:
            raise RuntimeError("DedupComposite is not fitted")
        if X.shape[1] != len(self.cols):
            raise ValueError(f"expected {len(self.cols)} features, got {X.shape[1]}")
        if self.shapes:
            from feature_composition.trees import apply_shapes

            assert self.shape_table is not None
            return self._shaped_blocks(apply_shapes(X, self.shape_table)) @ self.weights
        assert self.mu is not None and self.sd is not None
        Z = (X - self.mu) / self.sd
        return Z @ (block_matrix(self.labels, self.signs) @ self.weights)

    def transform_panel(self, panel: Panel) -> NDArray[np.float64]:
        """Composite signal for another panel of the same feature class (e.g. the hold-out).

        Checks the column order and that the same reflections were applied.
        """
        if list(panel.cols) != self.cols:
            raise ValueError("panel feature columns differ from the fitted composite")
        if list(panel.flipped) != self.flipped:
            raise ValueError(
                f"panel reflections {panel.flipped} differ from the fitted {self.flipped}; "
                "construct the hold-out Panel with reflect=composite.flipped"
            )
        return self.transform(panel.X)

    __call__ = transform

    # ------------------------------------------------------------------------- reading
    @property
    def feature_weights(self) -> NDArray[np.float64]:
        """Effective per-feature weights of the linear arm (``W @ block_weights``)."""
        if self.weights is None or self.labels is None or self.signs is None:
            raise RuntimeError("DedupComposite is not fitted")
        return block_matrix(self.labels, self.signs) @ self.weights

    def blocks(self) -> dict[int, list[str]]:
        """Block membership by label."""
        if self.labels is None:
            raise RuntimeError("DedupComposite is not fitted")
        return {
            int(b): [c for c, lab in zip(self.cols, self.labels, strict=True) if lab == b]
            for b in np.unique(self.labels)
        }

    def describe(self) -> str:
        """Human-readable summary of the frozen composite."""
        if self.weights is None:
            return "DedupComposite(unfitted)"
        lines = [
            f"DedupComposite(rho={self.rho}, block_weights={self.block_weights!r}, "
            f"shapes={self.shapes})",
            f"  {self.diagnostics['n_features']} variants -> {self.diagnostics['n_blocks']} ideas"
            + (f"; reflected {self.flipped}" if self.flipped else ""),
            f"  tau^2 (DL/PM/REML): {self.diagnostics['tau2_dl']:.2e} / "
            f"{self.diagnostics['tau2_pm']:.2e} / {self.diagnostics['tau2_reml']:.2e}",
        ]
        assert self.block_ic is not None
        for j, (b, members) in enumerate(self.blocks().items()):
            lines.append(
                f"  block {b}: {', '.join(members)}  (train IC {self.block_ic[j]:+.4f}, "
                f"weight {self.weights[j]:+.3f})"
            )
        return "\n".join(lines)

    # --------------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable frozen state."""
        if self.weights is None:
            raise RuntimeError("DedupComposite is not fitted")

        def arr(a: NDArray[Any] | None) -> list[Any] | None:
            return None if a is None else a.tolist()

        return {
            "rho": self.rho,
            "block_weights": self.block_weights,
            "shapes": self.shapes,
            "align_shaped_members": self.align_shaped_members,
            "n_stumps": self.n_stumps,
            "cols": self.cols,
            "flipped": self.flipped,
            "mu": arr(self.mu),
            "sd": arr(self.sd),
            "signs": arr(self.signs),
            "labels": arr(self.labels),
            "weights": arr(self.weights),
            "block_ic": arr(self.block_ic),
            "block_se": arr(self.block_se),
            "shape_table": arr(self.shape_table),
            "shaped_signs": arr(self.shaped_signs),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DedupComposite:
        """Rebuild a frozen composite from :meth:`to_dict` output."""
        obj = cls(
            rho=d["rho"],
            block_weights=d["block_weights"],
            shapes=d["shapes"],
            align_shaped_members=d.get("align_shaped_members", False),
            n_stumps=d.get("n_stumps", 600),
        )
        obj.cols, obj.flipped = list(d["cols"]), list(d["flipped"])
        for key in (
            "mu",
            "sd",
            "signs",
            "weights",
            "block_ic",
            "block_se",
            "shape_table",
            "shaped_signs",
        ):
            v = d.get(key)
            setattr(obj, key, None if v is None else np.asarray(v, dtype=float))
        obj.labels = np.asarray(d["labels"], dtype=int)
        obj.diagnostics = dict(d.get("diagnostics", {}))
        return obj

    def save(self, path: str | Path) -> None:
        """Write the frozen state as JSON."""
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> DedupComposite:
        """Read a frozen composite written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))
