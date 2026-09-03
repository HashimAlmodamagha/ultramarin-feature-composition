"""The playbook for a new feature class: cheap mechanical diagnostics with a measured record.

Four classes cannot certify a universal rule. What they can certify is a set of diagnostics
and a map from readings to actions:

1. **PC1 share and the merge tree.** Above ~0.45 with clean blocks, equal weighting will
   saturate the class; a diffuse spectrum (~0.3) means the partition matters and the plain
   benchmark is the likely casualty.
2. **The out-of-fold PLS-k sweep.** A peak at k = 1 means any sensible weighting works;
   rising past k >= 3 flags multi-directional signal. A research flag, not a production
   decision: both frozen multi-component fits inverted out of sample.
3. **tau^2 under all three estimators.** Zero: equal weights are optimal in-family. Positive
   under all three: the blocks truly differ; with many blocks or known identities the tilt
   deserves a fresh trial.
4. **Run the gate for shapes** (:mod:`feature_composition.gate`), which is expensive and
   therefore separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from feature_composition import linear
from feature_composition.clustering import (
    RECIPE_RHO,
    cluster_labels,
    eigen_shares,
    orientation_flips,
)
from feature_composition.composite import DedupComposite

if TYPE_CHECKING:
    from feature_composition.panel import Panel


@dataclass
class ReportCard:
    """Diagnostic readings for one feature class, plus the playbook's reading of them."""

    name: str
    n_features: int
    n_days: int
    n_rows: int
    pc1_share: float
    eigen_shares: list[float]
    reversed_variants: list[str]
    n_blocks: int
    blocks: dict[int, list[str]]
    pls_k: list[float]
    pls_oof_ic: list[float]
    naive_oof_ic: float
    tau2_dl: float
    tau2_pm: float
    tau2_reml: float
    block_ic: list[float]
    notes: list[str] = field(default_factory=list)

    @property
    def pls_best_k(self) -> int:
        return int(self.pls_k[int(np.argmax(self.pls_oof_ic))])

    def render(self) -> str:
        """Plain-text report card."""
        lines = [
            f"== {self.name}: {self.n_features} variants, {self.n_days:,} days, "
            f"{self.n_rows:,} rows ==",
            f"PC1 share {self.pc1_share:.2f}  (top-3 eigen shares "
            f"{', '.join(f'{s:.2f}' for s in self.eigen_shares[:3])})",
            f"orientation: {len(self.reversed_variants)} reversed variant(s) "
            f"{self.reversed_variants}",
            f"dedup at rho={RECIPE_RHO}: {self.n_features} variants -> {self.n_blocks} ideas",
        ]
        for b, members in self.blocks.items():
            lines.append(f"    block {b}: {', '.join(members)}")
        sweep = ", ".join(
            f"k={int(k)}: {ic:+.4f}" for k, ic in zip(self.pls_k, self.pls_oof_ic, strict=True)
        )
        lines.append(
            f"PLS-k out-of-fold IC: {sweep}  "
            f"(benchmark {self.naive_oof_ic:+.4f}; best k = {self.pls_best_k})"
        )
        lines.append(
            f"tau^2 DL/PM/REML: {self.tau2_dl:.2e} / {self.tau2_pm:.2e} / {self.tau2_reml:.2e}  "
            f"(block ICs {', '.join(f'{v:+.4f}' for v in self.block_ic)})"
        )
        lines.append("reading:")
        lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def diagnose(panel: Panel, rho: float = RECIPE_RHO, run_pls_sweep: bool = True) -> ReportCard:
    """Run the playbook's diagnostics on one training panel."""
    corr = panel.correlation()
    shares = eigen_shares(corr)
    reversed_variants = orientation_flips(corr, panel.cols)
    labels = cluster_labels(corr, rho)
    blocks = {
        int(b): [c for c, lab in zip(panel.cols, labels, strict=True) if lab == b]
        for b in np.unique(labels)
    }

    if run_pls_sweep:
        sweep = linear.pls_sweep(panel)
        naive_ref = float(np.mean(panel.oof_run(linear.naive_averaged)[0]))
    else:
        sweep = {"k": [], "mean": [], "std": []}
        naive_ref = float("nan")

    comp = DedupComposite(rho=rho, block_weights="eq").fit(panel)
    d = comp.diagnostics
    assert comp.block_ic is not None

    notes: list[str] = []
    pc1 = float(shares[0])
    if pc1 >= 0.45:
        notes.append(
            f"PC1 share {pc1:.2f} >= 0.45: essentially one factor; equal weighting will "
            "saturate this class and any supervision is a tax."
        )
    elif pc1 <= 0.35:
        notes.append(
            f"PC1 share {pc1:.2f}: diffuse spectrum, genuinely multi-factor; the partition "
            "matters here and the plain rank-and-average is the likely casualty."
        )
    else:
        notes.append(f"PC1 share {pc1:.2f}: intermediate; check the merge tree for clean blocks.")
    if reversed_variants:
        notes.append(
            f"{len(reversed_variants)} variant(s) reversed by PC1 alignment: fix the "
            "orientation before anything else, and take at least one sign from the target."
        )
    if run_pls_sweep:
        k_best = int(sweep["k"][int(np.argmax(sweep["mean"]))])
        if k_best == 1:
            notes.append(
                "PLS sweep peaks at k=1: one predictive direction; any sensible weighting works."
            )
        elif k_best >= 3:
            notes.append(
                f"PLS sweep peaks at k={k_best}: multi-directional signal in-sample. Research "
                "flag only; the study's frozen multi-component fits inverted out of sample."
            )
        else:
            notes.append(
                f"PLS sweep peaks at k={k_best}: weak second direction; treat as one factor."
            )
    tau_pos = [v > 0 for v in (d["tau2_dl"], d["tau2_pm"], d["tau2_reml"])]
    if all(tau_pos):
        notes.append(
            "tau^2 > 0 under all three estimators: the blocks truly differ. Keep equal "
            "weights (the tilt lost on the blind test) and flag the class for attention."
        )
    elif not any(tau_pos):
        notes.append(
            "tau^2 = 0 under all three estimators: equal block weights are optimal in-family; "
            "walk away."
        )
    else:
        notes.append("tau^2 estimators disagree: treat heterogeneity as unproven; equal weights.")
    notes.append(
        f"Selected composite: dedup at rho={rho}, {d['n_blocks']} equal-weight blocks, frozen. "
        "Run the shape gate separately before adopting learned shapes."
    )

    return ReportCard(
        name=panel.name,
        n_features=len(panel.cols),
        n_days=panel.n_days,
        n_rows=panel.n_rows,
        pc1_share=pc1,
        eigen_shares=[float(s) for s in shares],
        reversed_variants=reversed_variants,
        n_blocks=int(d["n_blocks"]),
        blocks=blocks,
        pls_k=list(sweep["k"]),
        pls_oof_ic=list(sweep["mean"]),
        naive_oof_ic=naive_ref,
        tau2_dl=float(d["tau2_dl"]),
        tau2_pm=float(d["tau2_pm"]),
        tau2_reml=float(d["tau2_reml"]),
        block_ic=[float(v) for v in comp.block_ic],
        notes=notes,
    )
