"""Reproduce the study's headline numbers from the Ultramarin parquet files.

The data is proprietary and not part of this repository. Point the script at a directory
holding ``feature_class_<n>_{in_sample,out_of_sample}.parquet`` and the matching
``*_target_*.parquet`` files and it will write, per class, into ``--out``:

* ``cv_<class>.csv``            purged-CV scorecard of the linear and cluster methods
* ``paired_<class>.csv``        paired Newey-West tests of every method against the benchmark
* ``pls_sweep_<class>.csv``     out-of-fold IC as PLS components are added
* ``holdout_<class>.csv``       frozen-on-in-sample scorecard on the out-of-sample split
* ``diagnostics_<class>.txt``   the playbook report card
* ``summary.md``                the numbers quoted in the README

Run with ``--shapes`` to add the learned-shape arm and the gate (needs xgboost; slow).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from feature_composition import cluster_methods, linear
from feature_composition.clustering import orientation_flips
from feature_composition.composite import DedupComposite
from feature_composition.cv import cross_validate, paired_against
from feature_composition.diagnostics import diagnose
from feature_composition.holdout import score_signal
from feature_composition.panel import Panel

HOLDOUT_KEYS = ("mean_ic", "nw_t", "ic_ir_ann", "port_ir_ann", "half_life", "ret_5d", "turn_l5")


def in_sample_methods(tuned: bool) -> dict[str, object]:
    m: dict[str, object] = {
        "naive_averaged": linear.naive_averaged,
        "ic_weight": linear.ic_weight,
        "pca_pc1": linear.pca_pc1,
        "uniqueness_reg": linear.uniqueness_reg,
        "eb_weight": linear.eb_weight,
        "cluster_eq": cluster_methods.cluster_eq,
        "cluster_eb": cluster_methods.cluster_eb,
    }
    if tuned:
        m.update(
            {
                "uniqueness_tuned": linear.uniqueness_tuned,
                "ridge_tuned": linear.ridge_tuned,
                "pls_tuned": linear.pls_tuned,
                "cluster_eq_tuned": cluster_methods.CLUSTER_METHODS["cluster_eq_tuned"],
                "cluster_ic_tuned": cluster_methods.CLUSTER_METHODS["cluster_ic_tuned"],
            }
        )
    return m


def run_class(
    data_dir: Path, n: int, out: Path, tuned: bool, shapes: bool, skip_cv: bool = False
) -> dict[str, object]:
    t0 = time.time()
    raw = Panel.load_class(data_dir, n, "in_sample")
    flips = orientation_flips(raw.correlation(), raw.cols)
    panel = Panel.load_class(data_dir, n, "in_sample", reflect=flips) if flips else raw
    print(f"\n{panel}  reflected={flips}", flush=True)

    # --- in-sample: diagnostics, purged CV, paired tests, PLS sweep ---------------------
    card = diagnose(panel)
    (out / f"diagnostics_c{n}.txt").write_text(card.render() + "\n")
    print(card.render(), flush=True)

    pl.DataFrame({"k": card.pls_k, "oof_ic": card.pls_oof_ic}).write_csv(
        out / f"pls_sweep_c{n}.csv"
    )
    table = paired = None
    if not skip_cv:
        table, oof = cross_validate(panel, in_sample_methods(tuned))  # type: ignore[arg-type]
        table.write_csv(out / f"cv_c{n}.csv")
        paired = paired_against(oof, "naive_averaged")
        paired.write_csv(out / f"paired_c{n}.csv")
        with pl.Config(tbl_rows=30, float_precision=4):
            print(table)
            print(paired.sort("ic_diff", descending=True))

    # --- hold-out: freeze on in-sample, score once --------------------------------------
    oos = Panel.load_class(data_dir, n, "out_of_sample", reflect=flips)
    print(oos, flush=True)
    Z_is, ic_is, _ = panel.fit(panel.all_days)
    mu, sd = panel.X.mean(0), panel.X.std(0)
    Zo = (oos.X - mu) / sd
    signs = np.where(ic_is >= 0, 1.0, -1.0)
    frozen: dict[str, NDArray[np.float64]] = {
        "naive_averaged": (oos.X * signs).mean(1),
        "ic_weight": Zo @ (ic_is / np.linalg.norm(ic_is)),
    }
    v = np.linalg.eigh(np.cov(Z_is, rowvar=False))[1][:, -1]
    v = -v if np.corrcoef(Z_is @ v, (panel.X * signs).mean(1))[0, 1] < 0 else v
    frozen["pca_pc1"] = Zo @ v
    for name, bw in (("cluster_eq", "eq"), ("cluster_eb", "eb")):
        comp = DedupComposite(block_weights=bw).fit(panel)  # type: ignore[arg-type]
        frozen[name] = comp.transform_panel(oos)
    gate_line = ""
    if shapes:
        from feature_composition.gate import gate_decision

        decision = gate_decision(panel)
        gate_line = str(decision)
        print(gate_line, flush=True)
        shaped = DedupComposite(shapes=True).fit(panel)
        frozen["cluster_eq_shaped"] = shaped.transform_panel(oos)
        frozen["selected (gated)"] = frozen[
            "cluster_eq_shaped" if decision.adopt_shapes else "cluster_eq"
        ]
    rows = []
    for name, sig in frozen.items():
        m = score_signal(oos, sig)
        rows.append({"method": name, **{k: m[k] for k in HOLDOUT_KEYS}})
    ho = pl.DataFrame(rows)
    ho.write_csv(out / f"holdout_c{n}.csv")
    with pl.Config(tbl_rows=30, float_precision=4):
        print(ho)
    print(f"class {n} done in {time.time() - t0:.0f}s", flush=True)
    return {
        "class": n,
        "n_features": len(panel.cols),
        "n_days": panel.n_days,
        "oos_days": oos.n_days,
        "pc1_share": card.pc1_share,
        "n_blocks": card.n_blocks,
        "flips": flips,
        "cv": table,
        "paired": paired,
        "holdout": ho,
        "gate": gate_line,
    }


def write_summary(results: list[dict[str, object]], out: Path) -> None:
    lines = ["# Reproduction summary", ""]
    lines.append(
        "| class | variants | ideas | PC1 share | in-sample days | hold-out days | reflected |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['class']} | {r['n_features']} | {r['n_blocks']} | {r['pc1_share']:.2f} | "
            f"{r['n_days']:,} | {r['oos_days']:,} | {r['flips']} |"
        )
    if results[0]["cv"] is not None:
        lines += ["", "## Purged-CV mean IC (in-sample)", ""]
        methods = list(results[0]["cv"]["method"])  # type: ignore[index]
        lines.append("| method | " + " | ".join(f"c{r['class']}" for r in results) + " |")
        lines.append("|---|" + "---|" * len(results))
        for m in methods:
            vals = []
            for r in results:
                t = r["cv"].filter(pl.col("method") == m)  # type: ignore[attr-defined]
                vals.append(f"{t['cv_mean_ic'][0]:.4f}" if t.height else "")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")
    lines += ["", "## Hold-out mean IC (frozen on in-sample)", ""]
    hmethods = list(results[0]["holdout"]["method"])  # type: ignore[index]
    lines.append("| method | " + " | ".join(f"c{r['class']}" for r in results) + " |")
    lines.append("|---|" + "---|" * len(results))
    for m in hmethods:
        vals = []
        for r in results:
            t = r["holdout"].filter(pl.col("method") == m)  # type: ignore[attr-defined]
            vals.append(f"{t['mean_ic'][0]:.4f} (t {t['nw_t'][0]:.2f})" if t.height else "")
        lines.append(f"| {m} | " + " | ".join(vals) + " |")
    for r in results:
        if r["gate"]:
            lines += ["", f"### Gate, class {r['class']}", "", "```", str(r["gate"]), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--tuned", action="store_true", help="include the slower tuned methods")
    ap.add_argument("--shapes", action="store_true", help="learned shapes + gate (needs xgboost)")
    ap.add_argument("--skip-cv", action="store_true", help="skip the in-sample CV (hold-out only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = [
        run_class(args.data_dir, n, args.out, args.tuned, args.shapes, args.skip_cv)
        for n in args.classes
    ]
    write_summary(results, args.out)
    print(f"\nwrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
