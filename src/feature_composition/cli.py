"""Command-line entry points.

``feature-composition diagnose --data-dir DATA --feature-class 1``
    Run the playbook diagnostics on one class of Ultramarin-format parquet files.

``feature-composition compare --data-dir DATA --feature-class 1``
    Purged-CV comparison of the linear and cluster methods against the benchmark.

``feature-composition freeze --data-dir DATA --feature-class 1 --out c1.json``
    Fit the selected composite on the in-sample split, optionally score the out-of-sample
    split under strict freeze, and save the frozen state as JSON.

``feature-composition demo``
    The whole pipeline end to end on a synthetic panel (no data needed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from feature_composition import cluster_methods, linear
from feature_composition.composite import DedupComposite
from feature_composition.cv import cross_validate, paired_against
from feature_composition.diagnostics import diagnose
from feature_composition.holdout import score_signal
from feature_composition.panel import Panel


def _load(args: argparse.Namespace, split: str = "in_sample") -> Panel:
    reflect = [c for c in (args.reflect or "").split(",") if c]
    return Panel.load_class(args.data_dir, args.feature_class, split, reflect=reflect)


def _methods(include_tuned: bool) -> dict[str, object]:
    m: dict[str, object] = {
        "naive_averaged": linear.naive_averaged,
        "ic_weight": linear.ic_weight,
        "pca_pc1": linear.pca_pc1,
        "uniqueness_reg": linear.uniqueness_reg,
        "eb_weight": linear.eb_weight,
        "cluster_eq": cluster_methods.cluster_eq,
        "cluster_eb": cluster_methods.cluster_eb,
    }
    if include_tuned:
        m.update(
            {
                "uniqueness_tuned": linear.uniqueness_tuned,
                "ridge_tuned": linear.ridge_tuned,
                "pls_tuned": linear.pls_tuned,
                "cluster_eq_tuned": cluster_methods.CLUSTER_METHODS["cluster_eq_tuned"],
            }
        )
    return m


def cmd_diagnose(args: argparse.Namespace) -> int:
    panel = _load(args)
    print(panel)
    print(diagnose(panel, rho=args.rho, run_pls_sweep=not args.no_pls).render())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    panel = _load(args)
    print(panel)
    table, oof = cross_validate(panel, _methods(args.tuned))  # type: ignore[arg-type]
    with pl.Config(tbl_rows=40, float_precision=4):
        print(table)
        print("\npaired Newey-West vs the benchmark (out-of-fold daily IC):")
        print(paired_against(oof, "naive_averaged").sort("ic_diff", descending=True))
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    panel = _load(args)
    print(panel)
    comp = DedupComposite(rho=args.rho, block_weights="eq", shapes=args.shapes).fit(panel)
    print(comp.describe())
    if args.out:
        comp.save(args.out)
        print(f"frozen composite written to {args.out}")
    if args.score_holdout:
        oos = Panel.load_class(
            args.data_dir, args.feature_class, "out_of_sample", reflect=comp.flipped
        )
        print(oos)
        bench = (oos.X * np.asarray(comp.signs)).mean(1)
        for name, sig in (("selected composite", comp.transform_panel(oos)), ("benchmark", bench)):
            m = score_signal(oos, sig)
            print(
                f"{name:20s} hold-out IC {m['mean_ic']:+.4f} (NW t {m['nw_t']:+.2f}), "
                f"ret_5d {m['ret_5d']:.3f}, turn_5d {m['turn_l5']:.3f}"
            )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from feature_composition.synthetic import make_panel_frame

    frame = make_panel_frame(
        n_days=args.n_days,
        n_names=args.n_names,
        block_sizes=(5, 3, 2, 1, 1),
        within_block_noise=0.5,
        reversed_variants=(1, 7),
        seed=args.seed,
    )
    cut = int(0.7 * frame["date"].n_unique()) + int(frame["date"].min())  # type: ignore[arg-type]
    train = Panel(frame.filter(pl.col("date") < cut), name="synthetic in-sample")
    print(train)
    card = diagnose(train)
    print(card.render())
    print()
    table, _ = cross_validate(train, _methods(False))  # type: ignore[arg-type]
    with pl.Config(tbl_rows=20, float_precision=4):
        print(table)
    comp = DedupComposite().fit(train)
    print()
    print(comp.describe())
    test = Panel(frame.filter(pl.col("date") >= cut + train.horizon), name="synthetic hold-out")
    bench = (test.X * np.asarray(comp.signs)).mean(1)
    print()
    for name, sig in (("selected composite", comp.transform_panel(test)), ("benchmark", bench)):
        m = score_signal(test, sig)
        print(
            f"{name:20s} hold-out IC {m['mean_ic']:+.4f} (NW t {m['nw_t']:+.2f}), "
            f"ret_5d {m['ret_5d']:.3f}, turn_5d {m['turn_l5']:.3f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feature-composition",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--data-dir", type=Path, required=True, help="directory of the parquet files"
        )
        sp.add_argument("--feature-class", type=int, required=True, help="class number, e.g. 1")
        sp.add_argument(
            "--reflect", default="", help="comma-separated variants to reflect (x -> 1-x)"
        )
        sp.add_argument("--rho", type=float, default=0.7, help="merge threshold on |corr|")

    d = sub.add_parser("diagnose", help="playbook diagnostics for one class")
    data_args(d)
    d.add_argument("--no-pls", action="store_true", help="skip the PLS-k out-of-fold sweep")
    d.set_defaults(func=cmd_diagnose)

    c = sub.add_parser("compare", help="purged-CV method comparison")
    data_args(c)
    c.add_argument("--tuned", action="store_true", help="include the slower tuned methods")
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("freeze", help="fit and save the selected composite")
    data_args(f)
    f.add_argument("--shapes", action="store_true", help="use learned stump shapes (needs xgboost)")
    f.add_argument("--out", type=Path, help="write the frozen composite as JSON")
    f.add_argument("--score-holdout", action="store_true", help="score the out_of_sample split")
    f.set_defaults(func=cmd_freeze)

    m = sub.add_parser("demo", help="end-to-end run on a synthetic panel")
    m.add_argument("--n-days", type=int, default=500)
    m.add_argument("--n-names", type=int, default=150)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
