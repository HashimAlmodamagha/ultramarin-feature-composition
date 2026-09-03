# Count Each Idea Once

**Building robust composites from clusters of correlated alpha-signal variants.**

[![CI](https://github.com/HashimAlmodamagha/ultramarin-feature-composition/actions/workflows/ci.yml/badge.svg)](https://github.com/HashimAlmodamagha/ultramarin-feature-composition/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

A quantitative research team rarely holds one definition of a factor. It holds a cluster of
correlated variants of the same idea (different lookbacks, normalizations, residualizations),
and those variants must somehow become one signal. This repository is my Berkeley MFE
industry project for Ultramarin on exactly that question:
**how should a cluster of correlated feature variants be combined into a single composite,
and does anything beat ranking each variant and averaging?**

It contains three things:

| | Where | What |
|---|---|---|
| **The report** | [`docs/Count_Each_Idea_Once_report.pdf`](docs/Count_Each_Idea_Once_report.pdf) | 28 pages: the study, its evidence and the selected methodology (LaTeX source in [`docs/report-source/`](docs/report-source/)) |
| **The seminar deck** | [`docs/Count_Each_Idea_Once_seminar_deck.pdf`](docs/Count_Each_Idea_Once_seminar_deck.pdf) | 30 slides presented at Ultramarin's research seminar, August 2026 |
| **The code** | [`src/feature_composition/`](src/feature_composition/) | A typed, tested Python package: the evaluation framework, every composition method in the study, the selected composite, and a diagnostic playbook for new feature classes. The 200-cell research notebook that produced the report is kept as the record in [`notebooks/`](notebooks/) |

The data (four anonymized feature classes and a blind hold-out, provided by Ultramarin
under NDA) is **not** included and is not needed to use the package: everything runs on any
panel in the same schema, and a synthetic generator ships with the tests.

## The finding

Every method in the study, from IC weighting through ridge, lasso, partial least squares,
gradient-boosted trees, redundancy-aware clustering and a random-effects estimator borrowed
from clinical meta-analysis, was scored under one fixed framework: daily rank information
coefficient (IC) with Newey-West significance, signal decay, factor turnover, purged 5-fold
cross-validation with a 21-day purge and embargo, and finally a strict freeze-on-in-sample
test on ~2,180 unseen trading days per class.

Three claims survived:

1. **Each cluster is worth what it is worth.** Every class has a hard accuracy ceiling that
   equal weighting essentially attains. Trees, PLS, GLS and tuned penalties cannot move it.
   After a family-wise multiplicity correction over all 56 method-versus-benchmark
   comparisons, nothing beats the benchmark on raw IC, including the study's own headline win.
2. **Deduplication is the one idea that survives every test.** Merging near-duplicate variants
   into blocks and weighting the blocks equally never loses anywhere, holds the shallowest
   worst case across 80 randomly perturbed feature sets, and delivers the study's only
   nominally significant win exactly where duplication bias binds (the one genuinely
   multi-factor class, +0.0028 daily IC, t = 2.26).
3. **The real edge is signal quality, not IC level.** Where the selected composite differs
   from the benchmark it decays more slowly and trades at 35-40% lower turnover.

The selected methodology, in one sentence: **count each idea once, weight ideas equally, learn
feature shapes only where the evidence is strong, and freeze everything.**

| step | rule |
|---|---|
| 1. Group the duplicates | Average-linkage clustering on `1 - |corr|`; merge variants sharing at least `|corr| = 0.7` (52 variants became 23 ideas across the four classes). The threshold is fixed, not tuned: leak-free tuning was offered twice and never beat it. |
| 2. Average within blocks | Flip negative-IC variants, then average each block's members. |
| 3. Weight ideas equally, freeze | The random-effects tau² detected real heterogeneity on one class; acting on it lost on the blind test. tau² stays as a diagnostic, never as weights. |
| 4. Shapes only past the gate | Per-feature response curves extracted from a depth-1 boosted fit, adopted only where a two-condition evidence bar clears (accuracy non-inferiority under purged CV plus a strict tradeability win). Empirically class 1 alone: double the half-life at roughly half the turnover, at the same accuracy. |

### Hold-out scorecard (frozen on in-sample, scored once)

Mean daily rank IC on ~2,180 unseen trading days per class, Newey-West t in parentheses;
`selected` is the shaped dedup composite on class 1 and the plain dedup composite elsewhere.

| class | variants → ideas | benchmark (rank-and-average) | dedup + equal weights | selected | 5-day turnover, benchmark → selected |
|---|---|---|---|---|---|
| 1 | 16 → 9 | 0.0118 (1.88) | 0.0119 (1.91) | **0.0154 (2.45)** | 0.62 → 0.39 |
| 2 | 10 → 4 | 0.0146 (2.43) | 0.0175 (2.66) | 0.0175 (2.66) | 0.03 → 0.04 |
| 3 | 6 → 2 | 0.0133 (2.64) | 0.0133 (2.64) | 0.0133 (2.64) | 0.03 → 0.03 |
| 4 | 20 → 8 | 0.0192 (3.94) | 0.0194 (3.69) | 0.0194 (3.69) | 0.26 → 0.17 |

These are the numbers produced by [`scripts/reproduce_study.py`](scripts/reproduce_study.py)
running this package on the data; they reproduce the research notebook's record to four
decimals (see [Reproducibility](#reproducibility)). Appendix A of the report carries the full
scorecard for every method.

## Quick start

```bash
pip install "ultramarin-feature-composition[trees] @ git+https://github.com/HashimAlmodamagha/ultramarin-feature-composition"
feature-composition demo          # the whole pipeline on a synthetic cluster, no data needed
```

The `trees` extra pulls in `xgboost`, which is only needed for the learned-shape arm; the
core package depends on `numpy`, `polars` and `scipy` alone.

### On your own data

A feature class is a table with `date` (integer trading-day index), `identifier`, the
variant columns (cross-sectional ranks in [0, 1]) and `target` (the residualized forward
return). The Ultramarin files were one parquet for the features and one for the target;
`Panel.from_parquet` joins them, `Panel(frame)` takes an already-joined frame.

```python
from feature_composition import DedupComposite, Panel, diagnose, gate_decision

train = Panel.from_parquet("data/class_in_sample.parquet", "data/class_target_in_sample.parquet")

# 1. read the class before touching a weight: PC1 share, merge tree, PLS-k sweep, tau²
print(diagnose(train).render())

# 2. decide, from in-sample evidence only, whether learned shapes are admitted
decision = gate_decision(train)          # needs xgboost; one stump fit per purged fold
print(decision)

# 3. freeze the selected composite and save it as plain JSON (no model library at runtime)
composite = DedupComposite(rho=0.7, block_weights="eq", shapes=decision.adopt_shapes).fit(train)
print(composite.describe())
composite.save("class_composite.json")

# 4. score it on unseen data, applying the same orientation the training panel had
test = Panel.from_parquet("data/class_out_of_sample.parquet",
                          "data/class_target_out_of_sample.parquet", reflect=composite.flipped)
signal = composite.transform_panel(test)
```

The same steps are available from the command line for files in the Ultramarin naming
convention (`feature_class_<n>_{in_sample,out_of_sample}.parquet` plus `*_target_*`):

```bash
feature-composition diagnose --data-dir data --feature-class 4
feature-composition compare  --data-dir data --feature-class 4 --tuned
feature-composition freeze   --data-dir data --feature-class 1 --shapes --out c1.json --score-holdout
```

## What is in the package

| module | contents | study |
|---|---|---|
| `metrics` | daily rank IC, Newey-West mean/t (numerically identical to statsmodels' HAC, without the dependency), decay and autocorrelation curves, half-life, weekly-book criterion, portfolio IR | Part II |
| `panel` | `Panel`: complete-case day-sorted panel, per-feature daily IC matrix, purged folds, inner validation split, out-of-fold runner | Parts II, VII |
| `cv` | purged k-fold with purge and embargo, chronological inner split, `cross_validate`, paired tests against an anchor | Part II |
| `clustering` | PC1 orientation alignment, redundancy blocks on `1 - |corr|`, block collapse matrix, the alternative partitions used as controls (linkages, IC-profile similarity, divisive, random placebo) | Parts I, VII, IX |
| `shrinkage` | DerSimonian-Laird, Paule-Mandel and REML tau², random-effects shrinkage, Hartung-Knapp interval, equal and adaptive block weights | Parts VIII, XII |
| `linear` | the benchmark, IC weighting, PCA PC1, the uniqueness (ridge-GLS) regression, OLS, ridge, lasso by coordinate descent, NIPALS PLS with the out-of-fold component sweep, the diversification penalty, feature-level random effects | Parts II, III, VII |
| `cluster_methods` | cluster-then-weight composites (`eq` / `ic` / `gls` / `eb` across blocks), threshold tuning, two-level partial pooling | Parts VII, VIII, IX |
| `trees` | boosted trees, exact stump-to-lookup-table extraction, shape application, quantile-bucketing control | Parts III, VII, VIII, X |
| `composite` | `DedupComposite`: the selected composite with fit / transform / JSON persistence and the tau² diagnostics | Parts XI, XII |
| `gate` | the two-condition evidence gate for learned shapes, `fit_selected` | Parts X, XI |
| `multiplicity` | Holm, Benjamini-Hochberg, Westfall-Young max-T by circular block bootstrap | Part XII |
| `diagnostics` | `diagnose`: the playbook report card for a new feature class | Report §10.1 |
| `holdout` | frozen-signal scorecard, trailing-window IC level monitor | Part XI |
| `synthetic` | synthetic clusters in the Ultramarin schema (blocks of near-duplicates, reversed variants, overlapping forward-return target) | tests, demo |

## Reproducibility

* **Tests.** `pytest` runs 49 tests on synthetic panels: fold purging, Newey-West against
  statsmodels, PLS-1 = IC weighting and PLS-p = OLS, stump extraction reproducing the
  booster's predictions, JSON round trips, the tau² estimators, the multiplicity corrections,
  and the property that motivated the whole study (dedup beats rank-and-average when one idea
  is re-implemented four times and another once).
* **Static checks.** `ruff` (lint and format) and `mypy --strict` on the package, run in CI on
  Python 3.10 and 3.12.
* **The study's numbers.** `scripts/reproduce_study.py --data-dir DATA --tuned --shapes`
  re-runs the in-sample cross-validation, the paired tests, the PLS sweep, the diagnostics and
  the frozen hold-out scorecard for all four classes with this package, and writes CSVs plus a
  summary. Run against the Ultramarin data, every value matches the research notebook's
  checkpoint to the four decimals the notebook reports (the CSVs are in [`results/`](results/)).

```bash
git clone https://github.com/HashimAlmodamagha/ultramarin-feature-composition
cd ultramarin-feature-composition
uv venv && uv pip install -e ".[dev]"      # or: pip install -e ".[dev]"
ruff check src tests scripts && mypy src && pytest
```

## Repository layout

```
src/feature_composition/   the package (see the module table above)
tests/                     pytest suite on synthetic panels
scripts/reproduce_study.py the study's headline numbers from the parquet files
results/                   CSVs written by that script on the Ultramarin data
notebooks/                 the executed 200-cell research notebook and its section summary
docs/                      report PDF, seminar deck PDF, report LaTeX source and figures
```

## Acknowledgements

The project was sponsored by Ultramarin, who provided the data, the proposal brief and the
blind hold-out, and supervised by Yves D'hondt, whose guidance shaped the study throughout.
The report and code are published with Ultramarin's permission; the data remains theirs.

## License

MIT for the code. The report and deck are © 2026 Hashim Almodamagha. If you build on the
work, please cite it (see [`CITATION.cff`](CITATION.cff)).
