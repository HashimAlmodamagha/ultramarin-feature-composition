# The research notebook

`Ultramarin.ipynb` is the complete, executed research record of the project: 200 cells,
Parts I to XII plus a checkpoint appendix, last run clean end to end on 2026-07-25. Every
number in the report and the seminar deck was produced by this notebook, and every figure in
the report is regenerated from its checkpoint by `docs/report-source/figures/make_v2_figures.py`.

The notebook is kept as it was run, with its outputs, because a study's evidence is the
sequence of questions it asked and the order in which the answers arrived. It is **not** the
recommended entry point for using the methodology: that is the `feature_composition` package
in `src/`, which is the refactored, typed and tested form of the same code. Section
`XII.9` of the notebook and the package's `scripts/reproduce_study.py` compute the same
headline numbers by two routes.

## Reading guide

`NOTEBOOK_SUMMARY.md` gives one or two sentences per section. The short version:

| Part | Question | Answer |
|---|---|---|
| I | What is in a cluster of variants? | Orientation artifacts (fixable) and redundancy (the structure to exploit) |
| II | Which weighting wins on class 1? | None: every sensible weighting ties at an IC ceiling of ~0.031 |
| III | Do PLS or trees break the tie? | No: one predictive direction, no nonlinear edge on accuracy |
| IV, V | What do trees change? | Not the IC, but the decay and turnover profile |
| VI | Does class 1 generalize? | Every class has its own ceiling; trees collapse off class 1 |
| VII | Weighting by uniqueness, done right | Dedup + equal weights never loses; supervision is a tax on one-factor classes |
| VIII | A tuning-free adaptive weight | DerSimonian-Laird random effects: a flawless detector of heterogeneity |
| IX | Stress tests | Partition choice, partial pooling, 80 perturbed feature sets: dedup holds the highest floor |
| X | The shapes inside the trees | Per-feature response curves, adopted behind a gate on class 1 alone |
| XI | The blind hold-out | Signals real, IC edge zero, quality edge real, every flagged fragility failed on schedule |
| XII | Statistical checks | 56 comparisons: nothing beats the benchmark after multiplicity correction; two negative results survive |

## Running it

The notebook expects the Ultramarin parquet files in a `Data/` directory next to it. That
data is proprietary and is not distributed with this repository, so the notebook cannot be
re-executed from a fresh clone; it can be read, and the package can be run on any panel in
the same schema (see the top-level README). Cell outputs that were kernel warnings have been
removed from the published copy; nothing else was changed.
