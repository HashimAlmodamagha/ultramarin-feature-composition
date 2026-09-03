# Reproduction results

These files were written by `scripts/reproduce_study.py --tuned --shapes` running this
package on the Ultramarin data (in-sample panels for estimation, the blind out-of-sample
panels for the frozen scorecard). They contain aggregate statistics only; no data rows.

Every value was checked against the research notebook's checkpoint and agrees to the
precision the notebook reports (four decimals for ICs, three for retention and turnover).

| file | contents |
|---|---|
| `summary.md` | the headline tables |
| `diagnostics_c<n>.txt` | the playbook report card: PC1 share, orientation, merge tree at rho = 0.7, PLS-k sweep, tau² under three estimators |
| `cv_c<n>.csv` | purged 5-fold CV: mean out-of-fold IC, across-fold std, worst fold, per method |
| `paired_c<n>.csv` | paired Newey-West test of each method's out-of-fold daily IC against the benchmark (`ic_diff`, `nw_t`) |
| `pls_sweep_c<n>.csv` | out-of-fold IC as PLS components are added (the predictive dimension) |
| `holdout_c<n>.csv` | frozen-on-in-sample scorecard on the hold-out: mean IC, Newey-West t, annualized IC-IR and portfolio IR, half-life, 5-day retention and turnover |

Method names follow the notebook: `naive_averaged` is the benchmark (rank, flip by the
sign of the training IC, average); `cluster_eq` / `cluster_eb` are the dedup composites at
the frozen rho = 0.7 with equal / random-effects block weights; `*_tuned` methods choose
their hyper-parameter on the purged inner validation slice; `cluster_eb_shaped` is the
shaped arm as the study recorded it (DerSimonian-Laird block weights on the shaped blocks),
`cluster_eq_shaped` the same with equal block weights, and `selected (gated)` the composite
the gate actually deploys (`cluster_eb_shaped` where shapes were adopted, `cluster_eq`
elsewhere).
