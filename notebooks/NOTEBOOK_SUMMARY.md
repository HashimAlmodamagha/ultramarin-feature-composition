# Ultramarin — Notebook Section Summary

*One–two sentences per section, in notebook order. Generated 2026-07-12 from the corrected
record; 200 cells as of 2026-07-24 (XII.9 gallery + errata sync added 07-22; re-synced to the
final 28-pp report and a Part VII merge-tree display cell added 07-24); companion to
`Ultramarin.ipynb`.*

## Header & Part I — EDA and the one-factor hypothesis
- **Intro:** Frames the problem — each feature class is a cluster of correlated variants proxying one latent signal, and the goal is to beat the naive rank-and-average composite; explains why PCA is the wrong tool (unsupervised, rewards redundancy) and defines the IC scorecard used throughout.
- **Joined feature + target tables:** Joins each class's feature and target panels on (date, identifier).
- **Preprocessing:** Removes missing-target and missing-feature rows to complete cases (~1.5M rows for c1; c2/c3 lose the most, ~12%).
- **Inferring the target's return horizon:** The target's triangular autocorrelation decay proves it is a ~21-day cumulative forward return — which retroactively justifies the 25-lag Newey–West window and the 21-day CV purge.
- **Within-cluster redundancy:** Shows the strong within-class correlations that naive averaging overweights — the structure the whole project exploits.
- **Sign alignment:** Reorients variants unsupervised (flip if PC1 loading is negative); c1 has 2 flipped duplicates, c2/c3 are coherent, c4's negatives are genuine multi-factor structure, not flips.
- **Original vs orientation vs redundancy:** Three views of each correlation matrix separating sign convention from true redundancy blocks.
- **Feature ↔ target correlations / |IC|:** Per-variant predictive strength — pooled correlation vs mean daily IC, then magnitudes with orientation removed.
- **Locking in class_1 orientation:** Persists the two PCA-flagged reflections (f4, f12 → 1−x) into the data itself, target-free and idempotent.

## Part II — Evaluation framework, baselines & benchmarks
- **The evaluation harness:** Defines daily rank IC, Newey–West (25-lag) significance, decay, and turnover — the fixed measuring apparatus every later part reuses.
- **Baselines and first ideas:** naive average, PCA PC1, IC-weighting, and the uniqueness (ridge-GLS) regression.
- **Purged 5-fold cross-validation:** Contiguous day-block folds with 21-day purge+embargo (Lopez de Prado), because random splits leak overlapping labels.
- **Regression benchmarks (OLS/ridge/lasso):** Textbook regressions with leak-free inner-validation tuning; least-squares objectives take a ~10% IC haircut vs rank-aligned methods.
- **Canonical fit / regularization path / why uniqueness fails:** Sweeping lambda shows OOS IC *improves* as you stop trusting the covariance matrix — c1 is near one-factor, so the inverse-covariance correction amplifies noise and IC-weighting wins on bias-variance.
- **Part II findings:** The cluster is ~one latent factor worth ~0.031 IC; weighting choices barely matter; next steps are nested tuning and supervised clustering.

## Part III — PLS & XGBoost
- **PLS verdict:** The tuner picks k=1 in all five folds — there is exactly one recoverable predictive direction, and PLS-1 = IC-weighting.
- **III.2 XGBoost + verdict:** An honestly-tuned tree matches the linear methods (0.0315 vs 0.0309) and prefers minimal capacity — ~0.031 is the *real* ceiling, not just the linear one.

## Part IV — Cross-method comparison & Part V — Interim verdict
- **Full comparison verdict:** All methods tie on IC, but trees reshape tradeability — slowest decay (11.1d half-life) and lowest 5-day turnover.
- **IV.2 rigor checks:** The paired out-of-fold test kills xgb's apparent IC edge (in-sample 0.047 was memorization) but its tradeability edge survives the same test — the study's first "claims must be earned out-of-fold" lesson.
- **Part V interim verdict:** Ship the parsimonious sign-aligned average for accuracy; trees are defensible only for a cost-sensitive weekly book.

## Part VI — Multi-class generalization
- **VI.1/VI.2 + cross-class verdict:** Every class is ~one factor and the naive average is the anchor everywhere; supervised weighting helps only where the cluster is diffuse (c4), and XGBoost collapses by 40–50% on c2–c4 — its c1 parity was the best case.

## Part VII — Weighting by uniqueness, done right
- **VII.1 predictive dimension:** Out-of-fold PLS-k sweeps show variance dimension ≠ predictive dimension — c4 looks multi-factor but predicts one-dimensionally; c3 is the reverse.
- **VII.2 redundancy-aware suite + verdict:** Cluster-then-weight methods vs tuned uniqueness vs a diversification penalty; `cluster_eq` (dedup + equal weights) is the only method that never loses to naive and the only nominal significant win (+0.0028 on c4, t 2.26), while supervision is a significant *tax* on one-factor c2.
- **VII.3 adaptive selection + verdict:** Letting inner validation pick the method per class is "never terrible, never optimal" — model selection is itself an estimation problem at this signal-to-noise.
- **VII.4 panel vs rolling vs daily + verdict:** No walk-forward estimator beats pooled estimation anywhere — the redundancy structure is time-stable; orientation is the one irreducible supervised bit.
- **VII.5 tree mechanism + verdicts (incl. VII.5b LightGBM):** Depth-1 stumps reproduce the full tree's entire profile and bucketing reproduces none of it — the edge is *learned additive per-feature transforms*, and LightGBM replicating everything makes the story tree-generic.
- **VII.6 bottom line:** The recipe emerges — diagnose, dedup at rho=0.7, equal-weight blocks, escalate only on diagnostic evidence.

## Part VIII — Evidence synthesis
- **VIII.1 DerSimonian–Laird + verdict:** Meta-analysis random-effects pooling gives tuning-free adaptivity — tau2=0 on c1–c3 (→ equal weights) and tau2>0 on c4 (→ tilt), recovering Part VII's hard-won map from training data alone; the tau2=0 follow-up note explains what DL does and doesn't claim vs GLS.
- **VIII.2 decay pharmacokinetics + verdict:** Two-compartment fits decompose every composite into fast/slow alpha; the tree's transforms double the slow share (0.54 vs 0.29) — the single number behind the tradeability story.
- **VIII.3 reading the stump model + verdict:** The learned transforms are exact, extractable lookup tables forming a graded *tail-depth detector* (75% of response in the top third of ranks); both concentration and gradation are load-bearing.

## Part IX — Stress-testing the recipe
- **IX.1 partial pooling:** Within-block DL shrinkage collapses to the block mean in 19/20 class-folds — the estimator, offered the middle depth, chooses full pooling; the dedup normalization itself is worth up to 30bp.
- **IX.2 partition choice:** Linkages, IC-profile similarity, divisive trees, and a random placebo at matched k — the partition matters only on c4, where the anchor wins and the placebo collapses.
- **IX.3 tradeability:** The cluster composites trade inside the linear-family band (no toll), but no reweighting buys the tree's decay profile.
- **IX.4 closing scorecard:** The two-track recommendation with full evidence — recipe for accuracy/robustness, shapes/stumps for the weekly book.
- **IX.5 feature-drop robustness:** Across 80 reduced menus the recipe (`cluster_eb`, the candidate at the time) is top-2 everywhere with the shallowest worst case (`cluster_eq` a 0.0001 behind); every rival craters somewhere — naive is last on 17/20 c4 menus, bottom-two on 19 — and the c3 GLS edge inverts.
- **IX.6 sponsor follow-ups:** Honest per-fold rho tuning can't beat fixed 0.7 (the ex-post best cell is not ex-ante selectable), and centroid linkage reproduces average linkage exactly.

## Part X — Learning the shapes inside the recipe
- **X.1 shaped recipe + gate verdict:** Passing variants through their stump curves before the recipe costs nothing on c1 and would be a disaster on c2/c3 — the weekly-book gate adopts c1 and refuses c4, but is noisy exactly where evidence is weak.
- **X.2 tradeability verdict:** The shaped recipe doesn't just inherit the tree's profile, it improves on it (13.5d half-life, 0.362 turnover in-sample) — and deploys as 16 lookup tables + one clustering, no tree library.

## Part XI — The blind hold-out
- **Erratum (2026-07-11):** Documents the c1 reflection bug (hold-out scored with f4/f12 unreflected), the one-line fix, and what changed — c1 linear ICs to ~0.0119 (t≈1.9), shapes barely moved, and the "Frobenius 7.4 drift" was ~97% artifact.
- **XI.0 recipe lock:** A final in-sample audit shows no universal fixed rho beats 0.7 — the recipe is frozen before touching the hold-out.
- **XI.1–XI.2 + Part XI verdict:** Under strict freeze on ~2,180 unseen days, c2–c4 are significantly positive, the recipe = naive on IC exactly as claimed, everything flagged fragile is what failed, and there are two casualties: the gate's weak-evidence c3 adoption (−0.0107) and the frozen c4 tilt (−0.0031 vs equal weights); c1's IC loss is pure alpha decay with a stable correlation structure.
- **XI.1x retired benchmarks (added 2026-07-12, post hoc):** `pca_pc1` and `pls_tuned` — retired in-sample in Parts II–III — are frozen under the identical protocol and appended into every hold-out scorecard. Out of sample: PC1 keeps a third of naive's IC on c4 (0.0063 vs 0.0192) and merely equals it on the one-factor classes; PLS-1 ≈ ic_weight where the tuner froze k=1 (c1, c4), and c3's honestly-tuned k=6 halves the class's IC (0.0070 vs 0.0133).
- **XI.3 gate v2 + verdict:** A two-stage in-sample evidence bar (CV non-inferiority + genuine tradeability win) adopts shapes on c1 alone and, verified once on the hold-out, recovers everything the frozen gate gave away.
- **XI.4 study in one figure:** CV→hold-out dumbbells for the three deployable choices, annotated with turnover.
- **XI.5 shaped v2 + verdict:** Sign-aligning the shaped block members — the pre-specified tidy-up — is statistically a wash (0.0155 vs 0.0154); v1 was safe all along.
- **XI.6 rolling vs freeze + verdict:** Quarterly re-estimation helps only the c4 tilt (0.0163→0.0182) — but frozen *equal weights* already sit at 0.0194, so the refit only repairs the tilt's self-inflicted wound; c1's decay is unrecoverable by any re-weighting.
- **XI.7 (superseded banner) + XI.7b plain words:** The audit trail of the withdrawn refit-iff-tau2>0 rule, and the final recipe in plain language — count each idea once, weight equally, shapes only where evidence is strong, freeze everything.
- **XI.8 Kendall-tau + verdict:** The one nominal win is concordance-robust (t 2.29 under tau) — not an artifact of Spearman.

## Part XII — Referee checks
- **XII.1 family-wise test + figure + verdict:** Of 56 method-vs-naive comparisons (incl. the XI.1x benchmarks), seven clear |t|>2 nominally and **zero** survive Holm, BH 5%, or the Westfall–Young max-T (bar ≈ 3.75); the only survivors of any correction (BH 10%) are two *negative* results — PCA's c4 collapse and c2's uniqueness tax. No certified way to beat naive; two near-certified ways to lose; the recipe's claim is the floor, not the mean.
- **XII.2 PM/REML/Hartung–Knapp + verdict:** Three estimators agree on the tau2 map (c4 heterogeneity is real, 5/5 folds), but all three tilts lose to equal weights on the hold-out (0.0163–0.0166 vs 0.0194) — detection is robust, monetization fails, and HK's vacuous small-k intervals explain why.
- **XII.3 eq vs eb + verdict:** Four independent reads (in-sample paired, frozen hold-out, menus, rolling arms) all point one way — ship `cluster_eq`, keep tau2 as a diagnostic.
- **XII.4 shapes vs menus + figure + verdict:** Across 20 menus with per-fold refits, shapes are free in-sample (0/20 significant losses), ahead OOS on 18–20/20, and the gate's single refusal cost +0.0003 on the one menu where shapes genuinely churned more; the c4 temptation is small and uncertifiable.
- **XII.5 combo verdict:** On the weekly level-plus-decay criterion, the shipped lineup (shaped c1 + cluster_eq) is ahead of naive on every class where anything differs — pooled +0.0019/day, t 1.72, suggestive and labeled as such; the retired benchmarks calibrate the scale — `pca_pc1` posts the study's only |t|>3 pooled result, a *deficit* (−0.0039/day, t −3.07), so the test has power and the shipped edge is competing with genuine zero.
- **XII.5b retired benchmarks OOS + figure + verdict:** The out-of-sample mirror of VII.1's sweep — hold-out IC of the frozen PLS-k signal as components are added, per class. Both retirements confirmed: PC1 collapses on c4 and equals naive elsewhere; PLS-1 ≈ ic_weight where k=1 froze; c2's tuned k=2 is the one honest tuning win (+0.0011, ns) while c3's defensible-in-sample k=6 halves IC out of sample — the study's sharpest "wins the inner split, dies frozen" exhibit.
- **XII.6 shipped default:** The final spec — dedup at rho=0.7, equal weights, frozen forever, shapes v2 behind gate-v2, tau2 as diagnostic — with a one-line justification for every change from XI.7.
- **XII.7 closing commentary:** The honest answer to "was any of this better than averaging?" — no on raw IC (proven), yes on signal quality, insurance, the combo criterion, and the catalogue of expensive upgrades the study cheaply killed.
- **XII.8 + XII.8b + XII.8c nutshell figures:** The whole study on one page (six claims, six panels — the retired benchmarks appear in panel A's callout, panel C's menu floors via XI.1y, and panel D's c1 decay), the dominance coda — shipped ≥ naive at every holding horizon on every class — and XII.8c, signal life across the board: hold-out decay for all 11 frozen methods × 4 classes with the lag-5 retention summary (the class sets the clock; only c1's shapes reset it; only c4's PCA breaks it, going negative past lag 12).
- **XII.9 report gallery + errata sync (2026-07-22; re-synced 2026-07-24 to the final 28-pp report — figure numbers updated, full 15-figure set rebuilt, merge-tree dendrogram Fig. 3 displayed at its Part VII home):** Regenerates the written report's figures from the checkpoint (`report/figures/make_v2_figures.py`) and displays the four beyond the notebook's record — the nutshell, the rolling 250-day hold-out IC (new analysis: c1's decay concentrates in the final year, final-250d mean −0.0397 vs 0.0227 before; c2–c4 cycle around positive means), the hold-out forest plot, and the 80-menu strip plot. Also records the report fact-check's corrections applied to the verdict cells (17/20 not "all 20"; ~40–60% tree collapse; 56-comparison family; eq-vs-eb floor attribution).

## Appendix
- **Checkpoint & fast resume:** Pickles all reusable products at the end of every full run and provides a one-click resume path so a kernel restart costs ~3 minutes instead of an hour.
