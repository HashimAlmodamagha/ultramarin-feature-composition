"""The evidence gate for learned shapes: extra capacity enters only past a mechanical bar.

Applied unconditionally, the stump shapes destroy most of a short one-factor class's IC
(hundreds of stumps on a short panel is the tree collapse all over again). The gate admits
them for a class only when in-sample evidence clears two conditions, both computed from
training data alone with no tuned constants:

* **Condition A, accuracy non-inferiority.** The paired Newey-West t of the shaped versus the
  unshaped composite, on the out-of-fold daily IC difference under purged CV, exceeds -2.
* **Condition B, strict tradeability win.** The full-sample shaped fit retains strictly more
  IC at a 5-day lag *and* turns over strictly less than the unshaped fit.

On the study's four classes the gate adopted shapes on class 1 alone; verified once on the
blind hold-out, that lineup kept everything the shapes had to give and refused the class
where they would have failed. Across 20 perturbed feature subsets it made 19 correct calls and
its single refusal prevented an out-of-sample blow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from feature_composition import metrics
from feature_composition.clustering import RECIPE_RHO
from feature_composition.composite import DedupComposite

if TYPE_CHECKING:
    from feature_composition.panel import Panel


@dataclass
class GateDecision:
    """What the gate saw and what it decided."""

    adopt_shapes: bool
    oof_diff: float
    oof_t: float
    pass_a: bool
    ret_5d_shaped: float
    ret_5d_linear: float
    turn_5d_shaped: float
    turn_5d_linear: float
    pass_b: bool
    per_fold_shaped: list[float] = field(default_factory=list)
    per_fold_linear: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        verdict = (
            "ADOPT shapes" if self.adopt_shapes else "REFUSE shapes (keep the linear composite)"
        )
        return (
            f"{verdict}\n"
            f"  A  accuracy non-inferiority: oof diff {self.oof_diff:+.4f}, NW t {self.oof_t:+.2f} "
            f"(need > -2) -> {'pass' if self.pass_a else 'fail'}\n"
            f"  B  tradeability win: ret_5d {self.ret_5d_shaped:.3f} vs {self.ret_5d_linear:.3f}, "
            f"turn_5d {self.turn_5d_shaped:.3f} vs {self.turn_5d_linear:.3f} "
            f"-> {'pass' if self.pass_b else 'fail'}"
        )


def gate_decision(
    panel: Panel,
    rho: float = RECIPE_RHO,
    n_stumps: int = 600,
    block_weights: str = "eq",
    t_floor: float = -2.0,
) -> GateDecision:
    """Run both conditions on a training panel and return the decision with its evidence.

    Fits one stump model per purged fold (the expensive part) plus one on the full panel.
    """
    linear = DedupComposite(rho=rho, block_weights=block_weights)  # type: ignore[arg-type]
    shaped = DedupComposite(rho=rho, block_weights=block_weights, shapes=True, n_stumps=n_stumps)  # type: ignore[arg-type]

    oof_s, oof_l = np.full(panel.n_days, np.nan), np.full(panel.n_days, np.nan)
    per_s, per_l = [], []
    for _, train, test in panel.folds():
        days = np.flatnonzero(test)
        oof_s[days] = panel.daily_ic(shaped.fit(panel, train).transform(panel.X), days)
        oof_l[days] = panel.daily_ic(linear.fit(panel, train).transform(panel.X), days)
        per_s.append(float(np.nanmean(oof_s[days])))
        per_l.append(float(np.nanmean(oof_l[days])))
    diff, t = metrics.paired_newey_west(oof_s, oof_l)
    pass_a = bool(t > t_floor)

    ts = panel.tradeability(shaped.fit(panel).transform(panel.X))
    tl = panel.tradeability(linear.fit(panel).transform(panel.X))
    ret_s, ret_l = float(ts["ret_5d"]), float(tl["ret_5d"])  # type: ignore[arg-type]
    turn_s, turn_l = float(ts["turn_l5"]), float(tl["turn_l5"])  # type: ignore[arg-type]
    pass_b = bool(ret_s > ret_l and turn_s < turn_l)

    return GateDecision(
        adopt_shapes=pass_a and pass_b,
        oof_diff=diff,
        oof_t=t,
        pass_a=pass_a,
        ret_5d_shaped=ret_s,
        ret_5d_linear=ret_l,
        turn_5d_shaped=turn_s,
        turn_5d_linear=turn_l,
        pass_b=pass_b,
        per_fold_shaped=per_s,
        per_fold_linear=per_l,
    )


def fit_selected(
    panel: Panel, rho: float = RECIPE_RHO, n_stumps: int = 600
) -> tuple[DedupComposite, GateDecision]:
    """The full selected methodology on one training panel: gate, then freeze.

    Returns the frozen composite (shaped if and only if the gate adopted shapes) together
    with the gate's evidence.
    """
    decision = gate_decision(panel, rho=rho, n_stumps=n_stumps)
    composite = DedupComposite(
        rho=rho, block_weights="eq", shapes=decision.adopt_shapes, n_stumps=n_stumps
    )
    return composite.fit(panel), decision
