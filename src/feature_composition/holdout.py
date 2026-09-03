"""Scoring frozen signals on a blind hold-out.

The protocol of the study's final test: every parameter (feature signs, weights, tuned
constants, the clustering itself, the shapes, the gate decision, even standardization
moments) is estimated once on the in-sample panel and applied unchanged to the unseen days.
Nothing is fit, tuned or selected on the hold-out; the hold-out ``Panel`` exists only to be
scored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from feature_composition import metrics

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from feature_composition.panel import Panel


def score_signal(panel: Panel, signal: NDArray[np.float64]) -> dict[str, object]:
    """The hold-out scorecard of one signal: IC level, significance, IR, decay and turnover.

    Returns the scalar metrics plus the daily IC series and the decay/autocorrelation curves
    under the keys ``daily_ic``, ``decay`` and ``autocorr``.
    """
    ic = panel.daily_ic(signal)
    clean = ic[~np.isnan(ic)]
    _, t = metrics.newey_west(clean)
    trade = panel.tradeability(signal)
    return {
        "mean_ic": float(clean.mean()),
        "ic_ir_ann": float(clean.mean() / clean.std(ddof=1) * np.sqrt(252)),
        "nw_t": float(t),
        "hit_rate": float((clean > 0).mean()),
        "port_ir_ann": metrics.portfolio_ir(signal, panel.y, panel.day_starts),
        "half_life": trade["half_life"],
        "ret_5d": trade["ret_5d"],
        "turn_l5": trade["turn_l5"],
        "n_days": len(clean),
        "daily_ic": ic,
        "decay": trade["decay"],
        "autocorr": trade["autocorr"],
    }


def rolling_mean_ic(daily_ic: NDArray[np.float64], window: int = 250) -> NDArray[np.float64]:
    """Trailing-window mean of a daily IC series (the level monitor of the deployment playbook).

    NaN until the window fills. The study's class 1 lost two-thirds of its IC in the final
    hold-out year while its correlation structure barely moved: the level, not the structure,
    is what a frozen composite must be monitored on.
    """
    x = np.where(np.isnan(daily_ic), 0.0, daily_ic)
    n = np.where(np.isnan(daily_ic), 0.0, 1.0)
    cs, cn = np.cumsum(x), np.cumsum(n)
    out = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        s = cs[i] - (cs[i - window] if i >= window else 0.0)
        c = cn[i] - (cn[i - window] if i >= window else 0.0)
        out[i] = s / c if c > 0 else np.nan
    return out
