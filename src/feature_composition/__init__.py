"""Count each idea once: robust composites from clusters of correlated alpha-signal variants.

The package is the refactored, tested form of the Berkeley MFE industry project for
Ultramarin (2026). Start with :class:`Panel` to load a feature class, :func:`diagnose` for the
playbook diagnostics, :class:`DedupComposite` for the selected composite and
:func:`gate_decision` for the learned-shape gate.
"""

from feature_composition.clustering import RECIPE_RHO, cluster_labels, pc1_sign_align
from feature_composition.composite import DedupComposite
from feature_composition.cv import cross_validate, paired_against, purged_kfold
from feature_composition.diagnostics import ReportCard, diagnose
from feature_composition.gate import GateDecision, fit_selected, gate_decision
from feature_composition.holdout import score_signal
from feature_composition.metrics import daily_rank_ic, newey_west, summarize_ic
from feature_composition.panel import Panel

__version__ = "1.0.0"

__all__ = [
    "RECIPE_RHO",
    "DedupComposite",
    "GateDecision",
    "Panel",
    "ReportCard",
    "__version__",
    "cluster_labels",
    "cross_validate",
    "daily_rank_ic",
    "diagnose",
    "fit_selected",
    "gate_decision",
    "newey_west",
    "paired_against",
    "pc1_sign_align",
    "purged_kfold",
    "score_signal",
    "summarize_ic",
]
