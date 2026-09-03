"""Shared style for the v2 report figures.

One fixed visual system: P052 (Palatino clone, matches the report's mathpazo
body), a validated categorical palette with fixed method->hue assignment,
recessive chrome, direct labels. All figures render as vector PDF.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- ink & chrome
INK      = "#0b0b0b"   # primary text
INK2     = "#52514e"   # secondary text
MUTED    = "#898781"   # axis labels, de-emphasis
GRID     = "#e1e0d9"   # hairline gridlines
BASELINE = "#c3c2b7"   # axis lines
SURFACE  = "#ffffff"

# ------------------------------------------------- method identity (fixed map)
# Categorical slots from the validated reference palette (light mode), assigned
# once and never cycled. naive is deliberately the neutral gray anchor.
METHOD_COLOR = {
    "naive_averaged":    INK2,       # benchmark anchor (neutral)
    "cluster_eq":        "#2a78d6",  # blue      - the recipe
    "cluster_eb_shaped": "#008300",  # green     - shaped arm / ship on c1
    "shaped":            "#008300",
    "cluster_eb":        "#1baf7a",  # aqua
    "ic_weight":         "#4a3aa7",  # violet
    "uniqueness_tuned":  "#eda100",  # yellow  (always direct-labeled)
    "uniqueness_reg":    "#eda100",
    "pls_tuned":         "#eb6834",  # orange
    "pca_pc1":           "#e34948",  # red
    "xgb_tuned":         "#e87ba4",  # magenta (always direct-labeled)
    "ridge_tuned":       "#8a89a5",
    "linreg":            "#b0aeb8",
}
METHOD_LABEL = {
    "naive_averaged":    "benchmark",
    "cluster_eq":        "dedup + equal weights",
    "cluster_eb":        "dedup + adaptive weights",
    "cluster_eb_shaped": "shaped composite",
    "ic_weight":         "IC weighting",
    "uniqueness_tuned":  "uniqueness (tuned)",
    "uniqueness_reg":    "uniqueness regression",
    "pls_tuned":         "PLS (tuned)",
    "pca_pc1":           "PCA PC1",
    "xgb_tuned":         "boosted trees",
    "ridge_tuned":       "ridge",
    "linreg":            "OLS",
}
CLASSES  = [f"feature_class_{i}" for i in range(1, 5)]
CLASS_LB = {c: f"class {i}" for i, c in enumerate(CLASSES, 1)}

# diverging pair for correlation heatmaps (blue <-> red, gray midpoint)
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"


def use_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["P052", "Palatino", "URW Palladio L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "figure.dpi": 110,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def div_cmap():
    """Diverging blue->gray->red for correlations."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "corr_div", [DIV_NEG, "#86b6ef", DIV_MID, "#f0a9a9", DIV_POS])


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    # PNG preview for eyeballing
    fig.savefig(str(path).replace(".pdf", "_preview.png"), dpi=170,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path)
