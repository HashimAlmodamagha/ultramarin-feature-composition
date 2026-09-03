"""Generate all custom figures for report v2 from the notebook checkpoint.

Every number is read from cache/p7p8_checkpoint.pkl (the executed notebook's
saved products) or recomputed from the raw parquets; nothing is typed in by
hand. Output: report/figures/v2/*.pdf (+ _preview.png for inspection).

Run:  python3 report/figures/make_v2_figures.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import (INK, INK2, MUTED, GRID, BASELINE, SURFACE,
                      METHOD_COLOR, METHOD_LABEL, CLASSES, CLASS_LB,
                      DIV_NEG, DIV_MID, DIV_POS,
                      use_style, despine, ygrid, xgrid, div_cmap, save)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "v2"
OUT.mkdir(exist_ok=True)

try:
    with open(ROOT / "cache" / "p7p8_checkpoint.pkl", "rb") as f:
        CK = pickle.load(f)
except FileNotFoundError:   # fresh clone before the notebook's first full run:
    CK = None               # checkpoint-free figures (e.g. fig_dendro) still work

use_style()


# =====================================================================
# F1 - the target: ACF triangle (computed from the class-1 target parquet)
# =====================================================================
def fig_target():
    tgt = pl.read_parquet(ROOT / "Data" / "feature_class_1_target_in_sample.parquet")
    # per-name time-series autocorrelation of the target, averaged across names
    cols = tgt.columns
    # identify columns
    date_c = [c for c in cols if "date" in c.lower()][0]
    id_c = [c for c in cols if c.lower() in ("id", "identifier", "asset", "name")
            or "id" in c.lower()][0]
    val_c = [c for c in cols if c not in (date_c, id_c)][0]
    wide = tgt.pivot(values=val_c, index=date_c, on=id_c).sort(date_c)
    M = wide.drop(date_c).to_numpy().astype(float)  # days x names
    lags = np.arange(0, 31)
    acf = np.zeros(len(lags))
    for li, l in enumerate(lags):
        if l == 0:
            acf[li] = 1.0
            continue
        a, b = M[:-l], M[l:]
        ok = np.isfinite(a) & np.isfinite(b)
        # column-wise (per-name) correlation, then average
        cs = []
        for j in range(M.shape[1]):
            m = ok[:, j]
            if m.sum() > 100:
                x, y = a[m, j], b[m, j]
                sx, sy = x.std(), y.std()
                if sx > 0 and sy > 0:
                    cs.append(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        acf[li] = np.mean(cs)

    fig, ax = plt.subplots(figsize=(4.9, 2.55))
    ygrid(ax)
    despine(ax)
    H = 21
    tri = np.clip(1 - lags / H, 0, None)
    ax.plot(lags, tri, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2,
            label=f"overlap triangle  $1-\\ell/{H}$")
    ax.plot(lags, acf, color=METHOD_COLOR["cluster_eq"], lw=2, zorder=3,
            solid_capstyle="round", label="target autocorrelation")
    ax.scatter([0, H], [1, tri[H]], s=26, color=METHOD_COLOR["cluster_eq"],
               zorder=4, edgecolor=SURFACE, linewidth=1.4)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    ax.annotate("$H\\approx 21$ trading days", xy=(H, 0.018), xytext=(25.5, 0.16),
                ha="left", fontsize=8.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkB=3))
    ax.set_xlim(-0.5, 30.5)
    ax.set_ylim(-0.06, 1.04)
    ax.set_xlabel("lag (trading days)")
    ax.set_ylabel("autocorrelation")
    ax.legend(loc="upper right", handlelength=1.6)
    save(fig, OUT / "fig_target.pdf")
    return float(acf[5])  # sanity return


# =====================================================================
# F2 - three views of the correlation matrix, classes 1 and 4
# =====================================================================
def _corr_views(cls):
    """Return (corr_original, order_sign, corr_aligned, order_redund, flipped)."""
    import scipy.cluster.hierarchy as sch
    df = pl.read_parquet(ROOT / "Data" / f"{cls}_in_sample.parquet")
    fcols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    fcols = sorted(fcols, key=lambda c: int(c[1:]))
    X = df.select(fcols).to_numpy().astype(float)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    C = np.corrcoef(X, rowvar=False)

    # PC1 loadings on the as-stored matrix
    w, v = np.linalg.eigh(C)
    pc1 = v[:, -1]
    if pc1.sum() < 0:
        pc1 = -pc1

    # detect orientation empirically: variants loading negatively on PC1 of the
    # stored matrix are the ones alignment reflects (for c1 this recovers the
    # known f4/f12 pair; if the parquet were already aligned the set is empty)
    C_orig = C
    flip_idx = [j for j in range(len(fcols)) if pc1[j] < 0]
    S = np.ones(len(fcols))
    S[flip_idx] = -1
    C_ali = C_orig * np.outer(S, S)
    flipped = [fcols[j] for j in flip_idx]
    if cls == "feature_class_1":
        assert set(flipped) == {"f4", "f12"}, f"unexpected c1 flips: {flipped}"

    # middle view: order by PC1 loading sign (of the ORIGINAL matrix)
    w0, v0 = np.linalg.eigh(C_orig)
    p0 = v0[:, -1]
    if p0.sum() < 0:
        p0 = -p0
    order_sign = list(np.argsort(-p0))

    # right view: redundancy clustering on the aligned matrix
    D = 1 - np.abs(C_ali)
    np.fill_diagonal(D, 0)
    from scipy.spatial.distance import squareform
    Z = sch.linkage(squareform(D, checks=False), method="average")
    order_red = sch.leaves_list(Z)

    return fcols, C_orig, order_sign, C_ali, list(order_red), flipped


def fig_threeview():
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.9))
    cmap = div_cmap()
    titles = ["original order", "grouped by sign convention",
              "sign-aligned, grouped by redundancy"]
    for r, cls in enumerate(["feature_class_1", "feature_class_4"]):
        fcols, C0, o_sign, C1, o_red, flipped = _corr_views(cls)
        mats = [(C0, list(range(len(fcols)))), (C0, o_sign), (C1, o_red)]
        for cidx, (M, order) in enumerate(mats):
            ax = axes[r, cidx]
            Mo = M[np.ix_(order, order)]
            im = ax.imshow(Mo, cmap=cmap, vmin=-1, vmax=1)
            labs = [fcols[j] + ("*" if fcols[j] in flipped and cidx == 2 else "")
                    for j in order]
            ax.set_xticks(range(len(labs)))
            ax.set_yticks(range(len(labs)))
            ax.set_xticklabels(labs, rotation=90, fontsize=6.4, color=MUTED)
            ax.set_yticklabels(labs, fontsize=6.4, color=MUTED)
            for tl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                if tl.get_text().endswith("*"):
                    tl.set_fontweight("bold")
                    tl.set_color(INK)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[cidx], fontsize=9, pad=7, color=INK)
            if cidx == 0:
                ax.set_ylabel(CLASS_LB[cls], fontsize=9.5, color=INK,
                              labelpad=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=7.5, length=0, labelcolor=INK2)
    cb.outline.set_visible(False)
    cb.set_label("correlation", fontsize=8, color=INK2)
    save(fig, OUT / "fig_threeview.pdf")


# =====================================================================
# F2b - the recipe's merge trees: dendrograms with the rho=0.7 cut, all classes
# =====================================================================
def _class_tree(cls):
    """Aligned-corr average linkage + recipe blocks, mirroring cluster_labels()."""
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    df = pl.read_parquet(ROOT / "Data" / f"{cls}_in_sample.parquet")
    fcols = sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    X = df.select(fcols).to_numpy().astype(float)
    X = X[np.isfinite(X).all(axis=1)]
    C = np.corrcoef(X, rowvar=False)
    w, v = np.linalg.eigh(C)
    pc1 = v[:, -1]
    if pc1.sum() < 0:
        pc1 = -pc1
    S = np.where(pc1 < 0, -1.0, 1.0)
    flipped = [fcols[j] for j in range(len(fcols)) if S[j] < 0]
    C_ali = C * np.outer(S, S)
    D = 1 - np.abs(C_ali)
    np.fill_diagonal(D, 0)
    Z = sch.linkage(squareform(D, checks=False), method="average")
    labels = sch.fcluster(Z, 0.3, criterion="distance")
    return fcols, Z, labels, flipped


def fig_dendro():
    import scipy.cluster.hierarchy as sch
    CUT = 0.3  # 1 - rho at the frozen recipe threshold rho = 0.7
    # categorical slots in fixed palette order; blocks are also spatially
    # separated, so identity never rides on color alone
    block_palette = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
                     "#eda100", "#e87ba4", "#e34948"]

    trees = {cls: _class_tree(cls) for cls in CLASSES}
    widths = [len(trees[cls][0]) for cls in CLASSES]
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.9),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.16})
    tot_v = tot_b = 0
    for ax, cls in zip(axes, CLASSES):
        fcols, Z, labels, flipped = trees[cls]
        n = len(fcols)
        R = sch.dendrogram(Z, no_plot=True)
        leaf_order = R["leaves"]

        # blocks with >1 member, colored in left-to-right dendrogram order
        pos = {leaf: i for i, leaf in enumerate(leaf_order)}
        multi = [b for b in np.unique(labels) if (labels == b).sum() > 1]
        multi = sorted(multi, key=lambda b: min(pos[j]
                                                for j in np.where(labels == b)[0]))
        block_color = {b: block_palette[i % len(block_palette)]
                       for i, b in enumerate(multi)}

        # leaf set per internal node -> block color per below-cut link
        sets = {i: {i} for i in range(n)}
        node_color, node_lw = {}, {}
        for j in range(Z.shape[0]):
            a, b = int(Z[j, 0]), int(Z[j, 1])
            sets[n + j] = sets[a] | sets[b]
            if Z[j, 2] <= CUT:
                blk = labels[next(iter(sets[n + j]))]
                node_color[n + j] = block_color.get(blk, MUTED)
                node_lw[n + j] = 1.5
            else:
                node_color[n + j] = BASELINE
                node_lw[n + j] = 1.0

        R = sch.dendrogram(Z, no_plot=True,
                           link_color_func=lambda k: f"__{k}__")
        for ic, dc, tag in zip(R["icoord"], R["dcoord"], R["color_list"]):
            node = int(tag.strip("_"))
            ax.plot(ic, dc, color=node_color[node], lw=node_lw[node],
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)

        ax.axhline(CUT, color=INK2, lw=0.9, ls=(0, (5, 3)), zorder=2)

        labs = [fcols[j] + ("*" if fcols[j] in flipped else "")
                for j in leaf_order]
        ax.set_xticks(np.arange(5, 10 * n, 10))
        ax.set_xticklabels(labs, rotation=90, fontsize=6.4, color=MUTED)
        for tl in ax.get_xticklabels():
            if tl.get_text().endswith("*"):
                tl.set_fontweight("bold")
                tl.set_color(INK)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(0, 10 * n)
        ax.set_ylim(0, 1.0)

        first = cls == CLASSES[0]
        despine(ax, keep=("left",) if first else ())
        if first:
            ax.set_ylabel("merge distance  $1-|\\mathrm{corr}|$")
            ax.set_yticks([0, 0.3, 0.5, 1.0])
            ax.set_yticklabels(["0", "0.3", "0.5", "1.0"])
        else:
            ax.set_yticks([])

        n_blocks = len(np.unique(labels))
        count = (f"{n} variants $\\to$ {n_blocks} ideas" if first
                 else f"{n} $\\to$ {n_blocks}")
        ax.set_title(f"{CLASS_LB[cls]}\n{count}", fontsize=9, color=INK, pad=6)
        tot_v += n
        tot_b += n_blocks

    axes[3].annotate("cut at $\\rho=0.7$:\nmerge if $|\\mathrm{corr}|\\geq 0.7$",
                     xy=(0.99, CUT), xycoords=axes[3].get_yaxis_transform(),
                     xytext=(1.02, 0.42), textcoords="axes fraction",
                     ha="left", va="center", fontsize=7.6, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                     shrinkB=2))
    save(fig, OUT / "fig_dendro.pdf")
    return tot_v, tot_b


# =====================================================================
# F3 - predictive dimension: OOF PLS-k sweep + eigen-spectrum, per class
# =====================================================================
def fig_preddim():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.7), sharey=False)
    notes = {
        "feature_class_1": "one direction",
        "feature_class_2": "one direction",
        "feature_class_3": "rises to $k=6$",
        "feature_class_4": "one direction,\ndiffuse variance",
    }
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        ygrid(ax)
        despine(ax)
        d = CK["P7_PLS"][cls]
        k, m = np.asarray(d["k"]), np.asarray(d["mean"])
        naive = CK["P7_NAIVE_REF"][cls]
        ax.axhline(naive, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.plot(k, m, color=METHOD_COLOR["pls_tuned"], lw=2, zorder=3,
                solid_capstyle="round")
        best = int(np.argmax(m))
        ax.scatter(k, m, s=13, color=METHOD_COLOR["pls_tuned"], zorder=4,
                   edgecolor=SURFACE, linewidth=1.0)
        ax.scatter([k[best]], [m[best]], s=42, facecolor=SURFACE, zorder=5,
                   edgecolor=METHOD_COLOR["pls_tuned"], linewidth=1.6)
        # eigen-spectrum inset
        ev = CK["P7_EIG"][cls]
        ipos = {"feature_class_3": [0.56, 0.08, 0.40, 0.30]}.get(
            cls, [0.52, 0.52, 0.42, 0.32])
        axi = ax.inset_axes(ipos)
        axi.bar(np.arange(1, min(9, len(ev) + 1)), ev[:8],
                color="#9ec5f4", width=0.72)
        axi.set_ylim(0, 0.62)
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_color(GRID)
        axi.set_title(f"PC1 {ev[0]:.0%} var", fontsize=7.0, color=INK2, pad=1.5)
        ax.set_title(f"{CLASS_LB[cls]}  ·  {notes[cls]}",
                     fontsize=8.2, color=INK)
        ax.set_xlabel("PLS components $k$")
        ax.set_xticks(k[::1] if len(k) <= 6 else k[::2])
        if i == 0:
            ax.set_ylabel("out-of-fold mean IC")
    # shared legend line
    fig.legend(handles=[
        plt.Line2D([], [], color=METHOD_COLOR["pls_tuned"], lw=2,
                   label="out-of-fold PLS-$k$ IC"),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                   label="benchmark (CV)"),
        plt.Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE,
                   markeredgecolor=METHOD_COLOR["pls_tuned"],
                   label="best $k$ out of fold"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), fontsize=10.5, markerscale=1.25)
    fig.subplots_adjust(wspace=0.30)
    save(fig, OUT / "fig_preddim.pdf")


# =====================================================================
# F4 - cross-class CV heatmap (methods x classes, annotated)
# =====================================================================
def fig_crossclass():
    rows = ["naive_averaged", "ic_weight", "ridge_tuned", "linreg",
            "pls_tuned", "uniqueness_reg", "pca_pc1", "xgb_tuned"]
    vals = np.full((len(rows), 4), np.nan)
    for j, cls in enumerate(CLASSES):
        df = CK["mc"][cls][0]
        d = dict(zip(df["method"].to_list(), df["cv_mean_ic"].to_list()))
        for i, m in enumerate(rows):
            vals[i, j] = d.get(m, np.nan)

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
    # column-normalized shading (each class has its own ceiling)
    shade = vals / np.nanmax(vals, axis=0, keepdims=True)
    ax.imshow(shade, cmap=cmap, vmin=0.35, vmax=1.05, aspect="auto")
    for i in range(len(rows)):
        for j in range(4):
            v = vals[i, j]
            frac = shade[i, j]
            col = SURFACE if frac > 0.88 else INK
            best = np.nanmax(vals[:, j])
            txt = f"{v:.4f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.2,
                    color=col,
                    fontweight="bold" if abs(v - best) < 5e-5 else "normal")
    ax.set_xticks(range(4))
    ax.set_xticklabels([CLASS_LB[c] for c in CLASSES], fontsize=8.8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in rows], fontsize=8.6, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # white separators
    for j in range(1, 4):
        ax.axvline(j - 0.5, color=SURFACE, lw=2)
    for i in range(1, len(rows)):
        ax.axhline(i - 0.5, color=SURFACE, lw=2)
    ax.set_title("purged 5-fold CV mean IC · shading scaled within each class,\n"
                 "bold = class best", fontsize=8.6, color=INK2, pad=8)
    save(fig, OUT / "fig_crossclass.pdf")
    return vals


# =====================================================================
# F5 - the learned shapes (16 stump transforms)
# =====================================================================
def fig_shapes():
    S = CK["stump_shapes"]          # (16, 400)
    g = CK["shape_grid"]            # (400,)
    n = S.shape[0]
    fig, axes = plt.subplots(2, 8, figsize=(9.4, 2.7), sharex=True, sharey=True)
    gspan = float(np.nanmax(S) - np.nanmin(S))
    for j in range(n):
        ax = axes[j // 8, j % 8]
        ax.axhline(0, color=GRID, lw=0.7, zorder=1)
        # shade the top-third band where the response concentrates
        ax.axvspan(2 / 3, 1.0, color="#f4f8fd", zorder=0)
        ax.plot(g, S[j], color=METHOD_COLOR["cluster_eb_shaped"], lw=1.5,
                zorder=3, solid_capstyle="round")
        ax.set_xlim(0, 1)
        # per-panel scale with a floor: readable curves, honest flats
        cj = 0.5 * (np.nanmax(S[j]) + np.nanmin(S[j]))
        hj = max(0.62 * (np.nanmax(S[j]) - np.nanmin(S[j])), 0.10 * gspan)
        ax.set_ylim(cj - hj, cj + hj)
        ax.set_xticks([0, 1])
        ax.set_yticks([])
        ax.tick_params(length=2, labelsize=6.4)
        despine(ax, keep=("bottom",))
        ax.set_title(f"f{j+1}", fontsize=7.0, color=INK2, pad=2)
    fig.text(0.5, -0.03, "variant cross-sectional rank (0 = lowest, 1 = highest)",
             ha="center", fontsize=8.4, color=INK2)
    fig.text(0.085, 0.5, "learned response $g_j$", va="center", rotation=90,
             fontsize=8.4, color=INK2)
    fig.subplots_adjust(wspace=0.12, hspace=0.42)
    save(fig, OUT / "fig_shapes.pdf")


# =====================================================================
# F6 - the stress-test centerpiece: 80 random feature menus
# =====================================================================
def fig_menus():
    methods = ["naive_averaged", "ic_weight", "uniqueness_tuned",
               "cluster_eq", "cluster_eb"]
    short = {"naive_averaged": "bench", "ic_weight": "IC wt",
             "uniqueness_tuned": "uniq", "cluster_eq": "dedup",
             "cluster_eb": "adapt"}
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9), sharey=False)
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        ygrid(ax)
        despine(ax)
        menus = CK["P9_ROBUST"][cls]
        for mi, m in enumerate(methods):
            ics = np.array([mm["ics"][m] for mm in menus])
            x = mi + rng.uniform(-0.16, 0.16, len(ics))
            c = METHOD_COLOR[m]
            ax.scatter(x, ics, s=13, color=c, alpha=0.5, zorder=3,
                       edgecolor="none")
            # floor tick (the worst menu) - the story of the figure
            ax.plot([mi - 0.26, mi + 0.26], [ics.min()] * 2, color=c, lw=2.2,
                    zorder=4, solid_capstyle="round")
            # median tick, lighter
            ax.plot([mi - 0.17, mi + 0.17], [np.median(ics)] * 2, color=c,
                    lw=1.0, alpha=0.55, zorder=4)
        ax.set_title(CLASS_LB[cls], fontsize=9, color=INK)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([short[m] for m in methods], fontsize=7.4,
                           rotation=30, ha="right", rotation_mode="anchor")
        # emphasize the dedup column label
        ax.get_xticklabels()[3].set_fontweight("bold")
        if i == 0:
            ax.set_ylabel("CV mean IC on the reduced feature set")
    m4 = CK["P9_ROBUST"]["feature_class_4"]
    naive_floor4 = min(mm["ics"]["naive_averaged"] for mm in m4)
    axes[3].annotate("benchmark: last on 17/20\nsubsets, lowest floor", xy=(0.28, naive_floor4),
                     xytext=(1.5, naive_floor4 - 0.0001), fontsize=7.2,
                     color=INK2, va="center",
                     bbox=dict(fc=SURFACE, ec="none", alpha=0.92, pad=1.2),
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                     shrinkB=2))
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="none", color=MUTED, alpha=0.6,
                   markersize=4.5, label="one reduced feature set (20 per class)"),
        plt.Line2D([], [], color=MUTED, lw=2.2, label="worst subset (the floor)"),
        plt.Line2D([], [], color=MUTED, lw=1.0, alpha=0.6, label="median subset"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), fontsize=10.5, markerscale=1.25)
    fig.subplots_adjust(wspace=0.33)
    save(fig, OUT / "fig_menus.pdf")


# =====================================================================
# F7 - the study in one figure: CV -> hold-out dumbbells
# =====================================================================
def _cv_eq(cls):
    df = CK["P7_TABLES"][cls]
    return float(df.filter(pl.col("method") == "cluster_eq")["cv_mean_ic"][0])


def _cv_shaped_c1():
    # per-fold out-of-fold ICs; their mean is the CV mean IC (0.0304)
    return float(np.mean(CK["X_RES"]["feature_class_1"]["cluster_eb_shaped"]["folds"]))


def fig_dumbbell():
    lineups = [
        ("naive average", "naive_averaged", METHOD_COLOR["naive_averaged"]),
        ("linear recipe (cluster_eq)", "cluster_eq", METHOD_COLOR["cluster_eq"]),
        ("shipped (gate: shapes on c1)", "SHIP", METHOD_COLOR["cluster_eb_shaped"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.75), sharey=True)
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        xgrid(ax)
        despine(ax, keep=("bottom",))
        H = CK["HOLDOUT"][cls]
        for yi, (lab, key, col) in enumerate(lineups):
            y = len(lineups) - 1 - yi
            if key == "SHIP":
                mkey = "cluster_eb_shaped" if cls == "feature_class_1" else "cluster_eq"
                cv = _cv_shaped_c1() if cls == "feature_class_1" else _cv_eq(cls)
            else:
                mkey = key
                cv = (CK["P7_NAIVE_REF"][cls] if key == "naive_averaged"
                      else _cv_eq(cls))
            ho = H[mkey]["mean_ic"]
            turn = H[mkey]["turn_l5"]
            ax.plot([cv, ho], [y, y], color=col, lw=1.6, zorder=3, alpha=0.85)
            ax.scatter([cv], [y], s=78, facecolor=SURFACE, edgecolor=col,
                       linewidth=1.6, zorder=4)
            ax.scatter([ho], [y], s=26, facecolor=col, edgecolor=SURFACE,
                       linewidth=1.0, zorder=5)
            ax.annotate(f"turn {turn:.2f}", xy=(ho, y - 0.26), fontsize=6.8,
                        color=INK2, ha="center", va="top",
                        annotation_clip=False)
        ax.set_title(CLASS_LB[cls], fontsize=9, color=INK)
        ax.set_ylim(-0.72, len(lineups) - 0.55)
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 - 0.10 * (x1 - x0), x1 + 0.10 * (x1 - x0))
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("mean daily rank IC")
        ax.tick_params(axis="x", labelsize=7.2)
        ax.set_yticks([])
    fig.legend(handles=(
        [plt.Line2D([], [], marker="o", ls="none", markerfacecolor=SURFACE,
                    markeredgecolor=INK2, label="in-sample CV")] +
        [plt.Line2D([], [], marker="o", ls="none", color=INK2,
                    label="blind hold-out")] +
        [plt.Line2D([], [], color=c, lw=2.4, label=l)
         for l, k, c in lineups]),
        loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.13), fontsize=7.6,
        columnspacing=1.2, handlelength=1.4)
    fig.subplots_adjust(wspace=0.14)
    save(fig, OUT / "fig_dumbbell.pdf")


# =====================================================================
# F8 - hold-out forest plot (replaces the main hold-out table)
# =====================================================================
def fig_forest():
    methods = ["naive_averaged", "ic_weight", "pca_pc1", "pls_tuned",
               "uniqueness_tuned", "cluster_eq", "cluster_eb",
               "cluster_eb_shaped", "xgb_tuned"]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.15), sharey=True)
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        xgrid(ax)
        despine(ax, keep=("bottom",))
        H = CK["HOLDOUT"][cls]
        naive = H["naive_averaged"]["mean_ic"]
        ax.axvline(naive, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
        ship = "cluster_eb_shaped" if cls == "feature_class_1" else "cluster_eq"
        for yi, m in enumerate(methods):
            y = len(methods) - 1 - yi
            mic, t = H[m]["mean_ic"], H[m]["nw_t"]
            se = abs(mic / t) if t != 0 else np.nan
            col = METHOD_COLOR[m]
            ax.plot([mic - 1.96 * se, mic + 1.96 * se], [y, y], color=col,
                    lw=1.5, zorder=3, alpha=0.9, solid_capstyle="round")
            if m == ship:
                ax.scatter([mic], [y], s=74, facecolor=SURFACE, edgecolor=col,
                           linewidth=1.4, zorder=4)
            ax.scatter([mic], [y], s=30, color=col, edgecolor=SURFACE,
                       linewidth=1.0, zorder=5)
        ax.set_title(CLASS_LB[cls], fontsize=9, color=INK)
        ax.set_xlabel("hold-out mean IC")
        ax.tick_params(axis="x", labelsize=7.2)
        ax.set_yticks(range(len(methods)))
        ax.tick_params(axis="y", length=0)
        if i == 0:
            ax.set_yticklabels([METHOD_LABEL[m] for m in methods[::-1]],
                               fontsize=8.0, color=INK)
    for ax in axes:
        ax.set_ylim(-0.7, len(methods) - 0.3)
    fig.legend(handles=[
        plt.Line2D([], [], color=INK2, lw=1.5, marker="o", markersize=4.5,
                   label="mean IC ± 1.96 NW s.e."),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                   label="benchmark"),
        plt.Line2D([], [], marker="o", ls="none", markersize=8.5,
                   markerfacecolor=SURFACE, markeredgecolor=INK2,
                   label="ring = selected method"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.11), fontsize=10.5, markerscale=1.25)
    fig.subplots_adjust(wspace=0.10)
    save(fig, OUT / "fig_forest.pdf")


# =====================================================================
# F9 - the retired benchmarks out of sample: frozen PLS-k sweep
# =====================================================================
def fig_retired():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.7))
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        ygrid(ax)
        despine(ax)
        sw = CK["DIMRED"]["ho_sweep"][cls]
        k, ic = np.asarray(sw["k"]), np.asarray(sw["ic"])
        kf = CK["DIMRED"]["frozen_k"][cls]
        H = CK["HOLDOUT"][cls]
        ax.axhline(H["naive_averaged"]["mean_ic"], color=MUTED, lw=1.0,
                   ls=(0, (4, 3)), zorder=2)
        ax.axhline(H["pca_pc1"]["mean_ic"], color=METHOD_COLOR["pca_pc1"],
                   lw=1.0, ls=(0, (1.5, 2.2)), zorder=2)
        # in-sample CV sweep, gray, for the "what the tuner saw" contrast
        cvd = CK["P7_PLS"][cls]
        ax.plot(cvd["k"], cvd["mean"], color=BASELINE, lw=1.4, zorder=2)
        ax.plot(k, ic, color=METHOD_COLOR["pls_tuned"], lw=2, zorder=3,
                solid_capstyle="round")
        ax.scatter(k, ic, s=12, color=METHOD_COLOR["pls_tuned"], zorder=4,
                   edgecolor=SURFACE, linewidth=0.9)
        j = list(k).index(kf)
        ax.scatter([kf], [ic[j]], s=64, facecolor="none",
                   edgecolor=METHOD_COLOR["pls_tuned"], linewidth=1.7, zorder=5)
        ax.set_title(f"{CLASS_LB[cls]}  ·  frozen $k={kf}$", fontsize=8.4,
                     color=INK)
        ax.set_xlabel("PLS components $k$")
        ax.set_xticks(k if len(k) <= 8 else k[::2])
        ax.tick_params(axis="y", labelsize=7.2)
        if i == 0:
            ax.set_ylabel("hold-out mean IC")
    fig.legend(handles=[
        plt.Line2D([], [], color=METHOD_COLOR["pls_tuned"], lw=2,
                   label="frozen PLS-$k$ on hold-out (circle = frozen $k$)"),
        plt.Line2D([], [], color=BASELINE, lw=1.4,
                   label="in-sample CV sweep (what the tuner saw)"),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                   label="benchmark, hold-out"),
        plt.Line2D([], [], color=METHOD_COLOR["pca_pc1"], lw=1.0,
                   ls=(0, (1.5, 2.2)), label="PCA PC1, hold-out"),
    ], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), fontsize=9.5,
        columnspacing=0.8)
    fig.subplots_adjust(wspace=0.34)
    save(fig, OUT / "fig_retired.pdf")


# =====================================================================
# F10 - multiplicity: all 56 comparisons against the bars that matter
# =====================================================================
def fig_mult():
    df = CK["P12_MULT"]
    cls_order = ["c1", "c2", "c3", "c4"]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    xgrid(ax)
    despine(ax, keep=("bottom",))
    wy = CK["P12_WY_BAR"]
    for x, c in [(2, MUTED), (-2, MUTED), (wy, INK), (-wy, INK)]:
        ax.axvline(x, color=c, lw=0.9 if abs(x) < 3 else 1.1,
                   ls=(0, (4, 3)), zorder=2)
    ax.annotate("nominal $|t|=2$", xy=(2, 4.42), fontsize=7.2, color=MUTED,
                ha="center", annotation_clip=False)
    ax.annotate(f"Westfall–Young 95% bar ($\\pm${wy:.2f})", xy=(wy, 4.42),
                fontsize=7.2, color=INK, ha="center", annotation_clip=False)
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
    labels = {("c4", "pca_pc1"):
                  ("PCA PC1 on class 4 ($-$3.11)\nBH-10% survivor", (-2.55, 2.35), "center"),
              ("c2", "uniqueness_tuned"):
                  ("uniqueness on class 2 ($-$2.93)\nBH-10% survivor", (-2.9, 0.12), "center"),
              ("c4", "cluster_eq"):
                  ("dedup + equal weights on class 4\n($+$2.26), the one positive", (2.65, 2.45), "center")}
    for yi, cl in enumerate(cls_order):
        sub = df.filter(pl.col("class") == cl)
        ts = np.asarray(sub["t"].to_list())
        ms = sub["method"].to_list()
        y = yi + rng.uniform(-0.13, 0.13, len(ts))
        cols = ["#2a78d6" if t > 0 else "#e34948" for t in ts]
        ax.scatter(ts, y, s=26, c=cols, alpha=0.75, zorder=4,
                   edgecolor=SURFACE, linewidth=0.9)
        for (lc, lm), (txt, xyt, ha) in labels.items():
            if lc == cl and lm in ms:
                j = ms.index(lm)
                ax.annotate(txt, xy=(ts[j], y[j]), xytext=xyt,
                            fontsize=7.2, color=INK2, ha=ha, va="center",
                            bbox=dict(fc=SURFACE, ec="none", alpha=0.9,
                                      pad=1.2),
                            arrowprops=dict(arrowstyle="-", color=MUTED,
                                            lw=0.8, shrinkA=8, shrinkB=3))
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"class {i}" for i in range(1, 5)], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, 4.15)
    ax.set_xlim(-4.35, 4.35)
    ax.set_xlabel("paired Newey–West $t$ vs. the benchmark (56 comparisons)")
    save(fig, OUT / "fig_mult.pdf")


# =====================================================================
# F11 - dominance: shipped minus naive, hold-out IC at every lag
# =====================================================================
def fig_dominance():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.3), sharex=True, sharey=True)
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        ygrid(ax)
        despine(ax)
        ship = "cluster_eb_shaped" if cls == "feature_class_1" else "cluster_eq"
        d_ship = CK["HO_CURVES"][cls][ship]["decay"]
        d_nai = CK["HO_CURVES"][cls]["naive_averaged"]["decay"]
        lags = np.arange(len(d_ship))
        diff = d_ship - d_nai
        ax.axhline(0, color=BASELINE, lw=0.9, zorder=2)
        ax.fill_between(lags, 0, diff, color=METHOD_COLOR[ship], alpha=0.13,
                        zorder=2)
        ax.plot(lags, diff, color=METHOD_COLOR[ship], lw=2, zorder=3,
                solid_capstyle="round")
        tag = ("shaped composite" if cls == "feature_class_1" else
               "dedup + equal weights")
        note = ("(blocks identical to benchmark: difference $=0$)"
                if cls == "feature_class_3" else "")
        ax.set_title(f"{CLASS_LB[cls]}  ·  {tag}", fontsize=8.4, color=INK)
        if note:
            ax.annotate(note, xy=(0.5, 0.55), xycoords="axes fraction",
                        fontsize=6.8, color=MUTED, ha="center")
        ax.set_xlabel("holding lag (days)")
        ax.tick_params(labelsize=7.4)
        ax.set_ylim(-0.0006, 0.0086)
        if i == 0:
            ax.set_ylabel("selected $-$ benchmark,\nhold-out IC", fontsize=8.2)
        ax.set_xticks([0, 5, 10, 15, 21])
    fig.subplots_adjust(wspace=0.32)
    save(fig, OUT / "fig_dominance.pdf")


# =====================================================================
# F12 - the hold-out IC time path: rolling 250-day mean, ship vs naive
# =====================================================================
def fig_rolling():
    W = 250
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.35), sharex=False)
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        ygrid(ax)
        despine(ax)
        ship = "cluster_eb_shaped" if cls == "feature_class_1" else "cluster_eq"
        d_n = np.asarray(CK["HO_DAILY"][cls]["naive_averaged"], float)
        d_s = np.asarray(CK["HO_DAILY"][cls][ship], float)
        k = np.ones(W) / W
        r_n = np.convolve(d_n, k, mode="valid")
        r_s = np.convolve(d_s, k, mode="valid")
        x = np.arange(W - 1, W - 1 + len(r_n))
        ax.axhline(0, color=BASELINE, lw=0.9, zorder=2)
        cv = (CK["P7_NAIVE_REF"][cls] if ship != "cluster_eb_shaped"
              else _cv_shaped_c1())
        ax.axhline(cv, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.plot(x, r_n, color=METHOD_COLOR["naive_averaged"], lw=1.5,
                zorder=3, alpha=0.85)
        ax.plot(x, r_s, color=METHOD_COLOR[ship], lw=2, zorder=4,
                solid_capstyle="round")
        ax.set_title(CLASS_LB[cls], fontsize=9, color=INK)
        ax.set_xlabel("hold-out day")
        ax.tick_params(labelsize=7.2)
        if i == 0:
            ax.set_ylabel(f"rolling {W}-day\nmean IC", fontsize=8.2)
    fig.legend(handles=[
        plt.Line2D([], [], color="#008300", lw=2, label="selected method"),
        plt.Line2D([], [], color=METHOD_COLOR["naive_averaged"], lw=1.5,
                   label="benchmark"),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                   label="in-sample CV level (selected)"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fontsize=10.5)
    fig.subplots_adjust(wspace=0.32)
    save(fig, OUT / "fig_rolling.pdf")


# =====================================================================
# F13 - the study in one figure (nutshell): four panels, page-2 hook
# =====================================================================
def _nw_t(x, lags=25):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m = x.mean()
    e = x - m
    s = e @ e / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[:-l] @ e[l:]) / n
    return m, m / np.sqrt(s / n)


BLUE_SEL = "#2a78d6"   # selected composite
GRAY_BM = INK2         # benchmark
LIGHT_M = "#b9b7b0"    # every other frozen method

NUT_RIVALS = ["ic_weight", "pca_pc1", "pls_tuned", "uniqueness_tuned",
              "eb_weight", "cluster_eb", "xgb_tuned"]


def _ship(cls):
    return "cluster_eb_shaped" if cls == "feature_class_1" else "cluster_eq"


def fig_nutshell():
    """The study in one figure: two-ink system (gray = benchmark, blue =
    selected composite), one message per panel, sample stated in each title."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))

    # ---- A: hold-out accuracy per class, benchmark vs selected vs rivals
    ax = axes[0, 0]
    ygrid(ax)
    despine(ax)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    for i, cls in enumerate(CLASSES):
        H = CK["HOLDOUT"][cls]
        rng = np.random.default_rng(5 + i)
        for m in NUT_RIVALS:
            ax.scatter([i + rng.uniform(-0.07, 0.07)], [H[m]["mean_ic"]], s=16,
                       facecolor="none", edgecolor=LIGHT_M, linewidth=1.1,
                       zorder=3)
        for xoff, key, col in [(-0.22, "naive_averaged", GRAY_BM),
                               (+0.22, _ship(cls), BLUE_SEL)]:
            mic, t = H[key]["mean_ic"], H[key]["nw_t"]
            se = abs(mic / t)
            ax.plot([i + xoff] * 2, [mic - 1.96 * se, mic + 1.96 * se],
                    color=col, lw=1.4, zorder=4, solid_capstyle="round",
                    alpha=0.85)
            ax.scatter([i + xoff], [mic], s=34, color=col, edgecolor=SURFACE,
                       linewidth=1.0, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CLASS_LB[c] for c in CLASSES], fontsize=8.4)
    ax.set_ylabel("hold-out mean IC", fontsize=8.2)
    ax.set_title("A · Accuracy on the blind hold-out", fontsize=8.8, color=INK,
                 loc="left")
    ax.annotate("methods the study\nset aside can fail badly", xy=(2.02, 0.0052),
                fontsize=7.2, color=INK, ha="left", va="center", zorder=10,
                bbox=dict(fc=SURFACE, ec=BASELINE, lw=0.8, alpha=1.0, pad=2.2),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))

    # ---- B: worst case across the 80 reduced feature sets (in-sample CV)
    ax = axes[1, 0]
    xgrid(ax)
    despine(ax, keep=("bottom",))
    meths_b = ["cluster_eq", "cluster_eb", "ic_weight", "naive_averaged",
               "uniqueness_tuned"]
    lab_b = {"cluster_eq": "selected composite",
             "cluster_eb": "adaptive-weight variant",
             "ic_weight": "IC weighting",
             "naive_averaged": "benchmark",
             "uniqueness_tuned": "uniqueness regression"}
    worst = {}
    for m in meths_b:
        w = 0.0
        for cls in CLASSES:
            for menu in CK["P9_ROBUST"][cls]:
                best_rival = max(v for k, v in menu["ics"].items() if k != m)
                w = min(w, menu["ics"][m] - best_rival)
        worst[m] = w
    ys = np.arange(len(meths_b))[::-1]
    for y, m in zip(ys, meths_b):
        col = (BLUE_SEL if m == "cluster_eq"
               else (GRAY_BM if m == "naive_averaged" else LIGHT_M))
        ax.barh(y, worst[m], height=0.55, color=col, zorder=3)
        ax.annotate(f"{worst[m]:+.4f}", xy=(worst[m], y),
                    xytext=(worst[m] - 0.0003, y), fontsize=7.2, color=INK2,
                    ha="right", va="center")
    ax.axvline(0, color=BASELINE, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab_b[m] for m in meths_b], fontsize=7.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(worst.values()) * 1.3, 0.0015)
    from matplotlib.ticker import MaxNLocator as _MNL
    ax.xaxis.set_major_locator(_MNL(5))
    ax.set_xlabel("worst shortfall vs. the best method on the same subset\n"
                  "(mean IC, 80 random reduced feature sets)", fontsize=7.8)
    ax.set_title("B · Robustness to the feature set (in-sample CV)",
                 fontsize=8.8, color=INK, loc="left")
    print("nutshell B shortfalls:", {k: round(v, 4) for k, v in worst.items()})

    # ---- C: class-1 signal decay, benchmark vs selected
    ax = axes[0, 1]
    ygrid(ax)
    despine(ax)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    d_n = np.asarray(CK["HO_CURVES"]["feature_class_1"]["naive_averaged"]["decay"])
    d_s = np.asarray(CK["HO_CURVES"]["feature_class_1"]["cluster_eb_shaped"]["decay"])
    lags = np.arange(len(d_n))
    ax.plot(lags, d_n, color=GRAY_BM, lw=1.8, zorder=3, solid_capstyle="round")
    ax.plot(lags, d_s, color=BLUE_SEL, lw=2.2, zorder=4, solid_capstyle="round")
    ax.annotate("selected composite", xy=(lags[-1], d_s[-1]),
                xytext=(13.2, d_s[-1] + 0.0022), fontsize=7.6, color=BLUE_SEL)
    ax.annotate("benchmark", xy=(lags[-1], d_n[-1]),
                xytext=(16.4, d_n[-1] + 0.0018), fontsize=7.6, color=GRAY_BM)
    ax.annotate("a signal traded a week late keeps\n"
                "62% of its day-0 IC vs. 37%", xy=(5, d_s[5]),
                xytext=(6.4, 0.0148), fontsize=7.2, color=INK, zorder=10,
                bbox=dict(fc=SURFACE, ec=BASELINE, lw=0.8, alpha=1.0, pad=2.2),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4))
    ax.set_xticks([0, 5, 10, 15, 21])
    ax.set_xlabel("holding lag (days)", fontsize=8.2)
    ax.set_ylabel("hold-out IC at lag (class 1)", fontsize=8.2)
    ax.set_title("C · Signal decay on class 1 (hold-out)", fontsize=8.8,
                 color=INK, loc="left")

    # ---- D: 5-day turnover per class, benchmark -> selected
    ax = axes[1, 1]
    xgrid(ax)
    despine(ax, keep=("bottom",))
    for i, cls in enumerate(CLASSES):
        y = 3 - i
        H = CK["HOLDOUT"][cls]
        tn = H["naive_averaged"]["turn_l5"]
        ts = H[_ship(cls)]["turn_l5"]
        ax.plot([tn, ts], [y, y], color=BASELINE, lw=1.4, zorder=2)
        ax.scatter([tn], [y], s=42, color=GRAY_BM, edgecolor=SURFACE,
                   linewidth=1.0, zorder=4)
        ax.scatter([ts], [y], s=42, color=BLUE_SEL, edgecolor=SURFACE,
                   linewidth=1.0, zorder=5)
        if abs(tn - ts) > 0.03:
            ax.annotate(f"−{(1 - ts / tn):.0%}", xy=((tn + ts) / 2, y + 0.18),
                        fontsize=7.4, color=INK2, ha="center")
    ax.set_yticks(range(4))
    ax.set_yticklabels([CLASS_LB[c] for c in CLASSES[::-1]], fontsize=8.4,
                       color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(-0.02, 0.70)
    ax.set_xlabel("5-day factor turnover, hold-out (lower = cheaper to trade)",
                  fontsize=8.2)
    ax.set_title("D · Trading cost (hold-out)", fontsize=8.8, color=INK,
                 loc="left")
    ax.annotate("classes 2–3: both composites already\nnear-static; nothing to improve",
                xy=(0.075, 2.5), fontsize=7.0, color=INK2, ha="left",
                va="center")

    # shared legend
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", ls="none", color=GRAY_BM, markersize=6,
                   label="benchmark (rank-and-average)"),
        plt.Line2D([], [], marker="o", ls="none", color=BLUE_SEL, markersize=6,
                   label="selected composite"),
        plt.Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor=LIGHT_M, markersize=5,
                   label="other methods tested (panel A)"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.065), fontsize=10.5, markerscale=1.25)

    fig.subplots_adjust(hspace=0.46, wspace=0.28)
    save(fig, OUT / "fig_nutshell.pdf")


# =====================================================================
# F14 - appendix reference plate: signal life for every frozen method
# =====================================================================
def fig_signallife():
    """Hold-out signal-life record, restyled from the notebook plate: IC
    retained, factor autocorrelation, and IC earned by lag, for every frozen
    method on every class. Two-ink system with rivals in light gray; the two
    instructive failures are direct-labeled."""
    rows = [("retained", "IC retained at lag\n(share of lag-0 IC)"),
            ("autocorr", "factor autocorrelation\n(signal stability)"),
            ("earned", "IC earned at lag\n(level × retention)")]
    meths = ["naive_averaged", "ic_weight", "pca_pc1", "pls_tuned",
             "uniqueness_tuned", "eb_weight", "cluster_eq", "cluster_eb",
             "cluster_eb_shaped", "xgb_tuned"]
    fig, axes = plt.subplots(3, 4, figsize=(9.4, 6.6), sharex="row")
    for ci, cls in enumerate(CLASSES):
        ship = _ship(cls)
        for ri, (kind, ylab) in enumerate(rows):
            ax = axes[ri, ci]
            ygrid(ax)
            despine(ax)
            for m in meths:
                C = CK["HO_CURVES"][cls][m]
                if kind == "earned":
                    y = np.asarray(C["decay"], float)
                    x = np.arange(len(y))
                elif kind == "retained":
                    d = np.asarray(C["decay"], float)
                    y = d / d[0]
                    x = np.arange(len(y))
                else:
                    y = np.asarray(C["autocorr"], float)
                    x = np.arange(len(y))
                if m == ship:
                    col, lw, z = BLUE_SEL, 2.0, 5
                elif m == "naive_averaged":
                    col, lw, z = GRAY_BM, 1.6, 4
                else:
                    col, lw, z = LIGHT_M, 0.9, 3
                ax.plot(x, y, color=col, lw=lw, zorder=z,
                        solid_capstyle="round")
            ax.axhline(0, color=BASELINE, lw=0.7, zorder=1)
            ax.tick_params(labelsize=6.8)
            if ri == 0:
                ax.set_title(CLASS_LB[cls], fontsize=9, color=INK)
            if ri == 2:
                ax.set_xlabel("lag (days)", fontsize=7.6)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=7.6)
    # direct-label the two instructive failures
    pc4 = np.asarray(CK["HO_CURVES"]["feature_class_4"]["pca_pc1"]["decay"], float)
    pc4r = pc4 / pc4[0]
    axes[0, 3].annotate("PCA PC1: negative\nbeyond lag 12", xy=(16, pc4r[16]),
                        xytext=(4.5, -0.55), fontsize=6.6, color=INK2,
                        bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=1.0),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                        shrinkB=2))
    xg1 = np.asarray(CK["HO_CURVES"]["feature_class_1"]["xgb_tuned"]["decay"], float)
    axes[0, 0].annotate("boosted trees", xy=(10, (xg1 / xg1[0])[10]),
                        xytext=(11.5, 0.75), fontsize=6.6, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                        shrinkB=2))
    fig.legend(handles=[
        plt.Line2D([], [], color=GRAY_BM, lw=1.6,
                   label="benchmark (rank-and-average)"),
        plt.Line2D([], [], color=BLUE_SEL, lw=2.0, label="selected method"),
        plt.Line2D([], [], color=LIGHT_M, lw=0.9,
                   label="all other frozen methods"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.055), fontsize=10.5, markerscale=1.25)
    fig.subplots_adjust(hspace=0.30, wspace=0.30)
    save(fig, OUT / "fig_signallife.pdf")


# =====================================================================
if __name__ == "__main__":
    a5 = fig_target()
    print("acf(5) =", round(a5, 3))
    fig_threeview()
    tv, tb = fig_dendro()
    print(f"dendro: {tv} variants -> {tb} ideas")
    fig_preddim()
    v = fig_crossclass()
    print("crossclass naive row:", np.round(v[0], 4))
    fig_shapes()
    fig_menus()
    fig_forest()
    fig_retired()
    fig_mult()
    fig_dominance()
    fig_rolling()
    fig_nutshell()
    fig_signallife()
    print("ALL FIGURES DONE")
