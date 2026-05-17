"""
make_figures_v3.py  —  Final clean version, all overlap issues resolved.
Only Fig 1 and Fig 4 needed further adjustment; others carried forward.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

COLORS = {
    "LR":        "#7BAFD4",
    "DL":        "#4472C4",
    "Debiased":  "#ED7D31",
    "FE-HITL":   "#70AD47",
    "threshold": "#FF0000",
    "gray":      "#A0A0A0",
    "layer1":    "#BDD7EE",
    "layer2":    "#FCE4D6",
    "layer3":    "#E2EFDA",
    "layer4":    "#FFF2CC",
    "arrow":     "#404040",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1  –  FIXED: each layer has 3 text rows (title / subtitle / diag),
#           each on its own line so nothing collides.
#           Box height increased to 1.6 to give vertical breathing room.
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # layer data: (y_bottom, height, color, linestyle,
    #              title, subtitle, diag_label)
    layers = [
        (7.1, 1.6, COLORS["layer1"], "solid",
         "Layer 1 · Data & Fairness Monitoring",
         "Real-time DI / EOD computation  ·  Trigger: DI < 0.8 or |EOD| > 0.1",
         "Diagnosis ①  Human obsolescence  →  Forced intervention"),

        (5.1, 1.6, COLORS["layer2"], "solid",
         "Layer 2 · Human Ethical Decision (Simulated)",
         "SHAP / LIME explainability  ·  Multi-option trade-off generation  ·  Value-preference labeling",
         "Diagnosis ②  Promethean shame  →  Multi-option generation"),

        (3.1, 1.6, COLORS["layer3"], "solid",
         "Layer 3 · Bias Correction & Optimization",
         "Constrained optimisation + imitation learning  ·  Adaptive boost / shrink",
         "Diagnosis ③  Technological imperialism  →  Value arbitration"),

        (0.8, 1.9, COLORS["layer4"], "dashed",
         "Layer 4 · Feedback & Update  [architectural placeholder — not validated in current study]",
         "Experience replay buffer  ·  Periodic fine-tuning for continual fairness",
         None),
    ]

    PAD_X = 0.28   # left text margin inside box
    BOX_W = 9.44   # box width (x from 0.28 to 9.72)
    BOX_X = 0.28

    for (yb, h, color, ls, title, sub, diag) in layers:
        box = FancyBboxPatch((BOX_X, yb), BOX_W, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#777777",
                             linewidth=1.5 if ls == "solid" else 1.0,
                             linestyle=ls, zorder=2)
        ax.add_patch(box)

        ytop = yb + h  # inner top of box

        # Row 1: title (bold, near top)
        ax.text(BOX_X + PAD_X, ytop - 0.25,
                title,
                fontsize=9.5, fontweight="bold", va="top", color="#1A1A1A", zorder=3)

        # Row 2: subtitle (italic, middle area)
        ax.text(BOX_X + PAD_X, ytop - 0.72,
                sub,
                fontsize=7.8, va="top", color="#3A3A3A", style="italic", zorder=3)

        # Row 3: diagnosis label (smaller, near bottom, purple)
        if diag:
            ax.text(BOX_X + PAD_X, yb + 0.22,
                    diag,
                    fontsize=7.5, va="bottom", color="#5B5EA6",
                    style="italic", zorder=3)

    # downward arrows between layers
    arrow_kw = dict(arrowstyle="-|>", color=COLORS["arrow"],
                    lw=1.6, mutation_scale=14, zorder=4)
    for (y_from, y_to) in [(7.1, 6.7), (5.1, 4.7), (3.1, 2.7)]:
        ax.annotate("", xy=(5.0, y_to), xytext=(5.0, y_from),
                    arrowprops=dict(**arrow_kw))

    # right-side future feedback arrow (dashed)
    ax.annotate("",
                xy=(9.88, 7.9), xytext=(9.88, 3.1),
                arrowprops=dict(arrowstyle="-|>", color="#AAAAAA",
                                lw=1.1, mutation_scale=11,
                                linestyle="dashed", zorder=4))
    ax.text(9.97, 5.5, "future\nfeedback", fontsize=6.2, color="#AAAAAA",
            ha="left", va="center", rotation=90, zorder=5)

    ax.set_title(
        "Fig. 1  Structure of the Fairness-Enhanced Human-in-the-Loop (FE-HITL) Framework\n"
        "(Layer 4 is an architectural placeholder; not validated in the current study)",
        fontsize=9, pad=8, style="italic")
    fig.tight_layout()
    fig.savefig("fig1_framework.png")
    plt.close(fig)
    print("✔  fig1_framework.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2  –  Carried forward from v2 (clean)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig2():
    models = ["LR", "DL", "Debiased-HITL", "FE-HITL"]
    colors = [COLORS["LR"], COLORS["DL"], COLORS["Debiased"], COLORS["FE-HITL"]]
    data = {
        "DI": {"mean": [0.633, 0.632, 0.679, 0.864], "sd": [0.043, 0.050, 0.046, 0.040]},
        "|EOD|": {"mean": [0.131, 0.130, 0.094, 0.025], "sd": [0.038, 0.043, 0.037, 0.019]},
        "R²": {"mean": [0.742, 0.736, 0.725, 0.702], "sd": [0.012, 0.013, 0.017, 0.024]},
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    x = np.arange(len(models))
    bar_w = 0.55
    for ax, (metric, vals) in zip(axes, data.items()):
        bars = ax.bar(x, vals["mean"], bar_w, color=colors,
                      edgecolor="white", linewidth=0.6,
                      yerr=vals["sd"], capsize=4,
                      error_kw=dict(elinewidth=1.0, ecolor="#404040", capthick=1.0))
        if metric == "DI":
            ax.axhline(0.8, color=COLORS["threshold"], lw=1.2, ls="--", zorder=5)
            ax.text(3.45, 0.81, "DI = 0.8", color=COLORS["threshold"],
                    fontsize=7, va="bottom", ha="right")
        if metric == "|EOD|":
            ax.axhline(0.1, color=COLORS["threshold"], lw=1.2, ls="--", zorder=5)
            ax.text(3.45, 0.102, "|EOD| = 0.1", color=COLORS["threshold"],
                    fontsize=7, va="bottom", ha="right")
        for bar, m, s in zip(bars, vals["mean"], vals["sd"]):
            ax.text(bar.get_x() + bar.get_width()/2, m+s+0.006, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "DL", "Debiased\n-HITL", "FE-HITL"], fontsize=7.5)
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.12) if metric in ("DI","R²") else ax.set_ylim(0, 0.28)
    # bracket on DI
    ax_di = axes[0]
    bkt_y = 1.01
    ax_di.plot([1,1,3,3],[bkt_y-0.015,bkt_y,bkt_y,bkt_y-0.015],
               color="#404040",lw=1.0,zorder=6)
    ax_di.text(2.0, bkt_y+0.008, "p < 0.001,  dz = 3.20",
               ha="center", va="bottom", fontsize=7.0, color="#404040")
    fig.suptitle(
        "Fig. 2  Fairness and predictive performance across models — agricultural dataset\n"
        "(n = 30 seeds, bias = 30%);  error bars = SD;  red dashed line = fairness threshold",
        fontsize=8.5, y=1.03, style="italic")
    fig.tight_layout()
    fig.savefig("fig2_agricultural_main.png")
    plt.close(fig)
    print("✔  fig2_agricultural_main.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3  –  Carried forward from v2 (clean)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    variants = ["Full\nFE-HITL", "w/o HED\n(−Forced\nIntervention)",
                "w/o MOG\n(−Multi-Option\nGen.)", "w/o F&U\n(placeholder;\nnot validated)"]
    di_mean  = [0.864, 0.639, 0.704, 0.864]
    di_sd    = [0.040, 0.058, 0.049, 0.040]
    r2_mean  = [0.702, 0.725, 0.724, 0.702]
    r2_sd    = [0.024, 0.018, 0.017, 0.024]
    bar_colors = [COLORS["FE-HITL"], "#C00000", COLORS["Debiased"], COLORS["gray"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5),
                             gridspec_kw={"wspace": 0.35})
    x = np.arange(len(variants))
    bar_w = 0.55
    from matplotlib.patches import Patch
    for panel_idx, (ax, means, sds, ylabel, title) in enumerate(zip(
        axes,
        [di_mean, r2_mean], [di_sd, r2_sd],
        ["DI", "R²"],
        ["Disparate Impact (DI)", "Coefficient of Determination (R²)"]
    )):
        for i,(m,s,c) in enumerate(zip(means,sds,bar_colors)):
            hatch = "///" if i==3 else None
            ax.bar(i, m, bar_w, color=c, edgecolor="white", linewidth=0.6,
                   hatch=hatch, alpha=0.85 if i==3 else 1.0,
                   yerr=s, capsize=4,
                   error_kw=dict(elinewidth=1.0,ecolor="#404040",capthick=1.0))
            ax.text(i, m+s+0.004, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
        if ylabel == "DI":
            ax.axhline(0.8, color=COLORS["threshold"],lw=1.2,ls="--",zorder=5)
            ax.text(3.42,0.802,"DI = 0.8",color=COLORS["threshold"],
                    fontsize=7,va="bottom",ha="right")
            ax.set_ylim(0,1.12)
            ax.text(0.98,0.99,
                    "ANOVA: F(3,116)=175.8, p<0.001, η²=0.82\nHED & MOG sig.  |  F&U: not validated (static)",
                    transform=ax.transAxes,
                    fontsize=6.8,ha="right",va="top",color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="#FFFDE7",
                              edgecolor="#CCCCCC",alpha=0.9))
        else:
            ax.set_ylim(0.0, 0.86)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
    hatch_patch = Patch(facecolor=COLORS["gray"], hatch="///", edgecolor="#555555",
                        label="w/o F&U (unvalidated placeholder)", alpha=0.85)
    fig.legend(handles=[hatch_patch], loc="lower center", ncol=1,
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5,-0.08))
    fig.suptitle(
        "Fig. 3  Ablation study: contribution of individual compensation mechanisms\n"
        "(n = 30 seeds; hatched bar = F&U, unvalidated placeholder in static setting)",
        fontsize=8.5, y=1.04, style="italic")
    fig.tight_layout()
    fig.savefig("fig3_ablation.png")
    plt.close(fig)
    print("✔  fig3_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4  –  FIXED: stat box moved to top-right (axes coords), completely clear
#           of bars and the "DI>1" box which stays top-left.
# ══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    models = ["LR", "DL", "Debiased-HITL", "FE-HITL"]
    colors = [COLORS["LR"], COLORS["DL"], COLORS["Debiased"], COLORS["FE-HITL"]]
    data = {
        # Updated to 50-seed results from code2
        "DI":       {"mean":[1.494,1.351,1.384,1.046], "sd":[0.338,0.543,0.600,0.345]},
        "|EOD|":    {"mean":[0.131,0.088,0.071,0.061], "sd":[0.090,0.079,0.066,0.046]},
        "Accuracy": {"mean":[0.746,0.733,0.733,0.731], "sd":[0.022,0.023,0.022,0.022]},
    }
    # Taller figure: extra headroom above DI bars for two separated annotation boxes
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.6))
    x = np.arange(len(models))
    bar_w = 0.55
    for ax, (metric, vals) in zip(axes, data.items()):
        bars = ax.bar(x, vals["mean"], bar_w, color=colors,
                      edgecolor="white", linewidth=0.6,
                      yerr=vals["sd"], capsize=4,
                      error_kw=dict(elinewidth=1.0, ecolor="#404040", capthick=1.0))
        for bar, m, s in zip(bars, vals["mean"], vals["sd"]):
            offset = 0.06 if metric == "DI" else (0.007 if metric == "|EOD|" else 0.001)
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + offset,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "DL", "Debiased\n-HITL", "FE-HITL"], fontsize=7.5)
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight="bold")

        if metric == "DI":
            # ylim extended to 4.4 — bars top out at ~2.0, boxes sit at y≈3.5–4.0
            ax.set_ylim(0, 4.4)
            ax.axhline(1.0, color=COLORS["threshold"], lw=1.2, ls="--", zorder=5)
            ax.text(3.45, 1.02, "DI=1.0 (parity)", color=COLORS["threshold"],
                    fontsize=6.5, va="bottom", ha="right")
            ax.axhline(0.8, color="#FF8000", lw=0.9, ls=":", zorder=5)
            ax.text(3.45, 0.82, "DI=0.8", color="#FF8000",
                    fontsize=6.5, va="bottom", ha="right")

            # LEFT box: DI>1 explanation — sits above bars, left half of axis
            ax.text(0.78, 3.95,
                    "DI > 1: over-prediction of\nnegative outcome for females",
                    ha="center", va="center", fontsize=6.5, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8DC",
                              edgecolor="#CCCCCC", alpha=0.95), zorder=6)

            # RIGHT box: stat annotation — sits above bars, right half of axis
            ax.text(2.62, 3.95,
                    "FE-HITL vs DL:\nt(49)=−4.521, p=0.0002***\ndz=−0.639",
                    ha="center", va="center", fontsize=6.5, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD",
                              edgecolor="#AAAAAA", alpha=0.95), zorder=6)

        elif metric == "|EOD|":
            ax.set_ylim(0, 0.34)
            ax.axhline(0.1, color=COLORS["threshold"], lw=1.2, ls="--", zorder=5)
            ax.text(3.45, 0.102, "|EOD|=0.1", color=COLORS["threshold"],
                    fontsize=7, va="bottom", ha="right")

        else:  # Accuracy — updated to reflect FE vs Debiased p=0.032
            ax.set_ylim(0.68, 0.82)
            ax.text(0.5, -0.24,
                    "FE-HITL vs DL: n.s. (p = 0.177)\n"
                    "FE-HITL vs Debiased-HITL: p = 0.032 (Δ < 0.003, negligible)",
                    transform=ax.transAxes,
                    fontsize=6.8, ha="center", va="top",
                    color="#555555", style="italic")

    fig.suptitle(
        "Fig. 4  Fairness and classification performance on the UCI German Credit dataset\n"
        "(n = 50 seeds);  error bars = SD;  DI > 1 = over-prediction of negative outcome for females",
        fontsize=8.5, y=1.02, style="italic")
    fig.tight_layout()
    fig.savefig("fig4_german_credit.png")
    plt.close(fig)
    print("✔  fig4_german_credit.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5  –  Carried forward from v2 (clean)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig5():
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.6, 11.5)
    ax.axis("off")

    def step_box(cx, cy, w, h, color, title, sub, ls="solid"):
        box = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#666666",
                             linewidth=1.4 if ls=="solid" else 1.0,
                             linestyle=ls, zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy+0.18, title,
                ha="center", va="center", fontsize=9, fontweight="bold",
                zorder=4, color="#1A1A1A")
        ax.text(cx, cy-0.22, sub,
                ha="center", va="center", fontsize=7.5, style="italic",
                zorder=4, color="#3A3A3A", multialignment="center")

    def down_arrow(cx, y_from, y_to):
        ax.annotate("", xy=(cx, y_to), xytext=(cx, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["arrow"],
                                    lw=1.4, mutation_scale=14, zorder=5))

    step_cx = 7.2
    box_w   = 8.8
    box_h   = 1.1
    step_ys = [10.4, 8.85, 7.3, 5.75, 4.2, 2.55]
    step_colors = [COLORS["layer1"], COLORS["layer1"],
                   COLORS["layer2"], COLORS["layer2"],
                   COLORS["layer3"], COLORS["layer4"]]
    step_ls = ["solid","solid","solid","solid","solid","dashed"]
    steps = [
        ("Step 1 · Original Model Prediction",
         "DL model: 28 units (42 below average) for farmer in region A"),
        ("Step 2 · Fairness Alert Triggered",
         "DI = 0.48 < threshold 0.8  →  system triggers human intervention"),
        ("Step 3 · Explainability Presentation",
         "SHAP: low historical output → contribution −35% to unfairness"),
        ("Step 4 · Multi-Option Generation",
         "A: 35 units (DI=0.72, ΔR²=−2%)  |  B: 40 units (DI=0.80, ΔR²=−5%)  |  C: 45 units (DI=0.85, ΔR²=−8%)"),
        ("Step 5 · Human Ethical Decision (Simulated)",
         "Expert selects Scheme C: 45 units;  label: 'regional balance + disadvantaged priority'"),
        ("Step 6 · Model Update  [F&U placeholder]",
         "Human decision fed back;  bias reduced for similar region-A farmers"),
    ]
    for (title, sub), y, color, ls in zip(steps, step_ys, step_colors, step_ls):
        step_box(step_cx, y, box_w, box_h, color, title, sub, ls)
    for i in range(len(step_ys)-1):
        down_arrow(step_cx, step_ys[i]-box_h/2-0.02,
                   step_ys[i+1]+box_h/2+0.02)

    sidebar_defs = [
        (9.625, 1.65, COLORS["layer1"], "Data &\nFairness\nMonitoring"),
        (6.525, 1.65, COLORS["layer2"], "Human\nEthical\nDecision"),
        (4.20,  1.20, COLORS["layer3"], "Bias\nCorrection"),
        (2.55,  1.20, COLORS["layer4"], "F&U\n(placeholder)"),
    ]
    for (ym,h,c,lbl) in sidebar_defs:
        rect = FancyBboxPatch((0.1, ym-h/2), 1.8, h,
                               boxstyle="round,pad=0.06",
                               facecolor=c, edgecolor="#BBBBBB",
                               linewidth=0.8, alpha=0.7, zorder=1)
        ax.add_patch(rect)
        ax.text(1.0, ym, lbl, ha="center", va="center",
                fontsize=7, color="#333333", zorder=2, multialignment="center")

    ax.text(step_cx, 1.82,
            "⚠  Step 6 (F&U) is an architectural placeholder — not validated in current study",
            ha="center", va="top", fontsize=7.5,
            color="#888888", style="italic", zorder=6)

    ax.set_title(
        "Fig. 5  Operational workflow of the FE-HITL framework — agricultural case\n"
        "(Farmer: region A, 2.3 ha, 5 yrs;  DL initial: 28 units  →  corrected: 45 units)",
        fontsize=9, pad=8, style="italic")
    fig.tight_layout()
    fig.savefig("fig5_workflow.png")
    plt.close(fig)
    print("✔  fig5_workflow.png")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("\nAll 5 figures saved (300 dpi PNG).")