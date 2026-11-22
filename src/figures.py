"""Render fig:paired from the paired-frame results.

Reads results/paired_view_auc.json (same-view AUC on the frame pairs) and
results/crossview.json (Cohen's kappa between the two views' labels on the same frame
pairs), the sources build_parameters.py reads for tab:crossview, so the figure, the table
and the text cannot drift apart. No value is hand-typed here.

Each point is one modifier: its same-view AUC on the paired frames (x) against the
cross-view kappa of its label on the same frames (y), each with its 95% bootstrap interval
over the clip pairs. The two pure-contact modifiers are left out because they are too rare
on the paired frames for either quantity (sec:crossview).

Output: manuscript/fig_paired.pdf (vector), included by main.tex as fig:paired.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import MANUSCRIPT, RESULTS  # portable repo paths

PAIRED = RESULTS / "paired_view_auc.json"
XVIEW = RESULTS / "crossview.json"
OUT = MANUSCRIPT / "fig_paired.pdf"

# Row labels mirror tab:crossview so a reader moving between the two meets the same names.
DISPLAY = {
    "upperarm_abduction": "Shoulder abduction",
    "trunk_bending": "Trunk lateral bend",
    "upperarm_across": "Shoulder across-body",
    "trunk_twist": "Trunk twist",
    "lowerarm_outbody": "Lower-arm out-of-body",
    "wrist_bending": "Wrist deviation",
    "wrist_twisted": "Wrist twist (any)",
    "wrist_twisted_end": "Wrist twist (end range)",
    "neck_bending": "Neck lateral bend",
    "legs_supported": "Leg support",
}
TOO_RARE = ("trunk_supported", "upperarm_supported")
# Label placement only, set so that no two labels overlap at the figure's printed size; it
# changes no plotted value. ("offset", dx, dy, ha) sets the label beside its point in
# points; ("at", x, y, ha) sets it at data coordinates with a leader line to the point.
LABEL_AT = {
    "legs_supported": ("offset", 7, -3, "left"), "trunk_twist": ("offset", 7, -3, "left"),
    "neck_bending": ("offset", -7, 3, "right"), "wrist_bending": ("offset", -7, 1, "right"),
    "wrist_twisted_end": ("offset", -7, -9, "right"),
    "trunk_bending": ("at", 0.93, 0.50, "right"), "upperarm_abduction": ("at", 0.99, 0.62, "right"),
    "lowerarm_outbody": ("at", 0.80, 0.36, "right"), "wrist_twisted": ("at", 0.99, -0.06, "right"),
    "upperarm_across": ("at", 0.80, 0.27, "right"),
}

# Wong colorblind-safe hues, each with its own marker so the two groups stay separable
# in grayscale.
STYLE = {"orientation": ("#0072B2", "o"), "support": ("#E69F00", "s")}

# Figure is included at \linewidth = 468pt = 6.48in, so it renders unscaled and these
# point sizes are literal at final size (ticks >= 8pt).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
})


def run():
    pa = json.load(open(PAIRED))["per_modifier"]
    xv = json.load(open(XVIEW))["per_modifier"]
    fig, ax = plt.subplots(figsize=(6.48, 3.9))
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=0)
    ax.axvline(0.5, color="0.6", lw=0.8, ls="--", zorder=0)
    seen = set()
    for m, name in DISPLAY.items():
        fam = "orientation" if pa[m]["family"] == "orientation" else "support"
        color, marker = STYLE[fam]
        x, (xlo, xhi) = pa[m]["same_worker_pooled"], pa[m]["same_worker_pooled_ci95"]
        y, (ylo, yhi) = xv[m]["kappa"], xv[m]["kappa_ci95"]
        label = ("Orientation modifier" if fam == "orientation" else "Leg support")
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt=marker, color=color, ms=6, mec="white", mew=0.8, elinewidth=1,
                    capsize=0, label=None if fam in seen else label, zorder=3)
        seen.add(fam)
        kind, a, b, ha = LABEL_AT[m]
        if kind == "offset":
            ax.annotate(name, (x, y), xytext=(a, b), textcoords="offset points", fontsize=8,
                        ha=ha, va="center", color="0.15")
        else:
            ax.annotate(name, (x, y), xytext=(a, b), textcoords="data", fontsize=8, ha=ha,
                        va="center", color="0.15",
                        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6,
                                        shrinkA=2, shrinkB=4))
    ax.set_xlabel("Same-view AUC on the paired frames")
    ax.set_ylabel(r"Cross-view Cohen's $\kappa$")
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlim(0.45, 1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
