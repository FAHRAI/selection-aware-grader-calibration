"""Figures 1 and 2 of the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK = "#222"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#6b6b6b",
    "axes.linewidth": 0.8,
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.grid": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
}
MSE_ROWS = [
    ("SW − SN (registered, A)", "registered", "shadow_MSE_SW_minus_SN_matched"),
    ("SW − SN, 6 contexts (post hoc)", "post_hoc", "MSE_SW_minus_SN_J2"),
    ("SW − SN, 60 contexts (post hoc)", "post_hoc", "MSE_SW_minus_SN_J20"),
    ("SW − UW (post hoc)", "post_hoc", "MSE_SW_minus_UW_all"),
    ("SL − SW (post hoc)", "post_hoc", "MSE_SL_minus_SW_all"),
]
LEARNING_ROWS = [
    ("SW − SN (A)", "registered", "learning_SW_minus_SN_matched"),
    ("SW − SN, G (B)", "registered", "learning_SW_minus_SN_G"),
    ("SW − SN, C80 (B)", "registered", "learning_SW_minus_SN_C80"),
    ("SW − SN, C95 (B)", "registered", "learning_SW_minus_SN_C95"),
    ("SW − SF (C)", "registered", "learning_SW_minus_SF_matched"),
    ("SW − UW (C)", "registered", "learning_SW_minus_UW_matched"),
    ("SX − SF (post hoc)", "post_hoc", "learning_SX_minus_SF"),
]


def experiments(path):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.set_axis_off()
    ax.set_xlim(-0.1, 12.3)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f4f6f9", ec="#6b6b6b", lw=0.8
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, color=INK)

    arrow = dict(arrowstyle="-|>", color="#444", lw=0.8)
    ax.text(2.9, 4.7, "Shadow experiment (estimation)", ha="center", fontsize=9.5, color=INK)
    box(-0.05, 2.9, 2.95, 1.3, "NA teaching trajectory\n(fixed driver tape)")
    box(3.15, 2.9, 2.8, 1.3, "Same audited answers\nand propensities $\\pi_i$")
    box(1.0, 0.4, 4.0, 1.5, "Estimators SN, SW, SL update\nerror rates; they never change\nactions: compare final MSE")
    ax.annotate("", xy=(3.15, 3.55), xytext=(2.9, 3.55), arrowprops=arrow)
    ax.annotate("", xy=(3.0, 1.9), xytext=(4.0, 2.9), arrowprops=arrow)
    ax.text(9.2, 4.7, "Closed-loop experiment (learning)", ha="center", fontsize=9.5, color=INK)
    box(6.3, 2.9, 2.75, 1.3, "Controller G, C80\nor C95 selects skill")
    box(9.3, 2.9, 2.85, 1.3, "Grades, targeted\naudit, belief update")
    box(7.4, 0.4, 3.8, 1.5, "Policy's own error-rate\nestimate feeds later decisions:\ncompare delayed success")
    ax.annotate("", xy=(9.3, 3.55), xytext=(9.05, 3.55), arrowprops=arrow)
    ax.annotate("", xy=(10.2, 1.9), xytext=(10.6, 2.9), arrowprops=arrow)
    ax.annotate("", xy=(7.6, 2.9), xytext=(8.2, 1.9), arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.8, ls="--"))
    ax.plot([6.1, 6.1], [0.3, 4.9], color="#ccc", lw=0.8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def contrasts(results, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.9, 5.4), gridspec_kw={"height_ratios": [5, 7]})
    panels = [
        (top, MSE_ROWS, 1e4, "Difference in final MSE (×10$^{-4}$)", "(a) Error-rate estimation (shadow)"),
        (bottom, LEARNING_ROWS, 100, "Difference in delayed success (pp)", "(b) Learning (closed loop)"),
    ]
    for ax, rows, scale, xlabel, title in panels:
        y = list(range(len(rows)))[::-1]
        ax.axvline(0, color="#999", lw=0.8)
        for yi, (_, section, key) in zip(y, rows, strict=True):
            row = results[section][key]
            ax.plot([row["lower"] * scale, row["upper"] * scale], [yi, yi], color=INK, lw=1.6, solid_capstyle="round")
            ax.plot(row["mean"] * scale, yi, "o", ms=4, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=10.5, loc="left")
    for x in (-0.1, 0.1):
        bottom.axvline(x, color="#777", lw=0.9, ls=":")
    bottom.text(
        0.125, 6.45, "dotted: ±0.1 pp margin\n(family A only)", fontsize=8.5, color="#555", ha="left", va="bottom"
    )
    bottom.set_ylim(-0.6, 7.3)
    bottom.set_xlim(-0.12, 0.24)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all(results_dir, figures_dir):
    results = json.loads((Path(results_dir) / "analysis.json").read_text())
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    experiments(figures_dir / "fig1_experiments.png")
    contrasts(results, figures_dir / "fig2_contrasts.png")
