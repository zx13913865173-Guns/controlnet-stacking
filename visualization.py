"""
Visualization module reproducing Figures 2-6 from the paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from .statistics import SlopeSummary

# Paper-style plotting defaults
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_slope_summary_panel(
    summaries: Dict[str, SlopeSummary],
    output_path: Optional[Path] = None,
):
    """
    Reproduce Figure 3: Metric-specific fixed-A non-normalized B-weight
    retention slope summary.
    Panel A: signed mean per-prompt slope s̄ with 95% CI.
    Panel B: SII = |s̄| across condition pairs (display only, no ranking).
    Panel C: numerical summaries table (MAS, median/IQR, pos/neg counts).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    directions = list(summaries.keys())
    s_bars = [summaries[d].mean_signed_slope for d in directions]
    ci_lowers = [summaries[d].ci_lower for d in directions]
    ci_uppers = [summaries[d].ci_upper for d in directions]
    siis = [summaries[d].sii for d in directions]

    yerr_lower = [s - cl for s, cl in zip(s_bars, ci_lowers)]
    yerr_upper = [cu - s for s, cu in zip(s_bars, ci_uppers)]

    # Panel A: Signed mean slopes with 95% CI
    ax = axes[0]
    colors = ["#d62728" if s < 0 else "#2ca02c" for s in s_bars]
    ax.bar(range(len(directions)), s_bars, color=colors, alpha=0.8,
           yerr=[yerr_lower, yerr_upper], capsize=4)
    ax.set_xticks(range(len(directions)))
    ax.set_xticklabels(directions, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Signed mean slope s̄")
    ax.set_title("(A) Signed mean slope s̄ (95% CI)")
    ax.axhline(0, color="black", linewidth=0.8)

    # Panel B: SII values
    ax = axes[1]
    ax.bar(range(len(directions)), siis, color="#1f77b4", alpha=0.8)
    ax.set_xticks(range(len(directions)))
    ax.set_xticklabels(directions, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("SII = |s̄|")
    ax.set_title("(B) SII across directions (display only)")

    # Panel C: Numerical summary text
    ax = axes[2]
    ax.axis("off")
    header = f"{'Direction':<25} {'s̄':>6} {'SII':>6} {'MAS':>6} {'n+':>3} {'n-':>3}"
    lines = [header, "─" * len(header)]
    for d in directions:
        s = summaries[d]
        lines.append(
            f"{d:<25} {s.mean_signed_slope:>6.2f} {s.sii:>6.2f} "
            f"{s.mas:>6.2f} {s.n_positive:>3} {s.n_negative:>3}"
        )
    ax.text(0, 1, "\n".join(lines), fontfamily="monospace", fontsize=9,
            va="top", ha="left")
    ax.set_title("(C) Numerical summaries")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    return fig


def plot_retention_curves(
    retention_data: Dict[str, Dict[float, np.ndarray]],
    b_weights: List[float],
    output_path: Optional[Path] = None,
):
    """
    Plot retention curves vs wB for each direction (Figure 5 style).
    """
    fig, axes = plt.subplots(1, len(retention_data), figsize=(5 * len(retention_data), 4))
    if len(retention_data) == 1:
        axes = [axes]

    for ax, (label, wb_data) in zip(axes, retention_data.items()):
        means = [np.mean(wb_data[w]) for w in b_weights]
        sems = [np.std(wb_data[w]) / np.sqrt(len(wb_data[w])) for w in b_weights]
        ax.errorbar(b_weights, means, yerr=sems, marker="o", capsize=3, linewidth=2)
        ax.set_xlabel("B weight (w_B)")
        ax.set_ylabel("Retention R_A")
        ax.set_title(label)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    return fig


def plot_endpoint_comparison(
    fixed_a_change: float,
    a_only_change: float,
    matched_diagnostic_change: Optional[float] = None,
    output_path: Optional[Path] = None,
):
    """
    Figure 4 (left): Fixed-A B-weight retention change vs A-only weight effect.
    Two bars are independent; not to be subtracted.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Fixed-A\n(Depth→Canny,\nB: 0.5→0.9)", "A-only\n(Canny: 0.5→0.1)"]
    values = [fixed_a_change, a_only_change]
    if matched_diagnostic_change is not None:
        labels.append("Matched R_A\n(0.5→B)")
        values.append(matched_diagnostic_change)

    colors = ["#ff7f0e", "#1f77b4", "#2ca02c"][:len(values)]
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_ylabel("Change in Canny-IoU retention")
    ax.set_title("Fixed-A vs A-only (independent designs, not subtracted)")
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom" if val > 0 else "top")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    return fig


def plot_progressive_ablation(
    l0: float, l1: float, l2: float, l3: float, l4: float,
    metric_name: str = "Canny-IoU",
    output_path: Optional[Path] = None,
):
    """
    Figure 4 (right): Five-level progressive ablation L0-L4.
    L0: full dual CN (Canny+Depth 5:5)
    L1: Canny only
    L2: Depth only
    L3: both units removed
    L4: both units loaded, all-white conditioning maps
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    levels = ["L0\nFull 5:5", "L1\nCanny only", "L2\nDepth only",
              "L3\nNo CN", "L4\nBlank maps"]
    values = [l0, l1, l2, l3, l4]
    ax.plot(levels, values, marker="o", linewidth=2, markersize=8, color="#d62728")
    ax.set_ylabel(f"{metric_name} compliance")
    ax.set_title("Progressive ablation (L0–L4)")
    ax.set_ylim(0, 1)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    return fig


def plot_preference_results(
    preferred_tool: int = 31,
    total_pairs: int = 40,
    output_path: Optional[Path] = None,
):
    """
    Figure 2 (right): Preference results (31/40 = 77.5% tool-default preferred).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    preferred_manual = total_pairs - preferred_tool
    sizes = [preferred_tool, preferred_manual]
    labels = [f"Tool-default\n({preferred_tool}/{total_pairs} = {preferred_tool/total_pairs*100:.1f}%)",
              f"Manual tuning\n({preferred_manual}/{total_pairs})"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#2ca02c", "#d62728"],
           startangle=90, textprops={"fontsize": 11})
    ax.set_title("Pairwise preference: tool-default vs manual tuning")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    return fig


def plot_slope_distribution(
    per_prompt_slopes: np.ndarray,
    direction_label: str,
    output_path: Optional[Path] = None,
):
    """Histogram of per-prompt slopes for a given direction."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_prompt_slopes, bins=15, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(per_prompt_slopes), color="red", linewidth=2, linestyle="--",
               label=f"Mean = {np.mean(per_prompt_slopes):.3f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Per-prompt OLS slope s_i")
    ax.set_ylabel("Count")
    ax.set_title(f"Slope distribution: {direction_label}")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    return fig
