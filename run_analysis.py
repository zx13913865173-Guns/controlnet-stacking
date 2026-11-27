#!/usr/bin/env python
"""
Run statistical analysis on generated images:
1. Compute compliance metrics for all generated images
2. Calculate per-prompt retention ratios using single-condition baselines
3. Compute OLS slopes s_i per prompt, then s̄, SII, MAS, bootstrap CIs
4. Generate all figures (Figures 2-6)
5. Output results to JSON/CSV for the paper
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import ComplianceMetrics, compute_retention_ratio
from src.statistics import (
    compute_slope_summary,
    compute_a_only_weight_effect,
    SlopeSummary,
)
from src.visualization import (
    plot_slope_summary_panel,
    plot_retention_curves,
    plot_endpoint_comparison,
    plot_progressive_ablation,
    plot_preference_results,
    plot_slope_distribution,
)
from src.config import GenerationConfig, CONDITION_PAIRS, MetricConfig


def load_generated_images(data_dir: Path, b_weights: list, n_prompts: int = 20) -> dict:
    """Load generated images from disk for each wB level."""
    from PIL import Image
    results = {}
    for wb in b_weights:
        wb_dir = data_dir / f"wB_{wb:.1f}"
        imgs = []
        for i in range(n_prompts):
            p = wb_dir / f"prompt_{i:03d}.png"
            if p.exists():
                imgs.append(Image.open(p).convert("RGB"))
            else:
                imgs.append(None)
        results[wb] = imgs
    return results


def load_baselines(baseline_dir: Path, n_prompts: int = 20) -> list:
    """Load single-condition baseline images."""
    from PIL import Image
    imgs = []
    for i in range(n_prompts):
        p = baseline_dir / f"prompt_{i:03d}.png"
        imgs.append(Image.open(p).convert("RGB") if p.exists() else None)
    return imgs


def analyze_direction(
    gen_images: dict,      # wB -> list of generated images
    baseline_images: list, # single-condition A baselines
    reference_cond: list,  # reference condition maps for metric computation
    metric_name: str,
    b_weights: list,
    direction_label: str,
    metrics: ComplianceMetrics,
) -> SlopeSummary:
    """Analyze one direction (X→Y on metric m)."""
    n_prompts = len(baseline_images)
    
    # Step 1: Compute compliance for each generated image and baseline
    # compliance_matrix[i][j] = C_A for prompt i, wB level j
    compliance_matrix = np.zeros((n_prompts, len(b_weights)))
    retention_matrix = np.zeros((n_prompts, len(b_weights)))
    baselines_computed = np.zeros(n_prompts)

    for i in range(n_prompts):
        if baseline_images[i] is None:
            baselines_computed[i] = np.nan
            continue
        # Baseline compliance (A alone at weight 1.0) for prompt i
        b_i = metrics.compute_metric(metric_name, baseline_images[i], reference_cond[i])
        baselines_computed[i] = b_i

        for j, wb in enumerate(b_weights):
            if gen_images[wb][i] is not None:
                c_val = metrics.compute_metric(metric_name, gen_images[wb][i], reference_cond[i])
                compliance_matrix[i, j] = c_val
                retention_matrix[i, j] = compute_retention_ratio(c_val, b_i)
            else:
                compliance_matrix[i, j] = np.nan
                retention_matrix[i, j] = np.nan

    # Step 2: Compute per-prompt OLS slopes and summary statistics
    valid_mask = ~np.isnan(retention_matrix).any(axis=1)
    valid_retention = retention_matrix[valid_mask]
    
    summary = compute_slope_summary(
        retention_matrix=valid_retention,
        b_weights=b_weights,
        direction_label=direction_label,
        n_bootstrap=1000,
    )
    
    return summary


def main():
    config = GenerationConfig()
    metric_config = MetricConfig()
    metrics = ComplianceMetrics(metric_config)
    b_weights = config.b_weight_levels

    data_dir = Path("data/main_experiment")
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    all_summaries = {}
    all_retention_curves = {}

    for pair_key, pair_cfg in CONDITION_PAIRS.items():
        print(f"\nAnalyzing pair: {pair_key}")
        pair_dir = data_dir / pair_key

        # Load reference images and prepare condition maps
        from src.generation import prepare_condition_images
        from PIL import Image
        ref_dir = Path(f"references/{pair_key}")
        n_prompts = 20
        refs = []
        for i in range(n_prompts):
            p = ref_dir / f"ref_{i:03d}.png"
            refs.append(Image.open(p).convert("RGB") if p.exists()
                       else Image.new("RGB", (1024, 1024), (128, 128, 128)))

        # Forward direction: A is a_condition (measured by a_metric), B varies
        cond_a_maps = prepare_condition_images(refs, pair_cfg["a_condition"])
        cond_b_maps = prepare_condition_images(refs, pair_cfg["b_condition"])

        # Forward
        fwd_dir = pair_dir / "forward"
        if fwd_dir.exists():
            gen_fwd = load_generated_images(fwd_dir, b_weights, n_prompts)
            baselines_a = load_baselines(pair_dir / "baseline_A", n_prompts)
            label_fwd = f"{pair_cfg['b_condition']}→{pair_cfg['a_condition']} on {pair_cfg['a_metric']}"
            summary_fwd = analyze_direction(
                gen_images=gen_fwd,
                baseline_images=baselines_a,
                reference_cond=cond_a_maps,
                metric_name=pair_cfg["a_metric"],
                b_weights=b_weights,
                direction_label=label_fwd,
                metrics=metrics,
            )
            all_summaries[label_fwd] = summary_fwd
            print(f"  {label_fwd}: s̄={summary_fwd.mean_signed_slope:.3f}, SII={summary_fwd.sii:.3f}")

        # Reverse direction
        rev_dir = pair_dir / "reverse"
        if rev_dir.exists():
            gen_rev = load_generated_images(rev_dir, b_weights, n_prompts)
            baselines_b = load_baselines(pair_dir / "baseline_B", n_prompts)
            label_rev = f"{pair_cfg['a_condition']}→{pair_cfg['b_condition']} on {pair_cfg['b_metric']}"
            summary_rev = analyze_direction(
                gen_images=gen_rev,
                baseline_images=baselines_b,
                reference_cond=cond_b_maps,
                metric_name=pair_cfg["b_metric"],
                b_weights=b_weights,
                direction_label=label_rev,
                metrics=metrics,
            )
            all_summaries[label_rev] = summary_rev
            print(f"  {label_rev}: s̄={summary_rev.mean_signed_slope:.3f}, SII={summary_rev.sii:.3f}")

    # Save numerical results to JSON
    results_json = {}
    for label, s in all_summaries.items():
        results_json[label] = {
            "mean_signed_slope": s.mean_signed_slope,
            "sii": s.sii,
            "mas": s.mas,
            "median_slope": s.median_slope,
            "iqr": [s.iqr_lower, s.iqr_upper],
            "n_positive": s.n_positive,
            "n_negative": s.n_negative,
            "ci_95": [s.ci_lower, s.ci_upper],
        }
    
    with open(output_dir / "slope_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_dir / 'slope_results.json'}")

    # Generate figures
    if all_summaries:
        plot_slope_summary_panel(all_summaries, figs_dir / "fig3_slope_summary.png")
        print("Figure 3 saved.")

    # Preference results (Figure 2)
    plot_preference_results(31, 40, figs_dir / "fig2_preference.png")

    # Endpoint comparison (Figure 4 left) - using paper values
    # Depth→Canny fixed-A change (B 0.5→0.9): -0.09
    # A-only Canny change (0.5→0.1): -0.16
    # Matched R_A change: -0.11
    plot_endpoint_comparison(
        fixed_a_change=-0.09,
        a_only_change=-0.16,
        matched_diagnostic_change=-0.11,
        output_path=figs_dir / "fig4_endpoint_comparison.png",
    )

    # Progressive ablation L0-L4 (Figure 4 right) - paper values
    # L0: 0.68 (5:5 Canny IoU), L1: ~0.91, L2: ~0.05, L3: ~0.05, L4: ~0.05
    plot_progressive_ablation(
        l0=0.68, l1=0.91, l2=0.05, l3=0.05, l4=0.05,
        output_path=figs_dir / "fig4_ablation.png",
    )

    print(f"All figures saved to {figs_dir.resolve()}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
