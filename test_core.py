#!/usr/bin/env python
"""
Quick unit test to verify statistical calculations and lookup tool
work correctly without requiring GPU or model downloads.
Run this first to confirm the pipeline logic is sound.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.statistics import (
    ols_slope,
    compute_per_prompt_slopes,
    compute_slope_summary,
    bootstrap_ci,
    compute_a_only_weight_effect,
)
from src.lookup_tool import DefaultLookupTool


def test_ols_slope():
    """Test basic OLS slope calculation."""
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = -0.38 * x + 0.9  # s̄ = -0.38 (Canny→Depth case)
    s = ols_slope(x, y)
    assert abs(s - (-0.38)) < 1e-10, f"Expected -0.38, got {s}"
    print("✓ OLS slope calculation correct")


def test_per_prompt_slopes():
    """Test per-prompt slope computation."""
    n_prompts = 20
    b_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    np.random.seed(42)
    
    # Simulate retention matrix with known slope -0.38 + noise
    x = np.array(b_weights)
    retention = np.zeros((n_prompts, len(b_weights)))
    for i in range(n_prompts):
        true_slope = -0.38 + np.random.normal(0, 0.1)
        retention[i] = true_slope * x + 0.9 + np.random.normal(0, 0.02, len(x))
    
    slopes = compute_per_prompt_slopes(retention, b_weights)
    s_bar = np.mean(slopes)
    assert abs(s_bar - (-0.38)) < 0.1, f"Expected ~-0.38, got {s_bar:.3f}"
    print(f"✓ Per-prompt slopes correct (mean s̄ = {s_bar:.3f}, expected ~-0.38)")


def test_slope_summary():
    """Test full slope summary with bootstrap CI."""
    n_prompts = 20
    b_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    np.random.seed(42)
    
    x = np.array(b_weights)
    retention = np.zeros((n_prompts, len(b_weights)))
    for i in range(n_prompts):
        true_slope = -0.38 + np.random.normal(0, 0.15)
        retention[i] = true_slope * x + 0.9 + np.random.normal(0, 0.03, len(x))
    
    summary = compute_slope_summary(retention, b_weights, "test_direction", n_bootstrap=500)
    
    assert summary.sii == abs(summary.mean_signed_slope), "SII should be |s̄|"
    assert summary.ci_lower < summary.mean_signed_slope < summary.ci_upper, "CI should contain mean"
    assert summary.n_positive + summary.n_negative <= n_prompts, "Counts should not exceed n"
    print(f"✓ Slope summary correct: s̄={summary.mean_signed_slope:.3f}, "
          f"SII={summary.sii:.3f}, MAS={summary.mas:.3f}")
    print(f"  95% CI: [{summary.ci_lower:.3f}, {summary.ci_upper:.3f}]")


def test_lookup_tool():
    """Test cell-qualified heuristic default lookup."""
    tool = DefaultLookupTool()
    
    # Canny+Depth CFG7.5 SDXL should return 38:62
    result = tool.lookup("canny_depth", 7.5, "sdxl")
    assert result.recommendation is not None
    assert abs(result.recommendation["canny"] - 0.38) < 1e-6
    assert abs(result.recommendation["depth"] - 0.62) < 1e-6
    assert result.ratio_evaluated == True
    print("✓ Lookup: canny_depth@CFG7.5/sdxl → canny:0.38, depth:0.62")
    
    # CFG15 should return "no recommendation"
    result15 = tool.lookup("canny_depth", 15.0, "sdxl")
    assert result15.recommendation is None
    assert result15.cell_measured == True
    assert result15.ratio_evaluated == False
    print("✓ Lookup: canny_depth@CFG15/sdxl → no recommendation (measured but no ratio eval)")
    
    # OpenPose+Canny should return "no recommendation"
    result_op = tool.lookup("openpose_canny", 7.5, "sdxl")
    assert result_op.recommendation is None
    print("✓ Lookup: openpose_canny@CFG7.5/sdxl → no recommendation")
    
    # SD1.5 should return "no recommendation" (not measured)
    result_sd15 = tool.lookup("canny_depth", 7.5, "sd15")
    assert result_sd15.recommendation is None
    assert result_sd15.cell_measured == False
    print("✓ Lookup: canny_depth@CFG7.5/sd15 → no recommendation (cell not measured)")


def test_a_only_effect():
    """Test A-only weight effect computation."""
    n_prompts = 20
    a_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    np.random.seed(42)
    
    x = np.array(a_levels)
    retention = np.zeros((n_prompts, len(a_levels)))
    for i in range(n_prompts):
        slope = 0.4 + np.random.normal(0, 0.05)  # positive slope with A weight
        retention[i] = slope * x + 0.3 + np.random.normal(0, 0.02, len(x))
    
    change = compute_a_only_weight_effect(retention, a_levels, from_level=0.5, to_level=0.1)
    print(f"✓ A-only weight effect (0.5→0.1): {change:.3f}")


def test_paper_values():
    """Verify that with paper-expected slopes, summaries match."""
    b_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    np.random.seed(42)
    n = 20
    
    # Canny → Depth on Depth-SSIM: s̄ = -0.38
    x = np.array(b_weights)
    ret = np.zeros((n, 5))
    for i in range(n):
        s = -0.38 + np.random.normal(0, 0.12)
        ret[i] = s * x + 0.85 + np.random.normal(0, 0.02, 5)
    s = compute_slope_summary(ret, b_weights, "canny->depth")
    assert abs(s.mean_signed_slope - (-0.38)) < 0.1
    assert abs(s.sii - 0.38) < 0.1
    print(f"✓ Paper value verification: Canny→Depth s̄={s.mean_signed_slope:.2f}, SII={s.sii:.2f} (expected -0.38/0.38)")

    # Depth → Canny on Canny-IoU: s̄ = -0.23
    ret2 = np.zeros((n, 5))
    for i in range(n):
        s = -0.23 + np.random.normal(0, 0.08)
        ret2[i] = s * x + 0.90 + np.random.normal(0, 0.02, 5)
    s2 = compute_slope_summary(ret2, b_weights, "depth->canny")
    assert abs(s2.mean_signed_slope - (-0.23)) < 0.1
    print(f"✓ Paper value verification: Depth→Canny s̄={s2.mean_signed_slope:.2f}, SII={s2.sii:.2f} (expected -0.23/0.23)")


if __name__ == "__main__":
    print("="*50)
    print("Running core logic tests (no GPU required)")
    print("="*50)
    
    test_ols_slope()
    test_per_prompt_slopes()
    test_slope_summary()
    test_a_only_effect()
    test_lookup_tool()
    test_paper_values()
    
    print("\n" + "="*50)
    print("All tests passed ✓")
    print("="*50)
