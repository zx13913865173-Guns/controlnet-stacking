"""
Statistical analysis: per-prompt OLS slopes, signed mean slope s̄,
SII (|s̄|), MAS (mean absolute slope), bootstrap confidence intervals.
Implements the exact definitions from Section 3.4.1 of the paper.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import pingouin as pg


@dataclass
class SlopeSummary:
    """Container for slope statistics for one direction (X→Y on metric m)."""
    direction: str           # e.g., "canny->depth on depth_ssim"
    mean_signed_slope: float # s̄ = N^{-1} Σ s_i
    sii: float               # SII = |s̄|
    mas: float               # MAS = N^{-1} Σ |s_i|
    median_slope: float
    iqr_lower: float
    iqr_upper: float
    n_positive: int
    n_negative: int
    ci_lower: float          # 95% CI lower for s̄
    ci_upper: float          # 95% CI upper for s̄
    per_prompt_slopes: np.ndarray


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS slope of y on x (simple linear regression)."""
    if len(x) < 2 or np.std(x) == 0:
        return 0.0
    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)


def compute_per_prompt_slopes(
    retention_matrix: np.ndarray,
    b_weights: List[float],
) -> np.ndarray:
    """
    For each prompt i, compute OLS slope s_i of R_{A,i} on w_B
    across the five w_B levels.
    
    Parameters
    ----------
    retention_matrix : np.ndarray, shape (n_prompts, n_b_levels)
        Retention values R_{A,i}(w_B) for each prompt and each B weight.
    b_weights : list of float
        The five B weight levels, e.g., [0.1, 0.3, 0.5, 0.7, 0.9].
    
    Returns
    -------
    per_prompt_slopes : np.ndarray, shape (n_prompts,)
        s_i for each prompt.
    """
    x = np.array(b_weights)
    n_prompts = retention_matrix.shape[0]
    slopes = np.zeros(n_prompts)

    for i in range(n_prompts):
        slopes[i] = ols_slope(x, retention_matrix[i])

    return slopes


def bootstrap_ci(
    slopes: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Prompt-level bootstrap for s̄ confidence interval.
    Resamples prompts (not within-prompt residuals) with 1,000 resamples.
    Reflects between-prompt variation only; does not propagate
    within-prompt OLS residual uncertainty.
    """
    rng = np.random.RandomState(random_state)
    n = len(slopes)
    boot_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = slopes[idx].mean()

    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(ci_lower), float(ci_upper)


def compute_slope_summary(
    retention_matrix: np.ndarray,
    b_weights: List[float],
    direction_label: str,
    n_bootstrap: int = 1000,
) -> SlopeSummary:
    """
    Compute full slope summary for one direction.
    
    s̄ = N^{-1} Σ s_i (signed mean per-prompt slope)
    SII = |s̄|
    MAS = N^{-1} Σ |s_i|
    Median, IQR, positive/negative counts
    95% CI for s̄ via prompt-level bootstrap
    """
    per_prompt_slopes = compute_per_prompt_slopes(retention_matrix, b_weights)

    s_bar = float(np.mean(per_prompt_slopes))
    sii = float(abs(s_bar))
    mas = float(np.mean(np.abs(per_prompt_slopes)))
    median_s = float(np.median(per_prompt_slopes))
    iqr_lower = float(np.percentile(per_prompt_slopes, 25))
    iqr_upper = float(np.percentile(per_prompt_slopes, 75))
    n_pos = int(np.sum(per_prompt_slopes > 0))
    n_neg = int(np.sum(per_prompt_slopes < 0))

    ci_lower, ci_upper = bootstrap_ci(per_prompt_slopes, n_bootstrap=n_bootstrap)

    return SlopeSummary(
        direction=direction_label,
        mean_signed_slope=s_bar,
        sii=sii,
        mas=mas,
        median_slope=median_s,
        iqr_lower=iqr_lower,
        iqr_upper=iqr_upper,
        n_positive=n_pos,
        n_negative=n_neg,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        per_prompt_slopes=per_prompt_slopes,
    )


def compute_a_only_weight_effect(
    retention_matrix_a_only: np.ndarray,
    a_levels: List[float],
    from_level: float = 0.5,
    to_level: float = 0.1,
) -> float:
    """
    A-only multi-level ablation: for each prompt, fit linear regression
    of R_{A,i} on w_A across five A-only levels; report fitted change
    over [from_level, to_level] interval.
    
    This is A's own weight effect, reported separately from B-weight
    retention change (not summed or subtracted).
    """
    n_prompts = retention_matrix_a_only.shape[0]
    per_prompt_changes = np.zeros(n_prompts)

    for i in range(n_prompts):
        slope = ols_slope(np.array(a_levels), retention_matrix_a_only[i])
        per_prompt_changes[i] = slope * (to_level - from_level)

    return float(np.mean(per_prompt_changes))


def condition_control_slope_contrast(
    slopes_real: np.ndarray,
    slopes_control: np.ndarray,
) -> np.ndarray:
    """
    Paired condition-control slope contrast:
    δ_{i,blank} = s_{i,real} - s_{i,blank}
    δ_{i,shuffle} = s_{i,real} - s_{i,shuffle}
    
    Components identical across B conditions cancel exactly;
    condition-specific total-weight interactions do NOT cancel.
    δ is a descriptive contrast, not a net B-addition effect.
    """
    return slopes_real - slopes_control


def run_paired_test(
    slopes_a: np.ndarray,
    slopes_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Paired t-test within metric and condition pair only.
    No cross-pair ANOVA; α = 0.05; uses Pingouin.
    Power calculation for n=20, two-sided α=0.05, dz=0.6: power=0.72.
    """
    result = pg.ttest(slopes_a, slopes_b, paired=True, correction=False)
    return {
        "t": float(result["T"].values[0]),
        "p": float(result["p-val"].values[0]),
        "d": float(result["cohen-d"].values[0]),
        "significant": float(result["p-val"].values[0]) < alpha,
    }


def binomial_preference_power(
    n: int = 40,
    p1: float = 0.65,
    alpha: float = 0.05,
) -> Dict:
    """
    Power calculation for preference test (n=40 pairs, one-sided binomial).
    Critical count c=26 at α=0.05.
    """
    from scipy.stats import binom
    c = binom.ppf(1 - alpha, n, 0.5)  # critical count
    power = 1 - binom.cdf(c, n, p1)
    return {
        "n": n,
        "critical_count": int(c),
        "p1": p1,
        "power": float(power),
    }
