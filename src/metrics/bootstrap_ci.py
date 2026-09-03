"""
Compute mean unique valid names per model with a percentile-bootstrap 95% CI over prompt units.
"""

from typing import List

import numpy as np
import pandas as pd

from src.metrics.ratios import unique_valid_at_budget


def accessible_coverage_bootstrap(
    accessible_df: pd.DataFrame, budget: int = 10, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """
    For each model: unique valid names among the first `budget` samples per prompt unit,
    averaged over the prompt units, with a bootstrap 95% CI obtained by resampling prompt units with replacement `n_boot` times.
    """
    per_unit = unique_valid_at_budget(
        accessible_df, budget, group_cols=["model_version", "profession", "prompt_key"]
    )
    rng = np.random.default_rng(seed)
    rows = []
    for model, sub in per_unit.groupby("model_version"):
        values = sub["unique_valid"].to_numpy()
        mean = float(values.mean())
        boot_means = [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(n_boot)
        ]
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        rows.append(
            {
                "model_version": model,
                "mean_unique_valid": mean,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_prompts": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)
