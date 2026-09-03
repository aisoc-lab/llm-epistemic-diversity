"""
Compute accessible-latent ratio Dacc(Q)/Dlat(Q), per model x profession x prompt format.
"""

from typing import List

import numpy as np
import pandas as pd

from src.data_loader import cumulative_unique_by_sample


def unique_valid_at_budget(
    df: pd.DataFrame, budget: int, group_cols: List[str] = ("model_version", "query")
) -> pd.DataFrame:
    """Unique valid names accumulated within the first `budget` samples/rounds/turns per group."""
    group_cols = list(group_cols)
    cum = cumulative_unique_by_sample(df, group_cols=group_cols, filter_valid=True)
    # groups that never reach `budget` samples: take their max x instead.
    group_max_x = cum.groupby(group_cols)["x"].transform("max")
    at_budget = cum[
        (cum["x"] == budget) | ((group_max_x < budget) & (cum["x"] == group_max_x))
    ]
    return at_budget[group_cols + ["cum_unique"]].rename(
        columns={"cum_unique": "unique_valid"}
    )


def accessible_latent_ratio(
    accessible_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    budget_acc: int = 100,
    budget_lat: int = 100,
) -> pd.DataFrame:
    """
    Dacc(Q)/Dlat(Q) per (model, profession, prompt_key).
    `latent_df` should be the expanded latent loader output.
    """
    acc = unique_valid_at_budget(
        accessible_df,
        budget_acc,
        group_cols=["model_version", "profession", "prompt_key"],
    )
    lat = unique_valid_at_budget(
        latent_df, budget_lat, group_cols=["model_version", "profession", "prompt_key"]
    )

    merged = acc.merge(
        lat,
        on=["model_version", "profession", "prompt_key"],
        suffixes=("_acc", "_lat"),
        how="outer",
    )
    merged["ratio"] = merged["unique_valid_acc"] / merged["unique_valid_lat"]
    merged["ratio"] = merged["ratio"].replace([np.inf, -np.inf], np.nan)
    return merged
