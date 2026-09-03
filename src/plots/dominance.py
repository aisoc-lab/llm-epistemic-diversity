"""Fig 5a: empirical CDF of top-entity dominance per query, one line per model."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plots.style import model_color, model_label, ordered_models


def top_entity_dominance(df: pd.DataFrame, valid_only: bool = True) -> pd.DataFrame:
    """For each (model, query), the max share of responses going to a single resolved_name."""
    sub = df[df["validity_label"] != False] if valid_only else df
    shares = (
        sub.groupby(["model_version", "query"])["resolved_name"]
        .apply(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
        .reset_index(name="top_entity_share")
    )
    return shares.dropna()


def plot_dominance_cdf(
    shares_df: pd.DataFrame, ax: Optional[plt.Axes] = None, n_thresholds: int = 100
):
    """
    Steeper curve = top entity rarely dominates, flatter = top entity frequently dominates.
    """
    ax = ax or plt.gca()
    if shares_df is None or shares_df.empty:
        print("No data for dominance CDF: nothing to plot.")
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
            transform=ax.transAxes,
        )
        return ax
    thresholds = np.linspace(0, 1, n_thresholds)
    any_lines = False
    for m in ordered_models(shares_df["model_version"].unique()):
        vals = shares_df.loc[
            shares_df["model_version"] == m, "top_entity_share"
        ].to_numpy()
        if len(vals) == 0:
            continue
        cdf = [np.mean(vals <= t) for t in thresholds]
        ax.plot(
            thresholds, cdf, color=model_color(m), label=model_label(m), linewidth=1.4
        )
        any_lines = True
    ax.set_xlabel("Fraction of samples (top entity per query)")
    ax.set_ylabel("Fraction of queries (≤ threshold)")
    if any_lines:
        ax.legend(fontsize=8)
    return ax
