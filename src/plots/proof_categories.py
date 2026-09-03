"""Figs 2(left)/5b/8: proof-strategy category plots for the proofs domain."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plots.style import model_label, ordered_models


def map_categories(raw_category: str) -> str:
    """
    Light normalization of judge-produced category strings so near-duplicate phrasings collapse to one label.
    """
    c = str(raw_category).strip().lower()
    if "invalid" in c:
        return "invalid"
    if "euclid" in c:
        return "euclidean"
    if "euler" in c:
        return "euler's theorem"
    if "group theory" in c or "group-theoretic" in c:
        return "group theory"
    if "induction" in c or "first solution" in c:
        return "induction / first solution"
    return raw_category.strip()


def unique_proofs_per_model(
    proofs_df: pd.DataFrame, problem_id: str, num_known_proofs: Optional[int] = None
) -> pd.DataFrame:
    """Fig 2 (left): unique valid proof categories per model for one problem, vs. a reference count."""
    sub = proofs_df[
        (proofs_df["problem_id"] == problem_id)
        & (proofs_df["category"].str.lower() != "invalid")
    ]
    if sub.empty:
        print(f"No proof data for problem_id={problem_id!r}.")
    counts = sub.groupby("model_version")["category"].apply(
        lambda s: s.map(map_categories).nunique()
    )
    out = counts.reset_index(name="n_unique_proofs")
    if num_known_proofs is not None:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "model_version": "Reference",
                            "n_unique_proofs": num_known_proofs,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return out


def plot_stacked_proof_categories(
    proofs_df: pd.DataFrame, problem_id: str, ax: Optional[plt.Axes] = None
):
    """Fig 5b: stacked bar of proof-category counts per model for one problem, gray = invalid."""
    ax = ax or plt.gca()
    sub = proofs_df[proofs_df["problem_id"] == problem_id].copy()
    if sub.empty:
        print(f"No proof data for problem_id={problem_id!r}.")
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
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    sub["category_norm"] = sub["category"].map(map_categories)
    models = ordered_models(sub["model_version"].unique())
    categories = [c for c in sub["category_norm"].unique() if c != "invalid"]
    cmap = plt.get_cmap("Set3")

    bottoms = np.zeros(len(models))
    for k, cat in enumerate(categories):
        heights = [
            len(sub[(sub["model_version"] == m) & (sub["category_norm"] == cat)])
            for m in models
        ]
        ax.bar(
            range(len(models)), heights, bottom=bottoms, label=cat, color=cmap(k % 12)
        )
        bottoms += np.array(heights)
    invalid_heights = [
        len(sub[(sub["model_version"] == m) & (sub["category_norm"] == "invalid")])
        for m in models
    ]
    if any(invalid_heights):
        ax.bar(
            range(len(models)),
            invalid_heights,
            bottom=bottoms,
            label="Invalid",
            color="lightgray",
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models], rotation=45, ha="right")
    if categories or any(invalid_heights):
        ax.legend(fontsize=7, ncol=2)
    return ax


def cumulative_unique_categories(
    proofs_df: pd.DataFrame, problem_id: str, max_samples: int = 10
) -> pd.DataFrame:
    """Fig 8: cumulative unique proof-category count vs. sample index, per model, for one problem."""
    sub = proofs_df[proofs_df["problem_id"] == problem_id].copy()
    if sub.empty:
        print(f"No proof data for problem_id={problem_id!r}.")
        return sub.assign(category_norm=[], cum_unique=[], x=[])
    sub["category_norm"] = sub["category"].map(map_categories)
    sub = sub[sub["category_norm"] != "invalid"]
    sub = sub.sort_values(["model_version", "sample_id"], kind="mergesort")
    first = ~sub.duplicated(["model_version", "category_norm"])
    sub["cum_unique"] = first.groupby(sub["model_version"]).cumsum().astype(int)
    sub["x"] = sub.groupby("model_version").cumcount().add(1).astype(int)
    return sub[sub["x"] <= max_samples]
