"""
Plot cumulative-unique-count grid plots.

For each (row=profession-or-problem, col=format-or-protocol) panel,
plot cumulative-unique-valid-count vs. sample/round/turn index, one line per model.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.plots.style import model_color, model_label, ordered_models


def plot_cumulative_grid(
    cum_df: pd.DataFrame,
    row_col: str,
    col_col: str,
    x_col: str = "x",
    y_col: str = "cum_unique",
    group_col: str = "model_version",
    max_x: Optional[int] = None,
    log_y: bool = False,
    row_order: Optional[List[str]] = None,
    col_order: Optional[List[str]] = None,
    figsize_per_cell: float = 2.5,
    suptitle: Optional[str] = None,
):
    """
    Generic grid of cumulative-gain line plots, e.g. rows=profession, cols=format (Figs 6/7),
    or rows=problem_id, cols=(single) (Fig 8), or rows=model, cols=protocol (Fig 11).

    Pass explicit `row_order`/`col_order` to force every expected panel to appear. Leave both
    as None to size the grid from whatever's actually present in `cum_df`.
    """
    rows = row_order or sorted(cum_df[row_col].unique())
    cols = col_order or sorted(cum_df[col_col].unique())
    n_rows, n_cols = len(rows), len(cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell * n_cols + 1, figsize_per_cell * n_rows + 0.5),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    models = ordered_models(cum_df[group_col].unique())
    empty_panels: List[tuple] = []
    for i, row_val in enumerate(rows):
        for j, col_val in enumerate(cols):
            ax = axes[i][j]
            panel = cum_df[(cum_df[row_col] == row_val) & (cum_df[col_col] == col_val)]
            if panel.empty:
                empty_panels.append((row_val, col_val))
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            for m in models:
                msub = panel[panel[group_col] == m].sort_values(x_col)
                if msub.empty:
                    continue
                ax.plot(
                    msub[x_col],
                    msub[y_col],
                    color=model_color(m),
                    label=model_label(m),
                    linewidth=1.2,
                )
            if log_y:
                ax.set_yscale("log")
            if max_x:
                ax.set_xlim(0, max_x)
            if i == 0:
                ax.set_title(str(col_val), fontsize=10)
            if j == 0:
                ax.set_ylabel(str(row_val), fontsize=10)

    if empty_panels:
        print(
            f"{len(empty_panels)}/{n_rows * n_cols} panel(s) with no data ({row_col}, {col_col}): {empty_panels}"
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        # fall back to any panel that has data, in case the first is empty
        for ax_row in axes:
            for ax in ax_row:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
            if handles:
                break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(models), 5),
            bbox_to_anchor=(0.5, -0.05),
        )
    else:
        print("No data at all for this grid: every panel is empty.")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def plot_cumulative_single_row(
    cum_df: pd.DataFrame, row_value, row_col: str, col_col: str, **kwargs
):
    """Fig 4a/4b: cumulative gain for a single profession row across all format columns."""
    sub = cum_df[cum_df[row_col] == row_value]
    return plot_cumulative_grid(
        sub, row_col=row_col, col_col=col_col, row_order=[row_value], **kwargs
    )
