"""Plotting functions for IRT results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_roc_comparison(
    roc_data: Dict[str, pd.DataFrame],
    output_path: Path,
    title: str = "ROC Curve Comparison",
    legend_labels: Optional[Dict[str, str]] = None,
    figsize: tuple = (10, 8),
    dpi: int = 200,
) -> None:
    """Plot ROC curves for multiple modalities on the same plot.

    Args:
        roc_data: Dict mapping modality name to DataFrame with fpr, tpr columns.
        output_path: Path to save the plot.
        title: Plot title.
        legend_labels: Optional mapping from modality name to display label.
        figsize: Figure size.
        dpi: Output DPI.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("Set2", n_colors=len(roc_data))

    for i, (modality, df) in enumerate(roc_data.items()):
        if df.empty:
            continue
        label = legend_labels.get(modality, modality) if legend_labels else modality
        ax.plot(df["fpr"], df["tpr"], label=label, color=colors[i], linewidth=2)

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_auc_multi_modality(
    data_list: List[Dict],
    output_path: Path,
    title: str = "AUC vs Training Data Size",
    xlabel: str = "Training Samples",
    ylabel: str = "AUC",
    legend_labels: Optional[Dict[str, str]] = None,
    palette: str = "Set2",
    start_from_origin: bool = True,
    figsize: tuple = (12, 7),
    legend_loc: str = "lower right",
    dpi: int = 200,
) -> None:
    """Plot AUC curves for multiple modalities.

    Args:
        data_list: List of dicts with 'name' and 'df' keys.
                   df should have 'train_samples' and 'auc' columns.
        output_path: Path to save the plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        legend_labels: Optional mapping from name to display label.
        palette: Seaborn color palette.
        start_from_origin: Whether to start axes from (0, 0).
        figsize: Figure size.
        legend_loc: Legend location.
        dpi: Output DPI.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette(palette, n_colors=len(data_list))

    for i, item in enumerate(data_list):
        name = item["name"]
        df = item["df"]

        if df.empty or "auc" not in df.columns:
            continue

        # Filter out None values
        plot_df = df.dropna(subset=["auc"])
        if plot_df.empty:
            continue

        label = legend_labels.get(name, name) if legend_labels else name

        ax.plot(
            plot_df["train_samples"],
            plot_df["auc"],
            marker="o",
            markersize=6,
            linewidth=2,
            label=label,
            color=colors[i],
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc=legend_loc, fontsize=10)

    if start_from_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
