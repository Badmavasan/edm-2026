"""Visualization functions for BKT results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Seaborn styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150


def plot_roc_comparison(
    roc_data_list: List[Dict],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot ROC curves for both modalities on same plot.

    Args:
        roc_data_list: List of dicts with keys: modality, fpr, tpr, auc.
        output_path: Path to save the plot.
        title_suffix: Suffix for the plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"error_independent": "#1f77b4", "error_dependent": "#ff7f0e"}

    for data in roc_data_list:
        modality = data["modality"]
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_val = data["auc"]

        label = f'{modality.replace("_", " ").title()} (AUC={auc_val:.3f})'
        ax.plot(fpr, tpr, color=colors.get(modality, "gray"), linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(f"ROC Curve Comparison{title_suffix}", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_auc_over_occurrence(
    step_metrics_df: pd.DataFrame,
    output_path: Path,
    modality: str,
) -> None:
    """Plot AUC over opportunity steps for a single modality.

    Args:
        step_metrics_df: DataFrame with step, auc, accuracy, f1 columns.
        output_path: Path to save the plot.
        modality: Modality name for the title.
    """
    df = step_metrics_df.copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["step"], df["auc"], marker="o", linewidth=2, label="AUC")
    ax.plot(df["step"], df["accuracy"], marker="s", linewidth=2, label="Accuracy")
    ax.plot(df["step"], df["f1"], marker="^", linewidth=2, label="F1")

    ax.set_xlabel("Opportunity Step (order_id)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        f"BKT Performance Over Steps - {modality.replace('_', ' ').title()}", fontsize=14
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_avg_predicted_prob_comparison(
    data_list: List[Dict],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot average predicted probability by occurrence for both modalities.

    Args:
        data_list: List of dicts with keys: modality, df (with occurrence_index, avg_predicted_prob, avg_actual).
        output_path: Path to save the plot.
        title_suffix: Suffix for the plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"error_independent": "#1f77b4", "error_dependent": "#ff7f0e"}

    for data in data_list:
        modality = data["modality"]
        df = data["df"]

        if df.empty:
            continue

        ax.plot(
            df["occurrence_index"],
            df["avg_predicted_prob"],
            marker="o",
            linewidth=2,
            color=colors.get(modality, "gray"),
            label=f'{modality.replace("_", " ").title()} (Predicted)',
        )

        # Also plot actual rate
        ax.plot(
            df["occurrence_index"],
            df["avg_actual"],
            marker="x",
            linewidth=1,
            linestyle="--",
            color=colors.get(modality, "gray"),
            alpha=0.7,
            label=f'{modality.replace("_", " ").title()} (Actual)',
        )

    ax.set_xlabel("Skill Occurrence Index", fontsize=12)
    ax.set_ylabel("Average Probability", fontsize=12)
    ax.set_title(f"Average Predicted Probability by Occurrence{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_data_csv(roc_data_list: List[Dict], output_path: Path) -> None:
    """Save ROC data to CSV for both modalities.

    Args:
        roc_data_list: List of dicts with keys: modality, fpr, tpr, auc.
        output_path: Path to save the CSV.
    """
    rows = []
    for data in roc_data_list:
        modality = data["modality"]
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_val = data["auc"]
        for f, t in zip(fpr, tpr):
            rows.append({"modality": modality, "fpr": f, "tpr": t, "auc": auc_val})

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_avg_pred_prob_csv(data_list: List[Dict], output_path: Path) -> None:
    """Save average predicted probability data to CSV.

    Args:
        data_list: List of dicts with keys: modality, df.
        output_path: Path to save the CSV.
    """
    all_dfs = []
    for data in data_list:
        df = data["df"].copy()
        if not df.empty:
            df["modality"] = data["modality"]
            all_dfs.append(df)

    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(output_path, index=False)
    else:
        pd.DataFrame().to_csv(output_path, index=False)


def load_and_plot_roc_from_csv(csv_path: Path, output_path: Path) -> None:
    """Load ROC data from CSV and plot.

    Args:
        csv_path: Path to the CSV with roc data.
        output_path: Path to save the plot.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    roc_data_list = []
    for modality in df["modality"].unique():
        mod_df = df[df["modality"] == modality]
        roc_data_list.append(
            {
                "modality": modality,
                "fpr": mod_df["fpr"].to_numpy(),
                "tpr": mod_df["tpr"].to_numpy(),
                "auc": mod_df["auc"].iloc[0] if "auc" in mod_df.columns else 0.5,
            }
        )

    plot_roc_comparison(roc_data_list, output_path)


def load_and_plot_avg_pred_prob_from_csv(csv_path: Path, output_path: Path) -> None:
    """Load average predicted probability data from CSV and plot.

    Args:
        csv_path: Path to the CSV with avg pred prob data.
        output_path: Path to save the plot.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    data_list = []
    for modality in df["modality"].unique():
        mod_df = df[df["modality"] == modality]
        data_list.append({"modality": modality, "df": mod_df})

    plot_avg_predicted_prob_comparison(data_list, output_path)


def plot_auc_over_training_samples(
    metrics_df: pd.DataFrame,
    output_path: Path,
    modality: str,
) -> None:
    """Plot AUC over training sample sizes for a single modality.

    Args:
        metrics_df: DataFrame with train_samples, auc columns.
        output_path: Path to save the plot.
        modality: Modality name for the title.
    """
    df = metrics_df.copy()
    if df.empty or "auc" not in df.columns:
        return

    # Filter out None AUC values
    df = df[df["auc"].notna()]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["train_samples"], df["auc"], marker="o", linewidth=2, color="#1f77b4")

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples (log scale)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(
        f"AUC vs Training Data Size - {modality.replace('_', ' ').title()}", fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_auc_over_training_samples_comparison(
    data_list: List[Dict],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot AUC over training samples for both modalities on same plot.

    Args:
        data_list: List of dicts with keys: modality, df (with train_samples, auc).
        output_path: Path to save the plot.
        title_suffix: Suffix for the plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"error_independent": "#1f77b4", "error_dependent": "#ff7f0e"}

    for data in data_list:
        modality = data["modality"]
        df = data["df"].copy()

        if df.empty or "auc" not in df.columns:
            continue

        # Filter out None AUC values
        df = df[df["auc"].notna()]
        if df.empty:
            continue

        ax.plot(
            df["train_samples"],
            df["auc"],
            marker="o",
            linewidth=2,
            color=colors.get(modality, "gray"),
            label=modality.replace("_", " ").title(),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples (log scale)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"AUC vs Training Data Size{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_and_plot_incremental_from_csv(csv_path: Path, output_path: Path) -> None:
    """Load incremental training data from CSV and plot AUC.

    Args:
        csv_path: Path to the CSV with incremental training data.
        output_path: Path to save the plot.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Check if this is a combined file with modality column
    if "modality" in df.columns:
        data_list = []
        for modality in df["modality"].unique():
            mod_df = df[df["modality"] == modality]
            data_list.append({"modality": modality, "df": mod_df})
        plot_auc_over_training_samples_comparison(data_list, output_path)
    else:
        # Single modality file
        plot_auc_over_training_samples(df, output_path, "unknown")


def plot_auc_multi_modality(
    data_list: List[Dict],
    output_path: Path,
    title: str = "AUC Comparison",
    xlabel: str = "Training Samples",
    ylabel: str = "AUC",
    x_column: str = "train_samples",
    y_column: str = "auc",
    legend_labels: Optional[Dict[str, str]] = None,
    figsize: tuple = (12, 7),
    palette: str = "Set2",
    start_from_origin: bool = True,
    marker_size: int = 10,
    line_width: float = 2.5,
    legend_loc: str = "lower right",
) -> None:
    """Plot AUC for multiple modalities on a single clean plot.

    Args:
        data_list: List of dicts with keys:
            - 'name': identifier for the modality
            - 'df': DataFrame with x_column and y_column
        output_path: Path to save the plot.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        x_column: Column name for x-axis values.
        y_column: Column name for y-axis values (typically 'auc').
        legend_labels: Dict mapping modality names to custom legend labels.
        figsize: Figure size as (width, height).
        palette: Seaborn color palette name.
        start_from_origin: Whether to start plot from (0, 0).
        marker_size: Size of markers on the line.
        line_width: Width of the lines.
        legend_loc: Legend location.
    """
    # Clean style
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors from palette
    colors = sns.color_palette(palette, n_colors=len(data_list))

    max_x = 0

    # Plot each modality
    for idx, data in enumerate(data_list):
        name = data.get("name", data.get("modality", f"Modality {idx + 1}"))
        df = data["df"].copy()

        if df.empty or y_column not in df.columns or x_column not in df.columns:
            continue

        # Filter out None/NaN values
        df = df[df[y_column].notna()]
        if df.empty:
            continue

        # Sort by x column
        df = df.sort_values(x_column)

        # Add origin point (0, 0) if start_from_origin
        if start_from_origin:
            origin = pd.DataFrame({x_column: [0], y_column: [0]})
            df = pd.concat([origin, df], ignore_index=True)

        max_x = max(max_x, df[x_column].max())

        # Get legend label
        if legend_labels and name in legend_labels:
            label = legend_labels[name]
        else:
            label = name.replace("_", " ").title()

        # Plot the line with markers
        ax.plot(
            df[x_column],
            df[y_column],
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            color=colors[idx],
            label=label,
        )

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Axis limits
    ax.set_xlim([0, max_x * 1.05])
    ax.set_ylim([0, 1.0])

    # Clean grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Legend
    ax.legend(
        loc=legend_loc,
        fontsize=12,
        framealpha=0.95,
        edgecolor="lightgray",
    )

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_metrics_multi_modality(
    data_list: List[Dict],
    output_path: Path,
    metrics: List[str] = None,
    title: str = "Metrics Comparison",
    xlabel: str = "Training Samples",
    x_column: str = "train_samples",
    legend_labels: Optional[Dict[str, str]] = None,
    use_log_scale: bool = True,
    figsize: tuple = (14, 8),
    palette: str = "husl",
) -> None:
    """Plot multiple metrics for multiple modalities using subplots.

    Args:
        data_list: List of dicts with 'name' and 'df' keys.
        output_path: Path to save the plot.
        metrics: List of metric column names to plot. Default: ['auc', 'accuracy', 'f1'].
        title: Main plot title.
        xlabel: X-axis label.
        x_column: Column name for x-axis values.
        legend_labels: Dict mapping modality names to custom legend labels.
        use_log_scale: Whether to use log scale for x-axis.
        figsize: Figure size.
        palette: Seaborn color palette name.
    """
    if metrics is None:
        metrics = ["auc", "accuracy", "f1"]

    sns.set_theme(style="whitegrid", palette=palette)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    n_modalities = len(data_list)
    colors = sns.color_palette(palette, n_colors=n_modalities)

    metric_titles = {
        "auc": "AUC",
        "accuracy": "Accuracy",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]

        for mod_idx, data in enumerate(data_list):
            name = data.get("name", data.get("modality", f"Modality {mod_idx + 1}"))
            df = data["df"].copy()

            if df.empty or metric not in df.columns or x_column not in df.columns:
                continue

            df = df[df[metric].notna()]
            if df.empty:
                continue

            if legend_labels and name in legend_labels:
                label = legend_labels[name]
            else:
                label = name.replace("_", " ").title()

            ax.plot(
                df[x_column],
                df[metric],
                marker="o",
                markersize=6,
                linewidth=2,
                color=colors[mod_idx],
                label=label,
            )

        if use_log_scale:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(metric_titles.get(metric, metric.title()), fontsize=11)
        ax.set_title(metric_titles.get(metric, metric.title()), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])

        if ax_idx == n_metrics - 1:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    """Example usage of multi-modality plotting functions."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Load incremental training data
    base_path = Path(__file__).parent.parent.parent
    results_dir = base_path / "results" / "global"

    # Load data for both modalities
    data_list = []
    legend_labels = {}

    for modality in ["error_independent", "error_dependent"]:
        csv_path = results_dir / modality / "incremental_training_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data_list.append({"name": modality, "df": df})
            # Custom legend labels
            if modality == "error_independent":
                legend_labels[modality] = "Error Independent"
            else:
                legend_labels[modality] = "Error Dependent"

    if not data_list:
        print("No data found. Run the BKT training first:")
        print("  python src/bkt/run_bkt.py")
        sys.exit(1)

    print(f"Loaded {len(data_list)} modalities")

    # Plot AUC comparison
    output_path = results_dir / "auc_multi_modality_comparison.png"
    plot_auc_multi_modality(
        data_list,
        output_path,
        title="BKT AUC vs Training Data Size",
        xlabel="Training Samples (log scale)",
        ylabel="AUC",
        legend_labels=legend_labels,
        palette="husl",
        ylim=(0.5, 1.0),
        figsize=(12, 7),
    )
    print(f"Saved: {output_path}")

    # Plot all metrics comparison
    output_path_metrics = results_dir / "metrics_multi_modality_comparison.png"
    plot_metrics_multi_modality(
        data_list,
        output_path_metrics,
        metrics=["auc", "accuracy", "f1"],
        title="BKT Metrics vs Training Data Size",
        legend_labels=legend_labels,
        palette="Set2",
    )
    print(f"Saved: {output_path_metrics}")