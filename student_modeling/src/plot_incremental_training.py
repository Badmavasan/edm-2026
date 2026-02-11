"""
Plot incremental training AUC for all models and modalities.

Creates a clean seaborn plot showing AUC vs training samples for:
- 3 algorithms: BKT, IRT, PFA
- 2 modalities: error_independent, error_dependent
= 6 lines total

Run with:
    python src/plot_incremental_training.py --dataset platform_a
    python src/plot_incremental_training.py --dataset platform_b
    python src/plot_incremental_training.py --dataset platform_a --fontsize 14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


# Configuration
MODELS = ['bkt', 'irt', 'pfa']
MODALITIES = ['error_independent', 'error_dependent']

# Clean display names
MODEL_NAMES = {
    'bkt': 'BKT',
    'irt': 'IRT',
    'pfa': 'PFA'
}

MODALITY_NAMES = {
    'error_independent': 'Error-Independent',
    'error_dependent': 'Error-Dependent'
}

# Color palette - distinct colors for models, line style for modality
MODEL_COLORS = {
    'bkt': '#2563eb',  # Blue
    'irt': '#16a34a',  # Green
    'pfa': '#dc2626',  # Red
}

MODALITY_STYLES = {
    'error_independent': '-',   # Solid
    'error_dependent': '--',    # Dashed
}

MODALITY_MARKERS = {
    'error_independent': 'o',   # Circle
    'error_dependent': 's',     # Square
}


def load_incremental_metrics(results_path: Path, dataset: str, model: str, modality: str) -> pd.DataFrame:
    """Load incremental training metrics for a specific model/modality."""
    path = results_path / dataset / model / 'global' / modality / 'incremental_training_metrics.csv'

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = df[df['auc'].notna()].copy()

    if df.empty:
        return None

    df['model'] = model
    df['modality'] = modality
    df['label'] = f"{MODEL_NAMES[model]} ({MODALITY_NAMES[modality]})"

    return df


def plot_incremental_auc(
    results_path: Path,
    dataset: str,
    output_path: Path,
    fontsize: int = 12,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    log_scale: bool = True,
):
    """
    Plot incremental training AUC for all models and modalities.

    Parameters:
    -----------
    results_path : Path
        Base path to results directory
    dataset : str
        Dataset name ('platform_a' or 'platform_b')
    output_path : Path
        Path to save the plot
    fontsize : int
        Base font size for labels and legend
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution for saved figure
    log_scale : bool
        Whether to use log scale for x-axis
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=fontsize / 10)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Load all data
    all_data = []

    for model in MODELS:
        for modality in MODALITIES:
            df = load_incremental_metrics(results_path, dataset, model, modality)
            if df is not None:
                all_data.append(df)

    if not all_data:
        print(f"No data found for dataset '{dataset}'")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Plot each combination
    for model in MODELS:
        for modality in MODALITIES:
            subset = combined_df[(combined_df['model'] == model) & (combined_df['modality'] == modality)]

            if subset.empty:
                continue

            label = f"{MODEL_NAMES[model]} ({MODALITY_NAMES[modality]})"

            ax.plot(
                subset['train_samples'],
                subset['auc'],
                color=MODEL_COLORS[model],
                linestyle=MODALITY_STYLES[modality],
                marker=MODALITY_MARKERS[modality],
                markersize=8,
                linewidth=2,
                label=label,
                markeredgecolor='white',
                markeredgewidth=1,
            )

    # Configure axes
    if log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Training Samples', fontsize=fontsize, fontweight='medium')
    ax.set_ylabel('AUC', fontsize=fontsize, fontweight='medium')

    # Configure ticks
    ax.tick_params(axis='both', labelsize=fontsize - 2)

    # Configure legend
    ax.legend(
        loc='lower right',
        fontsize=fontsize - 2,
        framealpha=0.95,
        edgecolor='gray',
    )

    # Set y-axis limits with some padding
    ymin = combined_df['auc'].min()
    ymax = combined_df['auc'].max()
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(max(0.4, ymin - padding), min(1.0, ymax + padding))

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    sns.despine()

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_comparison_both_datasets(
    results_path: Path,
    output_path: Path,
    fontsize: int = 12,
    figsize: tuple = (14, 6),
    dpi: int = 150,
):
    """
    Plot side-by-side comparison for both datasets.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=fontsize / 10)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    datasets = ['platform_a', 'platform_b']
    dataset_titles = {'platform_a': 'Platform A', 'platform_b': 'Platform B'}

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        for model in MODELS:
            for modality in MODALITIES:
                df = load_incremental_metrics(results_path, dataset, model, modality)

                if df is None or df.empty:
                    continue

                label = f"{MODEL_NAMES[model]} ({MODALITY_NAMES[modality]})"

                ax.plot(
                    df['train_samples'],
                    df['auc'],
                    color=MODEL_COLORS[model],
                    linestyle=MODALITY_STYLES[modality],
                    marker=MODALITY_MARKERS[modality],
                    markersize=7,
                    linewidth=2,
                    label=label if idx == 0 else None,  # Only label first subplot
                    markeredgecolor='white',
                    markeredgewidth=1,
                )

        ax.set_xscale('log')
        ax.set_xlabel('Training Samples', fontsize=fontsize, fontweight='medium')
        ax.set_ylabel('AUC' if idx == 0 else '', fontsize=fontsize, fontweight='medium')
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add dataset name as subtitle
        ax.text(0.02, 0.98, dataset_titles[dataset], transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold', va='top', ha='left')

    # Single legend for both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=fontsize - 2,
        framealpha=0.95,
    )

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot incremental training AUC')
    parser.add_argument('--dataset', type=str, default='platform_a',
                        choices=['platform_a', 'platform_b', 'both'],
                        help='Dataset to plot')
    parser.add_argument('--fontsize', type=int, default=12,
                        help='Base font size')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 6],
                        help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI')
    parser.add_argument('--linear', action='store_true',
                        help='Use linear scale instead of log scale')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: results/{dataset}/incremental_auc_comparison.png)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    results_path = script_dir / 'results'

    if args.dataset == 'both':
        output_path = args.output or results_path / 'incremental_auc_comparison_both.png'
        plot_comparison_both_datasets(
            results_path=results_path,
            output_path=Path(output_path),
            fontsize=args.fontsize,
            figsize=(14, 6),
            dpi=args.dpi,
        )
    else:
        output_path = args.output or results_path / args.dataset / 'incremental_auc_comparison.png'
        plot_incremental_auc(
            results_path=results_path,
            dataset=args.dataset,
            output_path=Path(output_path),
            fontsize=args.fontsize,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            log_scale=not args.linear,
        )


if __name__ == '__main__':
    main()
