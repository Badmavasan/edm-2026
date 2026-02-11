"""
Plot incremental training AUC by number of students for all models and modalities.

Creates a clean seaborn plot showing AUC vs number of training students for:
- 3 algorithms: BKT, IRT, PFA
- 2 modalities: error_independent, error_dependent
= 6 lines total (one plot per dataset, all on the same graph)

Uses incremental_training_metrics_by_students.csv from each modality directory.

Run with:
    python src/plot_incremental_training_by_students.py --dataset platform_a
    python src/plot_incremental_training_by_students.py --dataset platform_b
    python src/plot_incremental_training_by_students.py --dataset both
    python src/plot_incremental_training_by_students.py --dataset platform_a --fontsize 14
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# Configuration
MODELS = ['bkt', 'irt', 'pfa']
MODALITIES = ['error_independent', 'error_dependent']

MODEL_NAMES = {
    'bkt': 'BKT',
    'irt': 'IRT',
    'pfa': 'PFA',
}

MODALITY_NAMES = {
    'error_independent': 'Error-Independent',
    'error_dependent': 'Error-Dependent',
}

MODALITY_STYLES = {
    'error_independent': '-',
    'error_dependent': '--',
}

MODALITY_MARKERS = {
    'error_independent': 'o',
    'error_dependent': 's',
}


def _build_palette():
    """Return 3 clearly distinct colours that pair well together."""
    return {
        'bkt': '#1b9e77',  # Teal-green
        'irt': '#d95f02',  # Burnt orange
        'pfa': '#7570b3',  # Muted purple
    }


def load_incremental_by_students(
    results_path: Path, dataset: str, model: str, modality: str,
) -> pd.DataFrame | None:
    """Load incremental_training_metrics_by_students.csv for one model/modality."""
    path = (
        results_path / dataset / model / 'global' / modality
        / 'incremental_training_metrics_by_students.csv'
    )
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = df[df['auc'].notna()].copy()
    if df.empty:
        return None

    df['model'] = model
    df['modality'] = modality
    return df


def _set_linear_xticks(ax, x_max: float) -> None:
    """Set readable linear x-axis ticks based on data range."""
    nice_steps = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    # Pick a step that gives roughly 5-8 ticks
    step = nice_steps[0]
    for s in nice_steps:
        if x_max / s <= 8:
            step = s
            break
    ticks = np.arange(0, x_max + step, step)
    # Always include 0 and keep only ticks within data range + a small margin
    ticks = ticks[ticks <= x_max * 1.05]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))


def plot_incremental_by_students(
    results_path: Path,
    dataset: str,
    output_path: Path,
    fontsize: int = 12,
    figsize: tuple = (10, 6),
    dpi: int = 150,
):
    """Plot AUC vs number of training students for one dataset."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=fontsize / 10)

    model_colors = _build_palette()

    fig, ax = plt.subplots(figsize=figsize)

    all_data = []
    for model in MODELS:
        for modality in MODALITIES:
            df = load_incremental_by_students(results_path, dataset, model, modality)
            if df is not None:
                all_data.append(df)

    if not all_data:
        print(f"No by-students data found for dataset '{dataset}'")
        plt.close(fig)
        return

    combined = pd.concat(all_data, ignore_index=True)

    for model in MODELS:
        for modality in MODALITIES:
            subset = combined[
                (combined['model'] == model) & (combined['modality'] == modality)
            ]
            if subset.empty:
                continue

            label = f"{MODEL_NAMES[model]} ({MODALITY_NAMES[modality]})"
            ax.plot(
                subset['train_students'],
                subset['auc'],
                color=model_colors[model],
                linestyle=MODALITY_STYLES[modality],
                marker=MODALITY_MARKERS[modality],
                markersize=11,
                linewidth=3.5,
                label=label,
                markeredgecolor='white',
                markeredgewidth=1.8,
            )

    _set_linear_xticks(ax, combined['train_students'].max())

    ax.set_xlabel('Number of Training Students', fontsize=fontsize + 2, fontweight='bold',
                   labelpad=12)
    ax.set_ylabel('AUC', fontsize=fontsize + 2, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis='both', labelsize=fontsize + 1, width=1.5, length=5)

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=fontsize + 1,
        framealpha=0.95,
        edgecolor='gray',
        borderpad=1,
    )

    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
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
    """Plot side-by-side comparison for both datasets."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=fontsize / 10)

    model_colors = _build_palette()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    datasets = ['platform_a', 'platform_b']
    dataset_titles = {'platform_a': 'Platform A', 'platform_b': 'Platform B'}

    for idx, ds in enumerate(datasets):
        ax = axes[idx]

        x_max = 0
        for model in MODELS:
            for modality in MODALITIES:
                df = load_incremental_by_students(results_path, ds, model, modality)
                if df is None or df.empty:
                    continue

                x_max = max(x_max, df['train_students'].max())

                label = f"{MODEL_NAMES[model]} ({MODALITY_NAMES[modality]})"
                ax.plot(
                    df['train_students'],
                    df['auc'],
                    color=model_colors[model],
                    linestyle=MODALITY_STYLES[modality],
                    marker=MODALITY_MARKERS[modality],
                    markersize=10,
                    linewidth=3.5,
                    label=label if idx == 0 else None,
                    markeredgecolor='white',
                    markeredgewidth=1.8,
                )

        if x_max > 0:
            _set_linear_xticks(ax, x_max)

        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel('AUC' if idx == 0 else '', fontsize=fontsize + 2, fontweight='bold')
        ax.tick_params(axis='both', labelsize=fontsize + 1, width=1.5, length=5)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Centred dataset title with vertical padding from the top
        ax.text(
            0.5, 0.95, dataset_titles[ds], transform=ax.transAxes,
            fontsize=fontsize + 3, fontweight='bold', va='top', ha='center',
        )

    # Single x-axis label centred across both subplots
    fig.text(
        0.5, -0.02, 'Number of Training Students',
        ha='center', va='center',
        fontsize=fontsize + 2, fontweight='bold',
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=fontsize + 1,
        framealpha=0.95,
        edgecolor='gray',
        borderpad=1,
    )

    sns.despine()
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot incremental training AUC by number of students',
    )
    parser.add_argument(
        '--dataset', type=str, default='platform_a',
        choices=['platform_a', 'platform_b', 'both'],
        help='Dataset to plot',
    )
    parser.add_argument('--fontsize', type=int, default=12, help='Base font size')
    parser.add_argument(
        '--figsize', type=float, nargs=2, default=[10, 6],
        help='Figure size (width height)',
    )
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path (default: results/{dataset}/incremental_auc_by_students.png)',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    results_path = script_dir / 'results'

    if args.dataset == 'both':
        output_path = args.output or results_path / 'incremental_auc_by_students_both.png'
        plot_comparison_both_datasets(
            results_path=results_path,
            output_path=Path(output_path),
            fontsize=args.fontsize,
            figsize=(14, 6),
            dpi=args.dpi,
        )
    else:
        output_path = (
            args.output
            or results_path / args.dataset / 'incremental_auc_by_students.png'
        )
        plot_incremental_by_students(
            results_path=results_path,
            dataset=args.dataset,
            output_path=Path(output_path),
            fontsize=args.fontsize,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
        )


if __name__ == '__main__':
    main()
