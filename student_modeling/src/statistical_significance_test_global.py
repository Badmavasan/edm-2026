"""
Statistical Significance Testing for AUC Comparison - Global Level
Compares error_dependent vs error_independent modalities at global level.
Works with both platform_b and platform_a datasets.

Uses permutation test for comparing AUCs.

Run with: python src/statistical_significance_test_global.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm


def permutation_test_auc(y_true1, y_pred1, y_true2, y_pred2, n_permutations=500, show_progress=False):
    """
    Permutation test for comparing AUCs from independent samples.

    This is the main bottleneck - it runs n_permutations iterations,
    each computing 2 AUC scores. With large datasets this can be slow.

    Parameters:
    -----------
    n_permutations : int
        Number of permutations (default 500, reduced from 1000 for speed)
    show_progress : bool
        Whether to show tqdm progress bar
    """
    from sklearn.metrics import roc_auc_score

    # Compute observed AUCs
    try:
        auc1 = roc_auc_score(y_true1, y_pred1)
    except Exception:
        auc1 = np.nan

    try:
        auc2 = roc_auc_score(y_true2, y_pred2)
    except Exception:
        auc2 = np.nan

    if np.isnan(auc1) or np.isnan(auc2):
        return auc1, auc2, np.nan, np.nan

    observed_diff = auc1 - auc2

    # Combine data for permutation
    all_y = np.concatenate([y_true1, y_true2])
    all_pred = np.concatenate([y_pred1, y_pred2])
    n1 = len(y_true1)
    n_total = len(all_y)

    perm_diffs = []
    np.random.seed(42)

    # Use tqdm if show_progress is True
    iterator = range(n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc="    Permutations", leave=False)

    for _ in iterator:
        perm_indices = np.random.permutation(n_total)
        perm_y1 = all_y[perm_indices[:n1]]
        perm_pred1 = all_pred[perm_indices[:n1]]
        perm_y2 = all_y[perm_indices[n1:]]
        perm_pred2 = all_pred[perm_indices[n1:]]

        try:
            perm_auc1 = roc_auc_score(perm_y1, perm_pred1)
            perm_auc2 = roc_auc_score(perm_y2, perm_pred2)
            perm_diffs.append(perm_auc1 - perm_auc2)
        except Exception:
            continue

    if len(perm_diffs) < 100:
        return auc1, auc2, np.nan, np.nan

    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return auc1, auc2, observed_diff, p_value


def load_predictions(base_path: Path, modality: str) -> Optional[pd.DataFrame]:
    """Load predictions for a given modality at global level."""
    path = base_path / 'global' / modality / 'predictions.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) > 0 else None


def compare_modalities(pred_dep: pd.DataFrame, pred_indep: pd.DataFrame,
                       n_permutations: int = 500, show_progress: bool = False) -> Dict[str, Any]:
    """Compare two modalities and return test results."""
    auc1, auc2, diff, p_value = permutation_test_auc(
        pred_dep['actual'].values, pred_dep['predicted_prob'].values,
        pred_indep['actual'].values, pred_indep['predicted_prob'].values,
        n_permutations=n_permutations,
        show_progress=show_progress
    )

    return {
        'auc_error_dependent': auc1,
        'auc_error_independent': auc2,
        'auc_difference': auc1 - auc2 if not (np.isnan(auc1) or np.isnan(auc2)) else np.nan,
        'p_value': p_value,
        'test_type': 'Permutation',
        'significant_at_0.05': 'Yes' if p_value < 0.05 else 'No' if not np.isnan(p_value) else 'N/A',
        'n_samples_dep': len(pred_dep),
        'n_samples_indep': len(pred_indep)
    }


def apply_benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> tuple:
    """Apply Benjamini-Hochberg procedure for controlling FDR."""
    p_values = np.asarray(p_values)
    n = len(p_values)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]
    n_valid = len(valid_p)

    if n_valid == 0:
        return np.full(n, np.nan), np.full(n, False)

    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]

    ranks = np.arange(1, n_valid + 1)
    adjusted_p_sorted = np.minimum(sorted_p * n_valid / ranks, 1.0)

    for i in range(n_valid - 2, -1, -1):
        adjusted_p_sorted[i] = min(adjusted_p_sorted[i], adjusted_p_sorted[i + 1])

    adjusted_p_valid = np.empty(n_valid)
    adjusted_p_valid[sorted_indices] = adjusted_p_sorted

    adjusted_p_values = np.full(n, np.nan)
    adjusted_p_values[valid_mask] = adjusted_p_valid

    significant = np.full(n, False)
    significant[valid_mask] = adjusted_p_valid < alpha

    return adjusted_p_values, significant


def analyze_dataset(dataset_name: str, results_base_path: Path, models: List[str],
                    n_permutations: int = 500) -> List[Dict[str, Any]]:
    """Analyze all models for a given dataset at global level."""
    results = []
    dataset_path = results_base_path / dataset_name

    if not dataset_path.exists():
        print(f"  Dataset path not found: {dataset_path}")
        return results

    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name.upper()} - Global Level")
    print(f"{'='*60}")

    for model in tqdm(models, desc=f"  {dataset_name} models"):
        model_path = dataset_path / model
        if not model_path.exists():
            tqdm.write(f"    {model}: path not found")
            continue

        pred_dep = load_predictions(model_path, 'error_dependent')
        pred_indep = load_predictions(model_path, 'error_independent')

        if pred_dep is None or pred_indep is None:
            tqdm.write(f"    {model}: missing predictions")
            continue

        if len(pred_dep) < 10 or len(pred_indep) < 10:
            tqdm.write(f"    {model}: insufficient samples")
            continue

        # Show sample counts to explain why it might be slow
        tqdm.write(f"    {model}: {len(pred_dep)} + {len(pred_indep)} samples, running {n_permutations} permutations...")

        result = compare_modalities(pred_dep, pred_indep, n_permutations=n_permutations, show_progress=True)
        result['dataset'] = dataset_name
        result['model'] = model
        results.append(result)

        auc_dep = f"{result['auc_error_dependent']:.4f}" if not np.isnan(result['auc_error_dependent']) else 'N/A'
        auc_indep = f"{result['auc_error_independent']:.4f}" if not np.isnan(result['auc_error_independent']) else 'N/A'
        p_val = f"{result['p_value']:.4f}" if not np.isnan(result['p_value']) else 'N/A'
        tqdm.write(f"    {model}: AUC_dep={auc_dep}, AUC_indep={auc_indep}, p={p_val}, sig={result['significant_at_0.05']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Statistical significance tests at global level')
    parser.add_argument('--datasets', nargs='+', default=['platform_a', 'platform_b'],
                        help='Datasets to analyze')
    parser.add_argument('--models', nargs='+', default=['bkt', 'irt', 'pfa'],
                        help='Models to analyze')
    parser.add_argument('--permutations', type=int, default=500,
                        help='Number of permutations for test (default: 500, more = slower but more accurate)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    results_base_path = script_dir / 'results'
    output_path = results_base_path / 'auc_significance_tests_global.csv'

    print(f"Results base path: {results_base_path}")
    print(f"Output path: {output_path}")
    print(f"Permutations: {args.permutations}")
    print(f"\nNote: Each permutation computes 2 AUC scores. Large datasets = slow.")

    all_results = []

    for dataset in args.datasets:
        all_results.extend(analyze_dataset(dataset, results_base_path, args.models,
                                           n_permutations=args.permutations))

    df_results = pd.DataFrame(all_results)

    if len(df_results) > 0:
        print(f"\n{'='*60}")
        print("Applying Benjamini-Hochberg correction")
        print(f"{'='*60}")

        p_values = df_results['p_value'].values
        adjusted_p, significant_bh = apply_benjamini_hochberg(p_values, alpha=0.05)

        df_results['p_value_BH_adjusted'] = adjusted_p
        df_results['significant_after_BH'] = np.where(
            np.isnan(adjusted_p), 'N/A',
            np.where(significant_bh, 'Yes', 'No')
        )

        column_order = ['dataset', 'model', 'auc_error_dependent', 'auc_error_independent',
                        'auc_difference', 'p_value', 'p_value_BH_adjusted', 'test_type',
                        'significant_at_0.05', 'significant_after_BH',
                        'n_samples_dep', 'n_samples_indep']
        df_results = df_results[[c for c in column_order if c in df_results.columns]]

        numeric_cols = ['auc_error_dependent', 'auc_error_independent', 'auc_difference', 'p_value', 'p_value_BH_adjusted']
        for col in numeric_cols:
            if col in df_results.columns:
                df_results[col] = df_results[col].round(6)

    df_results.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    print(f"\nSummary:")
    print(f"Total comparisons: {len(df_results)}")
    if len(df_results) > 0:
        print(f"\n  Before correction (p < 0.05):")
        print(f"    Significant: {(df_results['significant_at_0.05'] == 'Yes').sum()}")
        print(f"    Non-significant: {(df_results['significant_at_0.05'] == 'No').sum()}")

        print(f"\n  After BH correction (FDR < 0.05):")
        print(f"    Significant: {(df_results['significant_after_BH'] == 'Yes').sum()}")
        print(f"    Non-significant: {(df_results['significant_after_BH'] == 'No').sum()}")

    return df_results


if __name__ == '__main__':
    main()
