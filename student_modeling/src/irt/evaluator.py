"""Metrics computation and evaluation for IRT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from tqdm import tqdm

if TYPE_CHECKING:
    from .config import IRTConfig


@dataclass
class Metrics:
    """Evaluation metrics."""

    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    n_samples: int


@dataclass
class IncrementalMetrics:
    """Metrics at a specific training sample size."""

    train_samples: int
    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    test_samples: int


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Compute evaluation metrics."""
    n_samples = len(y_true)

    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return Metrics(
            auc=None, accuracy=None, f1=None,
            precision=None, recall=None, n_samples=n_samples
        )

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    return Metrics(
        auc=auc, accuracy=accuracy, f1=f1,
        precision=precision, recall=recall, n_samples=n_samples
    )


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Compute ROC curve data."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    })


@dataclass
class CVFoldMetrics:
    """Metrics for a single cross-validation fold."""

    fold: int
    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    n_train: int
    n_test: int


def cross_validate_irt(
    irt_df: pd.DataFrame,
    n_students: int,
    n_items: int,
    config: "IRTConfig",
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Perform k-fold cross-validation for IRT model.

    Uses interaction-based split (IRT needs to see students in training).

    Args:
        irt_df: Full IRT data in long format.
        n_students: Number of students.
        n_items: Number of items.
        config: IRT configuration.
        n_folds: Number of cross-validation folds.

    Returns:
        Tuple of (fold_metrics_df, final_metrics_df, all_y_true, all_y_prob):
        - fold_metrics_df: Metrics for each fold
        - final_metrics_df: Mean metrics across folds
        - all_y_true: Concatenated true labels from all folds
        - all_y_prob: Concatenated predicted probabilities from all folds
    """
    from .model import IRTModel
    from .trainer import evaluate_irt_model

    # Adjust n_folds if there are fewer samples than requested folds
    n_samples = len(irt_df)
    if n_samples < n_folds:
        n_folds = max(2, n_samples)  # At least 2 folds for CV
        print(f"  Adjusted n_folds to {n_folds} due to limited samples ({n_samples})")

    # If still not enough samples for even 2-fold CV, return empty results
    if n_samples < 2:
        empty_fold_df = pd.DataFrame(columns=["fold", "auc", "accuracy", "f1", "precision", "recall", "n_train", "n_test"])
        empty_final_df = pd.DataFrame([{
            "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None,
            "n_folds": n_folds, "n_valid_folds": 0, "total_samples": len(irt_df)
        }])
        return empty_fold_df, empty_final_df, np.array([]), np.array([])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)

    fold_results: List[CVFoldMetrics] = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(
        kf.split(irt_df), total=n_folds, desc="Cross-validation"
    )):
        train_df = irt_df.iloc[train_idx].copy()
        test_df = irt_df.iloc[test_idx].copy()

        if len(train_df) < 10 or len(test_df) < 10:
            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                precision=None, recall=None, n_train=len(train_df), n_test=len(test_df)
            ))
            continue

        # Check both classes
        if len(np.unique(train_df["correct"])) < 2:
            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                precision=None, recall=None, n_train=len(train_df), n_test=len(test_df)
            ))
            continue

        try:
            # Create and train model
            model = IRTModel(
                n_students=n_students,
                n_items=n_items,
                learning_rate=config.learning_rate,
                regularization=config.regularization,
            )

            model.fit(
                train_df["student_idx"].values.astype(np.int64),
                train_df["item_idx"].values.astype(np.int64),
                train_df["correct"].values.astype(np.float64),
                max_iter=config.max_iter,
            )

            # Evaluate
            y_true, y_prob = evaluate_irt_model(model, test_df)

            if len(y_true) == 0:
                fold_results.append(CVFoldMetrics(
                    fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                    precision=None, recall=None, n_train=len(train_df), n_test=0
                ))
                continue

            # Store for ROC curve
            all_y_true.extend(y_true)
            all_y_prob.extend(y_prob)

            # Compute metrics
            metrics = compute_metrics(y_true, y_prob, config.threshold)

            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1,
                auc=metrics.auc,
                accuracy=metrics.accuracy,
                f1=metrics.f1,
                precision=metrics.precision,
                recall=metrics.recall,
                n_train=len(train_df),
                n_test=len(test_df),
            ))

        except Exception as e:
            print(f"  Warning: Fold {fold_idx + 1} failed: {e}")
            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                precision=None, recall=None, n_train=len(train_df), n_test=len(test_df)
            ))

    # Create fold metrics DataFrame
    fold_metrics_df = pd.DataFrame([asdict(r) for r in fold_results])

    # Compute mean metrics (excluding None values)
    mean_metrics = {
        "auc": fold_metrics_df["auc"].dropna().mean() if not fold_metrics_df["auc"].dropna().empty else None,
        "accuracy": fold_metrics_df["accuracy"].dropna().mean() if not fold_metrics_df["accuracy"].dropna().empty else None,
        "f1": fold_metrics_df["f1"].dropna().mean() if not fold_metrics_df["f1"].dropna().empty else None,
        "precision": fold_metrics_df["precision"].dropna().mean() if not fold_metrics_df["precision"].dropna().empty else None,
        "recall": fold_metrics_df["recall"].dropna().mean() if not fold_metrics_df["recall"].dropna().empty else None,
        "n_folds": n_folds,
        "n_valid_folds": fold_metrics_df["auc"].notna().sum(),
        "total_samples": len(irt_df),
    }
    final_metrics_df = pd.DataFrame([mean_metrics])

    return fold_metrics_df, final_metrics_df, np.array(all_y_true), np.array(all_y_prob)


def evaluate_incremental_training(
    irt_df: pd.DataFrame,
    n_students: int,
    n_items: int,
    config: "IRTConfig",
    sample_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Evaluate IRT model with incrementally increasing training data.

    Uses interaction-based split to allow proper IRT evaluation.

    Args:
        irt_df: Full IRT data in long format.
        n_students: Number of students.
        n_items: Number of items.
        config: IRT configuration.
        sample_sizes: Custom sample sizes. If None, uses config defaults.

    Returns:
        DataFrame with metrics at each training sample size.
    """
    from .model import IRTModel
    from .trainer import evaluate_irt_model
    from .data_transformer import split_by_interaction

    # Split by interaction (not by student)
    train_df, test_df = split_by_interaction(irt_df, config.test_ratio, config.random_seed)

    total_train_samples = len(train_df)

    if sample_sizes is None:
        sample_sizes = config.sample_sizes
        sample_sizes = [s for s in sample_sizes if s <= total_train_samples]
        if total_train_samples not in sample_sizes:
            sample_sizes.append(total_train_samples)

    results: List[IncrementalMetrics] = []

    for n_samples in tqdm(sample_sizes, desc="Incremental training"):
        # Take first n_samples from training data
        subset_df = train_df.iloc[:n_samples].copy()

        if len(subset_df) < 10:
            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=None, accuracy=None, f1=None,
                    precision=None, recall=None, test_samples=0,
                )
            )
            continue

        try:
            # Check if we have both classes
            if len(np.unique(subset_df["correct"])) < 2:
                results.append(
                    IncrementalMetrics(
                        train_samples=n_samples,
                        auc=None, accuracy=None, f1=None,
                        precision=None, recall=None, test_samples=0,
                    )
                )
                continue

            # Create and train model
            model = IRTModel(
                n_students=n_students,
                n_items=n_items,
                learning_rate=config.learning_rate,
                regularization=config.regularization,
            )

            model.fit(
                subset_df["student_idx"].values.astype(np.int64),
                subset_df["item_idx"].values.astype(np.int64),
                subset_df["correct"].values.astype(np.float64),
                max_iter=config.max_iter,
            )

            # Evaluate on test data
            y_true, y_prob = evaluate_irt_model(model, test_df)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                results.append(
                    IncrementalMetrics(
                        train_samples=n_samples,
                        auc=None, accuracy=None, f1=None,
                        precision=None, recall=None, test_samples=len(y_true),
                    )
                )
                continue

            metrics = compute_metrics(y_true, y_prob, config.threshold)

            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=metrics.auc,
                    accuracy=metrics.accuracy,
                    f1=metrics.f1,
                    precision=metrics.precision,
                    recall=metrics.recall,
                    test_samples=len(y_true),
                )
            )

        except Exception as e:
            print(f"  Warning: Training failed at {n_samples} samples: {e}")
            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=None, accuracy=None, f1=None,
                    precision=None, recall=None, test_samples=0,
                )
            )

    return pd.DataFrame([asdict(r) for r in results])
