"""Metrics computation and evaluation for PFA models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

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
    from .config import PFAConfig


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
    if len(np.unique(y_true)) < 2:
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


def cross_validate_pfa(
    pfa_df: pd.DataFrame,
    skill_to_idx: dict,
    config: "PFAConfig",
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Perform k-fold cross-validation for PFA model.

    Args:
        pfa_df: Full PFA data in long format.
        skill_to_idx: Skill to index mapping.
        config: PFA configuration.
        n_folds: Number of cross-validation folds.

    Returns:
        Tuple of (fold_metrics_df, final_metrics_df, all_y_true, all_y_prob):
        - fold_metrics_df: Metrics for each fold
        - final_metrics_df: Mean metrics across folds
        - all_y_true: Concatenated true labels from all folds
        - all_y_prob: Concatenated predicted probabilities from all folds
    """
    from .trainer import build_pfa_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_skills = len(skill_to_idx)

    # Adjust n_folds if there are fewer samples than requested folds
    n_samples = len(pfa_df)
    if n_samples < n_folds:
        n_folds = max(2, n_samples)  # At least 2 folds for CV
        print(f"  Adjusted n_folds to {n_folds} due to limited samples ({n_samples})")

    # If still not enough samples for even 2-fold CV, return empty results
    if n_samples < 2:
        empty_fold_df = pd.DataFrame(columns=["fold", "auc", "accuracy", "f1", "precision", "recall", "n_train", "n_test"])
        empty_final_df = pd.DataFrame([{
            "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None,
            "n_folds": n_folds, "n_valid_folds": 0, "total_samples": len(pfa_df)
        }])
        return empty_fold_df, empty_final_df, np.array([]), np.array([])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)

    fold_results: List[CVFoldMetrics] = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(
        kf.split(pfa_df), total=n_folds, desc="Cross-validation"
    )):
        train_df = pfa_df.iloc[train_idx].copy()
        test_df = pfa_df.iloc[test_idx].copy()

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
            # Build features
            X_train = build_pfa_features(train_df, n_skills)
            y_train = train_df["correct"].values
            X_test = build_pfa_features(test_df, n_skills)
            y_test = test_df["correct"].values

            # Scale
            scaler = StandardScaler()
            X_train[:, n_skills:] = scaler.fit_transform(X_train[:, n_skills:])
            X_test[:, n_skills:] = scaler.transform(X_test[:, n_skills:])

            # Train
            model = LogisticRegression(
                C=config.regularization,
                max_iter=1000,
                random_state=config.random_seed,
                solver="lbfgs",
            )
            model.fit(X_train, y_train)

            # Predict
            y_prob = model.predict_proba(X_test)[:, 1]

            # Store for ROC curve
            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)

            # Compute metrics
            metrics = compute_metrics(y_test, y_prob, config.threshold)

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
        "total_samples": len(pfa_df),
    }
    final_metrics_df = pd.DataFrame([mean_metrics])

    return fold_metrics_df, final_metrics_df, np.array(all_y_true), np.array(all_y_prob)


def evaluate_incremental_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    skill_to_idx: dict,
    config: "PFAConfig",
    sample_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Evaluate PFA model with incrementally increasing training data.

    Args:
        train_df: All training data.
        test_df: Test data (fixed).
        skill_to_idx: Skill to index mapping.
        config: PFA configuration.
        sample_sizes: Custom sample sizes. If None, uses config defaults.

    Returns:
        DataFrame with metrics at each training sample size.
    """
    from .trainer import train_pfa_model, evaluate_pfa_model, build_pfa_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_skills = len(skill_to_idx)
    total_train_samples = len(train_df)

    if sample_sizes is None:
        sample_sizes = config.sample_sizes
        sample_sizes = [s for s in sample_sizes if s <= total_train_samples]
        if total_train_samples not in sample_sizes:
            sample_sizes.append(total_train_samples)

    results: List[IncrementalMetrics] = []

    for n_samples in tqdm(sample_sizes, desc="Incremental training"):
        # Take first n_samples
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
            # Build features
            X_train = build_pfa_features(subset_df, n_skills)
            y_train = subset_df["correct"].values

            # Check if we have both classes
            if len(np.unique(y_train)) < 2:
                results.append(
                    IncrementalMetrics(
                        train_samples=n_samples,
                        auc=None, accuracy=None, f1=None,
                        precision=None, recall=None, test_samples=0,
                    )
                )
                continue

            X_test = build_pfa_features(test_df, n_skills)
            y_test = test_df["correct"].values

            # Scale
            scaler = StandardScaler()
            X_train[:, n_skills:] = scaler.fit_transform(X_train[:, n_skills:])
            X_test[:, n_skills:] = scaler.transform(X_test[:, n_skills:])

            # Train
            model = LogisticRegression(
                C=config.regularization,
                max_iter=1000,
                random_state=config.random_seed,
                solver="lbfgs",
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_prob, config.threshold)

            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=metrics.auc,
                    accuracy=metrics.accuracy,
                    f1=metrics.f1,
                    precision=metrics.precision,
                    recall=metrics.recall,
                    test_samples=len(y_test),
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
