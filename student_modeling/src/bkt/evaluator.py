"""Metrics computation and evaluation for BKT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from pyBKT.models import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

if TYPE_CHECKING:
    from src.bkt.config import BKTConfig


@dataclass
class Metrics:
    """Evaluation metrics."""

    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    n_samples: int


@dataclass
class StepMetrics:
    """Metrics at a specific opportunity step."""

    step: int
    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    train_rows: int
    eval_rows: int


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Metrics dataclass with AUC, accuracy, F1, and sample count.
    """
    n_samples = len(y_true)

    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return Metrics(auc=None, accuracy=None, f1=None, n_samples=n_samples)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return Metrics(auc=auc, accuracy=accuracy, f1=f1, n_samples=n_samples)


def evaluate_model(
    model: Model,
    test_df: pd.DataFrame,
    config: BKTConfig,
) -> Tuple[Metrics, pd.DataFrame]:
    """Evaluate a trained BKT model on test data.

    Args:
        model: Trained pyBKT model.
        test_df: Test data in pyBKT format.
        config: BKT configuration.

    Returns:
        Tuple of (overall metrics, predictions DataFrame).
    """
    if test_df.empty:
        return Metrics(auc=None, accuracy=None, f1=None, n_samples=0), pd.DataFrame()

    preds = model.predict(data=test_df)

    if "correct_predictions" not in preds.columns:
        raise RuntimeError("Missing correct_predictions in model output")

    # Filter out NaN values (pyBKT can produce NaN for sparse skills)
    valid_mask = ~preds["correct"].isna() & ~preds["correct_predictions"].isna()
    valid_preds = preds[valid_mask].copy()

    if len(valid_preds) == 0:
        return Metrics(auc=None, accuracy=None, f1=None, n_samples=0), pd.DataFrame()

    y_true = valid_preds["correct"].to_numpy(dtype=int)
    y_prob = valid_preds["correct_predictions"].to_numpy(dtype=float)

    metrics = compute_metrics(y_true, y_prob, config.threshold)

    # Predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "student_id": valid_preds["Anon Student Id"],
            "skill_name": valid_preds["skill_name"],
            "order_id": valid_preds["order_id"],
            "actual": y_true,
            "predicted_prob": y_prob,
        }
    )

    return metrics, predictions_df


def evaluate_over_steps(
    model: Model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BKTConfig,
) -> pd.DataFrame:
    """Evaluate AUC at each opportunity step (order_id).

    At step t:
    - Train on rows with order_id <= t
    - Predict on test rows with order_id <= t
    - Evaluate only rows where order_id == t

    Args:
        model: Trained pyBKT model (for reference, we retrain at each step).
        train_df: Training data.
        test_df: Test data.
        config: BKT configuration.

    Returns:
        DataFrame with step-by-step metrics.
    """
    if train_df.empty or test_df.empty:
        return pd.DataFrame(columns=["step", "auc", "accuracy", "f1", "train_rows", "eval_rows"])

    max_train = int(train_df["order_id"].max())
    max_test = int(test_df["order_id"].max())
    p90 = int(np.percentile(test_df["order_id"], config.order_id_percentile_for_max_steps))
    max_steps = min(p90, max_train, max_test)

    results: List[StepMetrics] = []

    for t in range(1, max_steps + 1):
        train_slice = train_df[train_df["order_id"] <= t].copy()
        test_prefix = test_df[test_df["order_id"] <= t].copy()

        if train_slice.empty or test_prefix.empty:
            break

        step_model = Model(seed=config.pybkt_seed, num_fits=config.pybkt_num_fits)
        step_model.fit(data=train_slice)

        preds = step_model.predict(data=test_prefix)
        eval_rows = preds[preds["order_id"] == t]

        if len(eval_rows) < config.min_test_rows_per_step:
            continue

        # Filter out NaN values (pyBKT can produce NaN for sparse skills)
        valid_mask = ~eval_rows["correct"].isna() & ~eval_rows["correct_predictions"].isna()
        valid_rows = eval_rows[valid_mask]

        if len(valid_rows) < config.min_test_rows_per_step:
            continue

        y_true = valid_rows["correct"].to_numpy(dtype=int)
        y_prob = valid_rows["correct_predictions"].to_numpy(dtype=float)

        m = compute_metrics(y_true, y_prob, config.threshold)
        results.append(
            StepMetrics(
                step=t,
                auc=m.auc,
                accuracy=m.accuracy,
                f1=m.f1,
                train_rows=len(train_slice),
                eval_rows=len(eval_rows),
            )
        )

    return pd.DataFrame([asdict(r) for r in results])


def compute_avg_pred_by_occurrence(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute average predicted probability by skill occurrence index.

    Args:
        predictions_df: DataFrame with 'order_id', 'predicted_prob', 'actual'.

    Returns:
        DataFrame with columns: occurrence_index, avg_predicted_prob, avg_actual, n_samples.
    """
    if predictions_df.empty:
        return pd.DataFrame(
            columns=["occurrence_index", "avg_predicted_prob", "avg_actual", "n_samples"]
        )

    grouped = (
        predictions_df.groupby("order_id")
        .agg(
            avg_predicted_prob=("predicted_prob", "mean"),
            avg_actual=("actual", "mean"),
            n_samples=("predicted_prob", "count"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={"order_id": "occurrence_index"})
    return grouped


def get_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Get ROC curve data for plotting.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.

    Returns:
        DataFrame with fpr, tpr, threshold columns.
    """
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


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


def cross_validate_bkt(
    bkt_df: pd.DataFrame,
    config: "BKTConfig",
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Perform stratified k-fold cross-validation for BKT model.

    Uses student-based split to avoid data leakage, with stratification
    on per-student majority label to ensure both classes in each fold.

    Args:
        bkt_df: Full BKT data in pyBKT format.
        config: BKT configuration.
        n_folds: Number of cross-validation folds.

    Returns:
        Tuple of (fold_metrics_df, final_metrics_df, all_y_true, all_y_prob):
        - fold_metrics_df: Metrics for each fold
        - final_metrics_df: Mean metrics across folds
        - all_y_true: Concatenated true labels from all folds
        - all_y_prob: Concatenated predicted probabilities from all folds
    """
    from tqdm import tqdm

    # Coerce correct column to numeric and drop rows that are still NaN
    bkt_df = bkt_df.copy()
    bkt_df["correct"] = pd.to_numeric(bkt_df["correct"], errors="coerce")
    bkt_df = bkt_df[bkt_df["correct"].notna()].copy()

    # Get unique students for student-based CV
    unique_students = bkt_df["Anon Student Id"].unique()

    # Adjust n_folds if there are fewer students than requested folds
    n_students = len(unique_students)
    if n_students < n_folds:
        n_folds = max(2, n_students)  # At least 2 folds for CV
        print(f"  Adjusted n_folds to {n_folds} due to limited students ({n_students})")

    # If still not enough students for even 2-fold CV, return empty results
    if n_students < 2:
        empty_fold_df = pd.DataFrame(columns=["fold", "auc", "accuracy", "f1", "precision", "recall", "n_train", "n_test"])
        empty_final_df = pd.DataFrame([{
            "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None,
            "n_folds": n_folds, "n_valid_folds": 0, "total_samples": len(bkt_df)
        }])
        return empty_fold_df, empty_final_df, np.array([]), np.array([])

    # Compute per-student majority label for stratification so each fold
    # gets a mix of "mostly-correct" and "mostly-incorrect" students.
    student_majority_label = (
        bkt_df.groupby("Anon Student Id")["correct"]
        .mean()
        .reindex(unique_students)
        .fillna(0.0)
        .round()
        .astype(int)
        .values
    )

    # Use StratifiedKFold if stratification is possible, else plain KFold
    if len(np.unique(student_majority_label)) < 2:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
        cv_iter = kf.split(unique_students)
    else:
        min_class_count = min(np.bincount(student_majority_label))
        if min_class_count < n_folds:
            n_folds = max(2, min_class_count)
            print(f"  Adjusted n_folds to {n_folds} due to minority class size ({min_class_count})")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)
        cv_iter = kf.split(unique_students, student_majority_label)

    fold_results: List[CVFoldMetrics] = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_student_idx, test_student_idx) in enumerate(tqdm(
        cv_iter, total=n_folds, desc="Cross-validation"
    )):
        train_students = set(unique_students[train_student_idx])
        test_students = set(unique_students[test_student_idx])

        train_df = bkt_df[bkt_df["Anon Student Id"].isin(train_students)].copy()
        test_df = bkt_df[bkt_df["Anon Student Id"].isin(test_students)].copy()

        if len(train_df) < 10 or len(test_df) < 10:
            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                precision=None, recall=None, n_train=len(train_df), n_test=len(test_df)
            ))
            continue

        # Filter test to skills in training
        train_skills = set(train_df["skill_name"].unique())
        test_df = test_df[test_df["skill_name"].isin(train_skills)].copy()

        if len(test_df) < 10:
            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                precision=None, recall=None, n_train=len(train_df), n_test=0
            ))
            continue

        try:
            # Train model
            model = Model(seed=config.pybkt_seed, num_fits=config.pybkt_num_fits)
            model.fit(data=train_df)

            # Predict
            preds = model.predict(data=test_df)

            if "correct_predictions" not in preds.columns:
                fold_results.append(CVFoldMetrics(
                    fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                    precision=None, recall=None, n_train=len(train_df), n_test=len(test_df)
                ))
                continue

            # Filter out NaN predictions (pyBKT can produce NaN for sparse skills)
            valid_mask = pd.notna(preds["correct"]) & pd.notna(preds["correct_predictions"])
            valid_preds = preds[valid_mask].copy()

            if len(valid_preds) < 10:
                fold_results.append(CVFoldMetrics(
                    fold=fold_idx + 1, auc=None, accuracy=None, f1=None,
                    precision=None, recall=None, n_train=len(train_df), n_test=len(valid_preds)
                ))
                continue

            y_true = valid_preds["correct"].astype(float).astype(int).to_numpy()
            y_prob = valid_preds["correct_predictions"].astype(float).to_numpy()

            # Store for ROC curve
            all_y_true.extend(y_true)
            all_y_prob.extend(y_prob)

            # Compute metrics
            metrics = compute_metrics(y_true, y_prob, config.threshold)

            # Additional metrics
            y_pred = (y_prob >= config.threshold).astype(int)
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))

            fold_results.append(CVFoldMetrics(
                fold=fold_idx + 1,
                auc=metrics.auc,
                accuracy=metrics.accuracy,
                f1=metrics.f1,
                precision=precision,
                recall=recall,
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
        "total_samples": len(bkt_df),
    }
    final_metrics_df = pd.DataFrame([mean_metrics])

    return fold_metrics_df, final_metrics_df, np.array(all_y_true), np.array(all_y_prob)


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


def evaluate_incremental_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BKTConfig,
    sample_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Evaluate BKT model performance with incrementally increasing training data.

    Trains with 10, 100, 500, 1000, 5000, 10000, ... samples and evaluates on test set.

    Args:
        train_df: Training data in pyBKT format.
        test_df: Test data in pyBKT format.
        config: BKT configuration.
        sample_sizes: Custom sample sizes to evaluate. If None, uses default progression.

    Returns:
        DataFrame with metrics at each training sample size.
    """
    from sklearn.metrics import precision_score, recall_score
    from tqdm import tqdm

    if train_df.empty or test_df.empty:
        return pd.DataFrame(
            columns=[
                "train_samples",
                "auc",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "test_samples",
            ]
        )

    # Generate sample sizes with more granularity
    if sample_sizes is None:
        sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        # Filter to only include sizes <= training data size
        sample_sizes = [s for s in sample_sizes if s <= len(train_df)]
        # Add the full training set size if not already included
        if len(train_df) not in sample_sizes:
            sample_sizes.append(len(train_df))

    results: List[IncrementalMetrics] = []

    for n_samples in tqdm(sample_sizes, desc="Incremental training"):
        # Take first n_samples from training data (preserving order)
        train_slice = train_df.head(n_samples).copy()

        if train_slice.empty:
            continue

        # Check skill coverage
        train_skills = set(train_slice["skill_name"].unique())

        # Filter test to only include skills present in training
        test_slice = test_df[test_df["skill_name"].isin(train_skills)].copy()

        if test_slice.empty:
            # Record with None metrics if no test data
            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=None,
                    accuracy=None,
                    f1=None,
                    precision=None,
                    recall=None,
                    test_samples=0,
                )
            )
            continue

        try:
            # Train model
            model = Model(seed=config.pybkt_seed, num_fits=config.pybkt_num_fits)
            model.fit(data=train_slice)

            # Predict on test
            preds = model.predict(data=test_slice)

            if "correct_predictions" not in preds.columns:
                results.append(
                    IncrementalMetrics(
                        train_samples=n_samples,
                        auc=None,
                        accuracy=None,
                        f1=None,
                        precision=None,
                        recall=None,
                        test_samples=len(test_slice),
                    )
                )
                continue

            # Filter out NaN values (pyBKT can produce NaN for sparse skills)
            valid_mask = ~preds["correct"].isna() & ~preds["correct_predictions"].isna()
            valid_preds = preds[valid_mask]

            if len(valid_preds) < 10:
                results.append(
                    IncrementalMetrics(
                        train_samples=n_samples,
                        auc=None,
                        accuracy=None,
                        f1=None,
                        precision=None,
                        recall=None,
                        test_samples=len(valid_preds),
                    )
                )
                continue

            y_true = valid_preds["correct"].to_numpy(dtype=int)
            y_prob = valid_preds["correct_predictions"].to_numpy(dtype=float)

            # Compute metrics
            if len(np.unique(y_true)) < 2:
                auc = None
            else:
                try:
                    auc = float(roc_auc_score(y_true, y_prob))
                except ValueError:
                    auc = None

            y_pred = (y_prob >= config.threshold).astype(int)
            accuracy = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))

            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=auc,
                    accuracy=accuracy,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    test_samples=len(test_slice),
                )
            )

        except Exception as e:
            # Record with None metrics if training fails
            print(f"  Warning: Training failed at {n_samples} samples: {e}")
            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=None,
                    accuracy=None,
                    f1=None,
                    precision=None,
                    recall=None,
                    test_samples=0,
                )
            )

    return pd.DataFrame([asdict(r) for r in results])
