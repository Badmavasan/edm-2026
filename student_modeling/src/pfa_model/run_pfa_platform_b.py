#!/usr/bin/env python3
"""PFA training for Platform B dataset.

Trains PFA models with two modalities (error-independent, error-dependent)
at two levels: global, exercise_tag.

Uses cross-validation with student-based split (no data leakage).

Run with: python src/pfa_model/run_pfa_platform_b.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import KFold


# =============================================================================
# CONFIGURATION - Platform B specific
# =============================================================================
STUDENT_COL = "id_compte"
STATUT_COL = "status"
TIMESTAMP_COL = "date_created"
EXPECTED_TASKS_COL = "expected_task_types"
TASKS_FROM_ERRORS_COL = "task_from_errors"
EXERCISE_TAG_COL = "exercise_tag"

LEVELS = ["global", "exercise_tag"]
MODALITIES = ["error_independent", "error_dependent"]

# Training parameters
TEST_RATIO = 0.20
RANDOM_SEED = 42
THRESHOLD = 0.5
REGULARIZATION = 1.0
N_FOLDS = 5


# =============================================================================
# DATA LOADING AND TRANSFORMATION
# =============================================================================
def parse_json_list(x: Any) -> List[str]:
    """Parse JSON list stored as string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Try JSON first (double quotes)
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if v is not None]
        except json.JSONDecodeError:
            pass
        # Fallback to Python literal (single quotes)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if v is not None]
        except (ValueError, SyntaxError):
            pass
    return []


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and preprocess the Platform B dataset."""
    df = pd.read_csv(path, low_memory=False)

    required = [STUDENT_COL, STATUT_COL, EXPECTED_TASKS_COL, TASKS_FROM_ERRORS_COL, TIMESTAMP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[STATUT_COL] = df[STATUT_COL].astype(str).str.strip().str.lower()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

    return df


def transform_to_pfa_format(
    df: pd.DataFrame,
    modality: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Transform data to PFA format with success/failure counts."""
    df = df.sort_values([STUDENT_COL, TIMESTAMP_COL]).reset_index(drop=True)

    all_skills = set()
    for _, row in df.iterrows():
        expected_tasks = parse_json_list(row[EXPECTED_TASKS_COL])
        error_tasks = parse_json_list(row[TASKS_FROM_ERRORS_COL])
        all_skills.update(expected_tasks)
        all_skills.update(error_tasks)

    skill_to_idx = {skill: idx for idx, skill in enumerate(sorted(all_skills))}

    student_skill_success = {}
    student_skill_failure = {}

    records = []

    for _, row in df.iterrows():
        student_id = row[STUDENT_COL]
        if pd.isna(student_id):
            continue
        student_id = str(student_id)
        statut = str(row[STATUT_COL]).strip().lower()
        expected_tasks = parse_json_list(row[EXPECTED_TASKS_COL])
        error_tasks = parse_json_list(row[TASKS_FROM_ERRORS_COL])

        if modality == "error_independent":
            correct_val = 1 if statut == "ok" else 0
            for skill in expected_tasks:
                if skill not in skill_to_idx:
                    continue

                key = (student_id, skill)
                success_count = student_skill_success.get(key, 0)
                failure_count = student_skill_failure.get(key, 0)

                records.append({
                    "student_id": student_id,
                    "skill": skill,
                    "skill_idx": skill_to_idx[skill],
                    "correct": correct_val,
                    "success_count": success_count,
                    "failure_count": failure_count,
                })

                if correct_val == 1:
                    student_skill_success[key] = success_count + 1
                else:
                    student_skill_failure[key] = failure_count + 1

        else:
            if statut == "ok":
                for skill in expected_tasks:
                    if skill not in skill_to_idx:
                        continue

                    key = (student_id, skill)
                    success_count = student_skill_success.get(key, 0)
                    failure_count = student_skill_failure.get(key, 0)

                    records.append({
                        "student_id": student_id,
                        "skill": skill,
                        "skill_idx": skill_to_idx[skill],
                        "correct": 1,
                        "success_count": success_count,
                        "failure_count": failure_count,
                    })

                    student_skill_success[key] = success_count + 1
            else:
                for skill in error_tasks:
                    if skill not in skill_to_idx:
                        continue

                    key = (student_id, skill)
                    success_count = student_skill_success.get(key, 0)
                    failure_count = student_skill_failure.get(key, 0)

                    records.append({
                        "student_id": student_id,
                        "skill": skill,
                        "skill_idx": skill_to_idx[skill],
                        "correct": 0,
                        "success_count": success_count,
                        "failure_count": failure_count,
                    })

                    student_skill_failure[key] = failure_count + 1

    return pd.DataFrame(records), skill_to_idx


def build_pfa_features(df: pd.DataFrame, n_skills: int) -> np.ndarray:
    """Build PFA feature matrix."""
    n_samples = len(df)
    X = np.zeros((n_samples, n_skills + 2), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        skill_idx = int(row["skill_idx"])
        X[i, skill_idx] = 1.0
        X[i, n_skills] = row["success_count"]
        X[i, n_skills + 1] = row["failure_count"]

    return X


# =============================================================================
# CROSS-VALIDATION WITH FULL METRICS
# =============================================================================
def cross_validate_pfa(pfa_df: pd.DataFrame, skill_to_idx: Dict[str, int]) -> Dict[str, Any]:
    """Perform k-fold cross-validation with student-based split.

    Returns comprehensive metrics including per-fold data.
    """
    # Filter out rows with NaN in essential columns
    pfa_df = pfa_df[
        pd.notna(pfa_df["correct"]) &
        pd.notna(pfa_df["student_id"]) &
        pd.notna(pfa_df["skill_idx"])
    ].copy()

    # Ensure correct column is numeric
    pfa_df["correct"] = pd.to_numeric(pfa_df["correct"], errors="coerce")
    pfa_df = pfa_df[pd.notna(pfa_df["correct"])].copy()

    empty_result = {
        "auc": None, "auc_std": None, "accuracy": None, "f1": None, "precision": None, "recall": None,
        "y_true": [], "y_prob": [], "fold_metrics": [],
        "total_train_samples": 0, "total_test_samples": 0
    }

    if pfa_df.empty:
        return empty_result

    unique_students = pfa_df["student_id"].unique()
    n_students = len(unique_students)
    n_skills = len(skill_to_idx)

    n_folds = min(N_FOLDS, n_students)
    if n_students < 2:
        return empty_result

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    all_y_true = []
    all_y_prob = []
    fold_metrics = []
    total_train = 0
    total_test = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_students)):
        train_students = set(unique_students[train_idx])
        test_students = set(unique_students[test_idx])

        train_df = pfa_df[pfa_df["student_id"].isin(train_students)].copy()
        test_df = pfa_df[pfa_df["student_id"].isin(test_students)].copy()

        # Filter out NaN in correct column
        train_df = train_df[pd.notna(train_df["correct"])].copy()
        test_df = test_df[pd.notna(test_df["correct"])].copy()

        n_train = len(train_df)
        n_test = len(test_df)
        total_train += n_train
        total_test += n_test

        if n_train < 20 or n_test < 10:
            fold_metrics.append({
                "fold": fold_idx + 1, "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None, "train_samples": n_train, "test_samples": n_test
            })
            continue

        if len(np.unique(train_df["correct"])) < 2:
            fold_metrics.append({
                "fold": fold_idx + 1, "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None, "train_samples": n_train, "test_samples": n_test
            })
            continue

        try:
            X_train = build_pfa_features(train_df, n_skills)
            y_train = train_df["correct"].astype(float).astype(int).values
            X_test = build_pfa_features(test_df, n_skills)
            y_test = test_df["correct"].astype(float).astype(int).values

            scaler = StandardScaler()
            X_train[:, n_skills:] = scaler.fit_transform(X_train[:, n_skills:])
            X_test[:, n_skills:] = scaler.transform(X_test[:, n_skills:])

            model = LogisticRegression(
                C=REGULARIZATION,
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)

            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)

            # Compute fold metrics
            if len(np.unique(y_test)) >= 2:
                fold_auc = roc_auc_score(y_test, y_prob)
            else:
                fold_auc = None

            fold_metrics.append({
                "fold": fold_idx + 1,
                "auc": fold_auc,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "train_samples": n_train,
                "test_samples": n_test
            })

        except Exception as e:
            print(f"    Fold {fold_idx + 1} failed: {e}")
            fold_metrics.append({
                "fold": fold_idx + 1, "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None, "train_samples": n_train, "test_samples": n_test
            })
            continue

    # Compute overall metrics
    if len(all_y_true) == 0:
        return {**empty_result, "fold_metrics": fold_metrics,
                "total_train_samples": total_train, "total_test_samples": total_test}

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # Compute mean AUC from folds
    valid_folds = [f for f in fold_metrics if f["auc"] is not None]
    auc_values = [f["auc"] for f in valid_folds]
    auc = float(np.mean(auc_values)) if auc_values else None
    auc_std = float(np.std(auc_values, ddof=1)) if len(auc_values) >= 2 else None

    return {
        "auc": auc,
        "auc_std": auc_std,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_prob": y_prob,
        "fold_metrics": fold_metrics,
        "total_train_samples": total_train,
        "total_test_samples": total_test,
    }


def save_pfa_parameters(
    pfa_df: pd.DataFrame,
    skill_to_idx: Dict[str, int],
    output_dir: Path,
) -> None:
    """Train final model on all data and save learned parameters to CSV."""
    pfa_df = pfa_df[pd.notna(pfa_df["correct"])].copy()
    pfa_df["correct"] = pd.to_numeric(pfa_df["correct"], errors="coerce")
    pfa_df = pfa_df[pd.notna(pfa_df["correct"])].copy()

    n_skills = len(skill_to_idx)

    if len(pfa_df) < 20 or len(np.unique(pfa_df["correct"])) < 2:
        return

    X = build_pfa_features(pfa_df, n_skills)
    y = pfa_df["correct"].astype(float).astype(int).values

    scaler = StandardScaler()
    X[:, n_skills:] = scaler.fit_transform(X[:, n_skills:])

    model = LogisticRegression(
        C=REGULARIZATION,
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(X, y)

    coef = model.coef_[0]
    records = []
    for skill_name, idx in sorted(skill_to_idx.items(), key=lambda x: x[1]):
        records.append({"parameter": "skill_beta", "name": skill_name, "value": coef[idx]})
    records.append({"parameter": "gamma_success", "name": "success_count", "value": coef[n_skills]})
    records.append({"parameter": "gamma_failure", "name": "failure_count", "value": coef[n_skills + 1]})
    records.append({"parameter": "intercept", "name": "intercept", "value": model.intercept_[0]})

    pd.DataFrame(records).to_csv(output_dir / "learned_parameters.csv", index=False)


# =============================================================================
# INCREMENTAL TRAINING
# =============================================================================
def split_students_no_leakage(pfa_df: pd.DataFrame) -> tuple:
    """Split data by students (no leakage)."""
    students = pfa_df["student_id"].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(students)

    n_test = int(len(students) * TEST_RATIO)
    test_students = set(students[:n_test])
    train_students = set(students[n_test:])

    train_df = pfa_df[pfa_df["student_id"].isin(train_students)].copy()
    test_df = pfa_df[pfa_df["student_id"].isin(test_students)].copy()

    return train_df, test_df


def evaluate_incremental_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_skills: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate with increasing training data.

    Returns two DataFrames:
    1. Metrics by sample count
    2. Metrics by student count
    """
    # Filter out NaN values upfront
    train_df = train_df[pd.notna(train_df["correct"])].copy()
    test_df = test_df[pd.notna(test_df["correct"])].copy()

    student_col = "student_id"

    # === SAMPLE-BASED INCREMENTAL TRAINING ===
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    sample_sizes = [s for s in sample_sizes if s <= len(train_df)]
    if len(train_df) not in sample_sizes:
        sample_sizes.append(len(train_df))

    sample_results = []

    for n_samples in tqdm(sample_sizes, desc="Incremental (samples)", leave=False):
        train_slice = train_df.head(n_samples).copy()
        n_students_in_slice = train_slice[student_col].nunique()

        if len(train_slice) < 20 or len(np.unique(train_slice["correct"])) < 2:
            sample_results.append({
                "train_samples": n_samples, "train_students": n_students_in_slice,
                "test_samples": len(test_df),
                "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None
            })
            continue

        try:
            X_train = build_pfa_features(train_slice, n_skills)
            y_train = train_slice["correct"].astype(float).astype(int).values
            X_test = build_pfa_features(test_df, n_skills)
            y_test = test_df["correct"].astype(float).astype(int).values

            scaler = StandardScaler()
            X_train[:, n_skills:] = scaler.fit_transform(X_train[:, n_skills:])
            X_test[:, n_skills:] = scaler.transform(X_test[:, n_skills:])

            model = LogisticRegression(
                C=REGULARIZATION,
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)

            if len(np.unique(y_test)) < 2:
                auc = None
            else:
                auc = roc_auc_score(y_test, y_prob)

            sample_results.append({
                "train_samples": n_samples, "train_students": n_students_in_slice,
                "test_samples": len(y_test), "auc": auc,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
            })

        except Exception:
            sample_results.append({
                "train_samples": n_samples, "train_students": n_students_in_slice,
                "test_samples": 0, "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None
            })

    # === STUDENT-BASED INCREMENTAL TRAINING ===
    unique_students = train_df[student_col].unique()
    total_students = len(unique_students)

    student_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    student_counts = [s for s in student_counts if s <= total_students]
    if total_students not in student_counts:
        student_counts.append(total_students)

    student_results = []

    for n_stud in tqdm(student_counts, desc="Incremental (students)", leave=False):
        selected_students = unique_students[:n_stud]
        train_slice = train_df[train_df[student_col].isin(selected_students)].copy()
        n_samples = len(train_slice)

        if len(train_slice) < 20 or len(np.unique(train_slice["correct"])) < 2:
            student_results.append({
                "train_students": n_stud, "train_samples": n_samples,
                "test_samples": len(test_df),
                "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None
            })
            continue

        try:
            X_train = build_pfa_features(train_slice, n_skills)
            y_train = train_slice["correct"].astype(float).astype(int).values
            X_test = build_pfa_features(test_df, n_skills)
            y_test = test_df["correct"].astype(float).astype(int).values

            scaler = StandardScaler()
            X_train[:, n_skills:] = scaler.fit_transform(X_train[:, n_skills:])
            X_test[:, n_skills:] = scaler.transform(X_test[:, n_skills:])

            model = LogisticRegression(
                C=REGULARIZATION,
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)

            if len(np.unique(y_test)) < 2:
                auc = None
            else:
                auc = roc_auc_score(y_test, y_prob)

            student_results.append({
                "train_students": n_stud, "train_samples": n_samples,
                "test_samples": len(y_test), "auc": auc,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
            })

        except Exception:
            student_results.append({
                "train_students": n_stud, "train_samples": n_samples,
                "test_samples": 0, "auc": None, "accuracy": None, "f1": None, "precision": None, "recall": None
            })

    return pd.DataFrame(sample_results), pd.DataFrame(student_results)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_roc_comparison(roc_data_list: List[Dict], output_path: Path, title_suffix: str = ""):
    """Plot ROC curves for both modalities."""
    plt.figure(figsize=(10, 8))

    colors = {"error_independent": "#2ecc71", "error_dependent": "#e74c3c"}
    labels = {"error_independent": "Error Independent", "error_dependent": "Error Dependent"}

    for data in roc_data_list:
        modality = data["modality"]
        plt.plot(
            data["fpr"], data["tpr"],
            color=colors.get(modality, "blue"),
            label=f'{labels.get(modality, modality)} (AUC = {data["auc"]:.4f})',
            linewidth=2
        )

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'PFA ROC Curve Comparison{title_suffix}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_incremental_auc(data_list: List[Dict], output_path: Path, title: str):
    """Plot AUC vs training data size for both modalities."""
    plt.figure(figsize=(12, 7))

    colors = {"error_independent": "#2ecc71", "error_dependent": "#e74c3c"}
    labels = {"error_independent": "Error Independent", "error_dependent": "Error Dependent"}

    for data in data_list:
        df = data["df"]
        name = data["name"]
        valid_df = df[df["auc"].notna()]
        if not valid_df.empty:
            plt.plot(
                valid_df["train_samples"], valid_df["auc"],
                color=colors.get(name, "blue"),
                label=labels.get(name, name),
                marker='o', linewidth=2, markersize=6
            )

    plt.xlabel('Training Samples', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print(" PFA Training - PLATFORM B Dataset")
    print("=" * 70)

    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "platform_b_dataset.csv"
    output_dir = base_path / "results" / "platform_b" / "pfa"

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        for modality in MODALITIES:
            (output_dir / level / modality).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading dataset...")
    wide_df = load_dataset(data_path)
    print(f"  Rows: {len(wide_df)}")
    print(f"  Students: {wide_df[STUDENT_COL].nunique()}")

    # Storage for all results
    global_results = []
    exercise_tag_results = []

    # ==========================================================================
    # GLOBAL LEVEL
    # ==========================================================================
    print("\n" + "=" * 50)
    print("Running Cross-Validation - Global Level")
    print("=" * 50)

    global_roc_data = []
    global_incremental_data = []

    for modality in MODALITIES:
        print(f"\n  {modality}...")

        pfa_df, skill_to_idx = transform_to_pfa_format(wide_df, modality)
        n_skills = len(skill_to_idx)

        if len(pfa_df) < 50:
            print(f"    Not enough data")
            continue

        print(f"    Samples: {len(pfa_df)}, Skills: {n_skills}")

        # Cross-validation
        cv_result = cross_validate_pfa(pfa_df, skill_to_idx)
        auc_str = f"{cv_result['auc']:.4f}" if cv_result['auc'] else "N/A"
        auc_std_str = f"{cv_result['auc_std']:.4f}" if cv_result['auc_std'] else "N/A"
        print(f"    CV AUC: {auc_str} +- {auc_std_str}")

        modality_dir = output_dir / "global" / modality

        # Save fold metrics
        fold_df = pd.DataFrame(cv_result["fold_metrics"])
        fold_df.to_csv(modality_dir / "fold_metrics.csv", index=False)

        # Save predictions
        if len(cv_result["y_true"]) > 0:
            pd.DataFrame({
                "actual": cv_result["y_true"],
                "predicted_prob": cv_result["y_prob"],
            }).to_csv(modality_dir / "predictions.csv", index=False)

        # Save learned parameters
        save_pfa_parameters(pfa_df, skill_to_idx, modality_dir)

        # Store for global summary
        global_results.append({
            "level": "global",
            "modality": modality,
            "auc": cv_result["auc"],
            "accuracy": cv_result["accuracy"],
            "f1": cv_result["f1"],
            "precision": cv_result["precision"],
            "recall": cv_result["recall"],
            "total_train_samples": cv_result["total_train_samples"],
            "total_test_samples": cv_result["total_test_samples"],
            "n_folds": len(cv_result["fold_metrics"]),
            "n_valid_folds": len([f for f in cv_result["fold_metrics"] if f["auc"] is not None]),
        })

        # ROC data
        if len(cv_result["y_true"]) > 0 and len(np.unique(cv_result["y_true"])) >= 2:
            fpr, tpr, thresholds = roc_curve(cv_result["y_true"], cv_result["y_prob"])
            global_roc_data.append({
                "modality": modality,
                "fpr": fpr,
                "tpr": tpr,
                "auc": cv_result["auc"] or 0.5,
            })
            # Save ROC data
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
                modality_dir / "roc_data.csv", index=False
            )

        # Incremental training
        print(f"    Running incremental training...")
        train_df, test_df = split_students_no_leakage(pfa_df)
        print(f"    Train: {len(train_df)}, Test: {len(test_df)}")

        if not train_df.empty and not test_df.empty:
            inc_metrics_samples, inc_metrics_students = evaluate_incremental_training(train_df, test_df, n_skills)
            inc_metrics_samples.to_csv(modality_dir / "incremental_training_metrics.csv", index=False)
            inc_metrics_students.to_csv(modality_dir / "incremental_training_metrics_by_students.csv", index=False)
            global_incremental_data.append({"name": modality, "df": inc_metrics_samples})

    # Save global accuracy summary
    if global_results:
        pd.DataFrame(global_results).to_csv(
            output_dir / "global_accuracy.csv", index=False
        )

    # Plot global ROC comparison
    if len(global_roc_data) >= 1:
        plot_roc_comparison(
            global_roc_data,
            output_dir / "global" / "roc_comparison.png",
            " (Global)"
        )

    # Plot global incremental training
    if len(global_incremental_data) >= 1:
        plot_incremental_auc(
            global_incremental_data,
            output_dir / "global" / "incremental_training_auc_comparison.png",
            "PFA: AUC vs Training Data Size (Platform B - Global)"
        )
        # Also save to root
        plot_incremental_auc(
            global_incremental_data,
            output_dir / "incremental_training_auc.png",
            "PFA: AUC vs Training Data Size (Platform B - Global)"
        )

    # ==========================================================================
    # EXERCISE TAG LEVEL
    # ==========================================================================
    print("\n" + "=" * 50)
    print("Running Cross-Validation - Exercise Tag Level")
    print("=" * 50)

    exercise_tags = sorted(wide_df[EXERCISE_TAG_COL].dropna().unique())

    for ex_tag in tqdm(exercise_tags, desc="Exercise Tags"):
        subset_df = wide_df[wide_df[EXERCISE_TAG_COL] == ex_tag].copy()
        ex_tag_roc_data = []

        for modality in MODALITIES:
            pfa_df, skill_to_idx = transform_to_pfa_format(subset_df, modality)

            if len(pfa_df) < 50:
                continue

            cv_result = cross_validate_pfa(pfa_df, skill_to_idx)

            modality_dir = output_dir / "exercise_tag" / modality / str(ex_tag)
            modality_dir.mkdir(parents=True, exist_ok=True)

            # Save fold metrics
            fold_df = pd.DataFrame(cv_result["fold_metrics"])
            fold_df.to_csv(modality_dir / "fold_metrics.csv", index=False)

            # Save predictions
            if len(cv_result["y_true"]) > 0:
                pd.DataFrame({
                    "actual": cv_result["y_true"],
                    "predicted_prob": cv_result["y_prob"],
                }).to_csv(modality_dir / "predictions.csv", index=False)

            # Save learned parameters
            save_pfa_parameters(pfa_df, skill_to_idx, modality_dir)

            # Store results
            exercise_tag_results.append({
                "exercise_tag": ex_tag,
                "modality": modality,
                "auc": cv_result["auc"],
                "accuracy": cv_result["accuracy"],
                "f1": cv_result["f1"],
                "precision": cv_result["precision"],
                "recall": cv_result["recall"],
                "total_train_samples": cv_result["total_train_samples"],
                "total_test_samples": cv_result["total_test_samples"],
            })

            # ROC data
            if len(cv_result["y_true"]) > 0 and len(np.unique(cv_result["y_true"])) >= 2:
                fpr, tpr, thresholds = roc_curve(cv_result["y_true"], cv_result["y_prob"])
                ex_tag_roc_data.append({
                    "modality": modality,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": cv_result["auc"] or 0.5,
                })
                pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
                    modality_dir / "roc_data.csv", index=False
                )

        # Plot ROC comparison for this exercise tag
        if len(ex_tag_roc_data) >= 1:
            plot_roc_comparison(
                ex_tag_roc_data,
                output_dir / "exercise_tag" / f"roc_comparison_{ex_tag}.png",
                f" (Exercise: {ex_tag})"
            )

    # Save per exercise accuracy
    if exercise_tag_results:
        pd.DataFrame(exercise_tag_results).to_csv(
            output_dir / "per_exercise_accuracy.csv", index=False
        )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" PFA Training Complete - PLATFORM B")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - global_accuracy.csv")
    print("  - per_exercise_accuracy.csv")
    print("  - incremental_training_auc.png")
    print("  - global/roc_comparison.png")
    print("  - global/*/fold_metrics.csv")
    print("  - global/*/incremental_training_metrics.csv")
    print("  - exercise_tag/roc_comparison_*.png")


if __name__ == "__main__":
    main()