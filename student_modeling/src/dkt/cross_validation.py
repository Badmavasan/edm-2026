"""Cross-validation for DKT models with student-stratified k-fold splits."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from .data_transformer import transform_to_sequences
from .trainer import train_dkt_model, evaluate_dkt_model

if TYPE_CHECKING:
    import pandas as pd
    from .config import DKTConfig


def cross_validate_dkt(
    df: pd.DataFrame,
    config: DKTConfig,
    modality: str,
    hidden_dim: int,
    n_folds: int = 5,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Run student-stratified k-fold cross-validation for DKT.

    Args:
        df: Raw DataFrame with student interactions.
        config: DKT configuration.
        modality: "error_independent" or "error_dependent".
        hidden_dim: Hidden dimension for the LSTM.
        n_folds: Number of folds.
        device: Torch device.

    Returns:
        Dict with aggregated metrics, per-fold metrics, and concatenated predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform data to sequences (single call for consistent skill mapping)
    sequences, skill_to_idx, student_ids = transform_to_sequences(df, config, modality)
    num_skills = len(skill_to_idx)

    # Build student -> sequence indices mapping
    student_to_seq_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sid in enumerate(student_ids):
        student_to_seq_indices[sid].append(idx)

    unique_students = np.array(list(student_to_seq_indices.keys()))

    if len(unique_students) < n_folds:
        print(f"  Warning: only {len(unique_students)} students, need >= {n_folds} for CV. Skipping.")
        return {
            "auc": None, "auc_std": None,
            "accuracy": None, "accuracy_std": None,
            "f1": None, "f1_std": None,
            "precision": None, "precision_std": None,
            "recall": None, "recall_std": None,
            "y_true": np.array([]), "y_prob": np.array([]),
            "fold_metrics": [],
            "total_train_samples": 0, "total_test_samples": 0,
        }

    # K-Fold on unique students
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)

    fold_metrics = []
    all_y_true = []
    all_y_prob = []

    # Override hidden_dim in config
    fold_config = dataclasses.replace(config, hidden_dim=hidden_dim)

    for fold_idx, (train_student_indices, test_student_indices) in enumerate(
        tqdm(kf.split(unique_students), total=n_folds, desc=f"CV {modality} hd={hidden_dim}")
    ):
        train_students = set(unique_students[train_student_indices])
        test_students = set(unique_students[test_student_indices])

        # Gather sequences
        train_sequences = []
        test_sequences = []
        for sid, seq_indices in student_to_seq_indices.items():
            for si in seq_indices:
                if sid in train_students:
                    train_sequences.append(sequences[si])
                elif sid in test_students:
                    test_sequences.append(sequences[si])

        # Collect skills present in training
        train_skill_ids = set()
        for seq in train_sequences:
            for skill_idx, _ in seq:
                train_skill_ids.add(skill_idx)

        # Filter test sequences: remove interactions with skills not in training
        filtered_test_sequences = []
        for seq in test_sequences:
            filtered = [(sk, c) for sk, c in seq if sk in train_skill_ids]
            if len(filtered) > 1:
                filtered_test_sequences.append(filtered)
        test_sequences = filtered_test_sequences

        if len(train_sequences) < 5 or len(test_sequences) < 5:
            continue

        # Train and evaluate
        model, _ = train_dkt_model(
            train_sequences, test_sequences, num_skills, fold_config, device
        )
        y_true, y_prob = evaluate_dkt_model(model, test_sequences, fold_config, device)

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            continue

        # Compute per-fold metrics
        y_pred = (y_prob >= config.threshold).astype(int)
        fold_auc = roc_auc_score(y_true, y_prob)
        fold_acc = accuracy_score(y_true, y_pred)
        fold_f1 = f1_score(y_true, y_pred, zero_division=0)
        fold_prec = precision_score(y_true, y_pred, zero_division=0)
        fold_rec = recall_score(y_true, y_pred, zero_division=0)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "auc": fold_auc,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "precision": fold_prec,
            "recall": fold_rec,
            "n_train": sum(len(s) for s in train_sequences),
            "n_test": sum(len(s) for s in test_sequences),
        })

        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    # Aggregate metrics across folds
    if not fold_metrics:
        return {
            "auc": None, "auc_std": None,
            "accuracy": None, "accuracy_std": None,
            "f1": None, "f1_std": None,
            "precision": None, "precision_std": None,
            "recall": None, "recall_std": None,
            "y_true": np.array([]), "y_prob": np.array([]),
            "fold_metrics": [],
            "total_train_samples": 0, "total_test_samples": 0,
        }

    metric_names = ["auc", "accuracy", "f1", "precision", "recall"]
    result: Dict[str, Any] = {}
    for m in metric_names:
        values = [fm[m] for fm in fold_metrics if fm[m] is not None]
        if values:
            result[m] = float(np.mean(values))
            result[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            result[m] = None
            result[f"{m}_std"] = None

    result["y_true"] = np.array(all_y_true)
    result["y_prob"] = np.array(all_y_prob)
    result["fold_metrics"] = fold_metrics
    result["total_train_samples"] = sum(fm["n_train"] for fm in fold_metrics)
    result["total_test_samples"] = sum(fm["n_test"] for fm in fold_metrics)

    return result
