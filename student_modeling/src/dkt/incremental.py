"""Incremental training evaluation for DKT (by samples and by students)."""

from __future__ import annotations

from typing import List, Dict, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from .trainer import train_dkt_model, evaluate_dkt_model

if TYPE_CHECKING:
    from .config import DKTConfig


def evaluate_incremental_by_samples(
    sequences: List[List[Tuple[int, int]]],
    student_ids: List[str],
    skill_to_idx: Dict[str, int],
    config: DKTConfig,
    device: torch.device = None,
) -> pd.DataFrame:
    """Evaluate DKT with increasing numbers of training interactions.

    Uses an 80/20 student split. Incrementally increases the number of
    training interactions (by taking sequences until N interactions are reached).

    Args:
        sequences: All sequences.
        student_ids: Corresponding student IDs.
        skill_to_idx: Skill-to-index mapping.
        config: DKT configuration.
        device: Torch device.

    Returns:
        DataFrame with columns: train_samples, train_students, test_samples,
        auc, accuracy, f1, precision, recall.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .data_transformer import split_sequences_by_student

    num_skills = len(skill_to_idx)

    train_sequences, test_sequences, train_students, test_students = (
        split_sequences_by_student(sequences, student_ids, config.test_ratio, config.random_seed)
    )

    total_interactions = sum(len(seq) for seq in train_sequences)

    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    sample_sizes = [s for s in sample_sizes if s <= total_interactions]
    if total_interactions not in sample_sizes:
        sample_sizes.append(total_interactions)

    results = []

    for n_samples in tqdm(sample_sizes, desc="Incremental (samples)"):
        # Take sequences until we reach n_samples interactions
        train_subset = []
        count = 0
        n_students_used = 0
        seen_students = set()
        for seq, sid in zip(train_sequences, train_students):
            if count >= n_samples:
                break
            train_subset.append(seq)
            count += len(seq)
            if sid not in seen_students:
                seen_students.add(sid)
                n_students_used += 1

        if len(train_subset) < 5:
            results.append({
                "train_samples": n_samples, "train_students": n_students_used,
                "test_samples": 0,
                "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None,
            })
            continue

        # Collect train skills for filtering test
        train_skill_ids = set()
        for seq in train_subset:
            for skill_idx, _ in seq:
                train_skill_ids.add(skill_idx)

        # Filter test sequences
        filtered_test = []
        for seq in test_sequences:
            filtered = [(sk, c) for sk, c in seq if sk in train_skill_ids]
            if len(filtered) > 1:
                filtered_test.append(filtered)

        if len(filtered_test) < 5:
            results.append({
                "train_samples": n_samples, "train_students": n_students_used,
                "test_samples": 0,
                "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None,
            })
            continue

        try:
            model, _ = train_dkt_model(
                train_subset, filtered_test, num_skills, config, device
            )
            y_true, y_prob = evaluate_dkt_model(model, filtered_test, config, device)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(y_true) < 10 or len(np.unique(y_true)) < 2:
                results.append({
                    "train_samples": n_samples, "train_students": n_students_used,
                    "test_samples": len(y_true),
                    "auc": None, "accuracy": None, "f1": None,
                    "precision": None, "recall": None,
                })
                continue

            y_pred = (y_prob >= config.threshold).astype(int)
            results.append({
                "train_samples": n_samples,
                "train_students": n_students_used,
                "test_samples": len(y_true),
                "auc": float(roc_auc_score(y_true, y_prob)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            })

        except Exception as e:
            print(f"  Warning: Incremental training failed at {n_samples} samples: {e}")
            results.append({
                "train_samples": n_samples, "train_students": n_students_used,
                "test_samples": 0,
                "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None,
            })

    return pd.DataFrame(results)


def evaluate_incremental_by_students(
    sequences: List[List[Tuple[int, int]]],
    student_ids: List[str],
    skill_to_idx: Dict[str, int],
    config: DKTConfig,
    device: torch.device = None,
) -> pd.DataFrame:
    """Evaluate DKT with increasing numbers of training students.

    Uses an 80/20 student split. Incrementally increases the number of
    training students.

    Args:
        sequences: All sequences.
        student_ids: Corresponding student IDs.
        skill_to_idx: Skill-to-index mapping.
        config: DKT configuration.
        device: Torch device.

    Returns:
        DataFrame with columns: train_students, train_samples, test_samples,
        auc, accuracy, f1, precision, recall.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .data_transformer import split_sequences_by_student

    num_skills = len(skill_to_idx)

    train_sequences, test_sequences, train_students_list, test_students_list = (
        split_sequences_by_student(sequences, student_ids, config.test_ratio, config.random_seed)
    )

    # Build student -> sequence mapping for train
    from collections import defaultdict
    student_to_train_seqs: Dict[str, List] = defaultdict(list)
    for seq, sid in zip(train_sequences, train_students_list):
        student_to_train_seqs[sid].append(seq)

    unique_train_students = list(student_to_train_seqs.keys())
    total_students = len(unique_train_students)

    student_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    student_counts = [s for s in student_counts if s <= total_students]
    if total_students not in student_counts:
        student_counts.append(total_students)

    results = []

    for n_students in tqdm(student_counts, desc="Incremental (students)"):
        selected_students = unique_train_students[:n_students]

        # Gather their sequences
        train_subset = []
        for sid in selected_students:
            train_subset.extend(student_to_train_seqs[sid])

        n_samples = sum(len(seq) for seq in train_subset)

        if len(train_subset) < 5:
            results.append({
                "train_students": n_students, "train_samples": n_samples,
                "test_samples": 0,
                "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None,
            })
            continue

        # Collect train skills for filtering test
        train_skill_ids = set()
        for seq in train_subset:
            for skill_idx, _ in seq:
                train_skill_ids.add(skill_idx)

        filtered_test = []
        for seq in test_sequences:
            filtered = [(sk, c) for sk, c in seq if sk in train_skill_ids]
            if len(filtered) > 1:
                filtered_test.append(filtered)

        if len(filtered_test) < 5:
            results.append({
                "train_students": n_students, "train_samples": n_samples,
                "test_samples": 0,
                "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None,
            })
            continue

        try:
            model, _ = train_dkt_model(
                train_subset, filtered_test, num_skills, config, device
            )
            y_true, y_prob = evaluate_dkt_model(model, filtered_test, config, device)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(y_true) < 10 or len(np.unique(y_true)) < 2:
                results.append({
                    "train_students": n_students, "train_samples": n_samples,
                    "test_samples": len(y_true),
                    "auc": None, "accuracy": None, "f1": None,
                    "precision": None, "recall": None,
                })
                continue

            y_pred = (y_prob >= config.threshold).astype(int)
            results.append({
                "train_students": n_students,
                "train_samples": n_samples,
                "test_samples": len(y_true),
                "auc": float(roc_auc_score(y_true, y_prob)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            })

        except Exception as e:
            print(f"  Warning: Incremental training failed at {n_students} students: {e}")
            results.append({
                "train_students": n_students, "train_samples": n_samples,
                "test_samples": 0,
                "auc": None, "accuracy": None, "f1": None,
                "precision": None, "recall": None,
            })

    return pd.DataFrame(results)
