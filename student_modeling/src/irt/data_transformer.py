"""Transform data to IRT format for both modalities."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from .config import IRTConfig


def parse_json_list_field(x: Any) -> List[str]:
    """Parse JSON list stored as string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if v is not None]
        except json.JSONDecodeError:
            pass
    return []


def transform_to_irt_format(
    df: pd.DataFrame,
    config: IRTConfig,
    modality: str,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Transform data to IRT format.

    IRT format: (student_id, item_id, correct)
    Items are skills in our case.

    Args:
        df: Wide format DataFrame.
        config: IRT configuration.
        modality: "error_independent" or "error_dependent".

    Returns:
        Tuple of (long_format_df, student_to_idx, item_to_idx):
        - long_format_df: DataFrame with student_idx, item_idx, correct
        - student_to_idx: Mapping from student ID to index.
        - item_to_idx: Mapping from item (skill) to index.
    """
    # Sort by student and timestamp
    df = df.sort_values([config.student_col, config.timestamp_col]).reset_index(drop=True)

    # Collect all students and items (skills)
    all_students = set()
    all_items = set()

    for _, row in df.iterrows():
        student_id = row[config.student_col]
        # Skip NaN/None student IDs
        if pd.notna(student_id):
            all_students.add(str(student_id))
        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
        error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])
        all_items.update(expected_tasks)
        all_items.update(error_tasks)

    student_to_idx = {s: idx for idx, s in enumerate(sorted(all_students))}
    item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building IRT data ({modality})"):
        student_id = row[config.student_col]
        # Skip rows with NaN student IDs
        if pd.isna(student_id):
            continue
        student_id = str(student_id)

        statut = str(row[config.statut_col]).strip().lower()
        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
        error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])

        student_idx = student_to_idx[student_id]

        if modality == "error_independent":
            # All expected skills get the outcome
            correct_val = 1 if statut == "ok" else 0
            for skill in expected_tasks:
                if skill in item_to_idx:
                    records.append({
                        "student_id": student_id,
                        "student_idx": student_idx,
                        "item": skill,
                        "item_idx": item_to_idx[skill],
                        "correct": correct_val,
                    })

        else:  # error_dependent
            if statut == "ok":
                # All expected skills are correct
                for skill in expected_tasks:
                    if skill in item_to_idx:
                        records.append({
                            "student_id": student_id,
                            "student_idx": student_idx,
                            "item": skill,
                            "item_idx": item_to_idx[skill],
                            "correct": 1,
                        })
            else:
                # Only error skills are incorrect
                for skill in error_tasks:
                    if skill in item_to_idx:
                        records.append({
                            "student_id": student_id,
                            "student_idx": student_idx,
                            "item": skill,
                            "item_idx": item_to_idx[skill],
                            "correct": 0,
                        })

    return pd.DataFrame(records), student_to_idx, item_to_idx


def split_by_interaction(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by interactions (not by student).

    This allows IRT to learn student abilities from training interactions
    and predict on held-out interactions from the same students.

    Args:
        df: Long format DataFrame.
        test_ratio: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (train_df, test_df).
    """
    rng = np.random.default_rng(seed)

    # Shuffle indices
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_test = max(1, int(len(df) * test_ratio))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    return train_df, test_df


def split_by_student(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split data by student (no data leakage).

    Note: For IRT, this means we can't predict on test students
    unless we estimate their abilities separately.

    Args:
        df: Long format DataFrame with student_id column.
        test_ratio: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (train_df, test_df, train_students, test_students).
    """
    rng = np.random.default_rng(seed)

    unique_students = list(df["student_id"].unique())
    rng.shuffle(unique_students)

    n_test = max(1, int(len(unique_students) * test_ratio))
    test_students = unique_students[:n_test]
    train_students = unique_students[n_test:]

    test_students_set = set(test_students)

    train_df = df[~df["student_id"].isin(test_students_set)].copy()
    test_df = df[df["student_id"].isin(test_students_set)].copy()

    return train_df, test_df, train_students, test_students
