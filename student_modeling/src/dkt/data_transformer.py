"""Transform data to DKT format for both modalities."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from .config import DKTConfig


def parse_json_list_field(x: Any) -> List[str]:
    """Parse JSON or Python list stored as string."""
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
        # Fallback: Python-style list with single quotes
        import ast
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if v is not None]
        except (ValueError, SyntaxError):
            pass
    return []


def transform_to_sequences(
    df: pd.DataFrame,
    config: DKTConfig,
    modality: str,
    skill_to_idx: Dict[str, int] = None,
) -> Tuple[List[List[Tuple[int, int]]], Dict[str, int], List[str]]:
    """Transform data to sequences for DKT.

    Args:
        df: Wide format DataFrame.
        config: DKT configuration.
        modality: "error_independent" or "error_dependent".
        skill_to_idx: Optional existing skill mapping.

    Returns:
        Tuple of (sequences, skill_to_idx, student_ids):
        - sequences: List of sequences, each is list of (skill_idx, correct) tuples.
        - skill_to_idx: Mapping from skill name to index.
        - student_ids: List of student IDs corresponding to sequences.
    """
    # Sort by student and timestamp
    df = df.sort_values([config.student_col, config.timestamp_col]).reset_index(drop=True)

    # Collect all skills first if no mapping provided
    if skill_to_idx is None:
        all_skills = set()
        for _, row in df.iterrows():
            expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
            error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])
            all_skills.update(expected_tasks)
            all_skills.update(error_tasks)
        skill_to_idx = {skill: idx for idx, skill in enumerate(sorted(all_skills))}

    # Build sequences per student
    student_sequences: Dict[str, List[Tuple[int, int]]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building sequences ({modality})"):
        student_id = row[config.student_col]
        statut = str(row[config.statut_col]).strip().lower()
        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
        error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])

        if student_id not in student_sequences:
            student_sequences[student_id] = []

        if modality == "error_independent":
            # All expected skills get the outcome
            correct_val = 1 if statut == "ok" else 0
            for skill in expected_tasks:
                if skill in skill_to_idx:
                    student_sequences[student_id].append((skill_to_idx[skill], correct_val))
        else:  # error_dependent
            if statut == "ok":
                # All expected skills are correct
                for skill in expected_tasks:
                    if skill in skill_to_idx:
                        student_sequences[student_id].append((skill_to_idx[skill], 1))
            else:
                # Only error skills are incorrect
                for skill in error_tasks:
                    if skill in skill_to_idx:
                        student_sequences[student_id].append((skill_to_idx[skill], 0))

    # Convert to list format
    sequences = []
    student_ids = []
    for student_id, seq in student_sequences.items():
        if len(seq) > 1:  # Need at least 2 interactions
            sequences.append(seq)
            student_ids.append(student_id)

    return sequences, skill_to_idx, student_ids


def split_sequences_by_student(
    sequences: List[List[Tuple[int, int]]],
    student_ids: List[str],
    test_ratio: float,
    seed: int,
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], List[str], List[str]]:
    """Split sequences by student (no data leakage).

    Args:
        sequences: List of sequences.
        student_ids: Corresponding student IDs.
        test_ratio: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (train_sequences, test_sequences, train_students, test_students).
    """
    rng = np.random.default_rng(seed)

    unique_students = list(set(student_ids))
    rng.shuffle(unique_students)

    n_test = max(1, int(len(unique_students) * test_ratio))
    test_students_set = set(unique_students[:n_test])

    train_sequences = []
    test_sequences = []
    train_students = []
    test_students = []

    for seq, student_id in zip(sequences, student_ids):
        if student_id in test_students_set:
            test_sequences.append(seq)
            test_students.append(student_id)
        else:
            train_sequences.append(seq)
            train_students.append(student_id)

    return train_sequences, test_sequences, train_students, test_students
