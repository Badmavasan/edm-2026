"""Transform data to PFA format for both modalities."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from .config import PFAConfig


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


def transform_to_pfa_format(
    df: pd.DataFrame,
    config: PFAConfig,
    modality: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Transform data to PFA format.

    PFA uses counts of successes and failures per skill as features.

    Args:
        df: Wide format DataFrame.
        config: PFA configuration.
        modality: "error_independent" or "error_dependent".

    Returns:
        Tuple of (long_format_df, skill_to_idx):
        - long_format_df: DataFrame with student_id, skill, correct, success_count, failure_count
        - skill_to_idx: Mapping from skill name to index.
    """
    # Sort by student and timestamp
    df = df.sort_values([config.student_col, config.timestamp_col]).reset_index(drop=True)

    # Collect all skills
    all_skills = set()
    for _, row in df.iterrows():
        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
        error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])
        all_skills.update(expected_tasks)
        all_skills.update(error_tasks)

    skill_to_idx = {skill: idx for idx, skill in enumerate(sorted(all_skills))}

    # Track success/failure counts per student per skill
    student_skill_success: Dict[str, Dict[str, int]] = {}
    student_skill_failure: Dict[str, Dict[str, int]] = {}

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building PFA data ({modality})"):
        student_id = row[config.student_col]
        # Skip rows with NaN student IDs
        if pd.isna(student_id):
            continue
        student_id = str(student_id)

        statut = str(row[config.statut_col]).strip().lower()
        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
        error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])

        if student_id not in student_skill_success:
            student_skill_success[student_id] = {}
            student_skill_failure[student_id] = {}

        if modality == "error_independent":
            # All expected skills get the outcome
            correct_val = 1 if statut == "ok" else 0
            for skill in expected_tasks:
                if skill not in skill_to_idx:
                    continue

                # Get current counts
                s_count = student_skill_success[student_id].get(skill, 0)
                f_count = student_skill_failure[student_id].get(skill, 0)

                records.append({
                    "student_id": student_id,
                    "skill": skill,
                    "skill_idx": skill_to_idx[skill],
                    "correct": correct_val,
                    "success_count": s_count,
                    "failure_count": f_count,
                })

                # Update counts
                if correct_val == 1:
                    student_skill_success[student_id][skill] = s_count + 1
                else:
                    student_skill_failure[student_id][skill] = f_count + 1

        else:  # error_dependent
            if statut == "ok":
                # All expected skills are correct
                for skill in expected_tasks:
                    if skill not in skill_to_idx:
                        continue

                    s_count = student_skill_success[student_id].get(skill, 0)
                    f_count = student_skill_failure[student_id].get(skill, 0)

                    records.append({
                        "student_id": student_id,
                        "skill": skill,
                        "skill_idx": skill_to_idx[skill],
                        "correct": 1,
                        "success_count": s_count,
                        "failure_count": f_count,
                    })

                    student_skill_success[student_id][skill] = s_count + 1
            else:
                # Only error skills are incorrect
                for skill in error_tasks:
                    if skill not in skill_to_idx:
                        continue

                    s_count = student_skill_success[student_id].get(skill, 0)
                    f_count = student_skill_failure[student_id].get(skill, 0)

                    records.append({
                        "student_id": student_id,
                        "skill": skill,
                        "skill_idx": skill_to_idx[skill],
                        "correct": 0,
                        "success_count": s_count,
                        "failure_count": f_count,
                    })

                    student_skill_failure[student_id][skill] = f_count + 1

    return pd.DataFrame(records), skill_to_idx


def split_by_student(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split data by student (no data leakage).

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
