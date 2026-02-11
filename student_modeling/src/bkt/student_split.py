"""Student-level train/test split with no data leakage."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def split_students_no_leakage(
    long_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Student-level split ensuring no data leakage.

    Ensures:
    1. No student appears in both train and test
    2. Every skill in test is present in train (skill coverage)

    Args:
        long_df: Long format DataFrame with 'Anon Student Id' and 'skill_name'.
        test_size: Fraction of students for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    students = long_df["Anon Student Id"].dropna().unique().tolist()

    if len(students) < 2:
        # Not enough students to split
        return long_df.copy(), pd.DataFrame()

    # Map student -> skills
    stud_skills: Dict[Any, Set[str]] = {}
    for sid, g in long_df.groupby("Anon Student Id"):
        stud_skills[sid] = set(g["skill_name"].astype(str).unique())

    # Count skill occurrences across students
    skill_counts: Dict[str, int] = {}
    for sid in students:
        for sk in stud_skills.get(sid, set()):
            skill_counts[sk] = skill_counts.get(sk, 0) + 1

    target_test_n = max(1, int(round(test_size * len(students))))
    shuffled = students[:]
    rng.shuffle(shuffled)

    test_students: List[Any] = []

    for sid in shuffled:
        if len(test_students) >= target_test_n:
            break

        skills = stud_skills.get(sid, set())
        # Only move to test if train retains at least one student per skill
        if all(skill_counts.get(sk, 0) >= 2 for sk in skills):
            test_students.append(sid)
            for sk in skills:
                skill_counts[sk] -= 1

    train_students = [sid for sid in students if sid not in set(test_students)]

    train_df = long_df[long_df["Anon Student Id"].isin(train_students)].copy()
    test_df = long_df[long_df["Anon Student Id"].isin(test_students)].copy()

    # Validate no leakage
    train_student_set = set(train_df["Anon Student Id"].unique())
    test_student_set = set(test_df["Anon Student Id"].unique())
    assert train_student_set.isdisjoint(test_student_set), "Data leakage detected!"

    # Validate skill coverage
    train_skills = set(train_df["skill_name"].unique())
    test_skills = set(test_df["skill_name"].unique())
    uncovered = test_skills - train_skills
    if uncovered:
        # Remove uncovered skills from test
        test_df = test_df[test_df["skill_name"].isin(train_skills)].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def filter_rare_skills(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter skills with fewer than min_rows in training.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        min_rows: Minimum number of rows required in training.

    Returns:
        Tuple of filtered (train_df, test_df).
    """
    if train_df.empty:
        return train_df, test_df

    counts = train_df["skill_name"].value_counts()
    keep_skills = set(counts[counts >= min_rows].index)

    if not keep_skills:
        return pd.DataFrame(), pd.DataFrame()

    train_df = train_df[train_df["skill_name"].isin(keep_skills)].copy()
    test_df = test_df[test_df["skill_name"].isin(keep_skills)].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
