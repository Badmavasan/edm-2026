"""Transform data to pyBKT format for both modalities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

try:
    from .data_loader import parse_json_list_field
except ImportError:
    from src.bkt.data_loader import parse_json_list_field

if TYPE_CHECKING:
    from src.bkt.config import BKTConfig


def transform_error_independent(
    df: pd.DataFrame,
    config: BKTConfig,
    filter_column: Optional[str] = None,
    filter_value: Any = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Error-Independent transformation.

    - ok submission: ALL skills in expected_type_tasks get correct=1
    - ko submission: ALL skills in expected_type_tasks get correct=0

    Args:
        df: Wide format DataFrame.
        config: BKT configuration.
        filter_column: Column to filter on (for level-specific transforms).
        filter_value: Value to filter for.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Long format DataFrame with pyBKT columns:
        - Anon Student Id, skill_name, correct, order_id
    """
    work_df = df.copy()
    if filter_column and filter_value is not None:
        work_df = work_df[work_df[filter_column] == filter_value].copy()

    records: List[Dict[str, Any]] = []

    iterator = work_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(work_df), desc="Error-independent transform")

    for _, row in iterator:
        sid = row[config.student_col]
        statut = row[config.statut_col]
        ts = row["_timestamp"]
        row_idx = row["_row_index"]

        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])

        if not expected_tasks:
            continue

        correct_val = 1 if statut == "ok" else 0

        for skill in expected_tasks:
            records.append(
                {
                    "Anon Student Id": sid,
                    "skill_name": str(skill),
                    "correct": correct_val,
                    "_timestamp": ts,
                    "_row_index": row_idx,
                }
            )

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        return pd.DataFrame(columns=["Anon Student Id", "skill_name", "correct", "order_id"])

    # Sort and compute order_id
    long_df = long_df.sort_values(
        ["Anon Student Id", "skill_name", "_timestamp", "_row_index"], kind="mergesort"
    )
    long_df["order_id"] = long_df.groupby(["Anon Student Id", "skill_name"]).cumcount() + 1

    return long_df[["Anon Student Id", "skill_name", "correct", "order_id"]].reset_index(drop=True)


def transform_error_dependent(
    df: pd.DataFrame,
    config: BKTConfig,
    filter_column: Optional[str] = None,
    filter_value: Any = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Error-Dependent transformation.

    - ok submission: ALL skills in expected_type_tasks get correct=1
    - ko submission: ONLY skills in tasks_from_errors get correct=0
                    (skills in expected but not in errors are NOT recorded)

    Args:
        df: Wide format DataFrame.
        config: BKT configuration.
        filter_column: Column to filter on (for level-specific transforms).
        filter_value: Value to filter for.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Long format DataFrame with pyBKT columns:
        - Anon Student Id, skill_name, correct, order_id
    """
    work_df = df.copy()
    if filter_column and filter_value is not None:
        work_df = work_df[work_df[filter_column] == filter_value].copy()

    records: List[Dict[str, Any]] = []

    iterator = work_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(work_df), desc="Error-dependent transform")

    for _, row in iterator:
        sid = row[config.student_col]
        statut = row[config.statut_col]
        ts = row["_timestamp"]
        row_idx = row["_row_index"]

        expected_tasks = parse_json_list_field(row[config.expected_tasks_col])
        error_tasks = parse_json_list_field(row[config.tasks_from_errors_col])

        if statut == "ok":
            # All expected tasks are correct
            for skill in expected_tasks:
                records.append(
                    {
                        "Anon Student Id": sid,
                        "skill_name": str(skill),
                        "correct": 1,
                        "_timestamp": ts,
                        "_row_index": row_idx,
                    }
                )
        elif statut == "ko":
            # Only error tasks are recorded (as incorrect)
            for skill in error_tasks:
                records.append(
                    {
                        "Anon Student Id": sid,
                        "skill_name": str(skill),
                        "correct": 0,
                        "_timestamp": ts,
                        "_row_index": row_idx,
                    }
                )
        # Note: for ko, skills in expected but NOT in errors are NOT recorded

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        return pd.DataFrame(columns=["Anon Student Id", "skill_name", "correct", "order_id"])

    # Sort and compute order_id
    long_df = long_df.sort_values(
        ["Anon Student Id", "skill_name", "_timestamp", "_row_index"], kind="mergesort"
    )
    long_df["order_id"] = long_df.groupby(["Anon Student Id", "skill_name"]).cumcount() + 1

    return long_df[["Anon Student Id", "skill_name", "correct", "order_id"]].reset_index(drop=True)
