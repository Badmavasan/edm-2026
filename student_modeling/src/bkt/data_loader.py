"""Data loading and preprocessing for BKT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.bkt.config import BKTConfig


def parse_json_list_field(x: Any) -> List[str]:
    """Parse JSON list stored as string.

    Handles:
    - None/NaN -> []
    - List -> list of strings
    - String -> JSON parse to list
    """
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


def load_dataset(path: Path, config: BKTConfig) -> pd.DataFrame:
    """Load and preprocess the dataset.

    Args:
        path: Path to the CSV file.
        config: BKT configuration.

    Returns:
        Preprocessed DataFrame with additional columns for timestamp and row index.
    """
    df = pd.read_csv(path, low_memory=False)

    # Validate required columns (filter out None for optional columns)
    required = [
        config.student_col,
        config.statut_col,
        config.expected_tasks_col,
        config.tasks_from_errors_col,
        config.timestamp_col,
    ]
    # Add optional columns only if they are configured
    if config.exercise_id_col:
        required.append(config.exercise_id_col)
    if config.exercise_type_col:
        required.append(config.exercise_type_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Preprocessing
    df[config.statut_col] = df[config.statut_col].astype(str).str.strip().str.lower()
    df["_timestamp"] = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    df["_row_index"] = np.arange(len(df), dtype=np.int64)

    # Sort by timestamp for correct ordering
    df = df.sort_values(["_timestamp", "_row_index"]).reset_index(drop=True)
    df["_row_index"] = np.arange(len(df), dtype=np.int64)

    return df
