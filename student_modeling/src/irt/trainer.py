"""IRT model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from .model import IRTModel

if TYPE_CHECKING:
    from .config import IRTConfig


@dataclass
class IRTTrainResult:
    """Result of IRT training."""

    model: IRTModel
    student_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    modality: str
    level: str
    subgroup: Optional[str]
    train_history: Dict[str, List[float]]


def train_irt_model(
    train_df: pd.DataFrame,
    n_students: int,
    n_items: int,
    config: IRTConfig,
) -> Tuple[IRTModel, Dict[str, List[float]]]:
    """Train an IRT model.

    Args:
        train_df: Training data in long format with student_idx, item_idx, correct.
        n_students: Total number of students.
        n_items: Total number of items.
        config: IRT configuration.

    Returns:
        Tuple of (trained model, training history).
    """
    # Create model
    model = IRTModel(
        n_students=n_students,
        n_items=n_items,
        learning_rate=config.learning_rate,
        regularization=config.regularization,
    )

    # Get training data
    student_indices = train_df["student_idx"].values.astype(np.int64)
    item_indices = train_df["item_idx"].values.astype(np.int64)
    correct = train_df["correct"].values.astype(np.float64)

    # Train
    history = model.fit(
        student_indices,
        item_indices,
        correct,
        max_iter=config.max_iter,
    )

    return model, history


def evaluate_irt_model(
    model: IRTModel,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate IRT model.

    Args:
        model: Trained IRT model.
        df: Data to evaluate (must have student_idx and item_idx columns).

    Returns:
        Tuple of (y_true, y_prob).
    """
    if len(df) == 0:
        return np.array([]), np.array([])

    # Filter to valid indices
    valid_mask = (
        (df["student_idx"] >= 0) & (df["student_idx"] < model.n_students) &
        (df["item_idx"] >= 0) & (df["item_idx"] < model.n_items)
    )
    valid_df = df[valid_mask]

    if len(valid_df) == 0:
        return np.array([]), np.array([])

    student_indices = valid_df["student_idx"].values.astype(np.int64)
    item_indices = valid_df["item_idx"].values.astype(np.int64)

    y_true = valid_df["correct"].values
    y_prob = model.predict_proba(student_indices, item_indices)

    return y_true, y_prob
