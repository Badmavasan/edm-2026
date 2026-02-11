"""PFA model training using logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from .config import PFAConfig


@dataclass
class PFATrainResult:
    """Result of PFA training."""

    model: LogisticRegression
    scaler: StandardScaler
    skill_to_idx: Dict[str, int]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    modality: str
    level: str
    subgroup: Optional[str]


def build_pfa_features(df: pd.DataFrame, n_skills: int) -> np.ndarray:
    """Build PFA feature matrix.

    Features per sample:
    - One-hot encoding of skill
    - Success count for that skill
    - Failure count for that skill

    Args:
        df: Long format DataFrame with skill_idx, success_count, failure_count.
        n_skills: Total number of skills.

    Returns:
        Feature matrix of shape (n_samples, n_skills + 2).
    """
    n_samples = len(df)

    # Features: skill one-hot + success_count + failure_count
    # Using skill_idx for one-hot, plus 2 count features
    X = np.zeros((n_samples, n_skills + 2), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        skill_idx = int(row["skill_idx"])
        X[i, skill_idx] = 1.0  # One-hot skill
        X[i, n_skills] = row["success_count"]  # Success count
        X[i, n_skills + 1] = row["failure_count"]  # Failure count

    return X


def train_pfa_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    skill_to_idx: Dict[str, int],
    config: PFAConfig,
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, List[float]]]:
    """Train a PFA model using logistic regression.

    Args:
        train_df: Training data in long format.
        test_df: Test data in long format.
        skill_to_idx: Skill to index mapping.
        config: PFA configuration.

    Returns:
        Tuple of (trained model, scaler, training history).
    """
    n_skills = len(skill_to_idx)

    # Build features
    X_train = build_pfa_features(train_df, n_skills)
    y_train = train_df["correct"].values

    X_test = build_pfa_features(test_df, n_skills)
    y_test = test_df["correct"].values

    # Scale features (only count features, not one-hot)
    scaler = StandardScaler()
    X_train[:, n_skills:] = scaler.fit_transform(X_train[:, n_skills:])
    X_test[:, n_skills:] = scaler.transform(X_test[:, n_skills:])

    # Train logistic regression
    model = LogisticRegression(
        C=config.regularization,
        max_iter=1000,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    # Training history (just final metrics for PFA)
    history = {
        "train_samples": [len(train_df)],
        "test_samples": [len(test_df)],
    }

    return model, scaler, history


def evaluate_pfa_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    n_skills: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate PFA model.

    Args:
        model: Trained logistic regression model.
        scaler: Feature scaler.
        df: Data to evaluate.
        n_skills: Number of skills.

    Returns:
        Tuple of (y_true, y_prob).
    """
    X = build_pfa_features(df, n_skills)
    X[:, n_skills:] = scaler.transform(X[:, n_skills:])

    y_true = df["correct"].values
    y_prob = model.predict_proba(X)[:, 1]

    return y_true, y_prob
