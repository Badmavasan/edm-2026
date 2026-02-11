"""Metrics computation and evaluation for DKT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

if TYPE_CHECKING:
    from .config import DKTConfig


@dataclass
class Metrics:
    """Evaluation metrics."""

    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    n_samples: int


@dataclass
class IncrementalMetrics:
    """Metrics at a specific training sample size."""

    train_samples: int
    auc: Optional[float]
    accuracy: Optional[float]
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    test_samples: int


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Compute evaluation metrics."""
    n_samples = len(y_true)

    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return Metrics(auc=None, accuracy=None, f1=None, n_samples=n_samples)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return Metrics(auc=auc, accuracy=accuracy, f1=f1, n_samples=n_samples)


def evaluate_incremental_training(
    train_sequences: List,
    test_sequences: List,
    num_skills: int,
    config: "DKTConfig",
    sample_sizes: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Evaluate DKT model with incrementally increasing training data.

    Args:
        train_sequences: All training sequences.
        test_sequences: Test sequences (fixed).
        num_skills: Number of unique skills.
        config: DKT configuration.
        sample_sizes: Custom sample sizes. If None, uses default progression.

    Returns:
        DataFrame with metrics at each training sample size.
    """
    from .trainer import train_dkt_model, evaluate_dkt_model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Count total interactions in training
    total_train_interactions = sum(len(seq) for seq in train_sequences)

    # Generate sample sizes based on interactions
    if sample_sizes is None:
        sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        sample_sizes = [s for s in sample_sizes if s <= total_train_interactions]
        if total_train_interactions not in sample_sizes:
            sample_sizes.append(total_train_interactions)

    results: List[IncrementalMetrics] = []

    for n_samples in tqdm(sample_sizes, desc="Incremental training"):
        # Select sequences up to n_samples interactions
        selected_sequences = []
        interaction_count = 0

        for seq in train_sequences:
            if interaction_count >= n_samples:
                break
            selected_sequences.append(seq)
            interaction_count += len(seq)

        if len(selected_sequences) < 2:
            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=None,
                    accuracy=None,
                    f1=None,
                    precision=None,
                    recall=None,
                    test_samples=0,
                )
            )
            continue

        try:
            # Train model
            model, _ = train_dkt_model(
                selected_sequences,
                test_sequences,
                num_skills,
                config,
                device,
            )

            # Evaluate
            y_true, y_prob = evaluate_dkt_model(model, test_sequences, config, device)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                auc = None
            else:
                auc = float(roc_auc_score(y_true, y_prob))

            y_pred = (y_prob >= config.threshold).astype(int)
            accuracy = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))

            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=auc,
                    accuracy=accuracy,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    test_samples=len(y_true),
                )
            )

        except Exception as e:
            print(f"  Warning: Training failed at {n_samples} samples: {e}")
            results.append(
                IncrementalMetrics(
                    train_samples=n_samples,
                    auc=None,
                    accuracy=None,
                    f1=None,
                    precision=None,
                    recall=None,
                    test_samples=0,
                )
            )

    return pd.DataFrame([asdict(r) for r in results])
