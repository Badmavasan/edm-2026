"""Configuration for PFA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PFAConfig:
    """Configuration for PFA training."""

    # Data paths
    data_path: Path = None
    output_base_dir: Path = None

    # Column names
    student_col: str = "compte_hash"
    statut_col: str = "statut"
    expected_tasks_col: str = "expected_type_tasks"
    tasks_from_errors_col: str = "tasks_from_errors"
    exercise_id_col: str = "exercise_id"
    exercise_type_col: str = "exercise_type"
    timestamp_col: str = "date_created"

    # Training parameters
    test_ratio: float = 0.2
    random_seed: int = 42
    threshold: float = 0.5

    # Model parameters
    regularization: float = 1.0  # C parameter for logistic regression

    # Incremental training sample sizes
    sample_sizes: List[int] = field(
        default_factory=lambda: [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    )
