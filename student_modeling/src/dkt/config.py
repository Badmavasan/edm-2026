"""Configuration for DKT training system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DKTConfig:
    """Configuration for DKT training."""

    # Paths
    data_path: Path = field(
        default_factory=lambda: Path("data/platform_a_dataset.csv")
    )
    output_base_dir: Path = field(default_factory=lambda: Path("results/dkt"))

    # Split parameters
    train_ratio: float = 0.80
    test_ratio: float = 0.20
    random_seed: int = 42

    # Model parameters
    hidden_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.2

    # Training parameters
    batch_size: int = 64
    max_seq_len: int = 200
    learning_rate: float = 0.001
    num_epochs: int = 50
    patience: int = 5  # Early stopping patience

    # Evaluation
    min_rows_per_skill_train: int = 20
    threshold: float = 0.5

    # Column names
    student_col: str = "compte_hash"
    exercise_id_col: str = "exercise_id"
    exercise_type_col: str = "exercise_type"
    statut_col: str = "statut"
    timestamp_col: str = "date_created"
    expected_tasks_col: str = "expected_type_tasks"
    tasks_from_errors_col: str = "tasks_from_errors"
