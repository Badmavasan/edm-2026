"""Configuration for BKT training system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BKTConfig:
    """Configuration for BKT training."""

    # Paths
    data_path: Path = field(default_factory=lambda: Path("data/platform_a_dataset.csv"))
    output_base_dir: Path = field(default_factory=lambda: Path("results"))

    # Split parameters
    train_ratio: float = 0.80
    test_ratio: float = 0.20
    random_seed: int = 42

    # pyBKT parameters
    pybkt_num_fits: int = 1
    pybkt_seed: int = 42

    # Evaluation
    min_rows_per_skill_train: int = 20
    min_test_rows_per_step: int = 50
    order_id_percentile_for_max_steps: int = 90
    threshold: float = 0.5

    # Column names
    student_col: str = "compte_hash"
    exercise_id_col: str = "exercise_id"
    exercise_type_col: str = "exercise_type"
    statut_col: str = "statut"
    timestamp_col: str = "date_created"
    expected_tasks_col: str = "expected_type_tasks"
    tasks_from_errors_col: str = "tasks_from_errors"


@dataclass
class LevelConfig:
    """Configuration for a specific level (global, exercise_type, exercise)."""

    name: str
    group_by_column: Optional[str]  # None for global, column name for others
    subgroups: List[str] = field(default_factory=list)  # Empty for global


# Level configurations
LEVEL_GLOBAL = LevelConfig(name="global", group_by_column=None)

LEVEL_EXERCISE_TYPE = LevelConfig(
    name="exercise_type",
    group_by_column="exercise_type",
    subgroups=["Console", "Design", "Robot"],
)

LEVEL_EXERCISE = LevelConfig(
    name="exercise",
    group_by_column="exercise_id",
    subgroups=[],  # Will be populated dynamically
)
