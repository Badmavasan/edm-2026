"""Dataset configuration for multi-dataset support.

Supports:
- platform_a: Global, exercise_type, exercise_tag levels
- platform_b: Global, exercise_tag levels only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LevelConfig:
    """Configuration for a specific analysis level."""
    name: str
    group_by_column: Optional[str]  # None for global
    subgroups: List[str] = field(default_factory=list)  # Empty = dynamically populated


@dataclass
class DatasetConfig:
    """Dataset-specific configuration.

    This class defines column mappings and available levels for each dataset.
    """
    # Dataset identification
    name: str
    data_path: Path

    # Column mappings (standardized names -> actual column names)
    student_col: str
    statut_col: str
    timestamp_col: str
    expected_tasks_col: str
    tasks_from_errors_col: str

    # Optional columns (may not exist in all datasets)
    exercise_id_col: Optional[str] = None
    exercise_type_col: Optional[str] = None
    exercise_tag_col: Optional[str] = None

    # Available levels for this dataset
    levels: List[LevelConfig] = field(default_factory=list)

    # Output directory
    output_base_dir: Path = field(default_factory=lambda: Path("results"))

    # Common parameters
    train_ratio: float = 0.80
    test_ratio: float = 0.20
    random_seed: int = 42
    threshold: float = 0.5

    def get_level_column(self, level_name: str) -> Optional[str]:
        """Get the column name for a specific level."""
        if level_name == "global":
            return None
        elif level_name == "exercise_type":
            return self.exercise_type_col
        elif level_name == "exercise_tag":
            return self.exercise_tag_col
        elif level_name == "exercise":
            return self.exercise_id_col
        return None

    def get_available_level_names(self) -> List[str]:
        """Get list of available level names."""
        return [level.name for level in self.levels]

    def has_level(self, level_name: str) -> bool:
        """Check if a level is available."""
        return level_name in self.get_available_level_names()


# =============================================================================
# PLATFORM A DATASET CONFIGURATION
# =============================================================================

def create_platform_a_config(base_path: Path) -> DatasetConfig:
    """Create configuration for Platform A dataset."""
    return DatasetConfig(
        name="platform_a",
        data_path=base_path / "data" / "platform_a_dataset.csv",
        output_base_dir=base_path / "results" / "platform_a",

        # Column mappings
        student_col="compte_hash",
        statut_col="statut",
        timestamp_col="date_created",
        expected_tasks_col="expected_type_tasks",
        tasks_from_errors_col="tasks_from_errors",

        # Subgroup columns
        exercise_id_col="exercise_id",
        exercise_type_col="exercise_type",
        exercise_tag_col="exercise_tag",

        # Available levels: global, exercise_type, exercise_tag
        levels=[
            LevelConfig(name="global", group_by_column=None),
            LevelConfig(
                name="exercise_type",
                group_by_column="exercise_type",
                subgroups=["Console", "Design", "Robot"],
            ),
            LevelConfig(
                name="exercise_tag",
                group_by_column="exercise_tag",
                subgroups=[],  # Populated dynamically
            ),
        ],
    )


# =============================================================================
# PLATFORM B DATASET CONFIGURATION
# =============================================================================

def create_platform_b_config(base_path: Path) -> DatasetConfig:
    """Create configuration for Platform B dataset."""
    return DatasetConfig(
        name="platform_b",
        data_path=base_path / "data" / "platform_b_dataset.csv",
        output_base_dir=base_path / "results" / "platform_b",

        # Column mappings (different from platform_a!)
        student_col="id_compte",
        statut_col="status",  # Note: 'status' not 'statut'
        timestamp_col="date_created",
        expected_tasks_col="expected_task_types",  # Note: different name
        tasks_from_errors_col="task_from_errors",  # Note: singular 'task'

        # Subgroup columns (platform_b uses exercise_tag as exercise_id)
        exercise_id_col="exercise_tag",  # Uses exercise_tag as the exercise identifier
        exercise_type_col=None,  # No exercise_type in platform_b
        exercise_tag_col="exercise_tag",

        # Available levels: global, exercise_tag only
        levels=[
            LevelConfig(name="global", group_by_column=None),
            LevelConfig(
                name="exercise_tag",
                group_by_column="exercise_tag",
                subgroups=[],  # Populated dynamically
            ),
        ],
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Pre-configured datasets (lazy initialization)
_CONFIGS = {}


def get_dataset_config(dataset_name: str, base_path: Path = None) -> DatasetConfig:
    """Get dataset configuration by name.

    Args:
        dataset_name: Either 'platform_a' or 'platform_b'
        base_path: Base path for data and output directories

    Returns:
        DatasetConfig for the specified dataset

    Raises:
        ValueError: If dataset_name is not recognized
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent

    key = (dataset_name, str(base_path))
    if key not in _CONFIGS:
        if dataset_name == "platform_a":
            _CONFIGS[key] = create_platform_a_config(base_path)
        elif dataset_name == "platform_b":
            _CONFIGS[key] = create_platform_b_config(base_path)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: 'platform_a', 'platform_b'"
            )

    return _CONFIGS[key]


# Shortcuts for direct import
PLATFORM_A_CONFIG = None  # Lazy
PLATFORM_B_CONFIG = None  # Lazy


def _init_default_configs():
    """Initialize default configs."""
    global PLATFORM_A_CONFIG, PLATFORM_B_CONFIG
    base_path = Path(__file__).parent.parent.parent
    PLATFORM_A_CONFIG = get_dataset_config("platform_a", base_path)
    PLATFORM_B_CONFIG = get_dataset_config("platform_b", base_path)
