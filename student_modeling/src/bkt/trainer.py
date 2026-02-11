"""BKT model training with tqdm progress bars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TYPE_CHECKING

import pandas as pd
from pyBKT.models import Model
from tqdm import tqdm

try:
    from .data_transformer import transform_error_dependent, transform_error_independent
    from .student_split import filter_rare_skills, split_students_no_leakage
except ImportError:
    from src.bkt.data_transformer import transform_error_dependent, transform_error_independent
    from src.bkt.student_split import filter_rare_skills, split_students_no_leakage

if TYPE_CHECKING:
    from src.bkt.config import BKTConfig, LevelConfig


@dataclass
class BKTTrainResult:
    """Result of BKT training."""

    model: Model
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    modality: str  # "error_independent" or "error_dependent"
    level: str  # "global", "exercise_type", "exercise"
    subgroup: Optional[str]  # None for global, type/exercise name for others


def train_bkt_model(train_df: pd.DataFrame, config: BKTConfig) -> Model:
    """Train a single pyBKT model.

    Args:
        train_df: Training data in pyBKT format.
        config: BKT configuration.

    Returns:
        Trained pyBKT Model.
    """
    model = Model(seed=config.pybkt_seed, num_fits=config.pybkt_num_fits)
    model.fit(data=train_df)
    return model


def train_all_models(
    wide_df: pd.DataFrame,
    config: BKTConfig,
    levels: List[LevelConfig],
) -> List[BKTTrainResult]:
    """Train all BKT models for both modalities and all levels.

    Args:
        wide_df: Wide format DataFrame from data_loader.
        config: BKT configuration.
        levels: List of level configurations.

    Returns:
        List of BKTTrainResult containing trained models and data.
    """
    results: List[BKTTrainResult] = []
    modalities = ["error_independent", "error_dependent"]

    # Count total jobs for progress bar
    total_jobs = 0
    for lvl in levels:
        if lvl.group_by_column is None:
            total_jobs += len(modalities)
        else:
            subgroups = lvl.subgroups
            if not subgroups:
                subgroups = wide_df[lvl.group_by_column].unique().tolist()
            total_jobs += len(modalities) * len(subgroups)

    pbar = tqdm(total=total_jobs, desc="Training BKT models")

    transform_funcs: dict[str, Callable] = {
        "error_independent": transform_error_independent,
        "error_dependent": transform_error_dependent,
    }

    for level in levels:
        if level.group_by_column is None:
            # Global level
            for modality in modalities:
                transform_fn = transform_funcs[modality]
                long_df = transform_fn(wide_df, config)

                if long_df.empty:
                    pbar.update(1)
                    continue

                train_df, test_df = split_students_no_leakage(
                    long_df, config.test_ratio, config.random_seed
                )
                train_df, test_df = filter_rare_skills(
                    train_df, test_df, config.min_rows_per_skill_train
                )

                if not train_df.empty and not test_df.empty:
                    model = train_bkt_model(train_df, config)
                    results.append(
                        BKTTrainResult(
                            model=model,
                            train_df=train_df,
                            test_df=test_df,
                            modality=modality,
                            level=level.name,
                            subgroup=None,
                        )
                    )
                pbar.update(1)
        else:
            # Per-subgroup levels
            subgroups = level.subgroups
            if not subgroups:
                subgroups = wide_df[level.group_by_column].unique().tolist()

            for subgroup in subgroups:
                for modality in modalities:
                    transform_fn = transform_funcs[modality]
                    long_df = transform_fn(
                        wide_df,
                        config,
                        filter_column=level.group_by_column,
                        filter_value=subgroup,
                    )

                    if long_df.empty:
                        pbar.update(1)
                        continue

                    train_df, test_df = split_students_no_leakage(
                        long_df, config.test_ratio, config.random_seed
                    )
                    train_df, test_df = filter_rare_skills(
                        train_df, test_df, config.min_rows_per_skill_train
                    )

                    if not train_df.empty and not test_df.empty:
                        model = train_bkt_model(train_df, config)
                        results.append(
                            BKTTrainResult(
                                model=model,
                                train_df=train_df,
                                test_df=test_df,
                                modality=modality,
                                level=level.name,
                                subgroup=str(subgroup),
                            )
                        )
                    pbar.update(1)

    pbar.close()
    return results
