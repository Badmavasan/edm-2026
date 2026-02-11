#!/usr/bin/env python3
"""DKT Experiment: Hidden dimension and data quantity ablation study.

Tests DKT with:
- Hidden dimensions: 8, 16, 32, 64, 128
- Both modalities: error_independent, error_dependent
- Various training data sizes
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import DKTConfig
from data_transformer import (
    transform_to_sequences,
    split_sequences_by_student,
)
from model import DKTModel, DKTDataset


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    # Paths
    data_path: Path = field(
        default_factory=lambda: Path("data/platform_a_dataset.csv")
    )
    output_dir: Path = field(default_factory=lambda: Path("results/dkt"))

    # Experiment parameters
    hidden_dims: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    modalities: List[str] = field(default_factory=lambda: ["error_independent", "error_dependent"])

    # Data quantity fractions (percentage of training data to use)
    data_fractions: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])

    # Fixed model parameters
    num_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 64
    max_seq_len: int = 200
    learning_rate: float = 0.001
    num_epochs: int = 50
    patience: int = 5

    # Split parameters
    test_ratio: float = 0.20
    random_seed: int = 42
    threshold: float = 0.5

    # Column names (can be overridden per dataset)
    student_col: str = "compte_hash"
    exercise_id_col: str = "exercise_id"
    exercise_type_col: str = "exercise_type"
    statut_col: str = "statut"
    timestamp_col: str = "date_created"
    expected_tasks_col: str = "expected_type_tasks"
    tasks_from_errors_col: str = "tasks_from_errors"
    exercise_tag_col: str = "exercise_tag"


def load_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(config.data_path, low_memory=False)
    df[config.statut_col] = df[config.statut_col].astype(str).str.strip().str.lower()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    return df


def train_model(
    train_sequences: List,
    test_sequences: List,
    num_skills: int,
    hidden_dim: int,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[DKTModel, Dict[str, List[float]]]:
    """Train a DKT model with specific hidden dimension."""

    train_dataset = DKTDataset(train_sequences, config.max_seq_len)
    test_dataset = DKTDataset(test_sequences, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = DKTModel(
        num_skills=num_skills,
        hidden_dim=hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            inputs, target_skills, labels, mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            predictions = model(inputs, target_skills)
            loss = criterion(predictions, labels)
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # Collect training predictions for AUC
            mask_bool = mask.bool()
            train_preds.extend(predictions[mask_bool].detach().cpu().numpy())
            train_labels.extend(labels[mask_bool].cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Compute training AUC
        try:
            train_auc = roc_auc_score(train_labels, train_preds)
        except ValueError:
            train_auc = 0.5
        history["train_auc"].append(train_auc)

        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, target_skills, labels, mask = [b.to(device) for b in batch]
                predictions = model(inputs, target_skills)
                loss = criterion(predictions, labels)
                loss = (loss * mask).sum() / mask.sum()
                val_losses.append(loss.item())
                mask_bool = mask.bool()
                all_preds.extend(predictions[mask_bool].cpu().numpy())
                all_labels.extend(labels[mask_bool].cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            val_auc = 0.5
        history["val_auc"].append(val_auc)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(
    model: DKTModel,
    sequences: List,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a DKT model."""
    model.eval()
    dataset = DKTDataset(sequences, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs, target_skills, labels, mask = [b.to(device) for b in batch]
            predictions = model(inputs, target_skills)
            mask_bool = mask.bool()
            all_preds.extend(predictions[mask_bool].cpu().numpy())
            all_labels.extend(labels[mask_bool].cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    n_samples = len(y_true)

    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return {"auc": None, "accuracy": None, "f1": None, "n_samples": n_samples}

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {"auc": auc, "accuracy": accuracy, "f1": f1, "n_samples": n_samples}


def run_experiment(config: ExperimentConfig) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Run the full experiment.

    Returns:
        Tuple of (results_df, all_histories) where all_histories contains training histories.
    """

    print("=" * 70)
    print(" DKT Experiment: Hidden Dimension & Data Quantity Ablation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading dataset...")
    df = load_dataset(config)
    print(f"  Rows: {len(df)}")
    print(f"  Students: {df[config.student_col].nunique()}")

    # Create a DKTConfig for data transformation
    dkt_config = DKTConfig(
        data_path=config.data_path,
        output_base_dir=config.output_dir,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
    )

    all_results = []
    all_histories = {}  # Store training histories for overfitting analysis

    # Calculate total number of experiments for progress bar
    total_experiments = len(config.modalities) * len(config.hidden_dims) * len(config.data_fractions)
    experiment_pbar = tqdm(total=total_experiments, desc="Total Progress", position=0)

    for modality in config.modalities:
        print(f"\n{'='*50}")
        print(f"Modality: {modality}")
        print("=" * 50)

        # Transform data to sequences
        print("  Transforming data to sequences...")
        sequences, skill_to_idx, student_ids = transform_to_sequences(df, dkt_config, modality)
        num_skills = len(skill_to_idx)

        print(f"  Total sequences: {len(sequences)}")
        print(f"  Number of skills: {num_skills}")

        # Split by student
        train_seq, test_seq, train_students, test_students = split_sequences_by_student(
            sequences, student_ids, config.test_ratio, config.random_seed
        )

        print(f"  Train sequences: {len(train_seq)}")
        print(f"  Test sequences: {len(test_seq)}")

        total_train_interactions = sum(len(seq) for seq in train_seq)
        print(f"  Total train interactions: {total_train_interactions}")

        for hidden_dim in config.hidden_dims:
            for data_fraction in config.data_fractions:
                # Select subset of training sequences based on fraction
                n_train_use = max(2, int(len(train_seq) * data_fraction))
                train_subset = train_seq[:n_train_use]
                n_interactions = sum(len(seq) for seq in train_subset)

                experiment_pbar.set_postfix({
                    "modality": modality[:10],
                    "hidden_dim": hidden_dim,
                    "data": f"{data_fraction:.0%}",
                    "seqs": n_train_use
                })

                try:
                    # Train model
                    model, history = train_model(
                        train_subset,
                        test_seq,
                        num_skills,
                        hidden_dim,
                        config,
                        device,
                    )

                    # Evaluate on test set
                    y_true, y_prob = evaluate_model(model, test_seq, config, device)
                    test_metrics = compute_metrics(y_true, y_prob, config.threshold)

                    # Evaluate on train set (for overfitting analysis)
                    y_train_true, y_train_prob = evaluate_model(model, train_subset, config, device)
                    train_metrics = compute_metrics(y_train_true, y_train_prob, config.threshold)

                    # Calculate overfitting gap
                    train_auc = train_metrics["auc"]
                    test_auc = test_metrics["auc"]
                    overfitting_gap = (train_auc - test_auc) if (train_auc and test_auc) else None

                    result = {
                        "modality": modality,
                        "hidden_dim": hidden_dim,
                        "data_fraction": data_fraction,
                        "train_sequences": n_train_use,
                        "train_interactions": n_interactions,
                        "test_sequences": len(test_seq),
                        "num_skills": num_skills,
                        # Test metrics
                        "test_auc": test_auc,
                        "test_accuracy": test_metrics["accuracy"],
                        "test_f1": test_metrics["f1"],
                        "n_test_samples": test_metrics["n_samples"],
                        # Train metrics (for overfitting analysis)
                        "train_auc": train_auc,
                        "train_accuracy": train_metrics["accuracy"],
                        "train_f1": train_metrics["f1"],
                        # Overfitting indicators
                        "overfitting_gap_auc": overfitting_gap,
                        "epochs_trained": len(history["train_loss"]),
                        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                        "loss_gap": (history["val_loss"][-1] - history["train_loss"][-1]) if history["train_loss"] else None,
                    }

                    # Store training history for this experiment (only for 100% data to avoid clutter)
                    if data_fraction == 1.0:
                        history_key = f"{modality}_hd{hidden_dim}"
                        all_histories[history_key] = {
                            "modality": modality,
                            "hidden_dim": hidden_dim,
                            "history": history,
                        }

                    auc_str = f"{test_auc:.4f}" if test_auc else "N/A"
                    gap_str = f"{overfitting_gap:.4f}" if overfitting_gap else "N/A"
                    tqdm.write(f"    [{modality}] hd={hidden_dim}, data={data_fraction:.0%} -> Test AUC: {auc_str}, Gap: {gap_str}")

                except Exception as e:
                    tqdm.write(f"      Error: {e}")
                    result = {
                        "modality": modality,
                        "hidden_dim": hidden_dim,
                        "data_fraction": data_fraction,
                        "train_sequences": n_train_use,
                        "train_interactions": n_interactions,
                        "test_sequences": len(test_seq),
                        "num_skills": num_skills,
                        "test_auc": None,
                        "test_accuracy": None,
                        "test_f1": None,
                        "n_test_samples": 0,
                        "train_auc": None,
                        "train_accuracy": None,
                        "train_f1": None,
                        "overfitting_gap_auc": None,
                        "epochs_trained": 0,
                        "final_train_loss": None,
                        "final_val_loss": None,
                        "loss_gap": None,
                    }

                all_results.append(result)
                experiment_pbar.update(1)

    experiment_pbar.close()

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.output_dir / "experiment_results.csv", index=False)
    print(f"\nResults saved to: {config.output_dir / 'experiment_results.csv'}")

    # Save training histories
    histories_dir = config.output_dir / "training_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    for key, hist_data in all_histories.items():
        hist_df = pd.DataFrame(hist_data["history"])
        hist_df["epoch"] = range(1, len(hist_df) + 1)
        hist_df.to_csv(histories_dir / f"{key}_history.csv", index=False)

    return results_df, all_histories


def plot_results(results_df: pd.DataFrame, all_histories: Dict[str, Dict], output_dir: Path) -> None:
    """Generate plots from experiment results including overfitting diagnostics."""

    sns.set_theme(style="whitegrid")
    plot_pbar = tqdm(total=12, desc="Generating plots")

    # ========================================================================
    # STANDARD PERFORMANCE PLOTS
    # ========================================================================

    # 1. Test AUC vs Hidden Dimension (for 100% data)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, modality in enumerate(["error_independent", "error_dependent"]):
        ax = axes[idx]
        modality_df = results_df[(results_df["modality"] == modality) & (results_df["data_fraction"] == 1.0)]

        if not modality_df.empty:
            ax.plot(modality_df["hidden_dim"], modality_df["test_auc"], marker="o", linewidth=2, markersize=10)
            ax.set_xlabel("Hidden Dimension", fontsize=12)
            ax.set_ylabel("Test AUC", fontsize=12)
            ax.set_title(f"{modality.replace('_', ' ').title()}\n(100% training data)", fontsize=14)
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128])
            ax.set_xticklabels([8, 16, 32, 64, 128])
            ax.set_ylim([0.5, 1.0])
            ax.grid(True, alpha=0.3)

    plt.suptitle("DKT: Test AUC vs Hidden Dimension", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "test_auc_vs_hidden_dim.png", dpi=200, bbox_inches="tight")
    plt.close()
    plot_pbar.update(1)

    # 2. Test AUC vs Data Fraction (by hidden dimension) - separate plot per modality
    for modality in ["error_independent", "error_dependent"]:
        fig, ax = plt.subplots(figsize=(10, 7))

        modality_df = results_df[results_df["modality"] == modality]
        colors = sns.color_palette("viridis", n_colors=len(results_df["hidden_dim"].unique()))

        for idx, hidden_dim in enumerate(sorted(results_df["hidden_dim"].unique())):
            hd_df = modality_df[modality_df["hidden_dim"] == hidden_dim].sort_values("data_fraction")
            if not hd_df.empty:
                ax.plot(
                    hd_df["data_fraction"] * 100,
                    hd_df["test_auc"],
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    color=colors[idx],
                    label=f"hidden_dim={hidden_dim}"
                )

        ax.set_xlabel("Training Data (%)", fontsize=12)
        ax.set_ylabel("Test AUC", fontsize=12)
        ax.set_title(f"DKT: Test AUC vs Training Data Size\n{modality.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        ax.set_ylim([0.5, 1.0])
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"test_auc_vs_data_fraction_{modality}.png", dpi=200, bbox_inches="tight")
        plt.close()
        plot_pbar.update(1)

    # ========================================================================
    # OVERFITTING DIAGNOSTIC PLOTS
    # ========================================================================

    # 3. Train vs Test AUC Comparison (CRITICAL FOR OVERFITTING DETECTION)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, modality in enumerate(["error_independent", "error_dependent"]):
        ax = axes[idx]
        modality_df = results_df[(results_df["modality"] == modality) & (results_df["data_fraction"] == 1.0)]
        modality_df = modality_df.sort_values("hidden_dim")

        if not modality_df.empty:
            ax.plot(modality_df["hidden_dim"], modality_df["train_auc"], marker="s", linewidth=2, markersize=10, color="#2ecc71", label="Train AUC")
            ax.plot(modality_df["hidden_dim"], modality_df["test_auc"], marker="o", linewidth=2, markersize=10, color="#e74c3c", label="Test AUC")

            # Fill the gap area to highlight overfitting
            ax.fill_between(modality_df["hidden_dim"], modality_df["train_auc"], modality_df["test_auc"], alpha=0.3, color="#f39c12", label="Overfitting Gap")

            ax.set_xlabel("Hidden Dimension", fontsize=12)
            ax.set_ylabel("AUC", fontsize=12)
            ax.set_title(f"{modality.replace('_', ' ').title()}", fontsize=14)
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128])
            ax.set_xticklabels([8, 16, 32, 64, 128])
            ax.set_ylim([0.5, 1.0])
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.suptitle("DKT: Train vs Test AUC (Overfitting Analysis)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "overfitting_train_vs_test_auc.png", dpi=200, bbox_inches="tight")
    plt.close()
    plot_pbar.update(1)

    # 4. Overfitting Gap vs Hidden Dimension
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"error_independent": "#1f77b4", "error_dependent": "#ff7f0e"}

    for modality in ["error_independent", "error_dependent"]:
        modality_df = results_df[(results_df["modality"] == modality) & (results_df["data_fraction"] == 1.0)]
        modality_df = modality_df.sort_values("hidden_dim")

        if not modality_df.empty:
            ax.plot(
                modality_df["hidden_dim"],
                modality_df["overfitting_gap_auc"],
                marker="o",
                linewidth=2,
                markersize=10,
                color=colors[modality],
                label=modality.replace("_", " ").title()
            )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="No Overfitting")
    ax.axhline(y=0.05, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Mild Overfitting (0.05)")
    ax.axhline(y=0.10, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Significant Overfitting (0.10)")

    ax.set_xlabel("Hidden Dimension", fontsize=12)
    ax.set_ylabel("Overfitting Gap (Train AUC - Test AUC)", fontsize=12)
    ax.set_title("DKT: Overfitting Gap vs Model Complexity", fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.set_xticklabels([8, 16, 32, 64, 128])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "overfitting_gap_vs_hidden_dim.png", dpi=200, bbox_inches="tight")
    plt.close()
    plot_pbar.update(1)

    # 5. Learning Curves (Train Loss vs Val Loss over epochs)
    if all_histories:
        for modality in ["error_independent", "error_dependent"]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Filter histories for this modality
            modality_histories = {k: v for k, v in all_histories.items() if v["modality"] == modality}

            # Plot Loss curves
            ax = axes[0]
            colors = sns.color_palette("viridis", n_colors=len(modality_histories))
            for idx, (key, hist_data) in enumerate(sorted(modality_histories.items())):
                history = hist_data["history"]
                hidden_dim = hist_data["hidden_dim"]
                epochs = range(1, len(history["train_loss"]) + 1)

                ax.plot(epochs, history["train_loss"], linestyle="-", color=colors[idx], linewidth=2, label=f"Train (hd={hidden_dim})")
                ax.plot(epochs, history["val_loss"], linestyle="--", color=colors[idx], linewidth=2, label=f"Val (hd={hidden_dim})")

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss (BCE)", fontsize=12)
            ax.set_title("Loss Curves", fontsize=14)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

            # Plot AUC curves
            ax = axes[1]
            for idx, (key, hist_data) in enumerate(sorted(modality_histories.items())):
                history = hist_data["history"]
                hidden_dim = hist_data["hidden_dim"]
                epochs = range(1, len(history["train_auc"]) + 1)

                ax.plot(epochs, history["train_auc"], linestyle="-", color=colors[idx], linewidth=2, label=f"Train (hd={hidden_dim})")
                ax.plot(epochs, history["val_auc"], linestyle="--", color=colors[idx], linewidth=2, label=f"Val (hd={hidden_dim})")

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("AUC", fontsize=12)
            ax.set_title("AUC Curves", fontsize=14)
            ax.set_ylim([0.5, 1.0])
            ax.legend(loc="lower right", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

            plt.suptitle(f"DKT Learning Curves: {modality.replace('_', ' ').title()}", fontsize=16, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / f"learning_curves_{modality}.png", dpi=200, bbox_inches="tight")
            plt.close()
            plot_pbar.update(1)

    # 6. Heatmap: Overfitting Gap by hidden_dim and data_fraction
    for modality in ["error_independent", "error_dependent"]:
        modality_df = results_df[results_df["modality"] == modality]

        pivot = modality_df.pivot_table(
            values="overfitting_gap_auc",
            index="hidden_dim",
            columns="data_fraction",
            aggfunc="mean"
        )

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn_r",  # Red = high gap (bad), Green = low gap (good)
                center=0,
                ax=ax
            )
            ax.set_xlabel("Data Fraction", fontsize=12)
            ax.set_ylabel("Hidden Dimension", fontsize=12)
            ax.set_title(f"DKT Overfitting Gap Heatmap: {modality.replace('_', ' ').title()}\n(Train AUC - Test AUC)", fontsize=14, fontweight="bold")

            plt.tight_layout()
            plt.savefig(output_dir / f"overfitting_heatmap_{modality}.png", dpi=200, bbox_inches="tight")
            plt.close()
            plot_pbar.update(1)

    # 7. Test AUC Heatmap (for reference)
    for modality in ["error_independent", "error_dependent"]:
        modality_df = results_df[results_df["modality"] == modality]

        pivot = modality_df.pivot_table(
            values="test_auc",
            index="hidden_dim",
            columns="data_fraction",
            aggfunc="mean"
        )

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                vmin=0.5,
                vmax=1.0,
                ax=ax
            )
            ax.set_xlabel("Data Fraction", fontsize=12)
            ax.set_ylabel("Hidden Dimension", fontsize=12)
            ax.set_title(f"DKT Test AUC Heatmap: {modality.replace('_', ' ').title()}", fontsize=14, fontweight="bold")

            plt.tight_layout()
            plt.savefig(output_dir / f"test_auc_heatmap_{modality}.png", dpi=200, bbox_inches="tight")
            plt.close()
            plot_pbar.update(1)

    # 8. Summary table
    summary = results_df.groupby(["modality", "hidden_dim"]).agg({
        "test_auc": ["mean", "std", "min", "max"],
        "train_auc": ["mean", "std"],
        "overfitting_gap_auc": ["mean", "std"],
        "test_accuracy": ["mean", "std"],
        "test_f1": ["mean", "std"],
    }).round(4)
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.to_csv(output_dir / "experiment_summary.csv")
    plot_pbar.update(1)
    plot_pbar.close()


def generate_readme(output_dir: Path) -> None:
    """Generate README explaining how to interpret the plots."""

    readme_content = """# DKT Experiment Results: Overfitting Analysis

## Overview

This experiment tests Deep Knowledge Tracing (DKT) models with different hidden dimensions
(8, 16, 32, 64, 128) across two modalities to diagnose potential overfitting issues.

## Modalities

- **Error Independent**: All expected skills receive the submission outcome (OK/KO)
- **Error Dependent**: Only skills from actual errors are marked as incorrect

---

## How to Interpret the Figures

### 1. Overfitting Detection Plots

#### `overfitting_train_vs_test_auc.png` (MOST IMPORTANT)
- **What it shows**: Train AUC (green) vs Test AUC (red) across hidden dimensions
- **Orange shaded area**: The "overfitting gap" between train and test performance
- **How to interpret**:
  - **Large gap** = OVERFITTING (model memorizes training data)
  - **Small/no gap** = Good generalization
  - **If Train AUC ≈ 1.0 but Test AUC << 1.0** → Severe overfitting

#### `overfitting_gap_vs_hidden_dim.png`
- **What it shows**: Overfitting gap (Train AUC - Test AUC) vs hidden dimension
- **Reference lines**:
  - Gray dashed (0.0): No overfitting
  - Orange dashed (0.05): Mild overfitting threshold
  - Red dashed (0.10): Significant overfitting threshold
- **How to interpret**:
  - **Gap increasing with hidden_dim** → Larger models overfit more
  - **Gap > 0.10** → Model is significantly overfitting
  - **Optimal hidden_dim**: Smallest model with acceptable test performance

#### `overfitting_heatmap_*.png`
- **What it shows**: Overfitting gap across hidden_dim × data_fraction
- **Color scale**: Green = low gap (good), Red = high gap (bad)
- **How to interpret**:
  - **Red cells at large hidden_dim + small data** → Classic overfitting pattern
  - **Green cells** → Good generalization
  - If all cells are red → Problem with data or model architecture

### 2. Learning Curves

#### `learning_curves_*.png`
- **Left panel**: Loss curves (Train vs Val) over epochs
- **Right panel**: AUC curves (Train vs Val) over epochs
- **How to interpret**:
  - **Train loss ↓, Val loss ↓** → Model is learning
  - **Train loss ↓, Val loss ↑** → OVERFITTING (stop training earlier)
  - **Train AUC >> Val AUC** → OVERFITTING
  - **Gap increases with epochs** → Model starts overfitting after N epochs

### 3. Performance vs Data Size

#### `test_auc_vs_data_fraction_*.png`
- **What it shows**: How test AUC changes with training data amount
- **How to interpret**:
  - **Lines converge at high data** → Sufficient data for all model sizes
  - **Large models worse at low data** → Overfitting with limited data
  - **All lines plateau early** → Data might not be limiting factor

#### `test_auc_heatmap_*.png`
- **What it shows**: Test AUC across hidden_dim × data_fraction
- **How to interpret**:
  - Look for the "sweet spot" (high AUC with reasonable hidden_dim)
  - Compare with overfitting heatmap to find best trade-off

---

## Diagnosing Your 0.99 AUC

If you're seeing AUC = 0.99, check:

1. **Is Train AUC also ~0.99?**
   - YES → Model might be correct OR data has patterns too easy to learn
   - NO (Train >> Test) → Likely overfitting

2. **Does gap increase with hidden_dim?**
   - YES → Smaller models generalize better, use hidden_dim=8 or 16
   - NO → Issue might not be model capacity

3. **Look at learning curves:**
   - Val loss increasing while train loss decreases? → Overfitting
   - Both decreasing? → Model is learning genuinely

4. **Check data fraction results:**
   - AUC high even with 10% data? → Data might have data leakage
   - AUC increases with data and levels off? → Normal behavior

---

## Common Overfitting Indicators

| Symptom | Diagnosis |
|---------|-----------|
| Train AUC = 0.99, Test AUC = 0.70 | Severe overfitting |
| Gap increases with hidden_dim | Model too complex |
| Val loss increases after epoch N | Train too long |
| Same high AUC for all data fractions | Possible data leakage |
| Learning curves diverge early | Overfitting from start |

---

## Recommendations Based on Results

- **If overfitting detected**: Use smallest hidden_dim with acceptable test AUC
- **If no overfitting but AUC too high**: Check for data leakage, class imbalance
- **If all models perform similarly**: Data characteristics dominate over model
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    print(f"  Saved: README.md")


def main(dataset_name: str = "platform_a"):
    """Main entry point.

    Args:
        dataset_name: Either 'platform_a' or 'platform_b'
    """
    base_path = Path(__file__).parent.parent.parent

    # Get dataset configuration
    from src.common.dataset_config import get_dataset_config
    dataset_config = get_dataset_config(dataset_name, base_path)

    config = ExperimentConfig(
        data_path=dataset_config.data_path,
        output_dir=dataset_config.output_base_dir / "dkt_experiment",
        hidden_dims=[8, 16, 32, 64, 128],
        modalities=["error_independent", "error_dependent"],
        data_fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
        # Column mappings from dataset config
        student_col=dataset_config.student_col,
        statut_col=dataset_config.statut_col,
        timestamp_col=dataset_config.timestamp_col,
        expected_tasks_col=dataset_config.expected_tasks_col,
        tasks_from_errors_col=dataset_config.tasks_from_errors_col,
    )

    print(f"\nDataset: {dataset_name.upper()}")

    # Run experiment
    results_df, all_histories = run_experiment(config)

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results_df, all_histories, config.output_dir)

    # Generate README
    generate_readme(config.output_dir)

    print("\n" + "=" * 70)
    print(f" Experiment Complete - {dataset_name.upper()}")
    print("=" * 70)
    print(f"\nOutput directory: {config.output_dir}")
    print("\nFiles generated:")
    print("  DATA:")
    print("    - experiment_results.csv (full results with train/test metrics)")
    print("    - experiment_summary.csv (aggregated stats)")
    print("    - training_histories/*.csv (epoch-by-epoch metrics)")
    print("  OVERFITTING DIAGNOSTICS:")
    print("    - overfitting_train_vs_test_auc.png (CRITICAL: shows gap)")
    print("    - overfitting_gap_vs_hidden_dim.png (gap vs model size)")
    print("    - overfitting_heatmap_*.png (gap across all conditions)")
    print("    - learning_curves_*.png (loss/AUC over epochs)")
    print("  PERFORMANCE:")
    print("    - test_auc_vs_hidden_dim.png")
    print("    - test_auc_vs_data_fraction_*.png")
    print("    - test_auc_heatmap_*.png")
    print("  DOCUMENTATION:")
    print("    - README.md (interpretation guide)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DKT overfitting experiment on specified dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["platform_a", "platform_b"],
        default="platform_a",
        help="Dataset to use (default: platform_a)"
    )
    args = parser.parse_args()
    main(args.dataset)
