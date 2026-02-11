"""DKT model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import DKTModel, DKTDataset

if TYPE_CHECKING:
    from .config import DKTConfig


@dataclass
class DKTTrainResult:
    """Result of DKT training."""

    model: DKTModel
    skill_to_idx: Dict[str, int]
    train_sequences: List
    test_sequences: List
    modality: str
    level: str
    subgroup: Optional[str]
    train_history: Dict[str, List[float]]


def train_dkt_model(
    train_sequences: List[List[Tuple[int, int]]],
    test_sequences: List[List[Tuple[int, int]]],
    num_skills: int,
    config: DKTConfig,
    device: torch.device = None,
) -> Tuple[DKTModel, Dict[str, List[float]]]:
    """Train a DKT model.

    Args:
        train_sequences: Training sequences.
        test_sequences: Test sequences (for early stopping).
        num_skills: Number of unique skills.
        config: DKT configuration.
        device: Torch device.

    Returns:
        Tuple of (trained model, training history).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = DKTDataset(train_sequences, config.max_seq_len)
    test_dataset = DKTDataset(test_sequences, config.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = DKTModel(
        num_skills=num_skills,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            inputs, target_skills, labels, mask = [b.to(device) for b in batch]

            optimizer.zero_grad()
            predictions = model(inputs, target_skills)

            # Masked loss
            loss = criterion(predictions, labels)
            loss = (loss * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

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

                # Collect for AUC
                mask_bool = mask.bool()
                all_preds.extend(predictions[mask_bool].cpu().numpy())
                all_labels.extend(labels[mask_bool].cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score
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

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_dkt_model(
    model: DKTModel,
    sequences: List[List[Tuple[int, int]]],
    config: DKTConfig,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate DKT model.

    Args:
        model: Trained DKT model.
        sequences: Sequences to evaluate.
        config: DKT configuration.
        device: Torch device.

    Returns:
        Tuple of (y_true, y_prob).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
