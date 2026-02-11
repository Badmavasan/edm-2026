"""DKT (Deep Knowledge Tracing) model using PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class DKTModel(nn.Module):
    """Deep Knowledge Tracing model.

    Architecture:
    - Embedding layer for (skill, correct) pairs
    - LSTM layers for sequence modeling
    - Linear output layer for prediction
    """

    def __init__(
        self,
        num_skills: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        """Initialize DKT model.

        Args:
            num_skills: Number of unique skills.
            hidden_dim: Hidden dimension for LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input: skill_id * 2 + correct (0 or 1)
        # This creates 2*num_skills possible inputs
        self.input_dim = num_skills * 2

        # Embedding for input (skill, correct) pairs
        self.embedding = nn.Embedding(self.input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer: predict probability for each skill
        self.output = nn.Linear(hidden_dim, num_skills)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        target_skills: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len) containing skill_id * 2 + correct.
            target_skills: Target skill indices of shape (batch, seq_len).

        Returns:
            Predictions of shape (batch, seq_len) for the target skills.
        """
        # Embed input
        embedded = self.embedding(x)  # (batch, seq_len, hidden_dim)
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        # Output for all skills
        output = self.output(lstm_out)  # (batch, seq_len, num_skills)
        output = self.sigmoid(output)

        # Gather predictions for target skills
        # target_skills: (batch, seq_len) -> (batch, seq_len, 1)
        target_skills = target_skills.unsqueeze(-1)
        predictions = torch.gather(output, 2, target_skills).squeeze(-1)  # (batch, seq_len)

        return predictions

    def predict_proba(
        self,
        x: torch.Tensor,
        target_skills: torch.Tensor,
    ) -> torch.Tensor:
        """Predict probabilities (same as forward, for clarity)."""
        return self.forward(x, target_skills)


class DKTDataset(torch.utils.data.Dataset):
    """Dataset for DKT model."""

    def __init__(
        self,
        sequences: list,
        max_seq_len: int = 200,
    ):
        """Initialize dataset.

        Args:
            sequences: List of sequences, each is a list of (skill_id, correct) tuples.
            max_seq_len: Maximum sequence length (will pad/truncate).
        """
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sequence.

        Returns:
            Tuple of (input, target_skills, labels, mask):
            - input: skill_id * 2 + correct for t-1 (shifted)
            - target_skills: skill_id at time t
            - labels: correct at time t
            - mask: valid positions mask
        """
        seq = self.sequences[idx]

        # Truncate if needed
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]

        seq_len = len(seq)

        # Create input (shifted by 1): predict t from t-1
        # First position gets a dummy input (skill=0, correct=0)
        inputs = [0]  # Dummy for first position
        target_skills = []
        labels = []

        for i, (skill_id, correct) in enumerate(seq):
            target_skills.append(skill_id)
            labels.append(correct)
            if i < len(seq) - 1:
                # Input for next step: current skill and correctness
                next_skill, next_correct = seq[i]
                inputs.append(next_skill * 2 + next_correct)

        # Pad sequences
        pad_len = self.max_seq_len - seq_len
        inputs = inputs[:seq_len]  # Ensure same length

        inputs = inputs + [0] * pad_len
        target_skills = target_skills + [0] * pad_len
        labels = labels + [0] * pad_len
        mask = [1] * seq_len + [0] * pad_len

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(target_skills, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
            torch.tensor(mask, dtype=torch.float),
        )
