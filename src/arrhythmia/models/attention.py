"""Temporal soft-attention head for sequence classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Soft attention over the time dimension.

    Projects each timestep to a scalar energy score, applies softmax across
    time, and returns the weighted sum of the hidden states.

    Args:
        hidden_size: Dimensionality of the input hidden states.
        attention_hidden: Width of the intermediate projection.
    """

    def __init__(self, hidden_size: int, attention_hidden: int = 64) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, attention_hidden, bias=True)
        self.v = nn.Linear(attention_hidden, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, C, T) — output of the last InceptionBlock.

        Returns:
            context: (B, C) — attention-weighted context vector.
            alpha:   (B, T) — attention weights (sum to 1 over T).
        """
        # (B, C, T) -> (B, T, C)
        h_t = h.permute(0, 2, 1)

        # Energy scores: (B, T, attention_hidden) -> (B, T, 1) -> (B, T)
        e = self.v(torch.tanh(self.W(h_t))).squeeze(-1)

        alpha = torch.softmax(e, dim=-1)  # (B, T)

        # Weighted sum: (B, T, 1) * (B, T, C) -> (B, C)
        context = (alpha.unsqueeze(-1) * h_t).sum(dim=1)

        return context, alpha
