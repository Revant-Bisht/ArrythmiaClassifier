"""Full InceptionTime + Temporal Attention model for multi-label ECG classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from arrhythmia.models.attention import TemporalAttention
from arrhythmia.models.inception_time import InceptionBlock


class InceptionTimeAttention(nn.Module):
    """Stack of InceptionBlocks followed by temporal soft-attention and a linear classifier.

    Architecture:
        (B, in_channels, 1000)
        → InceptionBlock × num_blocks     (B, num_filters*4, 1000)
        → TemporalAttention               (B, num_filters*4)
        → Linear → Sigmoid                (B, num_classes)

    Args:
        in_channels: Number of ECG leads (12 for PTB-XL).
        num_classes: Number of output classes (5 superdiagnostic classes).
        num_filters: Filters per branch inside each InceptionBlock.
        bottleneck_size: Bottleneck width inside each InceptionBlock.
        num_blocks: Number of stacked InceptionBlocks.
        kernel_sizes: Kernel sizes for the three parallel conv branches.
        attention_hidden: Hidden size of the attention projection.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        num_filters: int = 32,
        bottleneck_size: int = 32,
        num_blocks: int = 3,
        kernel_sizes: tuple[int, int, int] = (10, 20, 40),
        attention_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        hidden_size = num_filters * 4  # 128 with default num_filters=32

        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(
                InceptionBlock(
                    in_channels=in_channels if i == 0 else hidden_size,
                    num_filters=num_filters,
                    bottleneck_size=bottleneck_size,
                    kernel_sizes=kernel_sizes,
                    use_residual=(i == 0),
                )
            )
        self.inception_blocks = nn.Sequential(*blocks)

        self.attention = TemporalAttention(
            hidden_size=hidden_size,
            attention_hidden=attention_hidden,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)

        Returns:
            logits: (B, num_classes) — raw logits (apply sigmoid for probabilities).
            alpha:  (B, T)           — attention weights (returned for explainability).
        """
        h = self.inception_blocks(x)  # (B, hidden, T)
        context, alpha = self.attention(h)  # (B, hidden), (B, T)
        logits = self.classifier(self.dropout(context))  # (B, num_classes)
        return logits, alpha

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities without the attention weights."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)
