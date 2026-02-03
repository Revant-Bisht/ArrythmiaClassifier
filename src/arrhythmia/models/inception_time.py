"""InceptionTime block for 1-D time-series classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """Single Inception block with bottleneck, three parallel conv branches, and a MaxPool branch.

    Args:
        in_channels: Number of input channels.
        num_filters: Number of output filters per branch (total output = 4 × num_filters).
        bottleneck_size: Width of the bottleneck conv applied before the parallel branches.
        kernel_sizes: Tuple of three kernel sizes for the parallel conv branches.
        use_residual: Whether to add a learnable residual shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 32,
        bottleneck_size: int = 32,
        kernel_sizes: tuple[int, int, int] = (10, 20, 40),
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.use_residual = use_residual
        out_channels = num_filters * 4  # four branches concatenated

        # Bottleneck applied to all three parallel conv branches
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
        self.bn_bottleneck = nn.BatchNorm1d(bottleneck_size)

        # Parallel conv branches (operate on bottlenecked features)
        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_size,
                    num_filters,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        # MaxPool branch (operates directly on raw input)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False)

        self.bn_out = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Residual shortcut
        if use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]

        # Bottleneck
        h = self.relu(self.bn_bottleneck(self.bottleneck(x)))

        # Parallel conv branches — trim to T to handle even-kernel padding asymmetry
        branches = [conv(h)[..., :T] for conv in self.conv_branches]

        # MaxPool branch
        branches.append(self.conv_mp(self.maxpool(x))[..., :T])

        # Concatenate and normalise
        out = self.bn_out(torch.cat(branches, dim=1))

        if self.use_residual:
            out = out + self.shortcut(x)

        return self.relu(out)
