"""On-the-fly augmentation transforms for 12-lead ECG signals."""

from __future__ import annotations

import torch


class GaussianNoise:
    """Add i.i.d. Gaussian noise to every lead."""

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class LeadDropout:
    """Zero out each lead independently with probability *p*."""

    def __init__(self, p: float = 0.10) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(torch.full((x.shape[0], 1), 1 - self.p))
        return x * mask


class TimeShift:
    """Circularly shift the signal by a random number of samples in [-max_shift, max_shift]."""

    def __init__(self, max_shift: int = 50) -> None:
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(x, shift, dims=-1)


class Compose:
    """Apply a sequence of transforms."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def build_train_transform(
    gaussian_noise_std: float = 0.01,
    lead_dropout_prob: float = 0.10,
    time_shift_max: int = 50,
) -> Compose:
    return Compose(
        [
            GaussianNoise(std=gaussian_noise_std),
            LeadDropout(p=lead_dropout_prob),
            TimeShift(max_shift=time_shift_max),
        ]
    )
