"""1D Grad-CAM for InceptionTime + Attention ECG classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d


@dataclass
class GradCAMResult:
    heatmap: np.ndarray  # (T,) — Grad-CAM activation, normalised [0, 1]
    heatmap_smooth: np.ndarray  # (T,) — Gaussian-smoothed version for display
    attention: np.ndarray  # (T,) — temporal attention alpha weights
    predicted_probs: np.ndarray  # (5,) — sigmoid probabilities for all classes


class GradCAM1D:
    """Grad-CAM adapted for 1D convolutional models.

    Hooks into a target Conv1D layer, computes the gradient of a target
    class logit w.r.t. the feature map, and produces a per-timestep
    saliency heatmap.

    Usage (context manager — prevents hook leaks)::

        with GradCAM1D(model, model.inception_blocks[2]) as gc:
            result = gc.generate(x, class_idx=1)

    Args:
        model: The InceptionTimeAttention model (must be in eval mode).
        target_layer: The nn.Module to hook — typically the last InceptionBlock.
        smooth_sigma: Gaussian sigma applied to heatmap for display (raw heatmap
            is also returned unmodified).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        smooth_sigma: float = 5.0,
    ) -> None:
        self.model = model
        self.smooth_sigma = smooth_sigma

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self._gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> GradCAMResult:
        """Compute Grad-CAM for *class_idx* on input *x*.

        Args:
            x: Input tensor of shape (1, 12, T). Must be on the same device
               as the model. Batch size must be 1.
            class_idx: Index of the target class (0=NORM, 1=MI, 2=STTC, 3=CD, 4=HYP).

        Returns:
            GradCAMResult with heatmap, smoothed heatmap, attention weights,
            and predicted probabilities.
        """
        assert x.shape[0] == 1, "Batch size must be 1 for Grad-CAM."

        self.model.zero_grad()
        logits, alpha = self.model(x)

        # Backprop gradient of the target class logit (pre-sigmoid)
        score = logits[0, class_idx]
        score.backward()

        # activations: (1, C, T) → (C, T)
        A = self._activations[0]  # (C, T)
        G = self._gradients[0]  # (C, T)

        # Global Average Pool gradients over time → channel weights
        w = G.mean(dim=-1)  # (C,)

        # Weighted combination of activation maps
        cam = torch.einsum("c,ct->t", w, A).cpu().numpy()  # (T,)

        # ReLU — keep only positive contributions
        cam = np.maximum(cam, 0)

        # Normalise to [0, 1]
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max

        cam_smooth = gaussian_filter1d(cam, sigma=self.smooth_sigma)
        # Re-normalise after smoothing
        sm_max = cam_smooth.max()
        if sm_max > 1e-8:
            cam_smooth = cam_smooth / sm_max

        return GradCAMResult(
            heatmap=cam,
            heatmap_smooth=cam_smooth,
            attention=alpha[0].detach().cpu().numpy(),
            predicted_probs=torch.sigmoid(logits[0]).detach().cpu().numpy(),
        )

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> GradCAM1D:
        return self

    def __exit__(self, *args: object) -> None:
        self.remove_hooks()
