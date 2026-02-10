"""Training loop for InceptionTime + Temporal Attention."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from arrhythmia.training.metrics import compute_all_metrics, macro_auc_roc
from arrhythmia.utils.logging import get_logger

log = get_logger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


class Trainer:
    """Manages the full training lifecycle.

    Args:
        model: The model to train.
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        class_weights: (C,) tensor for weighted BCE loss.
        lr: Initial learning rate.
        weight_decay: L2 regularisation.
        num_epochs: Maximum number of training epochs.
        patience: Early-stopping patience (in epochs, measured on val macro AUC).
        label_smoothing: Label-smoothing ε applied before BCE.
        checkpoint_dir: Directory where the best checkpoint is saved.
        use_amp: Enable PyTorch Automatic Mixed Precision (requires CUDA).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        patience: int = 15,
        label_smoothing: float = 0.05,
        checkpoint_dir: str | Path = "checkpoints",
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = _get_device()
        log.info("Training on %s", self.device)
        self.model.to(self.device)

        self.class_weights = class_weights.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # AMP is only supported on CUDA; skip on MPS/CPU
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
        }

    def _smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        return y * (1 - eps) + eps / 2

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y_smooth = self._smooth_labels(labels)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, y_smooth, pos_weight=None, reduction="none"
        )
        return (bce * self.class_weights).mean()

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss = 0.0
        all_labels: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        with torch.set_grad_enabled(train):
            for signals, labels, _ in loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    logits, _ = self.model(signals)
                    loss = self._compute_loss(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * signals.size(0)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

        labels_np = np.concatenate(all_labels, axis=0)
        probs_np = np.concatenate(all_probs, axis=0)
        return total_loss / len(loader.dataset), labels_np, probs_np

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            dict with keys 'best_val_auc', 'best_epoch', 'history'.
        """
        best_val_auc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_loss, train_labels, train_probs = self._run_epoch(self.train_loader, train=True)
            val_loss, val_labels, val_probs = self._run_epoch(self.val_loader, train=False)

            self.scheduler.step()

            train_auc = macro_auc_roc(train_labels, train_probs)
            val_auc = macro_auc_roc(val_labels, val_probs)
            lr_now = self.scheduler.get_last_lr()[0]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)

            elapsed = time.time() - t0
            log.info(
                "Epoch %3d/%d | loss %.4f/%.4f | AUC %.4f/%.4f | lr %.2e | %.1fs",
                epoch,
                self.num_epochs,
                train_loss,
                val_loss,
                train_auc,
                val_auc,
                lr_now,
                elapsed,
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_auc, val_labels, val_probs)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    log.info(
                        "Early stopping — no improvement for %d epochs. Best val AUC: %.4f (epoch %d)",
                        self.patience,
                        best_val_auc,
                        best_epoch,
                    )
                    break

        log.info(
            "Training complete. Best val macro AUC: %.4f at epoch %d", best_val_auc, best_epoch
        )
        return {"best_val_auc": best_val_auc, "best_epoch": best_epoch, "history": self.history}

    def _save_checkpoint(
        self,
        epoch: int,
        val_auc: float,
        val_labels: np.ndarray,
        val_probs: np.ndarray,
    ) -> None:
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_auc": val_auc,
                "val_labels": val_labels,
                "val_probs": val_probs,
                "history": self.history,
            },
            path,
        )
        log.info("Checkpoint saved → %s  (val AUC %.4f)", path, val_auc)

    def evaluate(self, loader: DataLoader) -> dict:
        """Run inference on *loader* and return full metrics dict."""
        _, labels_np, probs_np = self._run_epoch(loader, train=False)
        metrics = compute_all_metrics(labels_np, probs_np, CLASS_NAMES)
        log.info(
            "Evaluation — macro AUC: %.4f  macro AUPRC: %.4f",
            metrics["macro_auc"],
            metrics["macro_auprc"],
        )
        for cls, auc_val in metrics["per_class_auc"].items():
            log.info("  %s AUC: %.4f", cls, auc_val)
        return metrics
