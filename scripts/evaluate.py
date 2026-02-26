"""Evaluate the best checkpoint on PTB-XL test fold 10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from torch.utils.data import DataLoader

from arrhythmia.data.dataset import PTBXLConfig, PTBXLDataset
from arrhythmia.models.inception_time_attention import InceptionTimeAttention
from arrhythmia.training.metrics import (
    compute_all_metrics,
    youden_threshold,
)
from arrhythmia.utils.logging import configure_root, get_logger

log = get_logger(__name__)

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

STRODTHOFF_BASELINES = {
    "Simple 1D CNN": 0.890,
    "LSTM + Bidir Attention": 0.907,
    "InceptionTime (ref)": 0.925,
    "xresnet1d101": 0.931,
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on PTB-XL test fold 10")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="checkpoints/test_results.json")
    parser.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        log.warning("Output already exists: %s  (use --force to re-run)", output_path)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    configure_root(cfg["logging"]["level"])
    device = _get_device()
    log.info("Evaluating on %s", device)

    # ── Reconstruct model ────────────────────────────────────────────────────
    mcfg = cfg["model"]
    model = InceptionTimeAttention(
        in_channels=mcfg["in_channels"],
        num_classes=mcfg["num_classes"],
        num_filters=mcfg["num_filters"],
        bottleneck_size=mcfg["bottleneck_size"],
        num_blocks=mcfg["num_inception_blocks"],
        attention_hidden=mcfg["attention_hidden"],
        dropout=mcfg.get("classifier_dropout", 0.0),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    log.info("Loaded checkpoint — val AUC: %.4f (epoch %d)", ckpt["val_auc"], ckpt["epoch"])

    # ── Youden thresholds from validation set ────────────────────────────────
    val_labels = ckpt["val_labels"]
    val_probs = ckpt["val_probs"]
    thresholds = youden_threshold(val_labels, val_probs)
    log.info(
        "Youden thresholds (from val): %s",
        {n: round(float(t), 3) for n, t in zip(CLASS_NAMES, thresholds)},
    )

    # ── Test DataLoader ───────────────────────────────────────────────────────
    data_root = Path(cfg["data"]["raw_dir"])
    test_ds = PTBXLDataset(
        PTBXLConfig(
            root_dir=data_root,
            sampling_rate=cfg["data"]["sampling_rate"],
            folds=(10,),
            min_likelihood=cfg["data"]["min_likelihood"],
        )
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    log.info("Test set: %d records", len(test_ds))

    # ── Inference ────────────────────────────────────────────────────────────
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for signals, labels, _ in test_loader:
            logits, _ = model(signals.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    labels_np = np.concatenate(all_labels, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)

    # Save raw arrays for notebook plotting
    np.save(Path(args.checkpoint).parent / "test_labels.npy", labels_np)
    np.save(Path(args.checkpoint).parent / "test_probs.npy", probs_np)
    log.info("Saved test_labels.npy and test_probs.npy")

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_all_metrics(labels_np, probs_np, CLASS_NAMES)

    preds_np = (probs_np >= thresholds).astype(int)
    per_class_f1 = {
        name: float(f1_score(labels_np[:, i], preds_np[:, i], zero_division=0))
        for i, name in enumerate(CLASS_NAMES)
    }
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    ml_cm = multilabel_confusion_matrix(labels_np, preds_np)

    # ── Print benchmark table ─────────────────────────────────────────────────
    all_models = {**STRODTHOFF_BASELINES, "Ours (InceptionTime+Attn)": metrics["macro_auc"]}
    print("\n" + "─" * 48)
    print(f"{'Model':<32} {'Macro AUC-ROC':>12}")
    print("─" * 48)
    for name, auc_val in all_models.items():
        marker = " ◀" if name.startswith("Ours") else ""
        print(f"{name:<32} {auc_val:>12.4f}{marker}")
    print("─" * 48)
    print()

    log.info("Test macro AUC-ROC:  %.4f", metrics["macro_auc"])
    log.info("Test macro AUPRC:    %.4f", metrics["macro_auprc"])
    log.info("Test macro F1:       %.4f", macro_f1)
    for name in CLASS_NAMES:
        log.info(
            "  %-5s  AUC %.4f  AUPRC %.4f  F1 %.4f",
            name,
            metrics["per_class_auc"][name],
            metrics["per_class_auprc"][name],
            per_class_f1[name],
        )

    # ── Save JSON ────────────────────────────────────────────────────────────
    results = {
        "test_fold": 10,
        "n_samples": len(test_ds),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "val_auc": float(ckpt["val_auc"]),
        "macro_auc": float(metrics["macro_auc"]),
        "macro_auprc": float(metrics["macro_auprc"]),
        "macro_f1": macro_f1,
        "per_class_auc": {k: float(v) for k, v in metrics["per_class_auc"].items()},
        "per_class_auprc": {k: float(v) for k, v in metrics["per_class_auprc"].items()},
        "per_class_f1": per_class_f1,
        "youden_thresholds": {n: float(t) for n, t in zip(CLASS_NAMES, thresholds)},
        "confusion_matrix_per_class": ml_cm.tolist(),
        "baselines": STRODTHOFF_BASELINES,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved → %s", output_path)


if __name__ == "__main__":
    main()
