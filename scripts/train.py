"""Training entry point — reads configs/default.yaml."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from arrhythmia.data.dataset import PTBXLConfig, PTBXLDataset
from arrhythmia.data.transforms import build_train_transform
from arrhythmia.models.inception_time_attention import InceptionTimeAttention
from arrhythmia.training.trainer import Trainer
from arrhythmia.utils.logging import configure_root, get_logger

log = get_logger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InceptionTime + Attention on PTB-XL")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    configure_root(cfg["logging"]["level"])
    _set_seed(cfg["training"]["seed"])

    data_root = Path(cfg["data"]["raw_dir"])

    train_transform = (
        build_train_transform(
            gaussian_noise_std=cfg["augmentation"]["gaussian_noise_std"],
            lead_dropout_prob=cfg["augmentation"]["lead_dropout_prob"],
            time_shift_max=cfg["augmentation"]["time_shift_max"],
        )
        if cfg["augmentation"]["enabled"]
        else None
    )

    train_ds = PTBXLDataset(
        PTBXLConfig(
            root_dir=data_root,
            sampling_rate=cfg["data"]["sampling_rate"],
            folds=tuple(cfg["data"]["train_folds"]),
            min_likelihood=cfg["data"]["min_likelihood"],
        ),
        transform=train_transform,
    )
    val_ds = PTBXLDataset(
        PTBXLConfig(
            root_dir=data_root,
            sampling_rate=cfg["data"]["sampling_rate"],
            folds=tuple(cfg["data"]["val_folds"]),
            min_likelihood=cfg["data"]["min_likelihood"],
        ),
    )

    num_workers = cfg["training"]["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

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

    class_weights = train_ds.class_weights()
    log.info(
        "Class weights: %s",
        {k: round(float(v), 4) for k, v in zip(["NORM", "MI", "STTC", "CD", "HYP"], class_weights)},
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        num_epochs=cfg["training"]["num_epochs"],
        patience=cfg["training"]["early_stopping_patience"],
        label_smoothing=cfg["loss"]["label_smoothing"],
        checkpoint_dir=cfg["training"]["checkpoint_dir"],
        use_amp=cfg["training"]["mixed_precision"],
    )

    results = trainer.train()
    log.info("Best val macro AUC: %.4f (epoch %d)", results["best_val_auc"], results["best_epoch"])


if __name__ == "__main__":
    main()
