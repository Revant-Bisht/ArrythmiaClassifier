"""PTB-XL PyTorch Dataset."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from .labels import LABEL_MAP, SUPERCLASS_INDEX, SUPERCLASS_NAMES


class PTBXLDataset(Dataset):
    """PyTorch Dataset for the PTB-XL ECG database.

    Returns:
        signal  : FloatTensor of shape (12, T)  — 12 leads, T timesteps
        label   : FloatTensor of shape (5,)     — multi-hot superdiagnostic vector
        meta    : dict with age, sex, ecg_id    — useful for the demo UI
    """

    def __init__(
        self,
        root_dir: str | Path,
        sampling_rate: int = 100,
        folds: list[int] | None = None,
        transform=None,
    ) -> None:
        self.root = Path(root_dir)
        self.sampling_rate = sampling_rate
        self.transform = transform

        df = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        if folds is not None:
            df = df[df["strat_fold"].isin(folds)]

        # Only keep records that have ≥1 superdiagnostic label
        df["superclass"] = df["scp_codes"].apply(self._extract_superclasses)
        self.df = df[df["superclass"].apply(len) > 0].reset_index()

    # ── Label helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_superclasses(scp_codes: dict[str, float]) -> list[str]:
        """Map a record's SCP codes to unique superdiagnostic class names."""
        classes: set[str] = set()
        for code, likelihood in scp_codes.items():
            if likelihood >= 100 and code in LABEL_MAP:
                classes.add(LABEL_MAP[code])
        return list(classes)

    def _make_label_vector(self, superclasses: list[str]) -> torch.Tensor:
        vec = torch.zeros(len(SUPERCLASS_NAMES), dtype=torch.float32)
        for cls in superclasses:
            vec[SUPERCLASS_INDEX[cls]] = 1.0
        return vec

    # ── Signal loading ─────────────────────────────────────────────────────────

    def _load_signal(self, filename_lr: str) -> np.ndarray:
        """Load a 12-lead ECG waveform from the PTB-XL WFDB files.

        Returns array of shape (12, T).
        """
        if self.sampling_rate == 100:
            path = self.root / filename_lr
        else:
            # 500 Hz files stored under records500/
            path = self.root / filename_lr.replace("records100", "records500")

        record = wfdb.rdrecord(str(path.with_suffix("")))
        signal = record.p_signal  # shape (T, 12)
        signal = signal.T  # → (12, T)

        # Replace NaN (rare signal gaps) with zeros
        signal = np.nan_to_num(signal, nan=0.0)
        return signal.astype(np.float32)

    # ── Normalisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """Per-lead z-score normalisation."""
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[idx]

        signal = self._load_signal(row["filename_lr"])
        signal = self._normalize(signal)
        signal_tensor = torch.from_numpy(signal)

        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)

        label = self._make_label_vector(row["superclass"])

        meta = {
            "ecg_id": int(row["ecg_id"]),
            "age": float(row["age"]) if not pd.isna(row["age"]) else -1.0,
            "sex": int(row["sex"]) if not pd.isna(row["sex"]) else -1,
        }

        return signal_tensor, label, meta

    # ── Utility ────────────────────────────────────────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted BCE loss."""
        counts = torch.zeros(len(SUPERCLASS_NAMES))
        for superclasses in self.df["superclass"]:
            for cls in superclasses:
                counts[SUPERCLASS_INDEX[cls]] += 1
        weights = len(self.df) / (len(SUPERCLASS_NAMES) * counts.clamp(min=1))
        return weights
