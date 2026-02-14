"""PTB-XL PyTorch Dataset."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from ..utils import get_logger
from .labels import NUM_CLASSES, SCP_CODE_MAP, SUPERCLASSES, Superclass

log = get_logger(__name__)


@dataclass(frozen=True)
class PTBXLConfig:
    """Configuration for a PTB-XL dataset split."""

    root_dir: Path
    sampling_rate: int = 100
    folds: tuple[int, ...] = field(default_factory=tuple)
    min_likelihood: float = 100.0

    def __post_init__(self) -> None:
        if self.sampling_rate not in (100, 500):
            raise ValueError(f"sampling_rate must be 100 or 500, got {self.sampling_rate}")
        if not (0.0 <= self.min_likelihood <= 100.0):
            raise ValueError(f"min_likelihood must be in [0, 100], got {self.min_likelihood}")


class PTBXLDataset(Dataset):
    """PyTorch Dataset for the PTB-XL ECG database.

    Returns per ``__getitem__``:
        signal : FloatTensor (12, T)  — 12 leads, T timesteps
        label  : FloatTensor (5,)     — multi-hot superdiagnostic vector
        meta   : dict                 — ecg_id, age, sex
    """

    def __init__(
        self,
        config: PTBXLConfig,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.transform = transform

        log.info(
            "Loading PTB-XL from %s | folds=%s | fs=%d Hz | min_likelihood=%.0f%%",
            config.root_dir,
            list(config.folds) or "all",
            config.sampling_rate,
            config.min_likelihood,
        )

        df = pd.read_csv(config.root_dir / "ptbxl_database.csv", index_col="ecg_id")
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        if config.folds:
            df = df[df["strat_fold"].isin(config.folds)]

        df["superclass"] = df["scp_codes"].apply(
            lambda codes: self._extract_superclasses(codes, config.min_likelihood)
        )
        df = df[df["superclass"].apply(len) > 0].reset_index()

        # Filter out records whose waveform files are unreadable (e.g. corrupted download)
        readable_mask = df["filename_lr"].apply(lambda fn: self._is_readable(config.root_dir / fn))
        n_bad = (~readable_mask).sum()
        if n_bad:
            log.warning("Skipping %d unreadable record(s)", n_bad)
        self._df = df[readable_mask].reset_index(drop=True)

        log.info(
            "Loaded %d records — class counts: %s",
            len(self._df),
            self._class_counts(),
        )

    @staticmethod
    def _is_readable(path: Path) -> bool:
        """Return False if the .dat file is clearly truncated relative to the .hea spec."""
        hea = path.with_suffix(".hea")
        dat = path.with_suffix(".dat")
        if not hea.exists() or not dat.exists():
            return False
        try:
            header = wfdb.rdheader(str(path))
            # Expected bytes: samples × leads × bytes-per-sample (format 16 = 2 bytes)
            expected = header.sig_len * header.n_sig * 2
            return dat.stat().st_size >= expected
        except Exception:
            return False

    @staticmethod
    def _extract_superclasses(
        scp_codes: dict[str, float], min_likelihood: float
    ) -> list[Superclass]:
        seen: set[Superclass] = set()
        for code, likelihood in scp_codes.items():
            if likelihood >= min_likelihood and code in SCP_CODE_MAP:
                seen.add(SCP_CODE_MAP[code])
        return list(seen)

    @staticmethod
    def _make_label_vector(superclasses: list[Superclass]) -> torch.Tensor:
        vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for cls in superclasses:
            vec[cls.index] = 1.0
        return vec

    def _load_signal(self, filename_lr: str) -> np.ndarray:
        """Load a 12-lead waveform; returns float32 array of shape (12, T)."""
        if self.config.sampling_rate == 100:
            path = self.config.root_dir / filename_lr
        else:
            path = self.config.root_dir / filename_lr.replace("records100", "records500")

        record = wfdb.rdrecord(str(path.with_suffix("")))
        signal: np.ndarray = record.p_signal.T  # (T, 12) → (12, T)
        return np.nan_to_num(signal, nan=0.0).astype(np.float32)

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """Per-lead z-score normalisation."""
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row = self._df.iloc[idx]

        signal = self._normalize(self._load_signal(row["filename_lr"]))
        signal_t = torch.from_numpy(signal)

        if self.transform is not None:
            signal_t = self.transform(signal_t)

        label = self._make_label_vector(row["superclass"])

        meta = {
            "ecg_id": int(row["ecg_id"]),
            "age": float(row["age"]) if not pd.isna(row["age"]) else -1.0,
            "sex": int(row["sex"]) if not pd.isna(row["sex"]) else -1,
        }
        return signal_t, label, meta

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted BCE loss."""
        counts = torch.zeros(NUM_CLASSES)
        for superclasses in self._df["superclass"]:
            for cls in superclasses:
                counts[cls.index] += 1
        return len(self._df) / (NUM_CLASSES * counts.clamp(min=1))

    def _class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {cls.value: 0 for cls in SUPERCLASSES}
        for superclasses in self._df["superclass"]:
            for cls in superclasses:
                counts[cls.value] += 1
        return counts
