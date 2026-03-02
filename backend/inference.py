from __future__ import annotations

import numpy as np
import onnxruntime as ort
from scipy.special import expit

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ONNXSession:
    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    def run(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if signal.ndim == 2:
            signal = signal[np.newaxis]
        logits, attention = self._session.run(None, {"ecg": signal.astype(np.float32)})
        probs = expit(logits[0]).astype(np.float32)
        return probs, attention[0]

    def predict(self, signal: np.ndarray) -> dict:
        probs, attention = self.run(signal)
        predicted_idx = int(probs.argmax())
        return {
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probs": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
            "attention": attention.tolist(),
        }
