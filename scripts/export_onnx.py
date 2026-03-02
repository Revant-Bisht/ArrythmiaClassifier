"""Export best_model.pt to ONNX and benchmark inference latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from arrhythmia.models.inception_time_attention import InceptionTimeAttention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="checkpoints/model.onnx")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    model = InceptionTimeAttention(
        in_channels=mcfg["in_channels"],
        num_classes=mcfg["num_classes"],
        num_filters=mcfg["num_filters"],
        bottleneck_size=mcfg["bottleneck_size"],
        num_blocks=mcfg["num_inception_blocks"],
        attention_hidden=mcfg["attention_hidden"],
        dropout=0.0,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, mcfg["in_channels"], 1000)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=cfg["export"]["opset_version"],
        input_names=["ecg"],
        output_names=["logits", "attention"],
        dynamic_axes={"ecg": {0: "batch_size"}},
    )
    print(f"Exported → {output_path}  ({output_path.stat().st_size / 1e6:.2f} MB)")

    with torch.no_grad():
        pt_logits, pt_alpha = model(dummy)

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_alpha = sess.run(None, {"ecg": dummy.numpy()})

    np.testing.assert_allclose(pt_logits.numpy(), onnx_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pt_alpha.numpy(), onnx_alpha, rtol=1e-4, atol=1e-4)
    print(
        f"Output verification passed — max diff logits: "
        f"{float(np.abs(pt_logits.numpy() - onnx_logits).max()):.2e}, "
        f"attention: {float(np.abs(pt_alpha.numpy() - onnx_alpha).max()):.2e}"
    )

    n_runs = 200
    x_np = dummy.numpy()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    pt_ms = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"ecg": x_np})
    onnx_ms = (time.perf_counter() - t0) / n_runs * 1000

    print(f"PyTorch : {pt_ms:.2f} ms/sample")
    print(f"ONNX    : {onnx_ms:.2f} ms/sample")
    print(f"Speedup : {pt_ms / onnx_ms:.2f}x")


if __name__ == "__main__":
    main()
