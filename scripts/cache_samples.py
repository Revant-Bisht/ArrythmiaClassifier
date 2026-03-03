"""Find the highest-confidence correct prediction per class, run 5x Grad-CAM,
and write complete API response JSONs to backend/cache/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from arrhythmia.data.dataset import PTBXLConfig, PTBXLDataset
from arrhythmia.explainability.gradcam import GradCAM1D
from arrhythmia.models.inception_time_attention import InceptionTimeAttention
from backend.reports import generate_report

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_II = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="backend/cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    test_ds = PTBXLDataset(
        PTBXLConfig(
            root_dir=Path(cfg["data"]["raw_dir"]),
            sampling_rate=cfg["data"]["sampling_rate"],
            folds=(10,),
            min_likelihood=cfg["data"]["min_likelihood"],
        )
    )

    best: dict[int, tuple] = {}

    with torch.no_grad():
        for signal, label, _ in test_ds:
            logits, _ = model(signal.unsqueeze(0))
            probs = torch.sigmoid(logits)[0].numpy()

            for c in range(5):
                current_best_conf = best[c][2] if c in best else -1.0
                if label[c] == 1.0 and float(probs[c]) > current_best_conf:
                    best[c] = (signal, label, float(probs[c]), probs.copy())

            if len(best) == 5 and all(best[c][2] > 0.98 for c in range(5)):
                break

    print(f"Samples found: {len(best)}/5")

    for c, name in enumerate(CLASS_NAMES):
        if c not in best:
            print(f"  SKIP {name}: no sample found")
            continue

        signal_t, _, conf, all_probs = best[c]
        x = signal_t.unsqueeze(0)

        gradcam_per_class: dict[str, list[float]] = {}
        attention_weights: list[float] = []

        for target_c, target_name in enumerate(CLASS_NAMES):
            with GradCAM1D(model, model.inception_blocks[2]) as gc:
                result = gc.generate(x, class_idx=target_c)
            gradcam_per_class[target_name] = result.heatmap_smooth.tolist()
            if target_c == c:
                attention_weights = result.attention.tolist()

        probs_dict = {n: float(p) for n, p in zip(CLASS_NAMES, all_probs)}
        top_cam = gradcam_per_class[name]
        report = generate_report(name, conf, top_cam)

        payload = {
            "sample_id": name,
            "predicted_class": name,
            "confidence": conf,
            "probs": probs_dict,
            "attention": attention_weights,
            "gradcam": top_cam,
            "gradcam_per_class": gradcam_per_class,
            "signal_lead2": signal_t[LEAD_II].tolist(),
            "report": report,
        }

        out_path = output_dir / f"sample_{name}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)

        print(
            f"  {name}: confidence {conf:.4f}  flagged regions: {len(report['flagged_regions'])}  → {out_path.name}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
