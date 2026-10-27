"""Fine-tune a pre-trained YOLO object detection model to find vehicle damage.

Two runs with identical settings, so the value of transfer learning is measured rather
than assumed:

    pretrained  YOLO11n starting from weights learned on the COCO dataset (transfer learning)
    scratch     the same architecture starting from random weights

Both are scored on the held-out test split (never used for training or model selection):
box mAP at IoU 0.5 and 0.5 to 0.95, precision and recall, overall and per damage class.

Usage:
    python src/train.py                       # 20 epochs each, Apple GPU if available
    python src/train.py --epochs 25 --skip-scratch
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "dataset.yaml"
RUNS = ROOT / "runs"


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def scores(metrics, names: dict) -> dict:
    """Pull the numbers we report out of an Ultralytics metrics object."""
    out = {
        "box_map50": round(float(metrics.box.map50), 3), "box_map50_95": round(float(metrics.box.map), 3),
        "precision": round(float(metrics.box.mp), 3), "recall": round(float(metrics.box.mr), 3),
        "per_class": {},
    }
    for idx, cls in enumerate(metrics.box.ap_class_index):
        out["per_class"][names[int(cls)]] = {
            "box_map50": round(float(metrics.box.ap50[idx]), 3),
            "precision": round(float(metrics.box.p[idx]), 3),
            "recall": round(float(metrics.box.r[idx]), 3),
        }
    return out


def run(name: str, weights: str, epochs: int, imgsz: int, device: str, batch: int) -> dict:
    from ultralytics import YOLO

    start = time.time()
    model = YOLO(weights)
    model.train(data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, device=device, batch=batch,
                project=str(RUNS), name=name, exist_ok=True, seed=7, deterministic=True,
                patience=15, plots=True, verbose=False)
    minutes = (time.time() - start) / 60
    best = RUNS / name / "weights" / "best.pt"
    trained = YOLO(str(best))
    test = trained.val(data=str(DATA_YAML), split="test", imgsz=imgsz, device=device,
                       project=str(RUNS), name=f"{name}_test", exist_ok=True, plots=True, verbose=False)
    result = {"weights_start": weights, "epochs": epochs, "train_minutes": round(minutes, 1),
              **scores(test, trained.names)}
    print(f"{name:<10} mAP50 {result['box_map50']:.3f}  mAP50-95 {result['box_map50_95']:.3f}  "
          f"precision {result['precision']:.3f}  recall {result['recall']:.3f}  ({minutes:.0f} min)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolo11n")
    parser.add_argument("--skip-scratch", action="store_true")
    args = parser.parse_args()
    if not DATA_YAML.exists():
        raise SystemExit("Run python src/prepare_data.py first.")

    device = pick_device()
    print(f"Training on {device}")
    results = {"device": device, "imgsz": args.imgsz}
    results["pretrained"] = run("pretrained", f"{args.model}.pt", args.epochs, args.imgsz, device, args.batch)
    if not args.skip_scratch:
        results["scratch"] = run("scratch", f"{args.model}.yaml", args.epochs, args.imgsz, device, args.batch)

    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(RUNS / "pretrained" / "weights" / "best.pt", ROOT / "models" / "damage_detector.pt")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved models/damage_detector.pt and reports/metrics.json")


if __name__ == "__main__":
    main()
