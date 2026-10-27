"""Unpack the Roboflow export, convert labels to standard bounding boxes, and check the data.

Expects a Roboflow export zip of the Car Damages Segmentation dataset (CC BY 4.0) in the
project folder or in ~/Downloads. The export used here is "YOLOv8 OBB" (rotated boxes, four
corner points). The script:

1. extracts it to data/roboflow/
2. converts every label to an axis-aligned YOLO detection box (class, centre x, centre y,
   width, height), the tightest upright box around the rotated one, in data/detect/
3. writes data/dataset.yaml with absolute paths, so training works from any folder
4. counts images and damage instances per class and split, checks every label
   (coordinates between 0 and 1, class id in range, box not empty), and writes
   reports/dataset_summary.md

Usage:
    python src/prepare_data.py                 # finds the zip in the project folder
    python src/prepare_data.py --zip path/to/export.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "roboflow"
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def find_zip() -> Path:
    candidates = list(ROOT.glob("*.zip")) + list((Path.home() / "Downloads").glob("Car Damages Segmentation*.zip"))
    zips = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise SystemExit("No dataset zip found in the project folder. Download the YOLOv8 export from Roboflow first.")
    return zips[0]


def check_label_file(path: Path, n_classes: int) -> tuple[list[int], list[str]]:
    """Return the class ids in a YOLO label file and any problems found."""
    classes, problems = [], []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls, coords = int(parts[0]), [float(x) for x in parts[1:]]
        except ValueError:
            problems.append(f"{path.name}:{i} not numeric")
            continue
        if not 0 <= cls < n_classes:
            problems.append(f"{path.name}:{i} class {cls} out of range")
        if len(coords) == 4:
            if coords[2] <= 0 or coords[3] <= 0:
                problems.append(f"{path.name}:{i} empty box")
        elif len(coords) < 6 or len(coords) % 2:
            problems.append(f"{path.name}:{i} polygon needs at least 3 x,y points")
        if any(c < -0.01 or c > 1.01 for c in coords):
            problems.append(f"{path.name}:{i} coordinate outside 0 to 1")
        classes.append(cls)
    return classes, problems


def to_detection_line(line: str) -> str | None:
    """Rotated box or polygon -> upright YOLO box. Already-upright boxes pass through unchanged."""
    parts = line.split()
    if len(parts) == 5:
        return line.strip()
    if len(parts) < 7:
        return None
    xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
    x1, x2 = max(min(xs), 0.0), min(max(xs), 1.0)
    y1, y2 = max(min(ys), 0.0), min(max(ys), 1.0)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return f"{parts[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def convert(src_root: Path, dst_root: Path) -> int:
    """Copy images and write upright-box labels for every split. Returns labels dropped as empty."""
    import shutil
    dropped = 0
    for folder in SPLITS.values():
        src_img, src_lbl = src_root / folder / "images", src_root / folder / "labels"
        if not src_img.exists():
            continue
        dst_img, dst_lbl = dst_root / folder / "images", dst_root / folder / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in src_img.iterdir():
            shutil.copy(img, dst_img / img.name)
            lbl = src_lbl / (img.stem + ".txt")
            lines = []
            if lbl.exists():
                for raw in lbl.read_text().splitlines():
                    if not raw.strip():
                        continue
                    out = to_detection_line(raw)
                    if out is None:
                        dropped += 1
                    else:
                        lines.append(out)
            (dst_lbl / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    return dropped


def summarise(root: Path, names: list[str]) -> dict:
    summary = {"classes": names, "splits": {}, "problems": []}
    for split, folder in SPLITS.items():
        img_dir, lbl_dir = root / folder / "images", root / folder / "labels"
        if not img_dir.exists():
            continue
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        counts, no_label = Counter(), 0
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists() or not lbl.read_text().strip():
                no_label += 1
                continue
            cls, probs = check_label_file(lbl, len(names))
            counts.update(names[c] for c in cls if 0 <= c < len(names))
            summary["problems"] += probs
        summary["splits"][split] = {"images": len(images), "images_without_damage": no_label,
                                    "instances": dict(counts)}
    return summary


def write_report(summary: dict, source: str) -> None:
    names = summary["classes"]
    lines = ["# Dataset summary", "", f"Source: {source}", "",
             "Licence: CC BY 4.0 (Roboflow Universe, Car Damages Segmentation v2, project ai-proyect).", "",
             "Labels: Roboflow YOLOv8 OBB export, converted to upright bounding boxes.", "",
             "| Class | " + " | ".join(summary["splits"]) + " |", "|---|" + "---|" * len(summary["splits"])]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(str(s["instances"].get(n, 0)) for s in summary["splits"].values()) + " |")
    lines.append("| **Images** | " + " | ".join(str(s["images"]) for s in summary["splits"].values()) + " |")
    lines += ["", f"Label problems found: {len(summary['problems'])}"]
    lines += [f"- {p}" for p in summary["problems"][:20]]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    zpath = args.zip or find_zip()
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(OUT)
    yaml_files = list(OUT.rglob("data.yaml"))
    if not yaml_files:
        raise SystemExit("No data.yaml in the zip. Make sure the export format was YOLOv8 or YOLOv11.")
    src_yaml = yaml_files[0]
    cfg = yaml.safe_load(src_yaml.read_text())
    root = DATA / "detect"
    dropped = convert(src_yaml.parent, root)
    names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][k] for k in sorted(cfg["names"])]

    dataset = {"path": str(root), "names": {i: n for i, n in enumerate(names)}}
    for split, folder in SPLITS.items():
        if (root / folder / "images").exists():
            dataset[split] = f"{folder}/images"
    if "test" not in dataset:
        dataset["test"] = dataset["val"]  # fall back so evaluation still runs
    (DATA / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    summary = summarise(root, names)
    write_report(summary, zpath.name)
    (ROOT / "reports" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Classes: {names}")
    for split, s in summary["splits"].items():
        print(f"{split:<6} {s['images']:>5} images  {sum(s['instances'].values()):>5} damage instances")
    print(f"Converted rotated boxes to upright boxes ({dropped} empty labels dropped).")
    print(f"Label problems: {len(summary['problems'])}. Wrote data/dataset.yaml and reports/dataset_summary.md")


if __name__ == "__main__":
    main()
