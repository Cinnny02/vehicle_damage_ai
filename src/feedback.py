"""Human-in-the-loop: record what the assessor decided, and turn corrections into training data.

Every review is appended to feedback/review_log.jsonl. Accepted and relabelled findings are
also written as YOLO detection labels under feedback/dataset/, next to a copy of the image,
so a future training run can learn from the assessors' corrections. Rejected findings are
kept in the log (they show where the model produces false alarms) but not in the labels.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK = ROOT / "feedback"


def polygon_to_yolo(points: list[list[float]], width: int, height: int, cls: int) -> str:
    """One YOLO detection line (class, centre x, centre y, width, height, all 0..1) around the points."""
    xs = [min(max(x / width, 0), 1) for x, _ in points]
    ys = [min(max(y / height, 0), 1) for _, y in points]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def save_review(image_path: Path, width: int, height: int, decisions: list[dict], class_ids: dict[str, int],
                reviewer: str, folder: Path = FEEDBACK) -> Path:
    """decisions: [{label, confidence, decision ('accept'|'reject'|'relabel'), new_label, polygon}]"""
    folder.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "image": image_path.name,
              "reviewer": reviewer, "decisions": [{k: v for k, v in d.items() if k != "polygon"} for d in decisions]}
    with (folder / "review_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    kept = []
    for d in decisions:
        if d["decision"] == "reject":
            continue
        label = d["new_label"] if d["decision"] == "relabel" else d["label"]
        if label in class_ids and d.get("polygon"):
            kept.append(polygon_to_yolo(d["polygon"], width, height, class_ids[label]))
    img_dir, lbl_dir = folder / "dataset" / "images", folder / "dataset" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if image_path.exists():
        shutil.copy(image_path, img_dir / image_path.name)
    label_path = lbl_dir / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return label_path


def agreement(folder: Path = FEEDBACK) -> dict[str, dict]:
    """How often assessors accepted the model's findings, per damage class."""
    log = folder / "review_log.jsonl"
    stats: dict[str, Counter] = defaultdict(Counter)
    if not log.exists():
        return {}
    for line in log.read_text(encoding="utf-8").splitlines():
        for d in json.loads(line)["decisions"]:
            stats[d["label"]][d["decision"]] += 1
    return {label: {**c, "accept_rate": round(c["accept"] / sum(c.values()), 2)} for label, c in stats.items()}
