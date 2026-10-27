"""Turn detected damage into a draft repair line list, and decide what needs a person to check.

The rules are deliberately simple and live in repair_rules.csv, so an estimator can change them
without touching code. They are illustrative only: real labour times would come from the
business's own repair data and industry time guides.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "repair_rules.csv"

SIZE_BANDS = [(0.02, "small"), (0.08, "medium"), (1.01, "large")]  # share of the image the damage covers


@dataclass
class Finding:
    label: str
    confidence: float
    area_fraction: float  # damage mask area / image area
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def size_band(area_fraction: float) -> str:
    for limit, band in SIZE_BANDS:
        if area_fraction < limit:
            return band
    return "large"


def load_rules(path: Path = RULES) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["damage"].strip().lower(), r["size"].strip().lower()): r for r in csv.DictReader(f)}


def repair_lines(findings: list[Finding], rules: dict | None = None) -> list[dict]:
    rules = rules or load_rules()
    lines = []
    for f in findings:
        band = size_band(f.area_fraction)
        rule = rules.get((f.label.lower(), band)) or rules.get((f.label.lower(), "any"))
        if rule is None:
            lines.append({"damage": f.label, "size": band, "operation": "Assess manually",
                          "labour_hours": None, "note": "No rule for this damage type"})
            continue
        lines.append({"damage": f.label, "size": band, "operation": rule["operation"],
                      "labour_hours": float(rule["labour_hours"]), "note": rule.get("note", "")})
    return lines


def needs_review(findings: list[Finding], low_conf: float = 0.5) -> list[str]:
    """Reasons an assessor should look at this claim before it goes further."""
    reasons = []
    if not findings:
        reasons.append("no damage found: check the photo covers the damaged area")
    weak = [f for f in findings if f.confidence < low_conf]
    if weak:
        reasons.append(f"{len(weak)} finding(s) below {low_conf:.0%} confidence")
    if any(size_band(f.area_fraction) == "large" for f in findings):
        reasons.append("large damage area: may need structural inspection")
    return reasons
