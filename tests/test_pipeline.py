"""Fast tests that need no dataset or GPU: pytest -q tests"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from feedback import agreement, polygon_to_yolo, save_review  # noqa: E402
from prepare_data import check_label_file  # noqa: E402
from repair import Finding, needs_review, repair_lines, size_band  # noqa: E402


def test_label_checker_catches_bad_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n5 0.1 0.1 0.2 0.1 0.2 0.2\n1 0.1 0.1 0.2\n2 0.1 1.5 0.2 0.1 0.2 0.2\n")
    classes, problems = check_label_file(f, n_classes=3)
    assert classes == [0, 5, 1, 2]
    assert len(problems) == 3  # class out of range, too few points, coordinate outside 0..1


def test_size_bands():
    assert (size_band(0.01), size_band(0.05), size_band(0.3)) == ("small", "medium", "large")


def test_repair_lines_use_rules_and_flag_unknowns():
    rules = {("dent", "small"): {"operation": "PDR", "labour_hours": "1.0"},
             ("broken part", "any"): {"operation": "Replace", "labour_hours": "2.5"}}
    lines = repair_lines([Finding("Dent", 0.9, 0.01, (0, 0, 1, 1)), Finding("Broken part", 0.8, 0.2, (0, 0, 1, 1)),
                          Finding("Rust", 0.7, 0.01, (0, 0, 1, 1))], rules)
    assert [l["operation"] for l in lines] == ["PDR", "Replace", "Assess manually"]


def test_review_triggers():
    assert needs_review([]) == ["no damage found: check the photo covers the damaged area"]
    assert any("confidence" in r for r in needs_review([Finding("Dent", 0.3, 0.01, (0, 0, 1, 1))]))
    assert needs_review([Finding("Dent", 0.9, 0.01, (0, 0, 1, 1))]) == []


def test_polygon_to_yolo_normalises_and_clips():
    line = polygon_to_yolo([[0, 0], [200, 0], [250, 100]], 200, 100, 3)
    assert line.split()[0] == "3"
    assert [float(v) for v in line.split()[1:]] == [0.5, 0.5, 1.0, 1.0]


def test_rotated_box_becomes_upright_box():
    from prepare_data import to_detection_line
    out = to_detection_line("8 0.875 0.36 0.476 0.36 0.476 0.87 0.875 0.87")
    cls, cx, cy, w, h = out.split()
    assert cls == "8" and abs(float(cx) - 0.6755) < 1e-3 and abs(float(w) - 0.399) < 1e-3
    assert to_detection_line("1 0.2 0.2 0.1 0.1") == "1 0.2 0.2 0.1 0.1"
    assert to_detection_line("1 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5") is None


def test_feedback_keeps_accepted_and_relabelled_drops_rejected(tmp_path):
    img = tmp_path / "car.jpg"
    img.write_bytes(b"x")
    poly = [[10, 10], [50, 10], [50, 50]]
    decisions = [
        {"label": "Dent", "confidence": 0.9, "decision": "accept", "new_label": "Dent", "polygon": poly},
        {"label": "Dent", "confidence": 0.4, "decision": "reject", "new_label": "Dent", "polygon": poly},
        {"label": "Scratch", "confidence": 0.6, "decision": "relabel", "new_label": "Dent", "polygon": poly},
    ]
    out = save_review(img, 100, 100, decisions, {"Dent": 0, "Scratch": 1}, "CK", folder=tmp_path / "fb")
    assert len(out.read_text().strip().splitlines()) == 2
    assert all(l.startswith("0 ") for l in out.read_text().strip().splitlines())
    stats = agreement(tmp_path / "fb")
    assert stats["Dent"]["accept"] == 1 and stats["Dent"]["reject"] == 1


def test_rules_file_is_valid():
    import csv
    rows = list(csv.DictReader((ROOT / "repair_rules.csv").open()))
    assert rows and all(float(r["labour_hours"]) >= 0 for r in rows)
