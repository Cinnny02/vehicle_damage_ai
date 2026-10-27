"""Assessor review screen: the model suggests damage, a person accepts, corrects or rejects it.

    streamlit run src/review_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feedback import agreement, save_review  # noqa: E402
from repair import Finding, needs_review, repair_lines  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "damage_detector.pt"

st.set_page_config(page_title="Vehicle Damage Review", layout="wide")
st.title("Vehicle damage review")
st.caption("The model suggests damage. An assessor decides. Every decision is logged and can be used for retraining.")


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL))


def test_images() -> list[Path]:
    cfg = yaml.safe_load((ROOT / "data" / "dataset.yaml").read_text())
    folder = Path(cfg["path"]) / cfg["test"]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


if not MODEL.exists():
    st.error("No trained model yet. Run `python src/prepare_data.py` and `python src/train.py` first.")
    st.stop()

model = load_model()
names = model.names
class_ids = {v: k for k, v in names.items()}

tab_review, tab_metrics, tab_feedback = st.tabs(["Review a claim", "Model performance", "Assessor feedback"])

with tab_review:
    left, right = st.columns([3, 2])
    with left:
        source = st.radio("Image", ["Test image", "Upload a photo"], horizontal=True)
        if source == "Upload a photo":
            up = st.file_uploader("Vehicle photo", type=["jpg", "jpeg", "png"])
            if up is None:
                st.stop()
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getvalue())
            image_path = tmp
        else:
            imgs = test_images()
            idx = st.slider("Test image", 0, len(imgs) - 1, 0)
            image_path = imgs[idx]
        conf = st.slider("Show findings above this confidence", 0.05, 0.9, 0.25, 0.05)

    result = model.predict(str(image_path), conf=conf, max_det=20, verbose=False)[0]
    h, w = result.orig_shape
    findings, polygons = [], []
    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
        area = (x2 - x1) * (y2 - y1) / (w * h)  # share of the photo the damage box covers
        findings.append(Finding(names[int(result.boxes.cls[i])], float(result.boxes.conf[i]), area, (x1, y1, x2, y2)))
        polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    with left:
        st.image(result.plot()[:, :, ::-1], caption=image_path.name, width="stretch")

    with right:
        reasons = needs_review(findings)
        if reasons:
            st.warning("Needs an assessor:\n\n" + "\n".join(f"- {r}" for r in reasons))
        else:
            st.success("Model is confident. Assessor confirmation still required before estimating.")

        st.subheader("Findings")
        decisions = []
        for i, f in enumerate(findings):
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**{i + 1}. {f.label}**  \n{f.confidence:.0%} confidence")
            choice = c2.selectbox("Decision", ["accept", "reject", "relabel"], key=f"d{i}{image_path.name}")
            new_label = f.label
            if choice == "relabel":
                new_label = c2.selectbox("Correct label", list(class_ids), key=f"l{i}{image_path.name}")
            decisions.append({"label": f.label, "confidence": round(f.confidence, 3), "decision": choice,
                              "new_label": new_label, "polygon": polygons[i]})

        reviewer = st.text_input("Assessor initials", max_chars=5)
        if st.button("Save review", disabled=not reviewer):
            save_review(image_path, w, h, decisions, class_ids, reviewer)
            st.success("Saved to feedback/. Corrections will be included in the next training run.")

        kept = [Finding(d["new_label"] if d["decision"] == "relabel" else d["label"], f.confidence, f.area_fraction, f.box)
                for d, f in zip(decisions, findings) if d["decision"] != "reject"]
        st.subheader("Draft repair lines")
        lines = repair_lines(kept)
        if lines:
            df = pd.DataFrame(lines)
            st.dataframe(df, hide_index=True, width="stretch")
            hours = df.labour_hours.dropna().sum()
            st.metric("Indicative labour", f"{hours:.1f} hours")
        st.caption("Illustrative rules from repair_rules.csv. Not a real time guide.")

with tab_metrics:
    mpath = ROOT / "reports" / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        runs = {k: v for k, v in m.items() if isinstance(v, dict)}
        cols = ["box_map50", "box_map50_95", "precision", "recall", "train_minutes"]
        st.markdown("Scored on the held-out **test** split, never used for training.")
        st.dataframe(pd.DataFrame({k: {c: v.get(c) for c in cols} for k, v in runs.items()}).T, width="stretch")
        st.markdown("**Per damage class (fine-tuned model)**")
        st.dataframe(pd.DataFrame(runs["pretrained"]["per_class"]).T, width="stretch")
        for img in ["BoxPR_curve.png", "confusion_matrix_normalized.png"]:
            p = ROOT / "runs" / "pretrained_test" / img
            if p.exists():
                st.image(str(p), caption=img, width=600)
    else:
        st.info("Run training to see metrics.")

with tab_feedback:
    stats = agreement()
    if stats:
        st.markdown("How often assessors accepted the model's findings:")
        st.dataframe(pd.DataFrame(stats).T.fillna(0), width="stretch")
    else:
        st.info("No reviews saved yet.")
