#!/bin/zsh
# Prepares the data, fine-tunes the model (pre-trained and from scratch), tests, then opens the review app.
# Training takes one to two hours on an Apple-silicon Mac. caffeinate keeps the Mac awake meanwhile.
cd "$(dirname "$0")"
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
pytest -q tests
python src/prepare_data.py
caffeinate -i python src/train.py --epochs ${EPOCHS:-20}
echo "\nOpening the review app. Press Ctrl+C here to stop it."
streamlit run src/review_app.py
