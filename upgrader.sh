#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

git pull --ff-only
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
brew update && brew upgrade ffmpeg chromaprint
pytest -q
