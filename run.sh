#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"; pwd)"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -m dmp.cli train --config config.yaml
python3 -m dmp.cli eval --config config.yaml
python3 -m dmp.cli eda --config config.yaml
python3 -m dmp.cli live --config config.yaml --ticker AAPL
