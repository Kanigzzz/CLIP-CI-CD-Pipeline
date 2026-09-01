#!/bin/bash

set -euo pipefail

echo "Downloading assets from HF..."
python scripts/download_assets.py

echo "Starting API..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000