#!/bin/bash

set -e

echo "Downloading assets from HF..."
python scripts/download_assets.py

echo "Starting API..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000