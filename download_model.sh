#!/bin/bash
set -e

MODEL_REPO="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_DIR="llama3-8b-instruct"

echo "Creating model directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

echo "Running Python script to download model using HF_TOKEN..."
python3 - <<END
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not set")

snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL_DIR",
    local_dir_use_symlinks=False,
    token=token
)
END
