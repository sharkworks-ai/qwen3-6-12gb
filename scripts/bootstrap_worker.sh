#!/usr/bin/env bash
set -euo pipefail

echo "Qwen3.6 12GB worker bootstrap"
echo "This script intentionally performs only safe local setup."
echo "GPU drivers/CUDA should be installed by the host image or provisioning layer."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,search]"

echo
echo "Environment installed."
echo "Run: source .venv/bin/activate && qwen12g doctor"
echo
echo "Before training, verify:"
echo "  - nvidia-smi sees the RTX 5090"
echo "  - CUDA toolkit is recent enough for Blackwell"
echo "  - llama.cpp CUDA build succeeds"
echo "  - temporary GitHub/Hugging Face credentials are least-privilege"
