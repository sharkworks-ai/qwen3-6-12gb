#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 qwen-worker@training-host [docker-context-name]" >&2
  exit 2
fi

SSH_HOST="$1"
CONTEXT="${2:-qwen5090}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,search]"

if docker context inspect "${CONTEXT}" >/dev/null 2>&1; then
  echo "Docker context ${CONTEXT} already exists."
else
  qwen12g worker context-create "${SSH_HOST}" --context "${CONTEXT}"
fi

echo
echo "Validating remote worker..."
qwen12g worker doctor

echo
echo "Laptop control plane is ready."
