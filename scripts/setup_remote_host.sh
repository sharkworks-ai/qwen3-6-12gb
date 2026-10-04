#!/usr/bin/env bash
set -euo pipefail

# Run ONCE on the remote dual-5090 host as an administrator.
# Docker Engine, NVIDIA drivers, and NVIDIA Container Toolkit should already
# be installed using the host's supported installation method.

WORKER_USER="${QWEN12G_WORKER_USER:-qwen-worker}"
DATA_ROOT="${QWEN12G_DATA_ROOT:-/srv/qwen12g}"

if ! id "${WORKER_USER}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${WORKER_USER}"
fi

mkdir -p \
  "${DATA_ROOT}/hf-cache" \
  "${DATA_ROOT}/datasets" \
  "${DATA_ROOT}/checkpoints" \
  "${DATA_ROOT}/artifacts" \
  "${DATA_ROOT}/runs"

chown -R "${WORKER_USER}:${WORKER_USER}" "${DATA_ROOT}"

if getent group docker >/dev/null 2>&1; then
  usermod -aG docker "${WORKER_USER}"
else
  echo "ERROR: docker group not found. Install/configure Docker first." >&2
  exit 1
fi

echo "Prepared ${WORKER_USER} and ${DATA_ROOT}."
echo "Add the laptop controller's PUBLIC SSH key to:"
echo "  /home/${WORKER_USER}/.ssh/authorized_keys"
echo
echo "Then reconnect so docker-group membership takes effect."
