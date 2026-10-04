#!/usr/bin/env bash
set -euo pipefail

# Run ONCE on the remote dual-5090 host as an administrator.
# Docker Engine, NVIDIA drivers, and NVIDIA Container Toolkit should already
# be installed using the host's supported installation method.

WORKER_USER="${QWEN12G_WORKER_USER:-qwen-worker}"
WORKER_UID="${QWEN12G_WORKER_UID:-10001}"
DATA_ROOT="${QWEN12G_DATA_ROOT:-/srv/qwen12g}"

if ! id "${WORKER_USER}" >/dev/null 2>&1; then
  if getent passwd "${WORKER_UID}" >/dev/null 2>&1; then
    echo "ERROR: UID ${WORKER_UID} is already in use. Set QWEN12G_WORKER_UID and rebuild the worker image to match." >&2
    exit 1
  fi
  useradd --create-home --uid "${WORKER_UID}" --shell /bin/bash "${WORKER_USER}"
fi

ACTUAL_UID="$(id -u "${WORKER_USER}")"
if [[ "${ACTUAL_UID}" != "${WORKER_UID}" ]]; then
  echo "ERROR: ${WORKER_USER} has UID ${ACTUAL_UID}, expected ${WORKER_UID}." >&2
  echo "The host worker UID must match the container worker UID." >&2
  exit 1
fi

mkdir -p \
  "${DATA_ROOT}/hf-cache" \
  "${DATA_ROOT}/datasets" \
  "${DATA_ROOT}/checkpoints" \
  "${DATA_ROOT}/artifacts" \
  "${DATA_ROOT}/runs"

chown -R "${WORKER_USER}:${WORKER_USER}" "${DATA_ROOT}"

install -d -m 700 -o "${WORKER_USER}" -g "${WORKER_USER}" "/home/${WORKER_USER}/.ssh"
touch "/home/${WORKER_USER}/.ssh/authorized_keys"
chown "${WORKER_USER}:${WORKER_USER}" "/home/${WORKER_USER}/.ssh/authorized_keys"
chmod 600 "/home/${WORKER_USER}/.ssh/authorized_keys"

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
