# NVIDIA CUDA and AMD ROCm appliances

The existing `appliance/Dockerfile` and `appliance/compose.yml` remain the NVIDIA
path. AMD uses the separate `appliance/Dockerfile.rocm` and
`appliance/compose.rocm.yml`. Do not merge these compose files: NVIDIA device
requests and AMD device mappings are different.

## AMD host requirements

- Linux amdgpu driver, accessible `/dev/kfd` and `/dev/dri` render devices.
- Docker with device access. An LXC guest also requires its parent to permit
  nested containers and pass through those GPU devices; root inside a guest
  cannot grant missing parent permissions.
- A GPU supported by the chosen ROCm/PyTorch image. AMD inventory alone is not a
  compute compatibility test. Run the GPU connection test before training.
- Enough free storage for the ROCm image, datasets, multiple model copies and
  checkpoints. Reserve at least 80 GB free for initial experiments; full-model
  training needs substantially more. Start with at least 16 GB system RAM for
  small experiments; larger models require a separate memory budget.

The image uses AMD's versioned ROCm 7.2 / PyTorch 2.9.1 base and constrains its
PyTorch packages so installing application dependencies cannot silently replace
HIP-enabled torch with NVIDIA wheels. The llama.cpp build uses `GGML_HIP`.
`AMDGPU_TARGETS` controls llama.cpp compilation only; it does not add missing
architectures to the prebuilt PyTorch/rocBLAS libraries.

The RX 6700-family `gfx1031` host is an **experimental deployment target**. A
working Vulkan llama-server does not establish that ROCm training works. This
project does not automatically spoof its architecture with
`HSA_OVERRIDE_GFX_VERSION`. A build and an actual kernel test on that host are
required before any training compatibility claim.

## Build and start

From the repository root:

```bash
export QWEN12G_WEB_TOKEN="$(openssl rand -hex 24)"
export AMD_VIDEO_GID="$(getent group video | cut -d: -f3)"
export AMD_RENDER_GID="$(getent group render | cut -d: -f3)"
# Set this to the actual GPU architecture reported by rocminfo, if needed:
# export AMDGPU_TARGETS=gfx1100
export GIT_COMMIT="$(git rev-parse HEAD)"
docker compose -f appliance/compose.rocm.yml up -d --build
docker compose -f appliance/compose.rocm.yml exec qwen12g python3 -m appliance.smoke
```

Keep the token in a private env file for subsequent restarts. The service runs as
UID 10001 with the host's numeric video/render groups, without a privileged
container, Docker socket, or SSH credentials. It binds localhost by default;
use an SSH tunnel for remote access. Data persists in `qwen12g-data`.

The connection test runs matrix multiplication and backpropagation and checks
finite results. An unsupported GPU architecture, permissions issue, or missing
kernel library must be fixed before running experiments.

## Shared behavior and current limits

PyTorch on ROCm deliberately uses `torch.cuda` and `cuda:0` device names. Existing
`cuda_devices` configs and `CUDA_VISIBLE_DEVICES` selection remain valid; do not
set conflicting HIP/ROCR visibility variables. Numeric AMD inventory IDs follow
KFD GPU node order, not DRM card numbers. Verify the selected device names using
the connection test, especially on multi-GPU systems.

Telemetry reads AMD kernel sysfs counters in MiB and preserves the existing CSV
format. Missing counters remain unmeasured, never synthetic zero measurements.
NVIDIA continues to use nvidia-smi. Use `QWEN12G_GPU_BACKEND=cuda|rocm|auto` to
choose inventory (the ROCm compose file selects rocm).

The shared proof, reconstruction, QAT and packed inference paths use ROCm torch.
AMD cards without native BF16 use FP32 for those calculations; this increases
memory use. NVIDIA precision behavior is preserved. Real-model SFT/profile jobs
must use `load_in_4bit=false` on this ROCm image: bitsandbytes QLoRA is not bundled
or claimed as supported. Guided plans select full-precision loading on ROCm;
size the source model accordingly. NVIDIA retains its existing 4-bit option.

A successful synthetic proof does not validate coding quality or 262K context.
Run representative held-out data and actual long-context tests before promoting
an artifact.

## References

- [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [AMD ROCm PyTorch images](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/3rd-party/pytorch-install.html)
- [llama.cpp build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

## Host with limited local disk

Build the full ROCm image on a machine with sufficient disk, then build
`Dockerfile.rocm-network`. Export the **same full image's** `/opt/venv`,
`/opt/rocm-7.2.0`, and `/opt/llama.cpp` to `runtime/` on shared storage; do not mix
runtime versions. The smaller network image contains the matching OS libraries
and application source. Record the full image ID beside those exports.

Mount the share on the compute host first and set `QWEN12G_SHARED_ROOT` to that
mount. Merge `compose.rocm-network.yml` with `compose.rocm.yml` to bind `data/`
read-write and the runtime directories read-only. The SQLite database gets a
separate local Docker volume. Build images on the laptop and load only the
network image onto the GPU host; do not use `--build` on the GPU host.

SSHFS requires FUSE access from the LXC parent. Keep both machines awake and
connected during jobs. A disconnected share can stall or fail a run; restore
storage before resuming. Do not share a writable run directory with simultaneous
writers on different hosts. Keep the database off SSHFS. Verify write, fsync,
atomic rename and locking before starting jobs.

For slow links, an optional local copy of the exported Python environment can
reduce startup time. Set `QWEN12G_VENV_ROOT` to that directory (approximately
5.4 GB for the tested image). Keep its contents identical to the full image's
`/opt/venv`; update it together with the application image. The larger ROCm
libraries and experiment outputs can remain on the laptop.
