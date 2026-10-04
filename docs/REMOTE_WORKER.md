# Control Plane / Compute Plane Architecture

## Decision

The project uses a strict split:

- **Control plane:** the user's laptop.
- **Compute plane:** a remote host with 2× RTX 5090 running Docker.

The agent, source-of-truth repository checkout, experiment scheduler, search logic, credentials, and promotion decisions remain on the laptop.

The GPU host is an execution worker. It runs containers and stores large datasets/checkpoints/artifacts, but it does not host the autonomous controller.

## Topology

```text
Laptop
┌────────────────────────────────────────────┐
│ coding / research agent                    │
│ qwen12g controller                         │
│ git working tree                           │
│ experiment/search decisions                │
│ benchmark summaries                        │
└──────────────────┬─────────────────────────┘
                   │ SSH transport
                   │ Docker context
                   ▼
Dual RTX 5090 host
┌────────────────────────────────────────────┐
│ Docker Engine + NVIDIA Container Toolkit   │
│                                            │
│  qwen12g-worker containers                 │
│  ┌──────────────────────────────────────┐  │
│  │ CUDA / PyTorch / training stack      │  │
│  │ torchrun / profiling / quant / eval  │  │
│  │ GPU 0 + GPU 1                        │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  /srv/qwen12g                              │
│  ├── hf-cache                              │
│  ├── datasets                              │
│  ├── checkpoints                           │
│  ├── artifacts                             │
│  └── runs                                  │
└────────────────────────────────────────────┘
```

The final 12 GB GPU is a separate release-validation worker.

## Remote transport

Use Docker's SSH-backed context rather than an unauthenticated or TLS-exposed Docker TCP socket.

Example from the laptop:

```bash
docker context create qwen5090 \
  --docker host=ssh://qwen-worker@training-host

docker --context qwen5090 info
```

The controller defaults to the context name `qwen5090`, configurable in `configs/worker/dual5090.yaml`.

The SSH user will normally need permission to access the remote Docker daemon. Treat membership in the remote `docker` group as privileged host access.

## Source flow

The authoritative source is the laptop checkout / GitHub repository.

A run records an exact Git commit. Worker images are built from that source revision. Do not edit project source independently on the GPU host.

```text
agent edits on laptop
 -> tests / commit
 -> remote image build for exact source
 -> run container
 -> write run outputs to persistent worker storage
 -> fetch small summaries/metrics
 -> controller chooses next experiment
```

Large model artifacts are not pushed to Git.

## Worker modes

### distributed

Both GPUs participate in one job:

```text
GPU 0 ─┐
       ├── torchrun --nproc-per-node=2
GPU 1 ─┘
```

Use for SFT, recovery, and jobs that need combined distributed capacity.

### independent

One candidate per GPU:

```text
GPU 0 -> candidate A
GPU 1 -> candidate B
```

Use for cheap quantization/evaluation/profile searches when candidates fit one 32 GB card.

The experiment scheduler must record which mode each run uses.

## Persistent worker storage

Default host root:

`/srv/qwen12g`

Containers mount only the required subdirectories.

The worker is disposable, but checkpoints and run data may be expensive to reproduce. Back up or replicate important artifacts before host teardown.

## Security boundary

The controller may invoke only project-defined worker operations by default. It should not expose a generic "run arbitrary SSH command" tool to an autonomous agent.

Never mount these into the training/evaluation container:

- the laptop's `~/.ssh`;
- cloud credentials;
- broad GitHub credentials;
- `/var/run/docker.sock`.

Model-generated coding commands belong in separate unprivileged task sandboxes and must not run in the training container.

## Credentials

Preferred model:

- laptop holds GitHub write credentials and the SSH private key;
- remote worker has no GitHub write token;
- public model/dataset downloads use no token where possible;
- if Hugging Face authentication is required, use a narrowly scoped read token supplied only to the required job, never baked into an image.

Do not store tokens in committed Compose files.

## Failure model

The laptop controller must tolerate:

- SSH interruption;
- laptop sleep/restart while a detached remote container continues;
- worker reboot;
- container failure;
- partial benchmark output.

All long jobs use stable run IDs and persistent remote output directories so the controller can reconnect and resume observation.
