# Self-contained appliance architecture

## Objective

Replace the laptop-controller / remote-worker split on `dev` with one portable GPU appliance image.

A host needs only Linux, NVIDIA driver, Docker Engine, NVIDIA Container Toolkit, and persistent storage.

```text
Browser
  |
  v
Qwen12G container
  - authenticated FastAPI UI
  - SQLite experiment DB
  - registered-job scheduler
  - GPU telemetry
  - training/eval/compression processes
  - Git/Hugging Face publishers
  |
  +-- /data persistent volume
  +-- NVIDIA GPUs
```

No host Docker socket is mounted.

## Persistence

`/data` contains `db`, `runs`, `models`, `datasets`, `checkpoints`, `artifacts`, `hf-cache`, and `repos`.

## Web security

`QWEN12G_WEB_TOKEN` is mandatory. Login creates a signed HttpOnly SameSite=Strict cookie.
HTTPS deployments should set `QWEN12G_SECURE_COOKIE=1`.

The appliance should normally be reachable only through a private network, VPN, SSH tunnel,
or authenticated reverse proxy.

## Job security

The UI selects from a code-defined registry. It cannot send arbitrary command strings.
Future stages (`sft`, `profile`, `prune`, `recover`, `quantize`, `eval`) must be registered explicitly.

Each job executes through a durable runner and writes `run.json`, `stdout.log`, `stderr.log`,
`vram_trace.csv`, and `result.json`. The web process can restart and reconcile status from these files.

## Publishing

GitHub and Hugging Face credentials come from container secrets/environment variables.
GitHub publishing is restricted to paths beneath `/data`, and the token-bearing remote URL is removed after push.
Hugging Face publishing uses `huggingface_hub.HfApi`.

## Current functionality retained

- persistent experiment/run DB;
- stable run IDs;
- GPU telemetry and peak VRAM capture;
- smoke workload;
- persistent logs/results;
- status/stop/reconcile controls;
- system/storage visibility;
- optional GitHub/Hugging Face publishing.

The old remote-worker code may remain temporarily for comparison, but it is no longer the target architecture on `dev`.
