# Qwen3.6 12GB Training Appliance

The `dev` direction packages the project as one NVIDIA GPU container with a web UI.

## Quick start

```bash
export QWEN12G_WEB_TOKEN="$(openssl rand -hex 24)"
docker compose -f appliance/compose.yml up -d --build
```

Open `http://HOST:8080`.

The image contains the UI/API, run database, registered job scheduler, GPU telemetry,
persistent logs/results, artifact directories, and GitHub/Hugging Face publishing.

Persistent state lives under `/data`.

Optional secrets: `GITHUB_TOKEN` and `HF_TOKEN`.

Publishing is restricted to directories under `/data`. Credentials are not copied into
run manifests. The GitHub publisher removes the credential-bearing remote URL after push.

The appliance deliberately does not mount the Docker socket and the UI deliberately does
not provide an arbitrary command field.

For public or remote access, put the UI behind a VPN, SSH tunnel, or authenticated TLS
reverse proxy. Set `QWEN12G_SECURE_COOKIE=1` when the browser reaches it over HTTPS.
