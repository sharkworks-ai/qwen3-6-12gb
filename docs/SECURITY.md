# Remote Worker Security

The remote dual-RTX 5090 worker is the compute plane and must be treated as untrusted/disposable infrastructure. The agent and controller run on the laptop control plane.

Remote Docker is accessed through an SSH-backed Docker context. Do not expose the Docker daemon on an unauthenticated TCP socket.

## Rules

- Use a dedicated ephemeral SSH key.
- Do not forward a personal SSH agent.
- Keep GitHub write credentials on the laptop; the worker should not need them.
- If a worker-side Git credential becomes unavoidable, make it repository-scoped and read-only where possible.
- Use a narrowly scoped Hugging Face token only if model/dataset downloads require it.
- Never expose unrelated cloud credentials to the worker.
- Model-generated shell commands must run inside a sandbox/container without host secrets.
- Persist experiment state outside the worker or push it to approved storage before teardown.
- Revoke temporary credentials after the worker is destroyed.
- Do not mount `/var/run/docker.sock` into training or model-execution containers.
- Do not expose a generic arbitrary SSH executor to the autonomous agent; prefer project-defined worker operations.

## Model execution sandbox

Agent evaluation containers should:

- run as non-root;
- mount only the task repository;
- receive no SSH keys;
- receive no cloud tokens;
- have controlled networking;
- have CPU/memory/time limits;
- log all commands and file changes.
