# Remote Worker Security

The rented RTX 5090 worker is disposable and must be treated as untrusted infrastructure.

## Rules

- Use a dedicated ephemeral SSH key.
- Do not forward a personal SSH agent.
- Use a repository-scoped GitHub credential only if pushes are required.
- Use a narrowly scoped Hugging Face token only if model/dataset downloads require it.
- Never expose unrelated cloud credentials to the worker.
- Model-generated shell commands must run inside a sandbox/container without host secrets.
- Persist experiment state outside the worker or push it to approved storage before teardown.
- Revoke temporary credentials after the worker is destroyed.

## Model execution sandbox

Agent evaluation containers should:

- run as non-root;
- mount only the task repository;
- receive no SSH keys;
- receive no cloud tokens;
- have controlled networking;
- have CPU/memory/time limits;
- log all commands and file changes.
