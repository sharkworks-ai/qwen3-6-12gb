from __future__ import annotations

import subprocess
from pathlib import Path

from huggingface_hub import HfApi


def inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Publishing is restricted to the persistent /data tree")
    return candidate


def push_github(*, data_root: Path, source_dir: Path, repo_url: str, branch: str,
                commit_message: str, token: str | None) -> str:
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")
    source_dir = inside(data_root, source_dir)
    if not source_dir.is_dir():
        raise ValueError("Source directory does not exist")

    if not (source_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=source_dir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=source_dir, check=True)
    subprocess.run([
        "git", "-c", "user.name=Qwen12G Appliance", "-c", "user.email=qwen12g@localhost",
        "commit", "-m", commit_message,
    ], cwd=source_dir, check=False, capture_output=True, text=True)

    authenticated = repo_url
    if repo_url.startswith("https://github.com/"):
        authenticated = repo_url.replace(
            "https://github.com/", f"https://x-access-token:{token}@github.com/", 1
        )

    remotes = subprocess.run(["git", "remote"], cwd=source_dir, check=True,
                             capture_output=True, text=True).stdout.split()
    if "origin" in remotes:
        subprocess.run(["git", "remote", "set-url", "origin", authenticated],
                       cwd=source_dir, check=True)
    else:
        subprocess.run(["git", "remote", "add", "origin", authenticated],
                       cwd=source_dir, check=True)
    try:
        subprocess.run(["git", "push", "-u", "origin", f"HEAD:{branch}"],
                       cwd=source_dir, check=True)
    finally:
        subprocess.run(["git", "remote", "set-url", "origin", repo_url],
                       cwd=source_dir, check=False)
    return f"Pushed {source_dir} to {repo_url} ({branch})"


def push_huggingface(*, data_root: Path, source_dir: Path, repo_id: str,
                     repo_type: str, token: str | None, private: bool) -> str:
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")
    source_dir = inside(data_root, source_dir)
    if not source_dir.is_dir():
        raise ValueError("Source directory does not exist")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(source_dir), repo_id=repo_id, repo_type=repo_type,
                      commit_message="Publish from Qwen12G appliance")
    return f"Uploaded {source_dir} to {repo_id}"
