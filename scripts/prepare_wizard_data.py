"""Build the five wizard input files from pinned r0b0tlab distillation datasets.

Dataset splits keep the inputs disjoint: SFT and calibration use the multi-teacher
train split, teacher prompts its validation split and held-out diagnostics its test
split. Recovery uses the separate DeepSeek agentic corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

MULTI_TEACHER = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation"
MULTI_TEACHER_REVISION = "7a3473446840bcc397928cd8183d4b3ba3ca13a7"
AGENTIC = "r0b0tlab/deepseek-v4-pro-0813-agentic"
AGENTIC_REVISION = "2f075a280078b485be3e07b7fbf1413b88f6411a"

SOURCES = {
    "training": (
        MULTI_TEACHER,
        MULTI_TEACHER_REVISION,
        [
            "data/sft_code/train-00000-of-00002.parquet",
            "data/sft_code/train-00001-of-00002.parquet",
            "data/sft_agent/train-00000-of-00001.parquet",
        ],
    ),
    "calibration": (
        MULTI_TEACHER,
        MULTI_TEACHER_REVISION,
        [f"data/prompt_completion_text/train-0000{i}-of-00005.parquet" for i in range(5)],
    ),
    "prompts": (MULTI_TEACHER, MULTI_TEACHER_REVISION, ["data/sft/validation-00000-of-00001.parquet"]),
    "heldout": (MULTI_TEACHER, MULTI_TEACHER_REVISION, ["data/sft/test-00000-of-00001.parquet"]),
    "recovery": (AGENTIC, AGENTIC_REVISION, ["sft_openai/train-00000-of-00001.parquet"]),
}


def rows(repo, revision, files):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
        yield from pq.read_table(path).to_pylist()


def clean_message(message):
    # Parquet structs carry every optional field; empty ones confuse chat templates.
    keep = {"role": message["role"], "content": message.get("content") or ""}
    for key in ("reasoning_content", "tool_calls", "tool_call_id", "name"):
        if message.get(key):
            keep[key] = message[key]
    return keep


def conversation(row):
    record = {"messages": [clean_message(m) for m in row["messages"]]}
    if row.get("tools"):
        record["tools"] = row["tools"]
    return record


def calibration_text(row):
    # prompt_completion_text rows are already rendered with role tags.
    return f"{row.get('prompt_text') or ''}\n{row.get('completion_text') or ''}".strip()


def calibration_record(row):
    text = calibration_text(row)
    return {"text": text} if text else None


def teacher_prompt(row, max_chars):
    # Source prompts only: AYOT regenerates answers, and rejects assistant turns.
    if row.get("tools"):
        return None
    prompt = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            break
        prompt.append({"role": message["role"], "content": message.get("content") or ""})
    if not any(m["role"] == "user" for m in prompt):
        return None
    if sum(len(m["content"]) for m in prompt) > max_chars:
        return None
    return {"messages": prompt}


def heldout_text(row):
    return "\n\n".join(
        f"{m['role']}: {m.get('content') or ''}" for m in row["messages"] if m.get("content")
    )


def sample(items, limit, seed):
    items = [item for item in items if item]
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"rows": len(records), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data/datasets/r0b0tlab")
    parser.add_argument("--training", type=int, default=0, help="0 keeps every row")
    parser.add_argument("--recovery", type=int, default=0)
    parser.add_argument("--calibration", type=int, default=512)
    parser.add_argument("--prompts", type=int, default=512)
    parser.add_argument("--heldout", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    builders = {
        "training": (conversation, args.training),
        "recovery": (conversation, args.recovery),
        "calibration": (calibration_record, args.calibration),
        "prompts": (lambda r: teacher_prompt(r, args.max_prompt_chars), args.prompts),
        "heldout": (lambda r: {"text": heldout_text(r)}, args.heldout),
    }
    manifest = {"seed": args.seed, "files": {}}
    for name, (convert, limit) in builders.items():
        repo, revision, files = SOURCES[name]
        records = sample([convert(r) for r in rows(repo, revision, files)], limit, args.seed)
        if not records:
            raise ValueError(f"No usable rows for {name}")
        result = write_jsonl(output / f"{name}.jsonl", records)
        manifest["files"][f"{name}.jsonl"] = {
            **result,
            "source": repo,
            "revision": revision,
            "source_files": files,
        }
        print(f"{name}: {result['rows']} rows", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
