from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from appliance.gpu import compute_dtype
from appliance.stages.common import load_config, resolve_model_source, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    base = resolve_model_source(config["model"])
    adapter = Path(config["adapter"]).resolve()
    output = Path(config.get("output_dir", "/data/checkpoints/qwen36-merged"))

    if config.get("dry_run"):
        save_json(output.parent / "merge_plan.json", {"base": base, "adapter": str(adapter), "output": str(output)})
        print(json.dumps({"dry_run": True, "base": base, "adapter": str(adapter), "output": str(output)}))
        return

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForImageTextToText.from_pretrained(
        base,
        torch_dtype=compute_dtype(),
        device_map="auto",
        low_cpu_mem_usage=True,
        token=token,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True, max_shard_size="4GB")
    processor = AutoProcessor.from_pretrained(base, token=token)
    processor.save_pretrained(str(output))
    save_json(output / "merge_manifest.json", {"base": base, "adapter": str(adapter)})
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
