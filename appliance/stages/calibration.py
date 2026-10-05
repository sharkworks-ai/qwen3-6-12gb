from __future__ import annotations

import argparse
import os
from pathlib import Path

from appliance.stages.common import load_config
from appliance.stages.dataset_io import load_training_dataset, maybe_limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    from transformers import AutoProcessor

    model = config.get("model", "Qwen/Qwen3.6-35B-A3B")
    processor = AutoProcessor.from_pretrained(model, token=os.environ.get("HF_TOKEN") or None)
    dataset = maybe_limit(load_training_dataset(config), config)
    output = Path(config.get("output", "/data/artifacts/calibration.txt"))
    output.parent.mkdir(parents=True, exist_ok=True)

    maximum = int(config.get("max_samples", 512))
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            if written >= maximum:
                break
            if row.get("messages") is not None:
                text = processor.apply_chat_template(
                    row["messages"],
                    tools=row.get("tools"),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = str(row.get("text") or row.get("prompt") or "")
            if not text.strip():
                continue
            handle.write(text.replace("\x00", "") + "\n")
            written += 1
    print(f"wrote {written} calibration samples to {output}")


if __name__ == "__main__":
    main()
