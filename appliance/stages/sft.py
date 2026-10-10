from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from appliance.gpu import compute_dtype, require_kbit_support
from appliance.stages.common import load_config, resolve_model_source, save_json
from appliance.stages.dataset_io import load_training_dataset, maybe_limit


def build_lora_config(config: dict, model_config):
    from peft import LoraConfig

    text_config = getattr(model_config, "text_config", model_config)
    num_experts = int(getattr(text_config, "num_experts", 256))
    rank = int(config.get("lora_rank", 32))
    expert_rank = int(config.get("expert_rank", max(1, rank // num_experts)))

    target_modules = config.get("target_modules", "all-linear")
    target_parameters = config.get(
        "target_parameters",
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    rank_pattern = {
        "experts.gate_up_proj": expert_rank,
        "experts.down_proj": expert_rank,
    }

    return LoraConfig(
        r=rank,
        lora_alpha=int(config.get("lora_alpha", rank * 2)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        target_parameters=target_parameters,
        rank_pattern=rank_pattern,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    if config.get("dry_run"):
        save_json(Path(args.config).parent / "sft_plan.json", config)
        print(json.dumps({"dry_run": True, "stage": "sft", "config": config}, indent=2))
        return

    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    model_source = resolve_model_source(config.get("model", "Qwen/Qwen3.6-35B-A3B"))
    output_dir = Path(config.get("output_dir", "/data/checkpoints/qwen36-agent-sft"))
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    quantization_config = None
    require_kbit_support(config.get("load_in_4bit", True))
    if config.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(config.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype(),
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        torch_dtype=compute_dtype(),
        quantization_config=quantization_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        token=os.environ.get("HF_TOKEN") or None,
    )
    # Transformers 5 passes unknown loading kwargs to the model class, which rejects
    # use_cache, so disable the KV cache on the loaded configs instead.
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.use_cache = False
    processor = AutoProcessor.from_pretrained(model_source, token=os.environ.get("HF_TOKEN") or None)

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )

    peft_config = build_lora_config(config, model.config)
    dataset = maybe_limit(load_training_dataset(config), config)

    # TRL directly supports conversational/tool-calling datasets. For tool-use data,
    # keep `messages` and `tools` columns intact.
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(config.get("max_length", 16384)),
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        num_train_epochs=float(config.get("num_train_epochs", 1.0)),
        max_steps=int(config.get("max_steps", -1)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=int(config.get("logging_steps", 5)),
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        bf16=compute_dtype() == torch.bfloat16,
        tf32=bool(config.get("tf32", True)) and not bool(torch.version.hip),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=bool(config.get("packing", False)),
        assistant_only_loss=bool(config.get("assistant_only_loss", True)),
        report_to=config.get("report_to", "none"),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=int(config.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    resume = config.get("resume_from_checkpoint", False)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    if bool(config.get("merge_adapter", False)) and local_rank == 0:
        merged_dir = Path(config.get("merged_output_dir", str(output_dir) + "-merged"))
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(str(merged_dir))

    if local_rank == 0:
        save_json(
            output_dir / "training_manifest.json",
            {
                "model": model_source,
                "dataset": config.get("dataset_name") or config.get("dataset_path"),
                "lora_rank": int(config.get("lora_rank", 32)),
                "expert_rank": int(config.get("expert_rank", max(1, int(config.get("lora_rank", 32)) // 256))),
                "max_length": int(config.get("max_length", 16384)),
                "load_in_4bit": bool(config.get("load_in_4bit", True)),
            },
        )


if __name__ == "__main__":
    main()
