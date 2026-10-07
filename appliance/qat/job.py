from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from appliance.qat.parametrize import apply_fake_quant, apply_precision_map, remove_fake_quant
from appliance.stages.common import save_json
from appliance.quant.precision import digest
from transformers.trainer_utils import get_last_checkpoint
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"], torch_dtype="bfloat16", trust_remote_code=True, low_cpu_mem_usage=True
    )
    precision = (
        json.loads(Path(cfg["precision_map"]).read_text()) if cfg.get("precision_map") else None
    )
    matched = (
        apply_precision_map(model, precision)
        if precision
        else apply_fake_quant(model, cfg.get("mode", "q3_moe"))
    )
    if not matched:
        raise ValueError("No QAT tensors matched")
    tok = AutoTokenizer.from_pretrained(cfg["student_model"], trust_remote_code=True)
    ds = load_dataset("json", data_files=cfg["dataset"], split="train")
    ml = int(cfg.get("max_length", 8192))

    def enc(r):
        text = (
            tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            if "messages" in r
            else r.get("text", "")
        )
        z = tok(text, truncation=True, max_length=ml)
        z["labels"] = z["input_ids"].copy()
        return z

    ds = ds.map(enc, remove_columns=ds.column_names)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out / "trainer"),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 5e-6)),
        max_steps=int(cfg.get("max_steps", 200)),
        bf16=True,
        logging_steps=1,
        save_steps=int(cfg.get("save_steps", 100)),
        report_to=[],
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap" if int(cfg.get("num_processes", 2)) > 1 else "",
        fsdp_config={
            "use_orig_params": True,
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
        }
        if int(cfg.get("num_processes", 2)) > 1
        else None,
    )
    identity = digest(
        {"config": {k: v for k, v in cfg.items() if k != "resume"}, "precision": precision}
    )
    manifest = out / "qat-manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if not cfg.get("resume") or old.get("identity") != identity:
            raise ValueError("QAT resume config mismatch")
        if old.get("status") == "succeeded":
            return
    rank = int(os.environ.get("RANK", 0))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.use_cache = False
    tr = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    if rank == 0:
        save_json(manifest, {"identity": identity, "status": "running"})
    checkpoint = get_last_checkpoint(str(out / "trainer")) if (out / "trainer").exists() else None
    tr.train(resume_from_checkpoint=checkpoint if cfg.get("resume") else None)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        state = tr.accelerator.get_state_dict(tr.model_wrapped)
        if rank == 0:
            from appliance.qat.fake_quant import fake_quant_weight

            clean = {}
            modules = dict(model.named_modules())
            for name, value in state.items():
                if ".parametrizations." in name:
                    parent, tail = name.split(".parametrizations.", 1)
                    parameter = tail.removesuffix(".original")
                    spec = getattr(modules[parent].parametrizations, parameter)[0].spec
                    clean[parent + "." + parameter] = fake_quant_weight(value, spec)
                else:
                    clean[name] = value
            model.save_pretrained(out / "recovered", state_dict=clean, safe_serialization=True)
    else:
        remove_fake_quant(model)
        model.save_pretrained(out / "recovered", safe_serialization=True)
    if rank == 0:
        tok.save_pretrained(out / "recovered")
        save_json(
            manifest,
            {
                "identity": identity,
                "status": "succeeded",
                "matched_parameters": matched,
                "note": "Re-run target quant backend after QAT; this checkpoint is not the deployment artifact.",
            },
        )


if __name__ == "__main__":
    main()
