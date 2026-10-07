"""Seeded miniature hybrid MoE and clearly labelled synthetic data."""

from __future__ import annotations

import json

import torch

from appliance.runtime.packed import sha256
from appliance.stages.common import save_json

TRAIN = [
    "User: add two numbers. Assistant: <think>Add the inputs.</think> def add(a, b): return a + b",
    "User: repair subtraction. Assistant: <think>Use subtraction.</think> def sub(a, b): return a - b",
    'User: read a file. Assistant: <think>Call the file tool.</think> {"name":"read_file","arguments":{"path":"main.py"}}',
    'User: list files. Assistant: <think>Call the list tool.</think> {"name":"list_directory","arguments":{"path":"."}}',
    'User: test code. Assistant: <think>Run the tests.</think> {"name":"run_tests","arguments":{}}',
    "User: count items. Assistant: <think>Use len.</think> def count(items): return len(items)",
    "User: multiply numbers. Assistant: <think>Multiply both inputs.</think> def mul(a, b): return a * b",
    "User: sort values. Assistant: <think>Use sorted.</think> def order(values): return sorted(values)",
]
HELDOUT = [
    "User: divide two numbers. Assistant: <think>Divide the inputs.</think> def div(a, b): return a / b",
    "User: find largest value. Assistant: <think>Use max.</think> def largest(values): return max(values)",
]


def generate(cfg, output):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(int(cfg["seed"]))
    vocab = {token: index for index, token in enumerate(["<pad>", "<eos>", "<unk>", "<bos>"])}
    vocab.update(
        {
            token: len(vocab) + index
            for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
    )
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single="$A <eos>", special_tokens=[("<eos>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        bos_token="<bos>",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: <think>{% endif %}"
    hidden = int(cfg["hidden_size"])
    heads = hidden // 64
    key_heads = next(n for n in range(max(1, heads // 2), 0, -1) if heads % n == 0)
    config = Qwen3_5MoeTextConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden,
        num_hidden_layers=int(cfg["layers"]),
        num_attention_heads=heads,
        num_key_value_heads=2 if heads % 2 == 0 else 1,
        head_dim=64,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=key_heads,
        linear_num_value_heads=heads,
        num_experts=int(cfg["experts"]),
        num_experts_per_tok=int(cfg["top_k"]),
        moe_intermediate_size=int(cfg["expert_width"]),
        shared_expert_intermediate_size=int(cfg["expert_width"]),
        layer_types=[
            "full_attention" if i % 3 == 1 else "linear_attention"
            for i in range(int(cfg["layers"]))
        ],
        max_position_embeddings=262144,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=3,
        tie_word_embeddings=False,
    )
    config.qwen12g_proof_model = True
    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(output / "source", safe_serialization=True)
    tokenizer.save_pretrained(output / "source")
    dataset = output / "datasets"
    dataset.mkdir(exist_ok=True)
    for name, rows in [("train", TRAIN), ("calibration", TRAIN), ("heldout", HELDOUT)]:
        path = dataset / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"text": text}) + "\n" for text in rows))
        save_json(
            path.with_suffix(".manifest.json"),
            {
                "kind": "synthetic_proof_fixture",
                "sha256": sha256(path),
                "note": "Handwritten fixtures, not generated AYOT or benchmark evidence.",
            },
        )
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "native_context": 262144,
        "synthetic": True,
        "trained_from_scratch": True,
    }
