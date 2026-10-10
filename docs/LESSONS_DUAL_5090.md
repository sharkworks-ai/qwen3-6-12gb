# Lessons from bringing the pipeline up on 2x RTX 5090

This records what was learnt while validating the appliance, worker tooling and the
full train/prune/compress/recover pipeline on a dual RTX 5090 lab host with the real
Qwen3.6-35B-A3B model. Figures are measurements from that host unless marked as
estimates. Status is as of `71df7eb` on `dev`.

## Summary

- The pipeline as originally written could not train the full model on 2x32 GB:
  bitsandbytes 4-bit does not quantize Qwen3.6's fused expert tensors, so a "4-bit"
  replica is about 62 GiB per GPU.
- Pruning before SFT, running SFT and QAT as one process sharded across both GPUs, and
  training QAT as LoRA corrections through the quantizer makes every stage fit.
- Every stage has now run on the real model: profile, prune, SFT, merge inputs,
  baseline, 40-layer reconstruction and QAT recovery. A full end-to-end training run
  (`smoke-train-4`) was in progress when this was written.
- Library drift (Transformers 5, TRL 1.15, PEFT) caused most of the individual
  failures; each is fixed and listed below.

## Hardware and host

| Item | Value | Note |
|---|---|---|
| GPUs | 2x RTX 5090, 32,607 MiB each (31.4 GiB usable) | sm_120 |
| System RAM | 125 GB total, about 78 to 86 GB available | README assumed 192 GB |
| Disk | 1 TB | README assumed 2 TB |
| CPU | 64 cores | llama.cpp builds use `BUILD_JOBS=32` |

Plan memory and disk against the real figures, not the README.

## Model facts (Qwen3.6-35B-A3B, revision `995ad96`)

| Item | Value |
|---|---|
| Architecture | `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`), multimodal wrapper |
| Download size | 71.9 GB, 40 files |
| Parameters | 35.1B in total |
| Fused routed experts | 32.2B (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`), about 60 GiB BF16 |
| Ordinary `nn.Linear` | 2.4B |
| Experts | 256 per layer, 8 active per token |
| Layers | 40, hybrid linear attention (gated delta net) and full attention |
| Native context | 262,144 tokens |
| Vocabulary | 248,320 tokens |

Pruned to 128 experts per layer, the BF16 checkpoint is 37 GB on disk.

## Memory: what fits and what does not

| Stage | Approach | Peak GPU (GPU 0 / GPU 1) | Result |
|---|---|---|---|
| SFT, per-GPU 4-bit replica (DDP) | original design | about 62 GiB needed per GPU | does not fit |
| Profile, 4-bit `device_map=auto` | original design | needs CPU offload, which 4-bit forbids | fails |
| Profile, BF16 with CPU overflow | fixed | 27.5 / 27.9 GB | works |
| Baseline evaluation | BF16 `device_map=auto` | 19.5 / 20.1 GB | works |
| Reconstruction (GSQ 2/4-bit), 1 layer per window | original | up to 32.0 GB / idle | works only with expandable segments |
| SFT on pruned model, model-parallel 4-bit | fixed | 20.1 / 22.7 GB, about 28 s per step | works |
| QAT full-parameter FSDP | original design | about 140 GB estimated for a 128-expert student | does not fit |
| QAT LoRA through the quantizer, model-parallel | fixed | about 27.5 / 27.6 GB, about 17 s per step | works |

Key lessons:

- **bitsandbytes 4-bit only replaces `nn.Linear`.** Fused 3-D expert parameters stay
  BF16. On MoE models with fused experts, 4-bit loading saves almost nothing.
- **PEFT's `prepare_model_for_kbit_training` upcasts every non-quantized parameter to
  FP32.** On this model that doubles the BF16 experts and causes an OOM. Skip it for
  model-parallel training; PEFT freezes the base weights itself.
- **`device_map="auto"` fills GPU 0 first and puts the LM head on the last GPU.**
  Training needs headroom on GPU 0 for the head, the loss and optimiser state.
- **Full-vocabulary FP32 logits are large**: 4,096 tokens x 248,320 vocabulary is about
  4 GiB. Use a chunked (fused) LM-head loss for training.
- **Reconstruction sits at the 32 GB limit on GPU 0 while GPU 1 is idle**, because with
  one layer per window every window lands on the first device. Fragmentation alone
  caused an OOM until `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set.
- **Host RAM is the next constraint.** Profiling and baseline spill BF16 weights to CPU,
  and reconstruction holds the source in RAM. Available RAM dipped to 44 GB in one
  stage.

## Pipeline design changes

1. **Prune before SFT.** The wizard now profiles and prunes the base model, then runs
   SFT and merge on the pruned model. SFT also recovers pruning loss.
2. **Model-parallel SFT.** One process sharded across the selected GPUs
   (`model_parallel: true`, GPU-only `max_memory`) instead of one replica per GPU.
3. **LoRA QAT through the quantizer.** The base model is frozen and each quantized
   tensor trains `fake_quant(W + B @ A)`, with one adapter per expert. Removing the
   parametrization leaves the merged, fake-quantized weight for re-quantization.
   `trainable: "full"` keeps the FSDP path for larger hardware.
4. **QAT trains on the reconstruction's own grid.** QAT used fresh min/max scales. When
   GSQ leaves the extreme levels unused (common for 3/4-bit attention and shared
   tensors), min/max picks a smaller step and re-rounds every weight. This raised
   held-out loss from 1.51 to 2.48 before training. QAT now loads the packed per-group
   scales and rounds onto the same grid the runtime decodes.
5. **Disk budget.** QAT keeps one trainer checkpoint (it previously kept four per
   variant). `Stages.release()` deletes superseded outputs of finished stages and
   re-baselines their integrity hashes so resume still works: the pruned model after
   merge, and each variant's first-pass export and QAT state after re-quantization.
   `keep_intermediates` disables this; `release_source_after_merge` opts in to deleting
   the base checkpoint.

## Quality measurements (smoke settings, 8 samples)

| Model | Held-out loss |
|---|---|
| Pruned BF16 (128 of 256 experts) | 1.58 |
| After GSQ aggressive reconstruction (2 steps per layer) | 1.51 |
| Same, re-rounded by the old min/max QAT quantizer | 2.48 |

- Reconstruction quality is good even with very few steps.
- QAT step losses with the fixed loss scaling: 1.86, 1.76, 0.82 (from the old grid; the
  reconstruction-grid fix should start lower and is being confirmed in `smoke-train-4`).
- These are smoke-test diagnostics on tiny samples, not quality claims. Coding, tool use
  and long-context retention remain untested.

## Library compatibility issues and fixes

| Library | Problem | Fix |
|---|---|---|
| Transformers 5.19 | `from_pretrained(..., use_cache=False)` is passed to the model class, which rejects it | Set `use_cache` on the loaded config and text config |
| Transformers 5.19 | `warmup_ratio` removed | Pass the ratio as a fractional `warmup_steps` |
| Transformers 5.19 | `Trainer` sets `model_accepts_loss_kwargs` on the instance from the forward signature, overriding class attributes | Normalise the loss by `num_items_in_batch` in `compute_loss` |
| TRL 1.15 | SFT always uses a fused, chunked LM-head loss whose Triton kernel runs on the current device | Keep the LM head on the embeddings' GPU (`keep_head_with_embeddings`) |
| TRL 1.15 | Adds the MoE router aux loss across devices when the model is split | `router_aux_loss_coef=0.0`; the router is frozen in SFT |
| TRL 1.15 | `assistant_only_loss` needs `{% generation %}` markers that stock Qwen templates lack | Train with `get_training_chat_template()`; restore the original template before saving |
| PEFT | `ParamWrapper` (LoRA on fused expert `target_parameters`) rejects dropout | Default `lora_dropout` to 0 when experts are targeted |
| PEFT | `prepare_model_for_kbit_training` upcasts to FP32 | Skip it for model-parallel runs |
| bitsandbytes | Refuses CPU offload of a 4-bit model | Profile in BF16 with `max_memory` |
| torchao 0.18 | Two prebuilt extensions fail to load (Python 3.10 MXFP8 build, Hopper-only CUTLASS) | Harmless; nothing in the appliance uses them |

Two missing optional kernels (`causal_conv1d`, `flash-linear-attention`) make the
linear-attention layers fall back to slower PyTorch code. Installing them is a likely
speed-up.

## Data

- Inputs come from two public datasets, pinned by revision in
  `scripts/prepare_wizard_data.py`:
  - `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation` at `7a34734` for SFT,
    calibration, teacher prompts (validation split) and held-out (test split);
  - `r0b0tlab/deepseek-v4-pro-0813-agentic` at `2f075a2` for recovery.
- The datasets' own splits keep held-out data separate from training and prompts.
- Avoid the multi-teacher dataset's `canonical` folder; it contains two overlapping sets
  of train shards.
- Format conversions that were needed:
  - tool-call `arguments` are stored as JSON strings, but Qwen's chat template iterates
    them as a mapping, so parse them into objects;
  - drop empty optional message fields (`name`, `tool_call_id`, `tool_calls`);
  - `prompt_completion_text` uses `prompt_text` and `completion_text` columns;
  - the agentic `sft_openai` view has no `reasoning_content`.
- Render every training and recovery row through the real chat template before a run.
  This caught the tool-call problem in seconds rather than mid-run.
- Licence and provider terms for these datasets were reviewed and cleared before use.

## Surviving SSH drops on long runs

- **Run builds and pipelines detached** (`setsid nohup ... < /dev/null &`) from a script
  file, so a network drop cannot kill them. Watch with short, fresh SSH polls rather than
  one long session.
- **A backgrounded command chain gets empty stdin.** `cat > file && ... &` writes an empty
  file; write the file in the foreground first.
- **`pkill -f` can match the SSH command's own shell** when the pattern appears in it.
  Kill by PID.
- **`grep` buffers when writing to a file**; use `--line-buffered` for live logs.

## Other fixes made along the way

- The miniature proof reported a false VRAM pass for CPU runs on a GPU host.
- The worker image build failed on Ubuntu 24.04 (pip upgrade).
- `qwen12g worker doctor` counted the CUDA banner as GPUs (11 instead of 2), and
  `runs result` failed to parse JSON for the same reason.
- `.local/` (host controller state) was being copied into images.

## Open issues and next steps

- **Confirm `smoke-train-4`** completes end to end, including merge, re-quantization,
  packed evaluation and the context check, and that QAT now starts near the
  reconstruction's loss.
- **Spread reconstruction across both GPUs.** It peaks at 32.0 GB on GPU 0 with GPU 1
  idle; real-scale settings will likely OOM.
- **Extreme variant and AYOT** (teacher reasoning traces) are untested on hardware.
- **Recovery `max_length` defaults to 8192** in the presets; the smoke run used 4,096.
  Measure before raising it.
- **Host RAM headroom** is thin for BF16 offload and reconstruction; avoid other
  memory-heavy work on the host during runs.
- **Docker build cache eviction** costs 25 to 40 minutes per appliance change.
- **Optional kernels** (`causal_conv1d`, `flash-linear-attention`) for faster linear
  attention.
- **Release gates** (coding and tool-use benchmarks, full-context quality) remain
  untested; the smoke results are process checks only.

## Commits on `dev` from this work

| Commit | Change |
|---|---|
| `f51a3bd` | Trust optional extra root CAs in the worker image |
| `82dc5dd` | Fix worker image build on Ubuntu 24.04 |
| `60705f6` | Fix worker GPU count including CUDA image banner |
| `dd69110` | Fix runs result and storage reads including CUDA image banner |
| `b1a7cc2` | Fix false proof VRAM pass on CPU runs; trust extra CAs in appliance |
| `69910c4` | Bound wizard disk use: keep one QAT checkpoint, release superseded outputs |
| `4f622d9` | Add reproducible wizard input preparation from r0b0tlab datasets |
| `bc88a1a` | Fix SFT model load on Transformers 5 (`use_cache` kwarg rejected) |
| `5f1d5e9` | Prune before SFT, model-parallel SFT, LoRA QAT |
| `3fb5cf8` | Fix chat-template issues on real data; avoid GPU fragmentation |
| `efc0de7` | Make model-parallel SFT and LoRA QAT run on 2x RTX 5090 |
| `71df7eb` | QAT: fake-quantize on the reconstruction's own grid |
