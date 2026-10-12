# Lessons from bringing the pipeline up on 2x RTX 5090

This records what was learnt while validating the appliance, worker tooling and the
full train/prune/compress/recover pipeline on a dual RTX 5090 lab host with the real
Qwen3.6-35B-A3B model. Figures are measurements from that host unless marked as
estimates. Status is as of `c9fdd3c` on `dev`.

## Summary

- The pipeline as originally written could not train the full model on 2x32 GB:
  bitsandbytes 4-bit does not quantize Qwen3.6's fused expert tensors, so a "4-bit"
  replica is about 62 GiB per GPU.
- Pruning before SFT, running SFT and QAT as one process sharded across both GPUs, and
  training QAT as LoRA corrections through the quantizer makes every stage fit.
- A full training run (`smoke-train-5`) completed every stage end to end, and the
  compressed model fits the memory target (about 7.3 GB of weights, about 9 GB peak on
  one GPU). Its quality was destroyed by a re-quantization bug, now found and fixed.
- With all fixes, the aggressive variant keeps the model working at every stage
  (held-out loss 1.66 uncompressed, 1.86 after the full compression pipeline, smoke
  settings). A confirming end-to-end run (`smoke-train-6`) was in progress when this
  was written.
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
| Experts | 256 per layer, 8 active per token |
| Layers | 40, hybrid linear attention (gated delta net) and full attention |
| Native context | 262,144 tokens |
| Vocabulary | 248,320 tokens |

Parameter counts, read from the checkpoint files:

| Part | Original | Pruned to 128 experts |
|---|---|---|
| Routed experts | 33.02B | 16.51B |
| Routers | 21.5M | 10.7M |
| Vision encoder | 0.45B | 0.45B |
| Attention, shared experts, embeddings, LM head, norms | 2.47B | 2.47B |
| **Total** | **35.95B** | **19.43B** |

- Pruning halves only the routed experts and routers. Active parameters per text token
  stay at about 3.5B (8 routed experts plus everything else), so the pruned model is a
  "19B-A3B" by the same convention that names the original "35B-A3B".
- A model built by Transformers holds about 0.85B fewer parameters (35.1B, of which
  32.2B are experts) than the checkpoint files. The difference is most likely a
  multi-token-prediction head that is stored but not loaded; this is unconfirmed.
- The pruned BF16 checkpoint is 37 GB on disk. Quantization does not change parameter
  counts, only storage.

## Memory: what fits and what does not

| Stage | Approach | Peak GPU (GPU 0 / GPU 1) | Result |
|---|---|---|---|
| SFT, per-GPU 4-bit replica (DDP) | original design | about 62 GiB needed per GPU | does not fit |
| Profile, 4-bit `device_map=auto` | original design | needs CPU offload, which 4-bit forbids | fails |
| Profile, BF16 with CPU overflow | fixed | 28.1 / 28.0 GB | works |
| Baseline evaluation | BF16 `device_map=auto` | 19.3 / 19.6 GB | works |
| Reconstruction (GSQ 2/4-bit), 1 layer per window | original | up to 32.1 GB / idle | works only with expandable segments |
| SFT on pruned model, model-parallel 4-bit | fixed | 20.1 / 22.7 GB, about 28 s per step | works |
| Merge adapter | BF16 `device_map=auto` | 22.1 / 22.2 GB | works |
| QAT full-parameter FSDP | original design | about 140 GB estimated for a 128-expert student | does not fit |
| QAT LoRA through the quantizer, model-parallel | fixed | about 28.5 / 28.9 GB, about 17 s per step | works |
| Packed evaluation of the compressed model | eager packed runtime | 9.3 GB on one GPU | fits the 11.5 GiB target |
| 4,096-token context check | eager packed runtime | 8.8 GB on one GPU | fits the 11.5 GiB target |

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
- **A parent process can hold GPU memory its child needs.** The pipeline reconstructs
  in-process and then launches QAT as a subprocess; the parent still held about 9 GB of
  cached memory on GPU 0. Collect garbage and empty the CUDA cache before launching.
- **Host RAM is the next constraint.** Profiling and baseline spill BF16 weights to CPU,
  and reconstruction holds the source in RAM. Available RAM dipped to 44 GB in one
  stage.

## Disk

- **A single smoke run reached 180 GB** before its end-of-variant clean-up, and filled
  the disk once. Where it went (aggressive variant only):

  | Item | Size | Needed after |
  |---|---|---|
  | QAT directory: recovered model plus a trainer checkpoint holding the frozen model | 75 GB | Re-quantization only |
  | First-pass `research-hf` and `packed` | 40 GB | QAT only |
  | Merged SFT model | 36 GB | The whole run |
  | Reconstruction `teacher-cache`, `training-checkpoints`, `student-inputs` | 16 GB | Reconstruction only (resume state) |
  | Re-quantized bundle (final) | about 56 GB at peak | Deliverable |

- **The end-of-variant release works**: the same run dropped from 180 GB to 100 GB once
  re-quantization succeeded. The problem is the peak while the variant runs.
- **A disk guard makes long runs safe.** A small host script checks utilisation every
  10 seconds and stops the run's pipeline processes at 88%, below a 90% ceiling; the run
  then resumes once space is freed. Start it before the image build so it covers the
  whole run.
- **Test and diagnostic outputs add up quickly.** Isolated stage tests each wrote 35 to
  75 GB; keep them in one scratch directory and clear it as soon as each result is
  recorded.

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
   tensors), min/max picks a smaller step and re-rounds every weight. QAT now loads the
   packed per-group scales and rounds onto the grid the runtime decodes. GSQ scales are
   signed and some are exactly zero (a zero-scale group decodes to zeros); dividing by
   them produced NaN until zero scales were handled explicitly.
5. **All-zero groups get the tensor's typical step in GSQ initialization.** See the
   re-quantization bug below.
6. **Disk budget.** QAT keeps one trainer checkpoint (it previously kept four per
   variant). `Stages.release()` deletes superseded outputs of finished stages and
   re-baselines their integrity hashes so resume still works: the pruned model after
   merge, and each variant's first-pass export and QAT state after re-quantization.
   `keep_intermediates` disables this; `release_source_after_merge` opts in to deleting
   the base checkpoint.

## The re-quantization bug

`smoke-train-5` ran every stage, but the compressed model was broken (held-out loss
7.78 against 1.34 before compression; it generated 2 blank tokens on the context
check). A stage-by-stage rerun found where:

| Step | Held-out loss (6 samples) | Held-out loss (32 samples) |
|---|---|---|
| SFT model, uncompressed | 1.339 | 1.660 |
| First-pass reconstruction | 1.496 | 1.783 |
| QAT recovery | 1.493 | 1.784 |
| Re-quantization, before fix | 7.788 | 7.703 |
| **Re-quantization, with fix** | **1.590** | **1.859** |

- Ruled out first: the packed runtime (the BF16 export was equally broken), the
  save/reload round trip (identical loss) and QAT's save path (identical weights at
  learning rate 0).
- Cause: 2-bit reconstruction rounds many groups entirely to zero, up to 42% of expert
  `gate_up_proj` groups in early layers. GPTQ gives an all-zero group the range [-1, 1],
  a step of about 0.67 against weights near 0.01. GSQ's stochastic code search then
  turns any code that leaves 0 into a weight about 50 times too large. Expert
  `gate_up_proj` relative error reached 11 in the worst layer; every other tensor
  changed by 3 to 10%.
- Fix: after GPTQ initialization, all-zero groups take the tensor's median step. The
  pinned upstream GSQ code is unchanged.
- Lesson: when a run's output is broken, evaluate each stage's output separately before
  theorising. Automatic clean-up deleted the intermediates of the failing run, so the
  diagnosis needed a rerun with each output kept.

## Quality measurements (smoke settings)

Held-out loss is the average cross-entropy per token on unseen text; lower is better,
and perplexity is e^loss. Uniform guessing over the vocabulary scores ln(248,320) =
12.4. Compare only figures from the same samples and script.

| Measurement | Held-out loss |
|---|---|
| Pruned BF16, before SFT (pipeline baseline, 8 samples) | 1.58 |
| Reconstruction re-rounded by the old min/max QAT quantizer | 2.41 to 2.48 |
| Same reconstruction through the reconstruction-grid quantizer | 1.504 (as saved: 1.506) |
| Stage-by-stage results | see the table above |

- Reconstruction costs about +0.12 (perplexity about 13% higher) even with 2 steps per
  layer.
- Ten QAT steps change nothing measurable; re-running GSQ in re-quantization costs about
  +0.08. Both should be re-measured at real-scale settings.
- Differences under about 0.05 are noise at these sample sizes.
- These are smoke-test diagnostics, not quality claims. Coding, tool use and
  long-context retention remain untested.

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
| GSQ (GPTQ init) | All-zero groups get the range [-1, 1] | Median step for all-zero groups |
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

- **Confirm `smoke-train-6`** end to end on `c9fdd3c`, including packed evaluation and
  the context check on a working compressed model.
- **Reduce the per-variant disk peak**, estimated at about 95 GB per variant: delete
  reconstruction resume state on success, store only LoRA adapters and optimiser state
  in QAT checkpoints, and release the first-pass export as soon as QAT has saved.
- **Consider packing QAT output directly.** QAT now trains on the reconstruction's grid,
  so its output is already on the packed grid. Packing it with the first-pass scales
  would replace the second GSQ pass: no re-quantization loss, and about 20 minutes and
  56 GB saved per variant. This is a design change.
- **Spread reconstruction across both GPUs.** It peaks at 32.1 GB on GPU 0 with GPU 1
  idle; real-scale settings will likely OOM.
- **Extreme variant and AYOT** (teacher reasoning traces) are untested on hardware.
- **Recovery `max_length` defaults to 8192** in the presets. Measure memory before
  using it at scale.
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
| `146b7ce` | Document lessons from bringing the pipeline up on 2x RTX 5090 |
| `239d9d5` | QAT grid: handle signed and zero GSQ scales; free GPU before QAT |
| `c9fdd3c` | Fix re-quantization blow-up: typical GSQ step for all-zero groups |
