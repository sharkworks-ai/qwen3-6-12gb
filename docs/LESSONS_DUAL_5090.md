# Lessons from bringing the pipeline up on 2x RTX 5090

This records what was learnt while validating the appliance, worker tooling and the
full train/prune/compress/recover pipeline on a dual RTX 5090 lab host with the real
Qwen3.6-35B-A3B model. Figures are measurements from that host unless marked as
estimates. Status is as of `edd04d7` on `dev`.

## Summary

- The pipeline as originally written could not train the full model on 2x32 GB:
  bitsandbytes 4-bit does not quantize Qwen3.6's fused expert tensors, so a "4-bit"
  replica is about 62 GiB per GPU.
- Pruning before SFT, running SFT and QAT as one process sharded across both GPUs, and
  training QAT as LoRA corrections through the quantizer makes every stage fit.
- `smoke-train-10` completed every stage for both variants, including the extreme
  variant and AYOT for the first time. Both compressed bundles fit the memory target
  (9.9 GB and 9.2 GB peak on one GPU).
- The aggressive variant works: held-out loss 1.434 for the original model, 1.629 for
  the final bundle, and it retrieves a needle at 10%, 50% and 90% of a 4,096-token
  prompt. The extreme variant is too degraded (loss 2.88) and answers nothing; most of
  its loss comes from re-quantization.
- An independent review found ten weaknesses in cleanup, evidence and reproducibility;
  all were confirmed and fixed (see "Evidence and reproducibility").
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
| Reconstruction, quantizers spread by estimated load | fixed | 31.5 / 31.6 GB (whole aggressive pipeline) | works |
| Extreme window 0 with 3-bit sensitive experts | original preset | about 43 GiB estimated per GPU | does not fit |
| Extreme window 0 with 2-bit sensitive experts | fixed | 29.9 / 29.6 GB | works |
| SFT on pruned model, model-parallel 4-bit | fixed | 20.1 / 22.7 GB, about 28 s per step | works |
| Merge adapter | BF16 `device_map=auto` | 22.1 / 22.2 GB | works |
| QAT full-parameter FSDP | original design | about 140 GB estimated for a 128-expert student | does not fit |
| QAT LoRA through the quantizer, model-parallel | fixed | about 28.5 / 28.9 GB, about 17 s per step | works |
| Packed evaluation, aggressive / extreme | eager packed runtime | 9.9 / 9.2 GB on one GPU | fits the 11.5 GiB target |
| 4,096-token context check, forced to 64 new tokens | eager packed runtime | 8.8 / 8.0 GB on one GPU | fits the 11.5 GiB target |

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
- **Estimate quantizer memory before building it.** GSQ keeps per-level state, about
  12 bytes per level per weight (calibrated on a measured aggressive layer), so a 3-bit
  tensor costs twice a 2-bit one. `check_window_fits()` now fails with a clear message
  instead of an OOM partway through building quantizers.
- **Mixed precision across devices needs explicit casts.** CAT-Q's ternary quantizer
  computes in FP32 while the block runs in BF16 on GPU; CPU tests run FP32 throughout
  and never saw the mismatch. Quantizers now return the source weight's dtype.
- **Other GPU workloads on a shared host break runs.** A vLLM container started on the
  same GPUs mid-run took about 27 GB on each, and the next stage ran out of memory.
  Check that the GPUs are free before starting and keep other services off them for the
  whole run; the run resumes once they are clear.
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
  re-quantization succeeded. The problem was the peak while the variant runs.
- **Releasing earlier cut the peak.** Reconstruction now deletes its resume state on
  success, QAT checkpoints hold only the LoRA adapters, and the first-pass export goes
  as soon as QAT has saved. `smoke-train-10` ran both variants with disk utilisation
  peaking at 79% (from 61%), and its run directory held 115 GB at the end.
- **Record a deletion before doing it.** If a run stops partway through deleting, the
  next resume must know to finish the job rather than fail its integrity check. Each
  release now writes its intent first and a resume completes it.
- **Hugging Face cache blobs can be shared** between snapshots of the same repository.
  Deleting a snapshot's blobs must skip any that another snapshot links to.
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

7. **Extreme: 2-bit sensitive experts.** The extreme preset's 3-bit early-layer experts
   did not fit a reconstruction window; they now use 2-bit GSQ like the aggressive
   preset.
8. **AYOT budget.** The SFT model can reason for more than 2,048 tokens on a hard
   teacher prompt without closing `</think>`; one such prompt failed the whole stage.
   The budget is now 4,096 tokens, unfinished traces are skipped and recorded, and the
   stage fails only if fewer than 75% of prompts complete. In `smoke-train-9`, 7 of 8
   traces completed.

## Evidence and reproducibility

An independent review of `6c2e3b3` raised ten points. Each was confirmed against the
code and fixed in `4cc977c`:

| Finding | Fix |
|---|---|
| Cleanup ordering could break resume | Deletion intent recorded first; resume finishes it |
| Removing a model could damage sibling cached snapshots | Shared blobs are kept |
| Ranking did not require full context | `required_context_tokens` (default 250,000) plus a forced-length run must pass |
| Baseline was the SFT model, hiding pruning loss | Lineage of held-out loss from the original model to the final bundle |
| Weak needle test (same secret and position) | Position and secret vary per repeat, with decoys and a forced-length run |
| QAT trained on the whole conversation | Assistant tokens only by default (25% of recovery tokens) |
| Evaluation too small, averaged per record, weak contamination check | 32 x 1,024 tokens, token-weighted; held-out tasks checked against training, recovery and prompt data |
| Reconstruction sampling missed samples | Seeded shuffle per window; coverage recorded |
| Recovery decisions accepted missing metrics | Missing or invalid metrics give `insufficient_evidence` |
| QAT seeded after adapter init; open version ranges | Seed first; exact package pins and a pinned llama.cpp commit |

- **Check the test as well as the model.** The needle check fed raw text to a
  chat-tuned model, which answered with a blank line and stopped. Asked as a chat turn
  (thinking off, same prompt length), the aggressive bundle found the secret at every
  position tested.
- **Contamination: compare tasks, not every turn.** Comparing all user turns flagged
  templated follow-up turns shared by unrelated conversations.
  Comparing first user turns gave zero overlap on the r0b0tlab data.
- **Not yet covered:** long-context tests on repository-scale tasks and tool history,
  and an untouched final test set. Both need the isolated benchmark harness.

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

Lineage from `smoke-train-10` (32 held-out records x 1,024 tokens, token-weighted,
19,509 scored tokens; 10 SFT steps, 2 reconstruction steps per window, 10 QAT steps):

| Stage | Aggressive | Extreme |
|---|---|---|
| Original 35B | 1.434 | 1.434 |
| Pruned to 128 experts | 1.545 | 1.545 |
| SFT merged | 1.470 | 1.470 |
| First-pass reconstruction | 1.533 | 2.021 |
| After QAT | 1.534 | 2.019 |
| Final packed bundle | 1.629 | 2.882 |
| Needle found at 4,096 tokens (chat prompt) | 3 of 3 | 0 of 3 |

- Pruning costs about +0.11; ten SFT steps recover about two thirds of it.
- Reconstruction costs about +0.06 for aggressive and +0.55 for extreme.
- Ten QAT steps change nothing measurable.
- Re-quantization costs +0.10 for aggressive but +0.86 for extreme, the largest single
  loss in the pipeline. Packing QAT output on the first-pass grid (see open issues)
  would avoid this step.
- Figures in the older table above used 8 or 32 records averaged per record and are
  not directly comparable.
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

- **Extreme re-quantization loss (+0.86).** Find whether it is the second GSQ pass at
  2 bits or ternary CAT-Q, by evaluating each tensor group separately.
- **Consider packing QAT output directly.** QAT now trains on the reconstruction's grid,
  so its output is already on the packed grid. Packing it with the first-pass scales
  would replace the second GSQ pass: no re-quantization loss, and about 20 minutes and
  56 GB saved per variant. This is a design change.
- **Real-scale settings.** Smoke settings (2 reconstruction steps, 10 QAT steps) only
  prove the path works; QAT effect and reconstruction quality need real budgets.
- **Report field.** `baseline_logit_mse` is now empty because the baseline no longer
  saves logits; lineage replaces it.
- **Recovery `max_length` defaults to 8192** in the presets. Measure memory before
  using it at scale.
- **Host RAM headroom** is thin for BF16 offload and reconstruction; avoid other
  memory-heavy work on the host during runs.
- **Docker build cache eviction** costs 25 to 40 minutes per appliance change.
- **Optional kernels** (`causal_conv1d`, `flash-linear-attention`) for faster linear
  attention.
- **Release gates** (coding and tool-use benchmarks, full 250,000-token context,
  repository and tool-history recall) remain untested; the smoke results are process
  checks only.

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
| `0d51658` | Update 2x RTX 5090 lessons: re-quantization bug, disk, parameter counts |
| `6c2e3b3` | Cut per-variant disk peak and spread reconstruction across GPUs |
| `0d7f85d` | Extreme: 2-bit sensitive experts; check window memory before building |
| `7dd11bb` | AYOT: 4096-token budget; skip unfinished traces instead of failing |
| `15e323d` | Return quantizer outputs in the source weight dtype |
| `4cc977c` | Address independent review: safe cleanup, evidence gates, lineage, QAT loss |
| `edd04d7` | Ask the needle question as a chat turn |
