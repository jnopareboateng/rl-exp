# Track 02: THUDM / Zhipu (Z.ai) RL: slime, the GLM line, and the agent-RL repos

Date of research: 2026-10-05. Code read: `THUDM/slime` at HEAD commit `8c17b676` (2026-10-02, shallow clone, so no commit history). Everything marked **[code]** was read directly in the repo. **[paper]** means I read the arXiv text (through a summarizing fetch tool, so numbers are quoted from its output, not eyeballed from the PDF). **[web]** means a blog, news item, or docs page. **[inferred]** means my own reading between sources.

---

## 1. TL;DR

- **slime is the RL engine behind the whole GLM line.** Its README says it is the framework behind GLM-4.5, 4.6, 4.7, 5, 5.1, 5.2 and 5.3 **[code: README.md]**. The GLM-4.5 and GLM-5 reports confirm slime for 4.5 and the async agentic infrastructure for 5 **[paper]**. For 5.2 the Z.ai blog confirms it **[web]**. The README's lineage for 5.3 is the only source I found for slime's role there.
- **Architecture:** Ray orchestrates three parts. (1) Megatron-LM trains (TP/PP/CP/EP/ETP, CPU-offloaded Adam). (2) SGLang serves rollouts behind sgl-router. (3) A "data buffer" (prompt source, partial-rollout buffer, or a persistent straw queue) sits between them. The training backend is Megatron only; the rollout backend is SGLang only. This is deliberate. The only other-engine path is the community fork **vime** (vLLM) **[code + web]**.
- **Sync vs async is just a choice of rollout function.** The default `train.py` loop is strictly generate, then train, then weight-sync. It runs colocated (`--colocate`, offload plus CUDA-IPC weight push) or disaggregated (separate GPUs, NCCL or disk transport). "Fully async" means swapping `--rollout-function-path` to a background worker that keeps a warm pool of in-flight trajectories across training steps. A newer "PipelineRL" mode (`--flush-cache-interval`) also pauses generation in place and keeps KV across weight updates **[code]**.
- **Extension surface is ~20 function-path flags** (`--custom-generate-function-path`, `--custom-rm-path`, `--custom-loss-function-path`, `--custom-tis-function-path`, `--custom-advantage-function-path`, `--data-source-path`, ...). Agentic RL (SWE agents, search, tau-bench, multi-agent) plugs into these hooks. No separate agent framework is needed **[code]**.
- **Loss and train/inference mismatch get serious treatment.**
  - Estimators: GRPO, GSPO, CISPO, REINFORCE++, REINFORCE++-baseline, and PPO with a critic.
  - Mismatch tools: TIS, IcePop-style masking, MIS with sequence veto, OPSM, MoE routing replay (R2 and R3), top-p replay, FP8 rollout, and a brand-new "score centering" option (arXiv 2609.20807, Sept 2026).
  - GLM-5 gets a bitwise-aligned train/rollout path (|logprob diff| < 1e-6 gated in CI) **[code]**.
- **GLM-5.2 moved off GRPO.** Zhipu's SAO paper (arXiv 2607.07508) drops group sampling for single-rollout, critic-based PPO. It uses a skip-observation GAE and double-sided token masking against the rollout log-probs. GLM-5.2 was trained with it, using slime plus compaction and sub-agent rollouts **[paper + web]**. slime ships the pieces (PPO plus critic, custom advantage hook, TITO adapters, fan-out samples). I found no SAO-specific code in the public tree **[code, negative result]**.
- **Related THUDM work:** AgentRL (async multi-task agent RL, veRL-derived, Go controller for up to 10k sessions), WebRL (self-evolving online curriculum), AutoGLM, ComputerRL (API-GUI plus Entropulse), ScaleCUA, and DeepDive.
- **For a biologist:** a docking or folding oracle is a plain `async def reward(args, sample) -> float` behind `--custom-rm-path`. The hard part is not the hook. It is that slime runs every rollout coroutine on one shared asyncio loop, so a blocking oracle call stalls all generation. Wrap the oracle in a thread, process pool or service, and use the fully-async rollout so slow oracles do not idle the GPUs (section 6).

---

## 2. slime: architecture from the code

### 2.1 Process layout **[code: `train.py`, `slime/ray/*`]**

```
train.py
 ├─ create_placement_groups(args)           # Ray PGs: actor / critic / rollout GPUs
 ├─ create_rollout_manager(...)             # RolloutManager (Ray actor): SGLang engines + router + data source
 ├─ create_training_models(...)             # Megatron actor group (+ critic group if --advantage-estimator ppo)
 └─ loop over rollout_id:
      rollout_manager.generate()            # -> sglang_rollout.generate_rollout (or custom)
      actor.async_train(rollout_data_ref)   # -> Megatron fwd (ref/old logprob), advantages, fwd/bwd, optim step
      rollout_manager.training_completed()
      save_checkpoint(...) every --save-interval
      pause_rollout_admission(); actor.update_weights(); resume_...()
```

The Megatron actor is `slime/backends/megatron_utils/actor.py`, with `model.py` for the train step and `loss.py` for losses and advantages. The training backend is a single value: "slime currently uses Megatron-LM as its training backend" (`docs/en/get_started/usage.md`). The repo ships Megatron patches (a `megatron-sglang-aligned.patch` is referenced in the reproducibility doc).

Arguments come in three kinds:
- Megatron args pass through untouched.
- Every SGLang `ServerArgs` flag is exposed with a `--sglang-` prefix.
- slime's own flags live in `slime/utils/arguments.py` (~2,200 lines).

The router is the sgl-router / SGLang Model Gateway. v0.2.4 removed the separate `slime-router` **[web: releases page]**. For multi-turn agents there is a `--router-policy consistent_hashing` option keyed on the `X-SMG-Routing-Key` header, set from `sample.session_id`. This gives prefix-cache affinity per agent session (`sglang_rollout.py`, `docs/en/advanced/sglang-config.md`).

### 2.2 The Sample and the "data buffer" **[code: `slime/utils/types.py`, `slime/data/data_source.py`, `batch_builder.py`]**

The unit of exchange is `Sample`. Key fields:
- `prompt`, `tokens`, `response`, `response_length`, `label`
- `reward` (float or dict, picked by `--reward-key`)
- `loss_mask` (the token-level "who is trainable" mask)
- `rollout_log_probs` (from SGLang `return_logprob=True`)
- `weight_versions` (for staleness), `status` (PENDING, COMPLETED, TRUNCATED, ABORTED)
- `session_id`, `metadata`, `rollout_id` (siblings from one trajectory share it)
- MoE routing capture (`rollout_routed_experts`)
- top-p nucleus capture (`rollout_top_p_token_ids/offsets`)

The "Data Buffer" is `RolloutDataSourceWithBuffer`. It serves prompt groups (`get_samples`) and takes back aborted or partial groups (`add_samples`). Two buffer policies ship: `pop_first` and `pop_oldest` (by weight version).

Two data-transport modes exist:
- **Ray object store** (default).
- **straw** (`--rollout-data-transport straw`): a persistent, leased queue on shared JuiceFS storage (`slime/data/queue_data_source.py`, 1,380 lines). It gives crash recovery and debug replay.

Debug tooling is unusually complete: `--debug-rollout-only`, `--debug-train-only`, `--save/--load-debug-rollout-data`, a trace viewer, and CPU "plugin contract" tests for every hook.

### 2.3 Rollout loop **[code: `slime/rollout/sglang_rollout.py`]**

`generate_rollout_async` runs a DAPO-style loop:
1. Over-sample `--over-sampling-batch-size` prompt groups.
2. Fire each group through `generate_and_rm_group` as an asyncio task.
3. As groups finish, apply `--dynamic-sampling-filter-path` (e.g. `check_reward_nonzero_std`, which drops groups whose rewards have std < 1e-6).
4. Stop at `--rollout-batch-size` accepted groups.
5. Abort the rest on the server. With `--partial-rollout`, collect half-finished samples back into the buffer; `--mask-offpolicy-in-partial-rollout` zeroes the loss mask on their old tokens.

Per-sample flow in `generate_and_rm`:
1. Acquire the global generation semaphore, `sglang_server_concurrency × num_engines`.
2. Call `custom_generate_function_path` or the stock `generate`. Stock `generate` POSTs `input_ids` plus `return_logprob` to `/generate` and stores the returned token ids and log-probs. This is token-in-token-out.
3. Run optional sample hooks.
4. Compute the reward via `async_rm` (or `batched_async_rm` for `--group-rm`).

Two details matter for custom rewards:
- **The reward call happens after the generation semaphore is released.** Nothing in slime throttles your oracle. You must rate-limit it yourself.
- **The whole rollout runs on one background asyncio loop thread** (`slime/utils/async_utils.py`, `run(coro)`). A reward function that blocks synchronously blocks every in-flight generation and tool call.

### 2.4 Reward models **[code: `slime/rollout/rm_hub/__init__.py`]**

Built-in `--rm-type` values:
- `math` (veRL grader)
- `dapo`
- `deepscaler` (the default in all example scripts)
- `f1`
- `gpqa`
- `ifbench`
- `remote_rm` (POSTs `{prompt, response, label}` to `--rm-url` with 10 retries and exponential backoff)
- `random`
- A `boxed_` prefix extracts `\boxed{}` first.

A per-sample `custom_rm_path` (from eval dataset config) beats the global `--custom-rm-path`, which beats the `rm_type` dispatch. The `remote_rm` path is a ready-made HTTP oracle client.

### 2.5 Colocated vs disaggregated; sync vs async **[code + docs]**

| Mode | Flags | What happens |
|---|---|---|
| Colocated, synchronous | `--colocate` (+ offload) | Same GPUs time-share. After rollout, SGLang weights and KV are offloaded (`offload_rollout`). Megatron trains. Training is offloaded (`torch_memory_saver.pause/resume`). Weights are pushed to SGLang via **CUDA IPC** (flattened-tensor buckets, Gloo gather, Ray handle). Used for GLM-4.7 on 64 H100 and GLM-5.2 on 256 H100. |
| Disaggregated, synchronous | omit `--colocate`, set `--rollout-num-gpus` | Separate GPU pools. Weights go over an NCCL group from the trainer to the engines, or via disk. Generate, then train, then sync, still in lockstep. |
| Fully async | `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async` | A background asyncio worker keeps a fixed pool of trajectories in flight across training steps. Aborted groups are redirected back to the buffer. Needs separate GPUs. |
| PipelineRL-style | `--flush-cache-interval N` (<=0 or >1) | Weight update uses SGLang `pause_generation(mode="in_place")`, keeps unfinished sequences and their KV. Samples span policy versions, so use `--use-rollout-logprobs`. Auto-selects the fully-async rollout. |
| External engines | `--rollout-external-engine-addrs`, `--update-weight-transport disk` | SGLang owned by another system (even other GPU vendors). Weights move as HF checkpoints, or **XOR + zstd deltas** (`--update-weight-mode delta`) on a shared filesystem. The engine pulls via an slime-patched `/pull_weights`. |

The GLM-4.5 report describes the same split: synchronous colocated mode for reasoning/general RL, and disaggregated asynchronous mode for agentic tasks such as SWE, where "data generation is protracted" **[paper]**.

Other infrastructure in the tree:
- `--use-fault-tolerance`: heartbeat `/health_generate`, restart a hung engine, then re-sync its weights.
- PD (prefill/decode) disaggregation via a `--sglang-config` YAML with typed server groups (`regular`, `prefill`, `decode`, `placeholder`). The same YAML supports multi-model serving with `update_weights: false` for frozen reference or reward LLMs.
- EAGLE/MTP speculative decoding, with an option to train the MTP layer online during RL so the draft doesn't drift.

### 2.6 Algorithms and loss **[code: `slime/backends/megatron_utils/loss.py`, `slime/utils/ppo_utils.py`, `slime/data/batch_builder.py`]**

`--advantage-estimator` choices are `grpo`, `gspo`, `cispo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, and `ppo`. PPO builds a Megatron critic (`args.use_critic`) with `--num-critic-only-steps` for value warmup. `--custom-advantage-function-path` replaces all of them: it must fill `rollout_data["advantages"]` and `["returns"]`.

**Advantages for GRPO/GSPO/CISPO.**
- Reward normalization is done at batch-build time. It subtracts the group mean and then divides by group std plus 1e-6 (`--disable-grpo-std-normalization` gives the Dr.GRPO variant). It handles uneven group sizes.
- The scalar advantage is broadcast to every response token.
- `--normalize-advantages` additionally whitens across the DP×CP group, with a CP-correct masked reduction. A code comment records a bug where whitening across only DP made CP halves of one sequence get different affine transforms.

**Policy loss** (`policy_loss_function`). Per token, with `ppo_kl = old_logp - logp`:
- **PPO/GRPO:** `max(-r·A, -clip(r, 1-eps_clip, 1+eps_clip_high)·A)`, with optional dual-clip `eps_clip_c`. Asymmetric clip-higher is standard: `--eps-clip 0.2 --eps-clip-high 0.28` in the GLM-5.2 script.
- **GSPO:** the "KL" is the length-normalized sequence-level log-ratio expanded back to every token (`compute_gspo_kl`). This needs a CP all-gather. The GLM-4.7-355B script uses `--advantage-estimator gspo --eps-clip 1e-4 --eps-clip-high 2e-4`, i.e. GSPO's very small sequence-level clip range.
- **CISPO** (MiniMax-M1): `-sg(clip(r)) · A · logp`, so clipped tokens still carry gradient. `eps_clip >= 1` disables the lower bound.
- **REINFORCE:** `-A·logp`. Forced when score centering is on.
- **Add-ons:** entropy bonus, KL-to-reference loss (`k1`, `k2`, `k3`/`low_var_kl`, optional unbiased IS-ratio-weighted KL "from DeepSeek-V3.2"), and OPSM (off-policy sequence masking: zero the whole sequence if its mean `old - new` log-ratio exceeds `--opsm-delta` AND its advantage is negative).

**Loss aggregation.** The default is per-rollout token-weighted mean. `batch_builder` precomputes `rollout_mask_sums` so a rollout split across micro-batches or fanned out into several `Sample`s gets one correct denominator. Dr.GRPO-style constant normalizers use `--custom-pg-loss-reducer-function-path`.

**On-policy distillation** is orthogonal: `A_hat = A - λ·(logp_student - logp_teacher)` per sampled token. The teacher is an SGLang server or a Megatron checkpoint.

### 2.7 Train/inference mismatch handling **[code]**

This is where slime is deepest. In ascending order of intervention:

1. **Metrics only:** `--get-mismatch-metrics` logs `train_rollout_logprob_abs_diff`, `ois`, and similar, without changing the loss.
2. **TIS:** `--use-tis --tis-clip 2.0 --tis-clip-low 0` (in the GLM-5.2 script: low 0.5, high 2.0). The weight is `clamp(exp(train_logp - rollout_logp), low, high)` multiplied into the pg loss. `icepop_function` is the same ratio but *masks to zero* outside the band (IcePop). The GLM-5 paper's `pop(ρ, 1/β, β)` with β=2 matches this **[paper]**.
3. **Full MIS/RS helper** (`examples/train_infer_mismatch_helper/mis.py` + `mis.yaml`):
   - Weights can be truncated, clipped or masked, at token, sequence or geometric level.
   - There is a separate rejection-sampling bound.
   - A per-token **veto** (`rs_veto_threshold: 1e-4`) zeroes an entire sequence if any token ratio falls below it.
   - There is batch-normalization of the IS weights to mean 1.
   - A decoupled three-policy PPO form is also provided.
   - `--use-rollout-logprobs` skips the training-side recomputation and uses SGLang's log-probs as `π_old`. This is the same idea as GLM-5's "direct double-sided importance sampling" **[inferred match]**.
4. **MoE routing replay:** `--use-routing-replay` (R2) and `--use-rollout-routing-replay` (R3, arXiv 2510.11370; SGLang returns `routed_experts` and the trainer replays them).
5. **Top-p/top-k replay and score centering** (`slime/utils/score_centering.py`, `get_score_centering_terms` in loss.py). This implements arXiv 2609.20807 (Marek and Ryabinin, 2026-09-17). Score centering is an additive correction for the persistent "drift" between training and inference engines. In slime it requires `--pg-loss-type reinforce` and composes with the built-in TIS/IcePop bounds. The sampler's top-k or full top-p nucleus is recorded during rollout and used to compute head and tail mass corrections. I did not independently evaluate the paper's claims beyond its abstract.
6. **Precision alignment:**
   - BF16 training plus block-FP8 rollout is the "stable production path". At each weight sync slime quantizes BF16 weights using the `quantization_config` in the FP8 `--hf-checkpoint`.
   - FP8 KV cache: `--sglang-kv-cache-dtype fp8_e4m3`.
   - Experimental FP8 *training* with `--fp8-format e4m3 --fp8-recipe blockwise`, backed by an LMSYS blog arguing quantization error, not kernels, drives the BF16-train/FP8-rollout mismatch **[web]**. INT4 QAT and INT4 rollout are marked beta.
7. **Bitwise alignment for GLM-5** (`docs/en/advanced/reproducibility.md`, "Train/rollout log-prob alignment (GLM-5)"): batch-invariant DeepGEMM block-FP8 for dense and grouped-MoE layers; an fp32 MoE router; DeepEP low-latency rollout versus a custom Megatron DeepEP dispatch with slot-ordered FP32 reduction; deterministic NSA prefix cache; and `--freeze-indexer`. CI asserts `train_rollout_logprob_abs_diff < 1e-6` on a 6-layer GLM-5.2 (`tests/test_glm52_6layer_deterministic_e2e.py`) and exact zero hidden-state difference over layers 0-5 (`tests/test_glm52_layerwise_zero_e2e.py`).

### 2.8 Weight-update path **[code: `slime/backends/megatron_utils/update_weight/*`]**

- `update_weight_from_tensor.py` (colocated): gather over PP/EP/TP (NCCL), convert Megatron to HF names and layouts on GPU, quantize to FP8 if the rollout checkpoint is quantized, flatten into a bucket (`--update-weight-buffer-size`, 2 GiB in the 744B recipe), serialize CUDA-IPC handles, gather via Gloo, hand to SGLang over Ray. A code comment explains the flattened tensor is never reused because SGLang can return before its GPU copy finishes.
- `update_weight_from_distributed.py`: NCCL broadcast to remote engines.
- `update_weight_from_disk*.py`: HF safetensors directory per version (`weight_v000123/`); delta mode stores `new XOR old` (or overwrite) tensors, zstd level 1, with per-tensor xxh3-128 checksums. The trainer's baseline snapshot is seeded from `--hf-checkpoint` so it works even where the Megatron-to-HF round trip isn't byte-exact.
- Flush semantics: `--flush-cache-interval` (see 2.5).

### 2.9 Agentic and multi-turn examples **[code: `examples/*`, `slime/agent/*`]**

| Example | Pattern |
|---|---|
| `search-r1` | Custom generate loop with `<search>` and `<answer>` tags. A BUGFIX comment documents that SGLang kept emitting junk after `</search>`, which was then trained on (loss_mask=1). The fix is to pass `stop` tags and keep tokens and logprobs aligned natively. Format reward `format_score` 0.2 plus exact match. |
| `retool` | Tool-augmented math with a code sandbox. |
| `tau-bench` | OpenAI-style tool adapter with token deltas. |
| `multi_agent` | Parallel sub-agents inside one `custom_generate`. |
| `geo3k_vlm(_multi_turn)` | VLM plus multi-turn. |
| `strands_sglang` | Strands agent framework. |
| `coding_agent_rl` | Real **claude-code / Codex CLI** harness inside an E2B sandbox. The agent talks to a slime **AnthropicAdapter** that tokenizes message history each turn, records prompt/output token snapshots, and keeps model-generated tokens as trainable only while later prompts stitch onto them. Template and observation tokens get mask 0. A `TrajectoryManager` keeps a message tree; any prefix divergence (sub-agent dispatch, auto-compaction) forks a branch. Each leaf becomes a `Sample`. Segments are labeled `subagent`, `wipe` (pre-compaction), `final`, and share a `rollout_id`. Reward comes from running the dataset's tests on a `git diff` in a second clean sandbox (no test-cheating). |
| `fully_async`, `eval_multi_task`, `on_policy_distillation`, `delta_weight_sync` | Mode demos. |

`slime_plugins/rollout_buffer/` is the older approach: an external HTTP "rollout buffer" server where agent frameworks call slime's LLM endpoint and push trajectories.

### 2.10 Models and example scale **[code: `scripts/`, `docs/en/examples/`]**

Scripts exist for GLM-4 9B, GLM-4.7 (30B-A3B and 355B-A32B), GLM-5 and GLM-5.2 (744B-A40B), Qwen2.5/3/3.5/3.6/Next, DeepSeek-R1, Kimi-K2 (Instruct and Thinking), MiniMax-M2, Moonlight, and MiMo-7B.

The GLM-5.2 recipe (`scripts/run-glm5.2-744B-A40B.sh`):

| Setting | Value |
|---|---|
| Hardware | 32 nodes, 256 H100, colocated |
| Parallelism | TP4 · PP8 · CP8 (DP=1), EP32, expert-TP 1, uneven PP split 14/8×6/16 |
| Rollout | PD-disaggregated: 1 prefill engine (64 GPU) + 3 decode engines (192 GPU), EP64, DeepEP, mooncake over IB |
| Rollout precision | FP8 weights and FP8 KV, NSA/flashmla sparse attention |
| Speculative decoding | EAGLE using the model's own MTP layer |
| Training precision | BF16 train, CPU-offloaded Adam (`--optimizer-cpu-offload --use-precision-aware-optimizer`) |
| Batch | 8 prompts × 8 samples per rollout, global batch 64, 64k max response |
| Algorithm | GRPO, `eps 0.2/0.28`, TIS 0.5/2.0, no KL, no entropy bonus |
| Optimizer | AdamW, lr 1e-6, constant, wd 0.1, beta2 0.98 |
| Data | dapo-math-17k with `deepscaler` rule reward |

This is the *demo* recipe (math reward), not Zhipu's production mix. Treat it as the infrastructure template, not the GLM-5.2 training data or algorithm.

Release cadence per the GitHub releases page: v0.2.3 (YAML sglang_config, GLM-5, consistent hashing), v0.2.4 (profiling, VLM, sgl-router unification), v0.3.0 (agent module, sandbox coding-agent example, fully async first-class), v0.3.1 (weight-update speed, top-p masking, fused PPO), v0.3.2 (fully aligned GLM-5). The fetch tool printed the year as 2024 for all of these, which cannot be right given the contents (GLM-5 is a 2026 model). I read this as a typo for 2026 **[inferred]**.

---

## 3. GLM family: what is publicly known about the RL

### GLM-4.5 / 4.5-Air (arXiv 2508.06471) **[paper]**
- **Pipeline:** train expert models in three domains (Reasoning, Agent, General chat), then self-distill them into one hybrid-reasoning model on "millions" of samples.
- **Algorithm:** "builds upon GRPO, excluding the KL loss term." For agentic RL the objective is group-relative reward with only model-generated tokens optimized; environment feedback is excluded from the loss.
- **Reasoning RL tricks:**
  - A two-stage difficulty curriculum. Stage two uses problems with pass@8 = 0 but pass@512 > 0, giving AIME24 83.4% vs 81.8% baseline.
  - **Single-stage RL directly at 64K** rather than progressive length stages, which they say makes the model "unlearn" long-context ability.
  - Dynamic sampling temperature: raise the temperature when mean reward plateaus, picked on a held-out set with at most 1% degradation.
  - **Token-weighted mean loss** beats sequence-mean for code RL.
  - Science RL works better trained only on expert-verified multiple-choice questions.
- **Agentic rewards:**
  - Web search: final-answer accuracy for the whole trace.
  - SWE: verifiable test cases.
  - Function calling: strict binary reward, 1 only if format and arguments match exactly. There are step-wise and end-to-end variants.
- **General RL:**
  - About 5,000 prompts across 7/33/139 categories, scored by human and AI feedback.
  - Instruction following with 7 major / 151 minor constraint types, using deterministic rules plus a trained reward model plus a critique model.
  - A "pathology RL" stage targeting language mixing, repetition and formatting errors.
- **Iterative distillation:** do RL on the cold-start model, replace the SFT data with the RL model's own responses, then resume RL at higher difficulty.
- **Infra:** slime, hybrid colocated/synchronous and disaggregated/asynchronous. BF16 training with online block-wise FP8 rollout. A Docker-based high-concurrency agent runtime with a unified HTTP endpoint and a centralized data pool.

### GLM-4.6 / 4.7 / 4.5V / 4.6V
- **GLM-4.6** (200K context): the official docs page I could fetch discloses no training or RL method **[web, negative]**. GLM-4.7 added "interleaved" and "preserved thinking" per a Z.ai X post and press coverage, which I did not open in full **[web, unverified]**. slime ships GLM-4.7 recipes (30B-A3B on 8 GPUs, 355B-A32B on 64 H100 with GSPO) **[code]**.
- **GLM-4.1V-Thinking / GLM-4.5V** (arXiv 2507.01006): the headline method is **RLCS, Reinforcement Learning with Curriculum Sampling**, across STEM, video, grounding, coding, GUI agents and long documents **[paper, abstract only]**.
- **GLM-4.6V**: the docs page says nothing about RL **[web, negative]**.
- **GLM-5V-Turbo** (2026-04-01): "30+ task joint RL" across STEM, grounding, video and tool use. It is pitched as avoiding the see-saw where vision gains hurt coding **[web: MarkTechPost]**.

### GLM-5 (arXiv 2602.15763, Feb 2026) **[paper]**
- **Pipeline:** SFT, then Reasoning RL, then Agentic RL, then General RL, then on-policy cross-stage distillation to avoid regression. Context is 202,752 tokens.
- **Reasoning RL:** GRPO with IcePop. The loss is `pop(ρ, 1/β, β) · min(r·A, clip(r, 1-0.2, 1+0.28)·A)` with ρ = π_train/π_infer, β=2. The KL term from the original IcePop is removed. Group size 32, batch size 32, "entirely on-policy". It trains math, science, code and tool-integrated reasoning jointly.
- **DSA-specific:** the sparse-attention indexer is **frozen** in RL. Indexer top-k must be deterministic (`torch.topk`); non-deterministic CUDA top-k caused "drastic performance degradation". slime exposes `--freeze-indexer` and deterministic alignment **[code]**.
- **Async agent RL:**
  - Inference and training are decoupled onto different GPUs; once enough trajectories finish they are sent to the trainer.
  - A "TITO Gateway" records exact token ids.
  - **Direct double-sided importance sampling:** `r = exp(logπθ - logπ_rollout)`, with tokens outside `[1-εl, 1+εh]` masked entirely. No `π_old` is stored.
  - Staleness is handled by logging the rollout weight version and discarding samples where `w' - w0 > τ`. The text I got did not give τ.
  - Samples lost to environment collapse are excluded; a group is padded by repeating valid samples if more than half survive, otherwise dropped.
  - A stateful router maps each rollout ID to a fixed DP rank by consistent hashing for KV reuse.
- **Environments:** more than 10k verifiable SWE environments across 9 languages; thousands of terminal environments (Docker build accuracy over 90%); search built on over 2M retained web pages, with a keep-recent-k=5 context policy (BrowseComp 55.3 to 62.0); slide generation with three reward levels (static HTML attributes, runtime-rendered geometry, visual perceptual features).
- **General RL:** a hybrid of rule rewards, outcome reward models (low variance, hackable), and generative reward models (robust, higher variance), plus human responses as style anchors.
- **Cross-stage distillation:** advantage is `sg[log π_teacher_infer - log π_train]`, group size 1, batch 1024.

### GLM-5.1 (2026-04-08), GLM-5.2 (2026-06-17), GLM-5.3 (Aug 2026)
- **5.1:** 754B MoE, 200K context, MIT, "new asynchronous RL infrastructure". Claims SOTA SWE-Bench Pro (58.4) and 8-hour autonomous runs **[web: MarkTechPost]**.
- **5.2:** 744B-A40B, 1M context, IndexShare (every 4 layers share an indexer; 2.9x fewer per-token FLOPs at 1M) **[web: HF blog, Raschka note]**. The post-training stack:
  - slime supports "white-box rollout, black-box rollout, compact trajectory, and sub-agent workflow."
  - Because compaction splits one rollout into a variable number of trainable traces, they moved from group-wise optimization to **critic-based PPO on individual rollouts** with a token-level critic.
  - More than ten expert models were merged by parallel on-policy distillation.
  - **Anti-hacking guard:** a rule filter flags suspected hacks (high recall), then an LLM judge checks intent (precision). The guard runs online and lets the rollout continue after a hacked action is caught.
  - Reported scores: Terminal-Bench 2.1 at 81.0 vs 63.5 for GLM-5.1; SWE-Bench Pro 62.1 vs 58.4.
- **SAO paper (arXiv 2607.07508, Hou, Li, Tang, Dong)** is the algorithm behind 5.2 **[paper]**.
  - *Problem:* GRPO groups create barriers in async systems because the group waits for its slowest member. Online agent settings often give only one trajectory per prompt.
  - *Objective:* `L = E[ f(r_t, ε_l, ε_h) · Â_t · log π_θ ]` where `f(x)=x` if `1-ε_l < x < 1+ε_h` else 0, and `r_t = exp(logπ_θ - logπ_rollout)` (two policies, no `π_old`). Reasoning ε_low/high is 0.3/5.0; coding is 0.8/3.0.
  - *Critic tricks:* K=2 value updates per policy step; policy lr 1e-6, critic lr 5e-6, 10-step critic warmup; a large value-pretraining corpus; **frozen attention** (only MoE projections train in RL; full-parameter gave 2-3x larger gradient norms).
  - *Skip-observation GAE:* GAE bridges action to action and skips environment-observation tokens so external noise doesn't propagate. λ_policy = 1 − 1/(1.5·l) (length-adaptive), λ_critic = 1.
  - *Evidence:* on a Qwen3-30B-A3B-Thinking-2507 backbone, SAO trains stably for ~1000 steps while GRPO collapses at ~160. AIME2025 goes 80.4 to 97.3 (GRPO 84.2), IMOAnswerBench 53.3 to 74.0, SWE-Bench Verified 23.0 to 29.8 (GRPO 27.0). Ablations: removing frozen attention −6.7 AIME; removing faster critic −2.3 AIME. These come from the authors' own paper.
- **5.3:** same base as 5.2; gains come from scaled post-training on more realistic, larger tasks, "SAO with compaction", and pipelines that synthesize environments and, for some tasks, the reward signal **[web: Z.ai docs]**. A Techmeme headline says weights follow in about two weeks. I did not verify whether they have been released **[web, unverified]**.

---

## 4. Related THUDM / Zhipu projects

| Project | What | RL detail | Source |
|---|---|---|---|
| **AgentRL** (`THUDM/AgentRL`, arXiv 2510.04206, MIT) | Scalable multi-turn multi-task agent RL | Fully async rollout/train on coroutine scheduling. Function-call env interface; containerized envs; a Go controller coordinating up to 10,000 sessions. **Cross-policy sampling** (draw actions from a pool of models to widen state coverage). **Per-task advantage normalization**. Built from veRL plus AgentBench code. Qwen2.5 3-32B and GLM-4-9B on five AgentBench tasks; min 16 H800 for 14B; Qwen2.5-32B reaches 70.4% average. | paper + repo |
| **ComputerRL** (arXiv 2508.14040) | Desktop agents | API-GUI action space; qemu-in-docker VM fleet over gRPC; step-level GRPO with rule verifiers; **Entropulse** alternates RL with SFT on successful rollouts to repair entropy collapse. GLM-4-9B-0414 and GLM-4.1V-9B-Thinking; OSWorld 48.9%. | paper |
| **ScaleCUA** (arXiv 2607.11185; `THUDM/SCALE-CUA`) | Computer-use agents | Online RL built on AgentRL, vLLM rollout and Megatron training. A VeriGen proposer/judger/checker trio synthesizes 24K verifiable GUI tasks. Frontier sampling keeps tasks near the success-rate frontier. OSWorld 68.7. | paper |
| **WebRL** (arXiv 2411.02337, ICLR'25) | Web agents | Self-evolving curriculum generated from failed tasks, a learned binary ORM, a KL-constrained update, and a replay buffer filtered by perplexity. Llama-3.1-8B 4.8 to 42.4% on WebArena-Lite; GLM-4-9B 6.1 to 43%. | paper |
| **AutoGLM** (arXiv 2411.00820) | GUI foundation agent | Intermediate-interface design (separate planning from grounding) plus self-evolving online curriculum RL. | paper |
| **DeepDive, T1, ReST-MCTS\*, ReST-RL** | Search agents with KG-synthesized hard questions and multi-turn RL; RL and test-time scaling; process-reward MCTS | | repo list only, not read in depth |

On "APEX or other RL infra": I found no THUDM project named APEX. The closest relatives are **APRIL** (arXiv 2509.18521, from RLsys-Foundation, not THUDM). It over-provisions rollouts, stops at the target count, and recycles unfinished responses; reported +22.5% rollout throughput (peak 44%) and +2.1% accuracy. The mechanism is built into slime as `--over-sampling-batch-size` plus `--partial-rollout`.

---

## 5. Forks and derivatives of slime

From the slime README unless noted:
- **Miles** (`radixark/miles`, RadixArk/LMSYS): a fork that "co-evolves" with slime. Adds true on-policy (0 KL via FA3 + DeepGEMM), LoRA and multi-LoRA, TITO, R3, MXFP8/NVFP4, P2P RDMA weight transfer, online-trained speculative decoding (25%+ rollout speedup over frozen MTP), and ROCm. Day-0 support is claimed for DeepSeek-V4, Kimi-K3, GLM-5.2 **[web: GitHub, LMSYS blog]**.
- **vime** (`vllm-project/vime`, vLLM project, June 2026): slime with the rollout backend swapped to vLLM plus vllm-router; `--vllm-*` args **[web]**.
- **Relax** (`redai-infra/Relax`): Ray Serve services with TransferQueue, Actor/Rollout/ActorFwd/Reference/Advantage on separate GPU clusters, and NCCL-broadcast "DCS" weight sync overlapping the next step; omni-modal.
- **Dressage** (Alibaba Accio): black-box agent RL (OpenCode, OpenClaw), sandbox/proxy layers, TITO and segment-aware training.
- **OpenClaw-RL**, **P1** (physics olympiad RL), **RLVE** (400 adaptive verifiable environments), **TritonForge** (Triton kernel generation with compile feedback), **APRIL**, **qqr/ArenaRL** (tournament-ranking rewards plus MCP), **AWS ART** (AgentCore).
- AMD publishes slime on Instinct GPUs; the repo has an `amd_tutorial.md`.
- The repo has run scripts for Kimi-K2, MiniMax-M2, DeepSeek-R1 and MiMo. That shows slime is used to *train other labs' architectures*, but I found no first-party statement that those labs use it in production.
- A Hugging Face survey of 16 async RL libraries (written against an older slime snapshot) characterizes slime as Megatron-centric, SGLang-only, abort-and-retry for partial rollouts, with TIS+OPSM for staleness and no LoRA **[web]**. LoRA is not in the main-line docs I read; Miles adds it.

---

## 6. How a biologist plugs in a custom reward (docking or structure oracle)

### 6.1 Mental model

slime trains an *LLM* policy. The policy emits tokens (a SMILES string, an amino-acid sequence, a tool call to a design pipeline, a Python snippet). Your oracle turns those tokens into a number. You supply two things: a dataset and `--custom-rm-path`. Everything else (SGLang sampling, group normalization, Megatron update, weight sync) is stock.

Dataset: JSONL with a `prompt` (chat messages or text), a `label` (anything your reward needs, e.g. a PDB id or target name), and optionally `metadata` (dict). Flags: `--input-key prompt --label-key label --metadata-key metadata --apply-chat-template`.

### 6.2 Minimal reward module (single-turn: "propose a ligand SMILES")

```python
# bio_reward.py  (on PYTHONPATH of the Ray workers)
import asyncio, re
from slime.utils.types import Sample

_ORACLE_SEM = asyncio.Semaphore(64)          # slime will NOT throttle you; do it here

def _parse_smiles(text: str):
    m = re.findall(r"<smiles>(.*?)</smiles>", text, re.S)
    return m[-1].strip() if m else None      # take the last tag, strict format

async def reward_func(args, sample: Sample, **kwargs) -> float:
    smi = _parse_smiles(sample.response)
    if smi is None:
        return -1.0                           # format failure
    # cheap validity + property gates BEFORE the expensive oracle
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -0.5
    target = sample.label                     # e.g. "6LU7"
    async with _ORACLE_SEM:
        # NEVER call a blocking docking function here; it would freeze the shared
        # asyncio loop that also drives every in-flight SGLang request.
        score = await asyncio.to_thread(run_vina_or_boltz, target, smi)   # or an HTTP call
    return float(max(-12.0, min(0.0, score)) / -12.0)   # normalise kcal/mol into [0,1]
```

Wire it up:

```bash
--prompt-data targets.jsonl --input-key prompt --label-key label --apply-chat-template
--custom-rm-path bio_reward.reward_func
--rollout-batch-size 32 --n-samples-per-prompt 8      # 256 oracle calls per step
--rollout-max-response-len 1024
--advantage-estimator grpo        # group-normalised advantages: scale-free reward is fine
```

### 6.3 Practical design decisions the code forces

1. **Blocking oracle = stalled GPUs.** All rollouts share one asyncio loop thread **[code: `slime/utils/async_utils.py`]**. Use `asyncio.to_thread`, a process pool, or an HTTP service. The built-in `--rm-type remote_rm --rm-url http://oracle:8000` already POSTs `{prompt, response, label}` with 10 retries and exponential backoff and expects the JSON body back. For a GPU docking farm or an AlphaFold/Boltz/ESMFold service, put a queue-backed HTTP server in front and let slime hit it.
2. **Group-level scoring.** With `--group-rm`, the function receives `list[Sample]` and returns `list[float]`. Use this to batch GPU folding calls (one batched ESMFold or Boltz call per group of 8) or to compute *relative* rewards such as diversity or novelty against the group.
3. **Multi-objective rewards.** Return a dict, e.g. `{"dock": ..., "qed": ..., "sa": ...}`. `--reward-key` selects the scalar used for training. For anything richer, `--custom-reward-post-process-path` takes the whole batch and can combine terms, or apply Pareto-style scalarization, before advantages. `--custom-rollout-log-function-path` can log each component. Keep a hard validity gate in the reward.
4. **Reward-hacking guards.** Docking scores are famous for being gamed (huge lipophilic molecules). Add penalty terms and a post-hoc filter. GLM-5.2's two-stage guard (cheap rule filter, then an LLM judge) is the pattern **[web]**. In slime, a `--rollout-sample-filter-path` hook can set `remove_sample` so flagged samples drop out of the loss.
5. **Slow oracles (minutes per call, wet-lab turnaround).** Use `fully_async_rollout`, which decouples `max_concurrent_tasks` from the batch and keeps the pool warm. Combine with `--use-tis` and `--use-rollout-logprobs`, because samples will be a few policy versions stale. Treat the staleness metrics (`slime/utils/staleness.py`) as a first-class dashboard.
6. **Dynamic sampling.** `--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` drops groups where every sample scored the same. In sparse-hit biology (most generated binders score zero) it saves oracle calls *only if you filter before scoring*, which the stock path doesn't do (rewards are computed first). A custom generate function can score cheaply first and skip the oracle on obvious failures.
7. **Not a protein LM.** slime/SGLang serve decoder LLMs with HF-style weights supported by a Megatron model provider. To RL a protein LM like ESM3, ProGen or a SMILES transformer, you need an SGLang-supported architecture and a Megatron spec plus HF converter (`slime/backends/megatron_utils/megatron_to_hf/`, `hf_to_megatron/`). Otherwise use a general LLM with a tool-calling design loop (below). `docs/en/advanced/arch-support-beyond-megatron.md` and `slime_plugins/models/hf_attention.py` suggest a path for unusual architectures; I did not test it **[inferred]**.

### 6.4 Agentic variant (tool-calling design loop)

Use `--custom-generate-function-path` so the model calls tools (RDKit, a docking service, a Boltz/AF3 wrapper, a literature search) across several turns, then gets a final reward. Follow the `search-r1` pattern:
- Loop: call SGLang with `stop` tags, parse the action, run the tool, append the observation.
- Set `loss_mask = 1` on model tokens and `0` on observation tokens.
- Collect `rollout_log_probs` aligned with those tokens.
- Do not re-tokenize text; the example's BUGFIX comment explains why.

For heavier harnesses (a Claude-Code-style computational chemist in a sandbox with Vina, GROMACS, and Python), copy `examples/coding_agent_rl`: sandbox per sample, an Anthropic/OpenAI-compatible adapter capturing tokens, grading in a *clean* second sandbox.

---

## 7. What the code shows that the papers don't

1. **"Async" is a rollout-function swap, not a different trainer.** `train.py` is a plain synchronous loop. Asynchrony and PipelineRL come from `generate_rollout_fully_async` plus `--flush-cache-interval`. The latter auto-switches the rollout function and refuses colocated, offload and `--release-train` configurations.
2. **Reward computation is outside the generation concurrency limit** and shares one event-loop thread with all generation. This is a practical trap for slow or blocking oracles.
3. **Group normalization lives in `batch_builder._post_process_rewards`, not in the loss.** The loss only sees a per-sample scalar advantage. Std normalization is on by default. `--custom-reward-post-process-path` replaces it.
4. **Fan-out `rollout_id` accounting.** Compaction and sub-agent segments share a `rollout_id`, and `rollout_mask_sums` gives one denominator per rollout. This is why slime can train GLM-5.2's compacted trajectories without over-weighting long ones. The docs also recommend dividing the reward by K across K segments.
5. **Open-source GLM-4.7 recipe uses GSPO with ~1e-4 clip bounds**, while GLM-5/5.2 use GRPO-style token clipping with IcePop/TIS. The papers talk about GRPO/IcePop and SAO; GSPO appears in the public scripts.
6. **No SAO in the public tree.** GLM-5.2's critic PPO, skip-observation GAE and `[0.3, 5.0]` masking are described in the paper, but a grep for SAO and skip-observation found nothing. The ingredients exist: `--advantage-estimator ppo`, `--num-critic-only-steps`, `--custom-advantage-function-path`, `--use-rollout-logprobs`, and the `custom_loss` hook. The skip-observation GAE and frozen-attention critic would be user code **[inferred]**.
7. **GLM-5 logprob alignment is engineered to bit-exactness.** Batch-invariant DeepGEMM, a custom DeepEP dispatch with ordered FP32 reduction, an fp32 router, a frozen indexer, and CI gates at 1e-6 and exactly-zero layer outputs. The papers only say the indexer is frozen and top-k is deterministic.
8. **Delta weight sync** (XOR+zstd, checksummed, seeded from `--hf-checkpoint`) is not mentioned in any GLM paper. It targets cross-datacenter or heterogeneous-vendor rollout fleets.
9. **Score centering (Sept 2026) is already wired into the loss**, including top-p replay of the sampler's nucleus. This appears a month after the paper.
10. **Search-R1 junk-token bug** (documented in code comments) is a concrete example of a silent RL failure: tokens after the stop tag trained at mask 1.
11. **The README claims about GLM-5.3** (that slime is behind it) are consistent with the 5.3 docs page, which says it inherits the 5.2 RL strategy. I did not find an independent GLM-5.3 technical report.

---

## 8. Transfer to biology

Items marked **(novel)** are my own proposals; I could find no published slime-for-biology use.

1. **Docking/structure oracle as `--custom-rm-path` (see section 6).** This is straightforward and is not novel.
2. **Cheap-then-expensive reward cascade inside `custom_generate` (novel framing).**
   - Gate order: format, then RDKit validity and PAINS/property filters, then a fast ML surrogate (e.g. a docking-score regressor), then real docking or Boltz/AF3 only for the top fraction.
   - Return the staged scores as a dict, selected by `--reward-key`. This borrows GLM-5's tiered slide reward (static, then rendered, then perceptual) **[paper]**, and uses `check_reward_nonzero_std`-style filtering to avoid spending oracle calls on all-zero groups.
3. **SAO-style single-rollout PPO for slow assays (novel for biology).**
   - GRPO needs 8-32 oracle evaluations per prompt in lockstep. SAO needs one rollout per prompt, a critic, and rollout-logprob importance masking.
   - That fits expensive oracles (MD free-energy, wet-lab batches) where you can afford one measurement per proposal. A new measurement arrives whenever the lab returns it, and the critic provides the baseline instead of sibling samples.
   - Build it from slime's `ppo` estimator plus a custom advantage hook; the skip-observation GAE is natural for tool-using design loops where tool outputs (docking logs) are observations.
4. **Fully-async rollout as a lab-in-the-loop scheduler (novel).** The fully-async worker already tolerates long-tail samples and tracks weight-version staleness per sample (`weight_versions`, `compute_staleness_metrics`). A wet-lab "reward service" that returns results hours later maps to a straw-queue pending task. The straw queue persists work and resumes after restarts. Samples labeled with the policy version that proposed them can be importance-corrected with TIS/OPSM when assay data finally comes back. Caveat: very large staleness violates the assumptions behind the clipping bounds, so expect to need explicit age limits like GLM-5's discard threshold τ.
5. **Anti-hack guard for scoring functions (adaptable pattern).** Copy GLM-5.2's rule-filter-then-LLM-judge pipeline: rules to catch docking exploits (excess MW, clashes, degenerate geometries); an LLM or independent oracle (e.g. a second docking engine) to confirm. Run it online so rollouts continue after a flagged action.
6. **Frozen-attention PPO for small-data domain adaptation (speculative, novel).** SAO's finding that training only MoE projections stabilizes a critic-based run on MoE LLMs could matter when RL data is a few thousand assay points.
7. **On-policy distillation to merge domain experts.** Train separate protein-design, small-molecule and genomics experts, then merge via `--use-opd` (student samples, teacher token logprobs), as GLM-5/5.2 did for ten-plus experts. This gives one generalist model without catastrophic forgetting.
8. **Curriculum by pass-rate, from GLM-4.5/4.1V.** Pick targets/binding-site tasks where the current policy has pass@8 = 0 but pass@many > 0 (the GLM-4.5 stage-two recipe), and cap the sampling pool with ScaleCUA-style frontier sampling (an EMA of per-task success). Directly applicable to protein-target curricula.
9. **Verifier synthesis for biology (novel adaptation of ScaleCUA's VeriGen / GLM-5 environment scaling).** A proposer agent writes a task plus a deterministic verifier (e.g. "design a peptide with predicted Kd < X to target Y per oracle Z"); a judge reviews the verifier for leaks; a checker dry-runs it. GLM-5.3 explicitly says it synthesizes the RL reward for a subset of tasks.
10. **R3/TIS matter more if you use MoE LLM backbones** for biological tool-use agents; for dense 7-30B models the plain `--use-tis` setting from the GLM-5.2 script is a reasonable default.
11. **Entropy-collapse remedy (ComputerRL's Entropulse)** is relevant for molecule generation, where diversity is part of the objective: alternate RL with SFT on top-scoring unique molecules to reset entropy.

---

## 9. Sources (V = verified by reading, I = inferred or secondhand, N = negative result)

**Code and docs (V, read in the shallow clone at `8c17b676`):**
- https://github.com/THUDM/slime. Key files: `train.py`, `README.md`, `slime/rollout/sglang_rollout.py`, `slime/rollout/rm_hub/__init__.py`, `slime/rollout/fully_async_rollout.py`, `slime/rollout/filter_hub/dynamic_sampling_filters.py`, `slime/ray/rollout.py`, `slime/data/batch_builder.py`, `slime/data/data_source.py`, `slime/utils/types.py`, `slime/utils/ppo_utils.py`, `slime/utils/score_centering.py`, `slime/utils/staleness.py`, `slime/utils/async_utils.py`, `slime/utils/arguments.py`, `slime/backends/megatron_utils/loss.py`, `slime/backends/megatron_utils/actor.py`, `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`, `examples/train_infer_mismatch_helper/{mis.py,mis.yaml,README.md}`, `examples/search-r1/generate_with_search.py`, `examples/coding_agent_rl/README.md`, `examples/fully_async/README.md`, `scripts/run-glm5.2-744B-A40B.sh`, `scripts/run-glm4.7-355B-A32B.sh`, `scripts/run-qwen3-4B.sh`, `docs/en/get_started/{customization,agent,quick_start,usage}.md`, `docs/en/advanced/{delta-weight-sync,low-precision,reproducibility,pd-disaggregation,sglang-config,external-rollout-engines,fault-tolerance,on-policy-distillation,speculative-decoding}.md`, `docs/en/examples/glm5.2-744B-A40B.md`, `slime_plugins/rollout_buffer/README.md`.
- https://github.com/THUDM/slime/releases: release notes (V; years printed as 2024, treated as a typo).
- https://github.com/THUDM: org page, star counts and repo list (V).
- https://github.com/THUDM/AgentRL: repo structure, veRL/AgentBench lineage, MIT (V).
- https://github.com/radixark/miles: feature list (V, README summary).
- https://github.com/RLsys-Foundation/APRIL: referenced from the slime README (I).

**Papers (V, via summarizing fetch):**
- https://arxiv.org/abs/2508.06471 and https://arxiv.org/pdf/2508.06471: GLM-4.5 (RL algorithm, infra, curricula).
- https://arxiv.org/html/2602.15763v2 and https://arxiv.org/pdf/2602.15763: GLM-5 (IcePop, async RL, TITO, DIS, environments). The summarizer did not return τ or GPU counts.
- https://arxiv.org/abs/2607.07508 and https://arxiv.org/pdf/2607.07508: SAO.
- https://arxiv.org/abs/2507.01006: GLM-4.1V-Thinking (RLCS; abstract only).
- https://arxiv.org/pdf/2510.04206: AgentRL.
- https://arxiv.org/pdf/2508.14040: ComputerRL.
- https://arxiv.org/html/2607.11185v1: ScaleCUA.
- https://arxiv.org/pdf/2411.02337: WebRL.
- https://arxiv.org/abs/2411.00820: AutoGLM.
- https://arxiv.org/abs/2509.18521: APRIL.
- https://arxiv.org/abs/2609.20807: Score Centering (abstract only).

**Blogs and news (mostly V; fetched via summarizer):**
- https://huggingface.co/blog/zai-org/glm-52-blog: GLM-5.2 (release date, IndexShare, slime modes, critic PPO, anti-hack).
- https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html: IndexShare (no RL content).
- https://docs.z.ai/guides/llm/glm-5.3: GLM-5.3 (same base as 5.2; SAO with compaction; synthesized rewards).
- https://www.marktechpost.com/2026/04/08/z-ai-introduces-glm-5-1-an-open-weight-754b-agentic-model-that-achieves-sota-on-swe-bench-pro-and-sustains-8-hour-autonomous-execution/: GLM-5.1.
- https://www.marktechpost.com/2026/04/01/z-ai-launches-glm-5v-turbo-a-native-multimodal-vision-coding-model-optimized-for-openclaw-and-high-capacity-agentic-engineering-workflows-everywhere/: GLM-5V-Turbo joint RL.
- https://docs.z.ai/guides/llm/glm-4.6 and https://docs.z.ai/guides/vlm/glm-4.6v: no RL content (N).
- https://www.tipranks.com/news/private-companies/zhipu-ai-highlights-new-rl-infrastructure-behind-glm-5-2-post-training: SAO news summary.
- https://lmsys.org/blog/2025-11-25-fp8-rl/: FP8 RL.
- https://www.lmsys.org/blog/2025-11-19-miles/: Miles.
- https://vllm.ai/blog/2026-06-09-announcing-vime: vime.
- https://huggingface.co/blog/async-rl-training-landscape: 16-library comparison (older slime snapshot; I).
- Not fetchable: z.ai/blog/glm-5.1, 5.2, 5.3 (pages render client-side and returned empty); the Zhihu "GLM 5.2 PPO" article (403); the X post by Zixuan Li on SAO (robots).
- Search results only, not opened: Techmeme headline on GLM-5.3 weights in "two weeks" (I, unverified); X post on GLM-4.7 "preserved thinking" (I).
