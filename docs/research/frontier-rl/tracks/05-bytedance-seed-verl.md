# Track 05: ByteDance Seed and verl (HybridFlow)

Date of survey: 2026-10-05. Code read: `verl-project/verl` main at commit `8718ca3` (2026-10-01, version `0.10.0.dev`) and `verl-project/verl-recipe` main. Papers were read through WebFetch of arXiv PDFs, so numbers quoted from papers are as extracted by that tool. Items I did not verify are marked **[unverified]** or **[inferred]**.

---

## 1. TL;DR

- **verl is the de facto open RL substrate** (Seed-initiated, now community-run under `github.com/verl-project`). Its core idea is a hybrid programming model. One single-controller driver process (Ray) expresses the RL control flow, while each model role runs as a multi-controller SPMD worker group (FSDP/FSDP2, Megatron-Core, VeOmni, TorchTitan, Automodel). Rollout is served by vLLM, SGLang or TRT-LLM replicas. HybridFlow reports a 1.53x to 20.57x throughput gain over earlier RLHF systems.
- **The architecture has changed a lot since the paper.**
  - The default trainer is now "V1" (`trainer.use_v1: true`). The driver passes only `KVBatchMeta` handles, and tensors live in a `TransferQueue` store.
  - Modes are `sync`, `colocate_async` and `separate_async` (partial rollout, staleness control, replay buffer with built-in DAPO group filtering and prompt refill).
  - Rewards are computed by a distributed, async `RewardLoop`, streamed per sample inside the agent loop.
  - Rollout is a server-based `AgentLoop`. A token-in/token-out "Continuous Token" builder keeps multi-turn tool trajectories on-policy.
  - Trainer-to-rollout weight sync goes through a `checkpoint_engine` (NCCL, NIXL, Mooncake, Kimi-style, and a lossless sparse "delta" sync).
- **Seed's published RL line is a PPO/critic lineage that grew out of long-CoT math, then went to agents.**
  - VC-PPO (value pretraining + decoupled GAE) became VAPO (adds length-adaptive GAE, token-level loss, clip-higher, positive-example NLL loss, group sampling).
  - DAPO is the critic-free sibling (clip-higher, dynamic sampling, token-level loss, soft overlong punishment, no KL).
  - Seed-Thinking-v1.5 (200B MoE, 20B active) and UI-TARS-2 (230B MoE, 23B active) use the VAPO tricks in production. Seed-Prover 1.5 uses VAPO with Lean-compiler rewards.
  - UI-TARS-2 reports PPO beating GRPO substantially on long-horizon agent tasks.
- **Reward design is the transferable lesson.** Seed uses binary verifiers (+1/-1) wherever possible. It moves to a *reasoning* LLM verifier (Seed-Thinking-Verifier, 99.3% vs 82.7% for the non-reasoning verifier on a human-labeled set) when matching is fuzzy. It uses pairwise generative RMs only for non-verifiable data. For formal proofs the verifier is the Lean compiler itself.
- **Recent verl work is dominated by training/inference mismatch.** It ships rollout-correction code (token/sequence IS, k1/k2/k3 rejection sampling, bypass mode), FP16 and FP8 recipes, router replay for MoE, a batch-invariant `vexact` rollout, and a score-centering loss tied to arXiv 2609.20807 (Sept 2026).
- **Seed biology and science is not RL.** The visible bio work is Protenix-v1 (AF3-class structure prediction), PXDesign and SeedProteo (diffusion binder design), DPLM (diffusion protein LMs), and an "Anew Labs" drug-discovery unit. I found no RL-for-biology paper from Seed. This leaves an open gap.
- **Open-source reach:** verl's README acknowledges adoption by about 40 organizations, including Alibaba Qwen, Moonshot (Kimi), Meituan, Microsoft Research, NVIDIA, Amazon and Mind Lab. Academic projects include Tsinghua SIA-Lab (DAPO, MemAgent), PRIME-RL, UIUC (Search-R1), HKUST (SimpleRL), TIGER-Lab (verl-tool) and NUS.

---

## 2. verl: architecture (code read)

### 2.1 Single-controller plus multi-controller

- `verl/single_controller/base/decorator.py` defines `@register(dispatch_mode=..., execute_mode=...)`. Dispatch modes are `ONE_TO_ALL`, `ALL_TO_ALL`, `DP_COMPUTE_PROTO` (chunk a `DataProto` over data-parallel ranks and concat on return), `DP_COMPUTE_PROTO_WITH_FUNC`, `DP_COMPUTE_METRIC`, and `DIRECT_ROLLOUT_METHOD`.
- `verl/single_controller/ray/base.py` provides `RayResourcePool`, `ResourcePoolManager`, `RayWorkerGroup`, and `create_colocated_worker_cls`, which fuses several roles into one process.
- A worker-group method call on the driver fans out to every rank, and the output is collected back. `docs/single_controller.rst` states the origin: the team wanted a toy single-process RLHF script to become distributed with minimal changes while keeping debuggability. The training loop is broken into named stages ("generate_sequences", "compute_advantages").
- `docs/hybrid_flow.rst` frames RL as a two-level dataflow. Control flow is the algorithm. Computation flow is the NN forward/backward. verl deliberately keeps control flow as a single process and computation as multi-process.
- **Cost of this choice:** data goes back and forth to the controller. The V1 trainer attacks exactly this cost (see 2.3).
- **Engines:** `verl/workers/engine/{fsdp,megatron,veomni,torchtitan,automodel,mindspeed}/transformer_impl.py` implement `BaseEngine` (`train_batch`, `infer_batch`, `forward_backward_batch(data, loss_fn)`, `get_per_tensor_param()` as a generator for weight sync). `verl/workers/engine_workers.py` holds `TrainingWorker` and `ActorRolloutRefWorker` (ref, actor and rollout are co-located; `compute_ref_log_prob`, `compute_log_prob`, `update_actor`).
- Megatron scale (docs/perf/dpsk.md, last updated 08/2025): DeepSeek-V3 671B runs from 96 H20 GPUs (TP8/PP12/EP8 with optimizer offload; about 1.6 TB CPU RAM per node). Qwen3-235B-A22B runs from 32 H20. MTP and quantization are disabled for RL.

### 2.2 The training step (legacy `RayPPOTrainer.fit`, still readable in `verl/trainer/ppo/ray_trainer.py`)

1. Repeat each prompt `rollout.n` times (interleaved) and tag it with a `uid`.
2. Generate through the async agent-loop manager. Rollout replicas are put to sleep afterward to free GPU memory.
3. Reward is read from `rm_scores`, which the reward loop has already produced.
4. Optional batch balancing across DP ranks by token count. This reorders data, and the code comment warns that it changes minibatching.
5. Recompute `old_log_probs` on the training engine ("decoupled mode"), **or** set `old_log_probs = rollout_log_probs` ("bypass mode", `algorithm.rollout_correction.bypass_mode`).
6. Ref log-prob and critic values if enabled.
7. Advantage on the driver. KL goes either in the reward (`use_kl_in_reward`) or as a loss (`use_kl_loss`, default `low_var_kl`, coef 0.001).
8. Update critic, then actor, then `checkpoint_manager.update_weights()`.

### 2.3 The V1 trainer (default now)

Files: `verl/trainer/ppo/v1/{trainer_base,trainer_sync,trainer_colocate_async,trainer_separate_async,replay_buffer,agent_loop_tq}.py`, and `verl/trainer/main_ppo.py` (`TaskRunnerV1` calls `tq.init(...)`).

- Data lives in **TransferQueue**: ZMQ-backed `SimpleStorage` with 8 storage units by default, or a Mooncake RDMA store (experimental). The controller handles `KVBatchMeta` keys and calls `tq.kv_batch_get/put`. `_compute_advantage` fetches only the fields it needs (`uid, response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values`) and writes back nested `advantages` and `returns`.
- `_step_once` runs: sample from replay buffer, reward (only if no reward loop is attached), balance batch, old_log_prob, ref, values, advantage, update critic, update actor.
- `separate_async` mode streams one PPO minibatch at a time.
  - It enforces `train_batch_size == parameter_sync_step * ppo_mini_batch_size`.
  - It forbids the `naive` checkpoint backend.
  - The standalone rollout pool can be "lent" idle trainer GPUs via `hybrid_rollout.enable_switch`, with adaptive thresholds.
- **Partial rollout:** `FullyAsyncLLMServerClient`. An aborted generation keeps its tokens and logprobs and resumes under a newer model version. The KV cache is rebuilt by prefill. A single trajectory can therefore span policy versions. Metrics `training/off_policy/trajectory_spans`, `trajectory_staleness` and `trajectory_staleness_worst` track this.
- **Staleness control:** `trainer.v1.sampler.max_off_policy_threshold` (default 8 model versions) with `max_off_policy_strategy` = `drop` or `wait`.
- **Decoupled-PPO bookkeeping:** `pi_old` weights are copied to CPU and restored for later minibatches, giving `N` saves and `2(N-1)` restores for `N` minibatches. This is skipped in bypass mode.
- **Replay buffer evicts and refills.** A prompt group is evicted if it is stale, DAPO-filtered or failed. The buffer then calls `refill_fn(k)` to fetch exactly `k` new prompts. **DAPO dynamic sampling is thus a first-class replay-buffer feature**, with the group classification cached per partition (`replay_buffer.py`, `_dapo_filtered_keys`). Validation is exempt.
- **Older experimental trainers** also exist: `verl/experimental/fully_async_policy` (contributed by Meituan; claims 2.35x to 2.67x speedup on Qwen2.5-7B with 128 GPUs), `one_step_off_policy`, and the recipe `partial_rollout` (APRIL-style cross-step interruption, synchronous otherwise).

### 2.4 Weight sync: `verl/checkpoint_engine/`

- Backends: `naive` (torch.distributed, colocated), `nccl`, `hccl`, `nixl`, `mooncake`, `kimi_ckpt_engine`.
- The README benchmark gives about 7 s for NCCL and NIXL on Qwen3-30B-A3B across 4x8 H100 with 400 Gb InfiniBand, and 5.9 s for Mooncake on 2x8 H100.
- `docs/advance/delta_weight_sync.md` argues that over 99% of BF16 weight bytes are unchanged step to step. It broadcasts only changed elements (the `delta_sharded` backend). Sync is lossless, bit-exact, with a per-flush checksum, and was measured 1.3x to 21x faster than a full broadcast at sizes from 0.5B to 235B. Both claims come from verl's own docs and I did not reproduce them.

### 2.5 Rollout and agent loop

- `verl/workers/rollout/{vllm_rollout,sglang_rollout,trtllm_rollout}` run as servers. `rollout.mode: async` is the default.
- `verl/experimental/agent_loop/agent_loop.py`: `AgentLoopManager` creates `AgentLoopWorker`s. Each sample runs as an asyncio task. Reward is computed right after each sample finishes (`_compute_score`) by a random reward-loop worker, so reward overlaps with rollout.
- `tool_agent_loop.py` is a state machine: `PENDING -> GENERATING -> PROCESSING_TOOLS -> TERMINATED`.
  - Tool-parser stop tokens end generation after each call.
  - Parallel calls are capped by `max_parallel_calls`.
  - Tool responses are truncated (`max_tool_response_length: 256` default, `middle` side).
  - Limits are `max_user_turns` and `max_assistant_turns`.
- **Token-level correctness:** it keeps `response_ids`, `response_mask` (1 = model-generated, 0 = tool/observation/boundary tokens) and `response_logprobs` jointly. `ct_merge_assistant_token` and `ct_merge_context_msg` (from `verl/utils/tokenizer/continuous_token.py`) append tokens instead of re-tokenizing the chat. The `MergeResult` docstring says CT-inserted boundary tokens "must not carry loss or model logprobs".
- Tools: `verl/tools/{base_tool,function_tool,tool_registry}.py`. YAML-configured `native` tools or `@function_tool` Python functions with schemas inferred from docstrings. Per-sample tool subsets are selectable with `extra_info.tool_selection`. Tools can return images (multimodal). Video returns raise `NotImplementedError`. Tool rewards go into `tool_rewards`, and the final reward function sees `num_turns` and `rollout_reward_scores`.

### 2.6 Rewards

- `verl/experimental/reward_loop/`: `RewardLoopManager` spawns `RewardLoopWorker`s (`reward.num_workers`).
- Reward managers (`reward_manager/`): `naive`, `dapo` (adds soft overlong penalty), `limited` (token-bucket RPM/TPM limiter for API rewards), `remote` (one Ray CPU actor per call, needed because `math-verify` misbehaves in threads, issue #3407), and `gdpo` (multi-component rewards).
- **Custom reward function interface:** any sync or async `fn(data_source, solution_str, ground_truth, extra_info, **kw)`. It returns a float or a dict with `score` plus extras (e.g. `acc`, `pred`), loaded via `reward.custom_reward_function.{path,name,reward_kwargs}` (`verl/trainer/ppo/reward.py`).
- A reward model can be a discriminative RM, a generative RM, or a hybrid of rules and a GenRM. A GenRM can run in a separate resource pool and be reached through a router (`reward_model.py`, `router/`). Streaming reward requires rule-based rewards or an RM with its own resource pool.
- Built-in scorers (`verl/utils/reward_score/`): `gsm8k`, `math_dapo` (+1/-1, strict `\boxed{}` option), `prime_math`, `prime_code`, `sandbox_fusion` (remote code execution with a concurrency semaphore), `search_r1_like_qa_em`, `geo3k`.

### 2.7 Algorithms registered in `verl/trainer/ppo/core_algos.py` (2,559 lines)

- **Advantage estimators:** GAE, GRPO, GRPO_VECTORIZED, GRPO_PASSK, GDPO (multi-reward), RLOO, RLOO_VECTORIZED, REINFORCE++ (and baseline), REMAX, OPO, GPG, OPTIMAL_TOKEN_BASELINE, and a multi-turn variant.
- **Policy losses:** `vanilla` (dual-clip PPO with separate `clip_ratio_low/high`, `clip_ratio_c`), `dppo_tv`, `dppo_kl`, `gspo`, `sapo`, `gpg`, `clip_cov`, `kl_cov`, `geo_mean`, `dro`, `cispo`, `bypass_mode`.
- `agg_loss` supports `token-mean`, `token-sum`, `seq-mean-token-sum`, `seq-mean-token-sum-norm` and `seq-mean-token-mean`. It takes global batch token counts so the loss is invariant to FSDP/Megatron parallelism (DAPO's token-level normalization done properly).
- **Defaults** (`actor/actor.yaml`): `clip_ratio: 0.2`, `clip_ratio_c: 3.0`, `loss_agg_mode: token-mean`, `use_kl_loss: false`, `ppo_epochs: 1`.

### 2.8 Recipes (`verl-recipe`, a separate repo since 2026-01)

- dapo, retool, prime, sppo, spin, spo, entropy (Clip-Cov and KL-Cov), fapo, genrm_remote, deepeyes, collabllm, langgraph_agent, swe_agent, tmax (terminal agents on Modal), echo (Microsoft, aux CE loss on observation tokens), rep_exp, flowrl, gvpo, gkd, partial_rollout, specRL (speculative decoding for rollout), low_precision (FP8), qat, nemo_gym, atropos, verl_tinker.
- Each recipe has a `REQUIRED_VERL.txt` that pins a commit. Several are stale against `main` (see section 4).
- Sibling repos in the same org, from the README news list (not read): `verl-vla` (v0.1.0, 2026-08), `verl-omni` (diffusion and omni-modal RL, v0.2.0), `uni-agent`, `verl-SpeCo`, `rl-insight`, `vexact`.

---

## 3. Seed's RL algorithm line

### 3.1 VC-PPO (arXiv 2503.01491)

- **Failure diagnosed:** vanilla PPO collapses on long CoT with Qwen2.5-32B (AIME 5.6, output length drops). There are two causes.
  - Value-model initialization from a reward model creates a positive bias that accumulates along the trajectory and favors short outputs.
  - With GAE lambda = 0.95, the reward signal decays as lambda^(T-t) over long sequences.
- **Fixes:** (a) *value pretraining* on samples from a frozen SFT policy with lambda = 1.0 Monte-Carlo returns, until value loss and explained variance converge (about 100 steps was best in the ablation); (b) *decoupled GAE* with lambda_critic = 1.0 and lambda_actor about 0.95.
- **Results:** 41.9 AIME at 8k context and 48.8 at 16k, versus GRPO 38.9. Removing decoupled GAE gives 29.4.

### 3.2 DAPO (arXiv 2503.14476; code `verl-recipe/dapo`)

- **Four changes to GRPO:**
  - *Clip-Higher:* eps_low = 0.2, eps_high = 0.28. The goal is to stop entropy collapse by letting low-probability tokens rise.
  - *Dynamic sampling:* drop groups where all samples are right or all wrong (accuracy 0 or 1) and refill until the batch is full. The paper's constraint is 0 < |correct| < G.
  - *Token-level loss:* normalize by the total token count in the batch instead of per sequence.
  - *Soft overlong punishment:* with L_max = 16,384 and L_cache = 4,096, there is no penalty up to 12,288 tokens, a linear penalty to 16,384, and -1 beyond.
- **Also:** no KL term, a rule-based reward of +1/-1, lr 1e-6 with warmup, 512 prompts x 16 samples per step.
- **Ablation** (AIME24 on Qwen2.5-32B, as extracted from the paper): GRPO 30, +overlong filtering 36, +clip-higher 38, +soft overlong 41, +token-level 42, +dynamic sampling 50.
- **verl implementation, checked in the config and recipe:**
  - `run_dapo_qwen2.5_32b.sh` uses `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `clip_ratio_c=10.0`, `loss_agg_mode=token-mean`, `max_response_length=20k`, an overlong buffer of 4k with penalty factor 1.0, `n=16`, `train_prompt_bsz=512`, and `gen_prompt_bsz=3x` (over-generate so filtering still fills the batch). `max_num_gen_batches=10`, `train_prompt_mini_bsz=32`, `grad_clip=1.0`, `weight_decay=0.1`, Ulysses SP=8, vLLM TP=4, 16 nodes.
  - Filtering keeps a prompt if `std(metric) > 0` or it has one sample (`dapo_ray_trainer.py` around lines 251-266).
  - The dapo reward manager implements `min(-exceed_len/buffer_len * factor, 0)` added to the score (`experimental/reward_loop/reward_manager/dapo.py`).
- **Reproduction table** (`verl-recipe/dapo/README.md`): 52% AIME24 with the full recipe on 16x8 H800, 50% without dynamic sampling, 44% for an early version without token-level loss or dynamic sampling (on H20). In verl's own reproduction, dynamic sampling was worth about 2 points, not the 8 reported in the paper's ablation. The ablation orders differ, so this is not a contradiction, but it shows the value is setup-dependent.

### 3.3 VAPO (arXiv 2504.05118)

- **Seven tricks on a value-based PPO,** reported as AIME24 scores on Qwen2.5-32B: vanilla PPO 5; +value-pretraining 11; +decoupled GAE 33; +length-adaptive GAE 45; +clip-higher 46; +token-level loss 53; +positive-example LM loss 54; +group sampling 55; full VAPO 60.4.
- **Length-adaptive GAE:** lambda_policy = 1 - 1/(alpha * l), alpha = 0.05, so the TD-error weighting is uniform across lengths.
- **Positive-example loss:** L = L_PPO + 0.1 * L_NLL on correct samples (self-imitation).
- **Other settings (as extracted):** actor lr 1e-6, critic lr 2e-6, value warmup of 50 steps, gamma = 1.0, about 5,000 steps. The paper claims no training crashes across repeated runs.
- **Open-source status:** verl has GAE, `clip_ratio_low/high` and critic workers. I did **not** find length-adaptive GAE or the positive-example NLL loss as built-in options in `core_algos.py` (`compute_gae_advantage_return` uses plain `lam`; `register_adv_est` has no such variant). **[Inferred from the grep of registered estimators; a custom estimator via `register_adv_est` would be needed.]**

### 3.4 Seed-Thinking-v1.5 (arXiv 2504.13914)

- **Model:** MoE, 20B active and 200B total. Reported results: AIME24 86.7, AIME25 74.0, Codeforces pass@8 55.0, GPQA Diamond 77.3, ARC-AGI 39.9. The paper introduces BeyondAIME (100 problems).
- **RL data:**
  - Verifiable: several hundred thousand STEM problems; competitive-programming problems with unit tests and checkers; 22 logic puzzle types. Trivial problems were removed using worst-of-N = 1 filtering.
  - Non-verifiable: about 100k creative, translation, QA and role-play prompts, kept only if score variance is high and the prompt is not too easy.
  - The SFT set is 400k (300k verifiable, 100k non-verifiable).
- **Rewards:**
  - Seed-Verifier is an LLM that judges answer equivalence using human-written principles (82.7% accuracy on the human-labeled set).
  - Seed-Thinking-Verifier reasons before it judges (99.3%). It is meant to resist reward hacking and handle corner cases.
  - Non-verifiable data uses a **pairwise generative reward model**. This stabilizes mixed-domain training.
- **Algorithm:** VAPO and DAPO tricks (value pretraining, decoupled GAE, length-adaptive GAE, dynamic sampling, clip-higher, token-level loss, positive-example LM loss), plus **online data distribution adaptation**.
- **Infra:**
  - A **Streaming Rollout System (SRS)** runs off-policy continuation from versioned snapshots. The ratio of on-policy completion is a parameter alpha. It claims 3x faster iteration than synchronous frameworks.
  - FP8 policy via post-training quantization.
  - TP/CP for attention and EP+FSDP for MoE.
  - The KARP algorithm for sequence-length balancing across DP ranks.
  - AutoTuner, and ByteCheckpoint for resharded resume.
- **Repo:** `ByteDance-Seed/Seed-Thinking-v1.5` is a PDF plus eval, with no training code.

### 3.5 ReTool (arXiv 2504.11536; verl recipe)

- **Paper:** cold-start SFT on synthesized code-integrated reasoning traces, then PPO with a code interpreter in the loop. Generation pauses at `</code>`, the sandbox runs asynchronously, and the output returns in `<interpreter>` tags (masked out of the loss). Reward is outcome-only (+1/-1). Qwen2.5-32B-Instruct reaches 67% AIME24 in 400 steps versus 40% for text-only RL. The model spontaneously learns code self-correction.
- **The verl recipe is a re-implementation, not the same algorithm.** Details are in section 4.

### 3.6 UI-TARS-2 (arXiv 2509.02544)

- **Model:** 532M vision encoder plus 23B active (230B total) MoE.
- **Algorithm:** PPO with the VC-PPO/VAPO machinery (value pretraining, decoupled GAE, length-adaptive GAE with alpha = 0.05, clip-higher). It reports that **PPO substantially beat GRPO with lower volatility** on these multi-turn tasks.
- **Rollout system:**
  - Stateful environments that keep execution state across tool calls.
  - A streaming training pool: training begins once a minimum batch is available, and unfinished traces continue next iteration.
  - Server-based asynchronous inference.
  - Environments are thousands of concurrent cloud VMs (Windows/Ubuntu/Android), browser sandboxes with a fake clock and crash recovery for HTML5 games, and a hybrid GUI-plus-SDK (terminal, files).
- **Rewards:** deterministic checks for games and QA, an LLM judge for browsing answers, and a generative outcome reward model for open-ended GUI tasks. The paper admits a "relatively high false positive rate" for that ORM.
- **Data flywheel:** the current RL model generates trajectories, a validation function splits them, high-quality ones go to SFT, the rest to continual pretraining, with human corrections during rollout to stay on-policy.
- **Multi-domain:** specialist agents are merged by **parameter interpolation** instead of joint RL.
- **Dynamics:** entropy **rises** during GUI/game RL, unlike math RL. Test-time interaction scaling stays steep far beyond the training step budget.
- **Results:** OSWorld 47.5, AndroidWorld 73.3, Online-Mind2Web 88.2, 15-game mean normalized score 59.8, SWE-Bench Verified 68.7, Terminal-Bench 45.3.

### 3.7 Seed-Prover and Seed-Prover 1.5 (Lean)

- **Seed-Prover 1.0 (arXiv 2507.23726):** lemma-style whole-proof generation, RL with Lean feedback, self-summarization, three test-time strategies. Proves 78.1% of formalized past IMO problems, saturates miniF2F, scores over 50% on PutnamBench, and fully proved 5 of 6 IMO 2025 problems. Seed-Geometry is a separate engine.
- **Seed-Prover 1.5 (arXiv 2512.17260):**
  - An *agentic* prover with tools: Lean via the LooKeng REPL, embedding-based Mathlib search (Lean v4.22.0), and Python. Inference limits are 64K tokens and 28 tool calls.
  - SFT cold start on synthetic tool-use data, then **VAPO with a +1 verified / -1 otherwise reward**. Training data is filtered by pass@4x8 so only problems not solved more than 3 times are kept.
  - A **sketch model** is trained with VAPO and a hybrid reward: Lean compile OK, at least 3 lemmas, structure check, and an LLM-judge rubric with natural-language score at least 0.7.
  - Test time is a hierarchy: natural-language prover, then sketch, then agentic Lean prover, with recursive lemma decomposition to depth 8.
  - Results: PutnamBench 88% (580/660), Fate-H 80%, Fate-X 33%, Putnam 2025 11/12 problems in 9 hours, IMO 2025 5/6. Compute is about 10 H20-days per problem (1.0 used 18; AlphaProof about 500 TPU-days per the paper). RL success rate rose from about 50% to about 90% during training.
- **Repo:** `github.com/ByteDance-Seed/Seed-Prover` has formal proofs (IMO 2025 P1-P5), not training code. Its README has no 2026 update. I found no Seed-Prover 2.

### 3.8 Seed1.5 to Seed2.0 and open models (2025 to 2026)

- **Seed1.6** blog: **AdaCoT**, with FullCoT, NoCoT and adaptive modes. An RL reward term penalizes overthinking. The CoT triggering rate is 37% on MMLU, 70% on MMLU-Pro and 90-100% on AIME/BeyondAIME. Multimodal training uses multi-stage rejection fine-tuning plus RL.
- **Seed-OSS-36B** (2025-08-20, Apache-2.0): dense, 64 layers, GQA, 512K context, trained on 12T tokens, with a user-set **thinking budget** (multiples of 512). The model card reports AIME24 91.7, AIME25 84.7, LiveCodeBench 67.4, SWE-Bench about 56%. No RL algorithm details are disclosed.
- **Seed1.8 model card** (arXiv 2603.20633, 2026-03): four thinking modes (no_think, think-low, -medium, -high), with search, code and GUI tools built in. It states benchmarks but, per my read, **discloses no RL algorithm, reward or infra**. BrowseComp-en is 67.6 per Seed's blog.
- **Seed2.0 model card** (arXiv 2607.00248; the PDF is dated 2026-02): Pro, Lite and Mini. From the fetched summary **[unverified; the summary was machine-generated]**: IMO 2025 gold (35/42), GPQA Diamond 88.9, AIME25 98.3, Codeforces Elo 3020, BrowseComp 77.3, plus science evals including BABE 50.0 (multimodal biology reasoning), SuperGPQA 68.7 and PhyBench 74.0. The report stresses "large-scale product feedback" and gives no RL details.

### 3.9 Seed on scientific and biology models (no RL found)

- **Protenix** (`github.com/bytedance/Protenix`, Apache-2.0): Protenix-v1 (Feb 2026) is an AF3-class, 368M-parameter, fully open structure model. It shows log-linear accuracy gains from sampling more candidates (inference-time scaling).
- **PXDesign** (bioRxiv 2025.08.15.670450): DiT-style backbone diffusion generator, ProteinMPNN sequences, Protenix and AF2-IG as orthogonal filters, and Foldseek clustering for selection. Wet-lab hit rates of 17-82% nanomolar on six of seven targets (20-73% in other summaries), 2x to 6x the baselines. **There is no reward-guided or RL step in the pipeline; the confidence models act as post-hoc filters.**
- **SeedProteo** (arXiv 2512.24192): diffusion binder design with self-conditioning, with reported 70-80% hit rates. Not RL.
- **DPLM, DPLM-2, DPLM-2.1** (ByteDance Research): multimodal diffusion protein LMs. I did not confirm any RL stage.
- **Anew Labs** (ByteDance AI drug discovery, about 36 core members per a news article): four oral small-molecule immunology candidates and the AnewSampling MD framework. **[Single secondary source.]**
- **Conclusion:** RL for biology is open ground at Seed. The pieces that exist (Protenix as an oracle, PXDesign as a generator) are exactly what an RL loop needs.

---

## 4. What the code shows that the papers do not

1. **The ReTool recipe is GRPO plus DAPO tricks, not the paper's PPO.** `run_qwen2-32b_dapo.sh` uses `adv_estimator=grpo`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `clip_ratio_c=10`, no KL, and `max_response_length=16384`. Its `compute_score` (`verl-recipe/retool/retool.py`) uses `math_dapo` (+1/-1, strict boxed) **plus a tool-use bonus on failures**: `if score < 0: score = min(-0.6, score + (num_turns - 2)/2 * 0.1)`. The paper claims outcome-only rewards. The recipe's tool wrapper also rewrites the model's code so its last line is wrapped in `print(...)`, so the sandbox always returns something. The recipe is pinned to verl v0.6.1 because the in-tree `SandboxFusionTool` was removed (docs/sglang_multiturn/sandbox_fusion.rst, 05/2026). The README results are 0.60 AIME25 mean@30 after 150 GRPO steps and 0.55 after 250 PPO steps.
2. **Dynamic sampling moved from trainer hacks to infrastructure.** Early DAPO re-looped batches (`gen_prompt_bsz = 3x`, `max_num_gen_batches=10`). In V1 it is an eviction-and-refill feature in the replay buffer, combined with staleness eviction and failure eviction. The refill unit is exactly one prompt group, so batch size stays fixed without wasted gradient computation.
3. **A GRPO sample with no sibling gets mean 0 and std 1** (`compute_grpo_outcome_advantage`: single-sample groups get `id2mean=0, id2std=1`). The advantage is then the raw score, a silent behavior worth knowing when `n=1`.
4. **Loss normalization is global, not per micro-batch.** `agg_loss` takes `batch_num_tokens`, `global_batch_size` and `dp_size` through `config.global_batch_info`, so token-mean is the same under FSDP or Megatron or different DP sizes. `seq-mean-token-sum-norm` divides by a fixed `loss_scale_factor` (Dr.GRPO-style) instead of the padded length.
5. **Bypass mode.** Setting `old_log_probs = rollout_log_probs` removes the whole old-log-prob recompute, which is the largest saving for long async rollouts. The code contains a warning that with PPO-clip the ratio pi_current/pi_rollout already acts as the IS correction, so IS weights must not be applied on top (double counting). Only the `reinforce` loss type applies explicit truncated IS weights.
6. **Rejection sampling for mismatch is configurable.** Options are `token_k1/k2/k3`, `seq_sum_k1...`, `seq_mean_*`, `seq_max_k2/k3`, chainable with commas (`rollout_corr_helper.py`). It comes from the "RL collapse" blog series (Liu, Li et al., 2025), the Trust Region Masking paper (arXiv 2512.23075), and score centering (arXiv 2609.20807), all as cited in the verl docs and code.
7. **Reward functions get `num_turns` and `tool_extra_fields` injected into `extra_info`** by the naive and remote managers. A tool can therefore put information such as a docking score or an assay readout on the trajectory and the reward function can use it.
8. **Reward managers use only the last sequence** (`data = data[-1:]`) for multi-sequence outputs. Multi-trajectory agents compute a single reward from the last one.
9. **Truncation and sanity checks.** The final output slices `response_ids[: response_length]`, so the tool-augmented trajectory is hard-truncated. `tokenization_sanity_check_mode: strict` checks incremental versus full tokenization.
10. **The reward function for `math_dapo` returns -1 for parse failures** (`is_correct_strict_box` returns 1 or -1), so format errors and wrong answers are the same signal.
11. **FAPO** (`verl-recipe/fapo/reward_fn_reasoning.py`, authors include Seed's Chi Zhang and Haibin Lin): a GenRM locates the earliest error step; if the final answer is correct but a step is flawed, the reward is reduced by a constant `FLAWED_REWARD_PENALTY = 1.0` (to 0.0 in the code). The paper may use a schedule **[not checked]**.
12. **Delta weight sync** assumes RL updates are over 99% sparse in BF16 (lossless). This is a data point about *how little* weights change per RL step, which matters for LoRA and sparse-update methods.
13. **FP16 and FP8 are first-class,** with recipes (`test_dapo_8b_megatron_fp16.sh` sets `rollout.dtype` and `megatron.dtype` to float16; `low_precision/` has FP8 rollout-only and end-to-end Megatron variants that need `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`).
14. **Native MCP tool type and the user-simulator `interaction` config are absent in this version of `tool_registry.py`;** only `native` and function tools remain. **[Inferred from grep of that file.]**

---

## 5. Who builds on verl

- **README says it is "adopted and contributed by":** ByteDance, Anyscale, LMSys, Alibaba Qwen, Shanghai AI Lab, Tsinghua, UC Berkeley, UCLA, UIUC, HKU, ke.com, All Hands AI, ModelBest, JD AI Lab, Microsoft Research, StepFun, Amazon, LinkedIn, Meituan, Camel-AI, OpenManus, Xiaomi, NVIDIA research, Baichuan, RedNote, SwissAI, Moonshot AI (Kimi), Baidu, Snowflake, Skywork, JetBrains, IceSword Lab, and others.
- **Named projects in the README:**
  - Reasoning/RLVR: TinyZero, SkyThought (NovaSky), simpleRL-reason (HKUST), Skywork-OR1, DeepScaleR (agentica/rllm), POLARIS, PRIME and the Entropy-Mechanism work (Tsinghua), FlowRL, ProRL, Logic-RL.
  - Tool/search/agent: Search-R1, ReSearch, ToRL, RAGEN, verl-agent (GiGPO), verl-tool (TIGER-Lab), OpenManus-RL, RL-Factory, DeepResearcher, MemAgent (SIA-Lab), rStar2-Agent (Microsoft), Agent Lightning (Microsoft), OpenTinker.
  - Multimodal: EasyR1, GUI-R1, VAGEN, VTool-R1, DeepEyes, Revisual-R1, OneThinker.
  - Formal/code: Code-R1, Kimina-Prover-RL (Numina), Seed-Coder.
  - Mid-2026 additions: Mind Lab (GRPO LoRA on a trillion-parameter model with 64 H800s, using Megatron-Bridge), SPEAR (Tencent Youtu), Dr. MAS, Rethinking-OPD (Tsinghua), DART-GUI.
- **Biology/chemistry on verl:** I did not find a prominent verl-based biology project. Cell-o1 (single-cell RL, arXiv 2506.02911) uses GRPO but does not name its framework in the part I read. LLMol (molecular generation with RDKit verifiable rewards, via a Pith summary of arXiv 2607.19044) does not name verl in the fetched text either. **[Checked only by search; these are likely generic examples of the pattern, not verl users.]**

---

## 6. Transfer to biology

*(Starred ideas are the ones I think are genuinely novel combinations. The rest are direct adaptations.)*

**A. Swap the math checker for a biological verifier inside verl's reward loop.**
- Write a `compute_score(data_source, solution_str, ground_truth, extra_info)` that parses a protein sequence, a SMILES/SELFIES string or a nucleotide sequence from the response. The verifier could be: RDKit property checks, a docking score, a Protenix or AF3-class ipTM/pLDDT, an ESM pseudo-likelihood, a binding-affinity predictor, or a motif/restriction-site checker.
- Because the reward loop is async and per-sample, slow oracles (folding at tens of seconds per sequence) overlap with rollout. Use the `limited` manager for rate-limited APIs. Use `remote` (one Ray actor per call) for CPU-heavy oracles like docking, which also isolates crashes the way the math-verify fix does.

**B. Seed-Prover as the template for formal-verifier biology.**
- *Principle:* an objective checker (Lean) plus a +1/-1 reward plus a tool-using agent (REPL, retrieval, Python) gave 50% to 90% training success. The biological analogue is a *cheap but trusted* computational oracle (Rosetta/FoldX energy gates, MD short-run stability, a retrosynthesis route checker) with tools for sequence search, structure retrieval and Python.
- *Filter RL data by pass@4x8,* excluding problems the model already solves. This is the same recipe as Seed-Prover 1.5 and works for design targets.
- ***Hierarchical sketch-then-verify for biology:*** train a "sketch" model that proposes a scaffold, a motif arrangement, a retrosynthetic skeleton or a pathway outline, rewarded for passing a structural validity check plus a rubric score, as the sketch model in Seed-Prover 1.5 is. Then an agentic fill-in model resolves each sub-part against the verifier. I have not seen this decomposition applied to protein or molecule design.

**C. Fix noisy oracles with the Seed-Thinking-Verifier and FAPO patterns.**
- A reasoning verifier (99.3% vs 82.7%) beat a non-reasoning one. For biology: when the oracle is a learned predictor (affinity, solubility, immunogenicity), train a *reasoning* critic that outputs the argument for or against the prediction. It can catch adversarial sequences that exploit the predictor, which is the biological form of reward hacking.
- ***FAPO-style "flawed positive" penalty for biology:*** a design may pass the main oracle for the wrong reason (e.g. high ipTM from a degenerate poly-X or repeat, an unrealistically low-complexity binder). A secondary check (a GenRM or rule set looking for degenerate or hallucinated features) reduces the reward. The code shows how to implement it as a one-line subtraction on top of the primary score.

**D. Long-latency wet-lab rewards need `separate_async` plus staleness control.**
- For plate-based assays with day-scale latency, use V1 `separate_async`. The replay buffer's `max_off_policy_threshold` and the `drop` or `wait` strategies are the staleness controls. Treat each design batch as a "prompt group" whose reward arrives late. The `wait` strategy blocks until in-flight groups finish so they remain trainable.
- ***Decoupled/bypass-mode PPO with explicit IS and k3 rejection sampling for lab-in-the-loop rounds:*** these tools were built for sampler/trainer mismatch, but the same math handles a design policy that has moved on since the batch was proposed. Combine this with delta-sync-style sparse updates when the generator is a LoRA-adapted protein LM.

**E. Value-model PPO (VC-PPO/VAPO) for sparse, delayed biological reward.**
- Reward arrives only at the end of a long trajectory (a full multi-step synthesis route, a long sequence design, an agent running a multi-instrument protocol). UI-TARS-2 found PPO beat GRPO on long-horizon agent tasks, and VC-PPO/VAPO show *why*: GAE with lambda = 0.95 decays to near zero. Use value pretraining plus decoupled and length-adaptive GAE (you must write the estimator via `register_adv_est`, as verl lacks it).
- ***Value model as a cheap surrogate for the oracle:*** pretrain the critic on historical (sequence, oracle score) pairs from a design campaign. The critic's per-token value then gives dense credit within a sequence. This is a standard idea in RL generally, but I know of no reported use for protein or molecule LMs trained with PPO at LLM scale.

**F. Tool-using agent for lab and simulator automation.**
- Use `tool_agent_loop` with YAML or `@function_tool` tools wrapping: BLAST/Foldseek search, AlphaFold/Protenix calls, docking, a simulator (OpenMM short runs), and a (simulated) liquid-handler API. The loss mask zeros out tool outputs, so the model is trained only on its own decisions. Follow ReTool's finding that code self-correction emerges from outcome rewards: expect self-correction of failed tool calls (invalid SMILES, failing folding runs). Add a small bonus on failure for using the tool, as the verl ReTool recipe does, to avoid collapse into no-tool answers.
- The `tool_selection` field enables per-sample tool menus, which maps to per-assay-type tool sets.

**G. Entropy and exploration are central in design.**
- DAPO's clip-higher, Clip-Cov/KL-Cov (in `core_algos.py` and `verl-recipe/entropy`) and the UI-TARS-2 observation that entropy *rises* in rich environments all matter for design diversity. A policy that collapses onto one binder family is a failure mode. Measure `actor/entropy` and add a sequence-diversity metric (Foldseek cluster count) next to it.
- ***Group-relative advantage across designs for the same target:*** DAPO's dynamic sampling (drop groups where all designs score the same) becomes "skip targets that are trivially easy or impossible," giving automatic curriculum. Sampling 16 designs per target is close to how PXDesign already works (generate many, filter, cluster), so the RL group structure is natural.

**H. Use Seed's own bio models as the reward environment.**
- Protenix is Apache-2.0 and PXDesign shows the Protenix ipTM filter enriches true binders. A closed loop would be: LLM or protein-LM policy, Protenix as verifier (ipTM, pLDDT, interface metrics), group-relative advantage, DAPO filtering. ***RL fine-tuning of a PXDesign-style generator against its own Protenix filter, with an independent second oracle (AF2-IG) held out as a Goodhart check*** is a concrete experiment with built-in overfitting detection. The paper reports orthogonal filters; training on one and evaluating on the other is the natural reward-hacking probe.

**I. Infrastructure notes for a small team.**
- verl supports LoRA RL on Megatron and FSDP, FP8/FP16 rollout and trainer, and works with models as large as 671B MoE. For a 1B to 30B protein or chemistry LM, use FSDP2 plus vLLM or SGLang, a `RewardLoop` with `reward.num_workers` tuned to oracle throughput, and `separate_async` only when the oracle is slower than generation.
- Training/inference mismatch is worse for non-text vocabularies and custom tokenizers (SELFIES, amino acids). Use `calculate_log_probs: True`, `bypass_mode`, and `rollout_rs: token_k1,seq_mean_k3` as a safety net, and watch the `rollout_corr/` metrics.

---

## 7. Sources

**Code and docs read directly (verified)**
- https://github.com/verl-project/verl (formerly volcengine/verl), main at `8718ca3`, 2026-10-01. Files read: `README.md`; `verl/single_controller/base/decorator.py`; `verl/single_controller/ray/base.py` (symbols); `verl/trainer/main_ppo.py`; `verl/trainer/ppo/{ray_trainer.py, core_algos.py, reward.py, rollout_corr_helper.py, score_centering.py}`; `verl/trainer/ppo/v1/{trainer_base.py, trainer_sync.py, trainer_separate_async.py, replay_buffer.py}`; `verl/trainer/config/{ppo_trainer.yaml, actor/actor.yaml, algorithm/rollout_correction.yaml, rollout/rollout.yaml, transfer_queue/transfer_queue.yaml}`; `verl/experimental/agent_loop/{agent_loop.py, tool_agent_loop.py}`; `verl/experimental/reward_loop/reward_manager/{dapo,naive,remote,limited}.py`; `verl/tools/{function_tool,tool_registry}.py`; `verl/utils/reward_score/{__init__,math_dapo,sandbox_fusion}`; `verl/utils/tokenizer/continuous_token.py`; `verl/workers/engine/base.py`; `verl/checkpoint_engine/README.md`.
- Docs read: `docs/{single_controller.rst, hybrid_flow.rst, advance/placement.rst, advance/v1_async_trainer.md, advance/reward_loop.rst, advance/delta_weight_sync.md, algo/baseline.md, algo/rollout_corr.md, perf/dpsk.md, sglang_multiturn/sandbox_fusion.rst}`, `verl/experimental/fully_async_policy/README.md`.
- https://github.com/verl-project/verl-recipe: `README.md`, `dapo/{README.md, run_dapo_qwen2.5_32b.sh, dapo_ray_trainer.py, config/dapo_trainer.yaml, REQUIRED_VERL.txt}`, `retool/{README.md, retool.py, run_qwen2-32b_dapo.sh, sandbox_fusion_tool_config.yaml, REQUIRED_VERL.txt}`, `fapo/{README.md, reward_fn_reasoning.py}`, `low_precision/README.md`, plus README heads of tmax, echo, sppo, spo, entropy, deepeyes, collabllm, partial_rollout, atropos.

**Papers and web pages (fetched via WebFetch)**
- https://arxiv.org/abs/2409.19256 : HybridFlow abstract (1.53x to 20.57x throughput, hybrid programming model, 3D-HybridEngine).
- https://arxiv.org/pdf/2503.14476 : DAPO (eps 0.2/0.28, dynamic sampling, token-level loss, overlong shaping, ablation). Numbers as extracted.
- https://arxiv.org/pdf/2504.05118 : VAPO (7 techniques, ablation, length-adaptive GAE formula, hyperparameters). A batch-size figure in the extraction looked wrong and is omitted.
- https://arxiv.org/pdf/2503.01491 : VC-PPO (value pretraining, decoupled GAE, AIME results).
- https://arxiv.org/pdf/2504.13914 : Seed-Thinking-v1.5 (data, verifiers, algorithm, SRS infra, results).
- https://seed.bytedance.com/en/blog/bytedance-s-latest-thinking-model-seed-thinking-v1-5-technical-details-disclosed : Seed's blog; I did not use claims that conflict with the paper (it says "Streaming Reasoning System" where the paper says "Streaming Rollout System").
- https://arxiv.org/pdf/2504.11536 : ReTool (PPO with interpreter masking, +1/-1 reward, 67% AIME24 in 400 steps).
- https://arxiv.org/pdf/2509.02544 : UI-TARS-2 (PPO, async stateful rollouts, VM infra, data flywheel, parameter-interpolation merge).
- https://arxiv.org/abs/2507.23726 : Seed-Prover 1.0 abstract.
- https://arxiv.org/html/2512.17260v1 and https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture : Seed-Prover 1.5.
- https://github.com/ByteDance-Seed/Seed-Prover : repo contents (proofs only, no training code).
- https://seed.bytedance.com/en/blog/introduction-to-techniques-used-in-seed1-6 : Seed1.6 AdaCoT.
- https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct : Seed-OSS-36B card.
- https://arxiv.org/abs/2603.20633 and https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model : Seed1.8 (few training details disclosed).
- https://arxiv.org/abs/2607.00248 and https://arxiv.org/pdf/2607.00248 : Seed2.0 model card. The detailed numbers (GPQA 88.9 etc.) come from a machine-generated summary of the PDF, so treat them as **[unverified]**.
- https://seed.bytedance.com/en/direction/ai_for_science : Seed AI-for-Science page.
- https://www.marktechpost.com/2026/02/08/bytedance-releases-protenix-v1-a-new-open-source-model-achieving-af3-level-performance-in-biomolecular-structure-prediction/ and https://github.com/bytedance/Protenix : Protenix-v1.
- https://protenix.github.io/pxdesign/ and https://www.biorxiv.org/content/10.1101/2025.08.15.670450v1 : PXDesign (generation plus Protenix and AF2-IG filtering; no RL).
- https://arxiv.org/abs/2512.24192 : SeedProteo (diffusion; no RL).
- https://startupfortune.com/bytedance-is-building-drug-discovery-ai-and-it-is-presenting-at-the-worlds-best-science-conferences/ : Anew Labs. Single secondary source, **[unverified]**.
- https://arxiv.org/html/2506.02911v1 : Cell-o1 (GRPO, batch-level binary reward on single-cell tasks).
- https://pith.science/paper/2607.19044 : LLMol summary (molecular generation with RDKit rewards).

**Not accessed or inferred**
- Seed2.0, Seed1.8 and Seed-OSS RL methodology: not disclosed in anything I could fetch.
- Whether verl exposes length-adaptive GAE or the positive-example NLL loss: inferred from absence in `core_algos.py` and its registry.
- arXiv 2512.23075, 2609.20807 and 2510.22543 (rollout correction, score centering, FAPO): cited from the verl docs and code comments. I did not open the papers.
- The GitHub search tool was unavailable (no account connected), so the list of verl-based projects comes from the README only.
