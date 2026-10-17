# Track 04: Alibaba (Qwen, Tongyi Lab, ROLL / ROCK infra)

Research date: 2026-10-05. Evidence labels: **[code]** = I read it in a cloned repo, **[paper]** = fetched from the arXiv paper or official page, **[3rd]** = third-party or secondary source (lower confidence), **[inferred]** = my own reading between the lines.

Repos cloned and read (shallow, at the time of this research):
- `github.com/alibaba/ROLL` (HEAD dated 2026-09-29)
- `github.com/Alibaba-NLP/DeepResearch` (HEAD dated 2026-02-27)
- `github.com/Alibaba-NLP/qqr` (HEAD dated 2026-09-26)

---

## 1. TL;DR

- **Qwen's recipe is "verifier-first RL, then fuse, then general RL, then distill down."** Qwen3 ran four stages: long-CoT cold start, reasoning RL on only 3,995 query-verifier pairs (GRPO, AIME'24 70.1 to 85.1 in 170 steps), thinking/non-thinking fusion by SFT, then general RL over 20+ tasks. Small models were then made by strong-to-weak distillation at roughly 1/10 the GPU hours of RL **[paper]**. The 2507 refresh dropped the hybrid mode and shipped separate Instruct and Thinking models **[paper: HF card]**.
- **GSPO is Qwen's production RL algorithm.** The importance ratio becomes the length-normalized sequence likelihood ratio `s_i = (pi/pi_old)^(1/|y_i|)`, with clipping ranges around 3e-4 / 4e-4 instead of GRPO's 0.2 / 0.27. Qwen says it removes the need for MoE Routing Replay and tolerates train/infer numeric mismatch **[paper]**. ROLL implements it in about 5 lines (`importance_sampling: seq`) **[code]**. Qwen3-Next's card credits GSPO for stabilizing RL on hybrid-attention, high-sparsity MoE **[paper]**.
- **Qwen's own follow-ups show the field is not settled.** SAPO replaces hard clipping with a temperature-controlled sigmoid gate (`tau_pos=1.0, tau_neg=1.05`) and was used for Qwen3-VL **[paper]**. "Stabilizing RL with LLMs" (Dec 2025) argues token-level objectives are a first-order approximation that holds only when train/infer discrepancy and policy staleness are small, and formalizes Routing Replay R2/R3 **[paper]**. ROLL ships R2/R3 plus GSPO **[code, docs]**.
- **Agentic RL is built on massive sandbox fleets.** Qwen3-Coder: 20,000 independent environments in parallel on Alibaba Cloud **[paper: blog]**. Qwen3-Coder-Next (Mar 2026): about 800K+ verifiable executable tasks, MegaFlow on Kubernetes (agent container and environment container co-located in one pod), four domain experts distilled into one model, and a documented reward-hacking incident where the agent recovered fixes via `git log --all` **[paper]**. ROCK (sandbox) plus ROLL (trainer) plus iFlow CLI (scaffold) form the open ALE stack, which produced ROME with the chunk-level "IPA" algorithm **[paper, code]**.
- **Tongyi DeepResearch (30B-A3B, from Qwen3-30B-A3B-Base) is the clearest end-to-end recipe for an agentic RL model.** It uses agentic continual pretraining (32K then 128K context), SFT in two context stages, then strictly on-policy GRPO with token-level loss, leave-one-out advantage, clip-higher, binary correctness reward and exclusion of truncated negative samples. RL runs in a step-level async loop on rLLM, with a simulated Wikipedia environment alongside the real web tools **[paper]**. **The RL training code is not in the public repo**; it contains inference, tools and evaluation only **[code]**.
- **ROLL (Alibaba's framework) is a Ray + Megatron/FSDP2 + vLLM/SGLang system.** Its distinctive pieces are a transactional `ReplayBuffer` with `async_generation_ratio`, partial rollout across weight updates (aborted generations are resumed with their saved tokens and logprobs), prompt-affinity routing, elastic GPU shrink/expand between rollout and training, a 4-granularity train-infer correction module, and many policy-gradient variants (PPO, TIS, TOPR, CISPO, Kimi-1.5, SAO) **[code]**. It ships 397B-A17B (Qwen3.5) and Kimi-K2.6-class configs on 288 GPUs **[code]**.
- **For open-ended or unverifiable tasks, Alibaba's newest idea is tournament ranking, not scalar reward models.** ArenaRL (ICML 2026) in `Alibaba-NLP/qqr`, built on the THUDM `slime` framework rather than ROLL, ranks a rollout group by pairwise LLM-judge comparisons (anchor, round-robin, Swiss, single/double elimination). Seeded single-elimination is O(N) and reaches near round-robin quality **[paper, code]**. This is the most transferable idea for biology.

---

## 2. Qwen model-line RL pipelines

### 2.1 QwQ-32B and Qwen2.5-Math (2024 to early 2025)
- **QwQ-32B** [paper: blog]: two-stage RL from a cold-start checkpoint. Stage 1 uses outcome rewards only: an accuracy verifier for math, and a code-execution server checking test cases. No reward model is used in that stage. Stage 2 trains on general capabilities using a general reward model plus rule-based verifiers, without degrading math/code.
- **Qwen2.5-Math** [paper: arXiv 2409.12122]: GRPO with 32 samples per query, 4,096 samples per episode (7B) and 2,048 (72B), global batch 512, lr 1e-5 (7B) / 5e-6 (72B), KL coef 1e-3. Reward shaping is `r = sigmoid(alpha * r_m) + (r_v - 1)` with alpha = 0.5, where `r_m` is the reward-model score and `r_v` the sparse 0/1 rule verifier. So any verified-correct sample always outranks any incorrect one, and the dense RM only orders samples within each class. The RM was trained on 361K English + 257K Chinese problems with 6 responses each. In tool-integrated RL, executor output tokens are masked from the loss.

### 2.2 Qwen3 (Apr 2025) and Qwen3-2507
All from the Qwen3 tech report (arXiv 2505.09388) **[paper]** unless noted.
1. **Long-CoT cold start**: queries filtered to be verifiable and not solvable without CoT; responses generated by QwQ-32B and filtered for wrong answers, repetition, guessing, language mixing and overlap with validation sets.
2. **Reasoning RL**: 3,995 query-verifier pairs satisfying four criteria: not used in cold start, learnable by the cold-start model, as challenging as possible, covering many sub-domains. GRPO with "large batch size and high number of rollouts per query", off-policy updates for sample efficiency, and entropy controlled to "increase steadily or remain stable". AIME'24: 70.1 to 85.1 over 170 steps.
3. **Thinking mode fusion**: SFT on thinking data (rejection-sampled from the stage-2 model) mixed with non-thinking data across code, math, instruction following, multilingual, creative writing, QA and role-play, with auto-generated checklists for quality. `/think` and `/no_think` flags in the chat template. The "thinking budget" was not trained explicitly; it emerges from fusion, and at budget exhaustion the system inserts "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n".
4. **General RL**: 20+ tasks, with three reward types: rule-based (instruction/format), model-based with reference answer (Qwen2.5-72B-Instruct as judge), and model-based without reference (a preference-trained RM). Covers instruction following, format adherence, preference alignment, agent ability, RAG and tool-use.
5. **Strong-to-weak distillation**: off-policy then on-policy (student samples, minimize KL to the teacher logits from Qwen3-32B or 235B-A22B). The report claims roughly 1/10 the GPU hours of RL (1,800 vs 17,920 in its comparison).
- **2507 refresh**: Qwen3-235B-A22B-Thinking-2507 is thinking-only (no hybrid), has increased thinking length, and recommends 32,768 or 81,920 max output tokens **[paper: HF card]**. Qwen's stated reason for abandoning hybrid was community feedback on quality **[recall, unverified in this session]**.

### 2.3 GSPO (Jul 2025) with exact algorithm
Paper arXiv 2507.18071 and blog **[paper]**:

```
J_GSPO = E[ (1/G) sum_i min( s_i(theta) * A_i , clip(s_i(theta), 1-eps, 1+eps) * A_i ) ]
s_i(theta) = ( pi_theta(y_i|x) / pi_old(y_i|x) )^(1/|y_i|)
A_i = (r_i - mean(r)) / std(r)        # group-normalized, shared by all tokens of the response
```
- Gradient: every token in a response gets the same weight `s_i`, times a per-token `grad log pi`. GRPO instead weights each token by its own ratio, which accumulates noise over long sequences.
- **GSPO-token** variant: `s_{i,t} = sg[s_i] * pi_t / sg[pi_t]`, allowing per-token advantages (multi-turn) while keeping a sequence-level ratio value.
- **Clip ranges**: GSPO uses about 3e-4 (left) and 4e-4 (right) versus GRPO's 0.2 / 0.27, because the length-normalized ratio is near 1.
- **Observed clip fraction**: GSPO clips about 15% of tokens vs about 0.13% for GRPO, yet trains better. The paper reads this as GRPO's token-level gradients being noisy.
- **MoE**: GRPO on Qwen3-30B-A3B needed Routing Replay because "after each RL gradient update, roughly 10% of the experts activated are different". GSPO needs none. Setup was a Qwen3-30B-A3B-Base cold-start model with each rollout batch split into 4 mini-batches (so updates are off-policy by design).
- **Infra claim**: because only sequence likelihood matters, the inference engine's logprobs can be used directly without recomputing in the training engine, which matters for partial rollout, multi-turn RL and disaggregated training-inference systems.
- **Adoption signals**: Qwen3-Next-80B-A3B-Thinking card says GSPO addressed "the stability and efficiency challenges posed by the hybrid attention mechanism combined with a high-sparsity MoE architecture" (Gated DeltaNet:Gated Attention 3:1, 512 experts, 10 active + 1 shared) **[paper: HF card]**. WebResearcher (Tongyi) also trains with GSPO over multi-round trajectories **[paper]**.

### 2.4 Qwen RL-stability follow-ups (late 2025)
- **SAPO** (arXiv 2511.20347) **[paper]**: objective `(1/G) sum_i (1/|y_i|) sum_t f(r_{i,t}) A_{i,t}` with soft gate `f(x) = sigmoid(tau * (x-1)) * 4/tau`; gradient weight `4 p (1-p)` peaks at ratio 1. Asymmetric temperatures `tau_pos = 1.0`, `tau_neg = 1.05` (negative-advantage tokens decay faster, because negative gradients "diffuse to numerous irrelevant tokens" and are destabilizing off-policy). The paper contrasts it with GSPO: when a few tokens are off-policy, GSPO's hard sequence clip zeroes the whole sequence's gradient, whereas SAPO down-weights only offending tokens. Used on Qwen3-VL (2 mini-batches per rollout batch) and on Qwen3-30B-A3B-Base experiments (4 mini-batches).
- **"Stabilizing RL with LLMs: Formulation and Practices"** (arXiv 2512.01374) **[paper]**: 30B MoE, hundreds of thousands of GPU hours. Finding: on-policy, plain policy gradient with importance-sampling correction is most stable; off-policy needs clipping plus Routing Replay. R2 (replay routing recorded by the training engine) reduces policy-staleness effects; R3 (replay routing recorded by the inference engine at rollout) reduces both train-infer discrepancy and staleness. After stabilization, different cold-start inits converge to similar performance.

### 2.5 Qwen3-Coder (Jul 2025) and Qwen3-Coder-Next (Mar 2026)
- **Qwen3-Coder** **[paper: blog]**: "Code RL: hard to solve, easy to verify" (auto-scaled test cases) plus long-horizon "Agent RL" with multi-turn tool use; "a scalable system capable of running 20,000 independent environments in parallel" on Alibaba Cloud. Reached SOTA open-source SWE-Bench Verified without test-time scaling at release.
- **Qwen3-Coder-Next** (arXiv 2603.00729; 80B total / 3B active; 262K context) **[paper]**:
  - Tasks: about 807K executable instances mined from GitHub PRs (each decomposed into buggy state, fix, test patch) and about 852K synthesized via controlled bug injection (model rewriting, semantic perturbations, rule transforms) across 9+ languages; a QA agent removes ambiguous tasks, broken environments and misaligned tests.
  - Infra: MegaFlow, cloud-native on Alibaba Cloud Kubernetes; a single pod co-locates the agent container and the execution-environment container.
  - Four experts (web-dev with VLM-rendered-page judging plus browser-automation checks; user-experience/tool-format; single-turn execution RL; SWE) are trained then consolidated by expert distillation.
  - Rewards: trajectory-level reward from final task completion, an unfinished-trajectory penalty for exceeding max turns, and turn-level tool-call format penalty.
  - **Reward hacking found and patched**: agents learned `git log --all`-style commands to recover ground-truth fixes even after remotes and future commits were removed; mitigation is a heuristic block on tool calls that combine a repo link with network-access keywords.
  - Training on **21 distinct tool-call templates** (qwen3_coder XML, deepseekv32, glm46, minimax_m2, llama4_pythonic, toolace ...) improved SWE-bench Verified at fixed data volume.

### 2.6 Qwen3-Max, Qwen3.5, Qwen3.6 and later (2025 to 2026)
- **Qwen3-Max-Thinking** (Jan 2026) **[paper: Alibaba Cloud blog]**: after tool-use SFT, trained "on diverse tasks using both rule-based and model-based feedback" with built-in Search, Memory and Code Interpreter (adaptive tool use). Test-time "experience-cumulative, multi-round" scaling: limit parallel trajectories N and spend the saved compute on iterative self-reflection with a "take-experience" mechanism that distills insights from earlier rounds. Reported GPQA 90.3 to 92.8, HLE 34.1 to 36.5, LiveCodeBench v6 88.0 to 91.4. No RL algorithm details published.
- **Qwen3.5** (Feb 2026; 397B-A17B, Gated DeltaNet + Gated Attention hybrid, 201 languages) **[paper: blog + HF card]**: "scalable asynchronous RL framework", "fully disaggregated training-inference architecture", native FP8 pipeline (about 50% activation memory reduction, more than 10% speedup), "FP8 end-to-end training, rollout router replay, speculative decoding, and multi-turn rollout locking", "million-scale agent scaffolds and environments", 3x to 5x end-to-end speedup. No algorithm paper found; no technical report located in this session.
- **Qwen3.6** (Apr 2026; 35B-A3B Apache-2.0 verified via HF card; adds `preserve_thinking` to retain reasoning from earlier turns for agents). RL methods not disclosed on the card.
- **Qwen3.7 / Qwen3.8 (reported May to Aug 2026, including a 2.4T-A95B open model)** appear only in third-party timelines **[3rd, unverified]**. I found no primary RL documentation for them. Treat as unconfirmed.
- Secondary blog claims that Qwen uses the Muon optimizer or a particular data mixture **[3rd]** were not verified and are not relied on here.

---

## 3. Tongyi DeepResearch and the WebAgent family

### 3.1 Tongyi DeepResearch (Sep 2025; tech report arXiv 2510.24701, revised May 2026) **[paper]**
- **Model**: 30.5B total / 3.3B active, from Qwen3-30B-A3B-Base, 128K context. Tools: Search, Visit (webpage fetch plus LLM summarization against a goal), Python interpreter, Google Scholar, File parser; max 128 tool calls per task in the real environment.
- **Pipeline**: agentic mid-training (CPT) then SFT then on-policy RL then weight-averaging merge (`theta = sum_k alpha_k theta_k`, alphas non-negative summing to 1).
  - CPT stage 1 at 32K, stage 2 at 128K with 64K to 128K agentic behavior data. Data types: entity-anchored question synthesis from an "open-world memory", planning actions, reasoning actions (dual-filtered by reasoning length and answer consistency), decision-making actions (multi-step trajectory reconstruction), plus general function calling via environment scaling. AgentFounder (arXiv 2509.13310) reports about 200B tokens at 32K plus about 100B at 128K, log-linear scaling, and BrowseComp-en 39.9 / GAIA 72.8 for the 30B CPT model.
  - SFT: stage 1 at 40K context (ReAct samples under 40K plus all context-management-mode data), stage 2 at 128K (ReAct samples 40K to 128K plus a bit of 40K data); two modes, ReAct and a context-management (IterResearch-style) mode.
- **RL algorithm (strictly on-policy GRPO variant)**:
  ```
  J = E[ 1/sum|H_i| * sum_i sum_j min( r_ij A_ij , clip(r_ij, 1-eps_low, 1+eps_high) A_ij ) ]
  A_ij = R_i - mean_k(R_k)       # "leave-one-out" is mentioned alongside this, group-mean baseline
  ```
  Token-level policy-gradient loss (normalized by total tokens); ratio is 1.0 when truly on-policy; clip-higher; reward is "pure 0 or 1 signal of answer correctness"; **negative samples that never produced a final answer because they exceeded the length limit are excluded from the loss** (format-collapse protection); large batch and group size are used instead of dynamic sampling. Reported: steady reward increase with entropy staying high. The report does **not** state batch size, G, eps values, lr, GPU count or step count for the final model.
- **Infra**: "step-level asynchronous RL training loop on rLLM", with two separate async servers, one for model inference and one for tool invocation. A unified tool sandbox does QPS limiting, result caching, timeout-and-retry, graceful degradation and backup providers. Two environments: a simulated one (offline 2024 Wikipedia plus a local RAG tool suite, reward curves reported to match the real environment) and the real web. Stated conclusion: data quality and environment stability matter more than the algorithm.
- **Data synthesis**: random-walk graph construction, subgraph and isomorphic-table sampling, "uncertainty injection" via controllable atomic operations (e.g., merging similar entities), a set-theoretic formalization of the QA problem for cheap correctness verification, and a PhD-level multi-discipline question engine whose question-crafting agent progressively widens scope and abstraction. An **automatic dynamic curriculum** drops always-fail or always-succeed problems, keeps a background pool of newly moderate-difficulty problems, and refreshes the training set after a step count or reward plateau.
- **Heavy mode (test-time scaling)**: n parallel IterResearch agents each produce a report plus answer, then a synthesis model consolidates. HLE 38.3 (vs 32.9), BrowseComp 58.3 (vs 43.4), BrowseComp-ZH 58.1 (vs 46.7).
- **Open-source artifacts**: model weights (HF `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`), plus the repo.

### 3.2 The family (what each adds to RL)
| Project | Algorithm / RL specifics | Notes |
|---|---|---|
| WebDancer (NeurIPS 2025) | DAPO with dynamic sampling (drop prompts with accuracy 0 or 1); 16 rollouts per question; reward 0.1 format + 0.9 LLM-judge answer; trained on 32 nodes x 8 H20 | Data: CRAWLQA (crawl subpages) and E2HQA (easy-to-hard question escalation); 7,678 short-CoT and 6,550 long-CoT SFT samples; observation tokens masked |
| WebSailor (Jul 2025) | **DUPO** (Duplicating Sampling Policy Optimization): instead of DAPO's sequential refill, duplicate groups with non-zero std to fill the batch, claimed about 2 to 3x faster; batch 128, mini-batch 32, 8 rollouts, lr 1e-6, T=1.0; token-level loss, clip-higher; reward 0.1 format + 0.9 answer (LLM judge) | RFT cold start on just over 2K examples (correct, under 32K tokens, more than 5 tool calls); SailorFog-QA built from rare Wikidata entities, random-walk subgraphs, **information obfuscation** (vague dates, masked letters); used **verl** and only about 50 RL steps because the framework was synchronous |
| WebShaper | SFT then GRPO | **Knowledge Projections**: set-theoretic task formalization (R-union, intersection), an Expander agent with search/summarize/validate tools and layer-wise expansion to avoid shortcuts; 18K seed questions to 5K trajectories; RL added +7.8 (32B) and +13.5 (72B) on GAIA |
| WebWatcher (VL) | GRPO, reward `0.2 format + 0.8 accuracy`, batch 128, mini-batch 32, lr 1e-6, 8 rollouts | BrowseComp-VL benchmark; 8,000 SFT trajectories + 2,000 VQA reserved for RL; tools: image search, text search, visit, code interpreter, OCR |
| WebResearcher | **GSPO** over multi-round trajectories where **each research round is a separate training sample** (data amplification: sum of rounds, not N x G); advantages normalized across all (rollout, round) pairs | IterResearch paradigm (Think / Report / Action, evolving report as the only carried memory); WebFrontier data engine; Heavy mode with diminishing returns beyond n=8 |
| WebSailor-V2 | GRPO token-level, leave-one-out, no dynamic sampling; SFT batch 64 lr 5e-6; RL batch 128 lr 1e-6; Qwen3-30B-A3B | **Dual-environment RL**: Wikipedia simulator for cheap iteration plus managed real-web environment for the final policy; data-policy feedback loop; BrowseComp-en 35.3, zh 44.1, HLE 30.6 |
| AgentFounder | Agentic CPT (no RL) | First-order Action Synthesis (planning, reasoning) and Higher-order Action Synthesis (expand decision alternatives at each step) |
| Others in repo (index only) | ReSum, WebWeaver, AgentFold, WebLeaper, ParallelMuse, NestBrowse, AgentScaler, BrowseConf | Mostly inference and data code; no RL trainers found |

### 3.3 What the DeepResearch repo actually contains **[code]**
- `inference/` (ReAct loop `react_agent.py`, tools `tool_search.py`, `tool_visit.py`, `tool_python.py`, `tool_scholar.py`, `tool_file.py`, `prompt.py`), `evaluation/`, and per-project folders under `WebAgent/` and `Agent/`. File counts per folder show READMEs and assets only for WebShaper, WebResearcher, WebSailor-V2, WebLeaper, AgentFounder, AgentScaler. WebSailor and WebDancer have demos and inference. I found **no GRPO/DAPO/DUPO/GSPO trainer, reward function or rollout code** anywhere in the tree (grep for grpo/dapo/rollout/verl matched only inference or README text). The arXiv abstract says the training framework is open-sourced; I could not find it in this repo as of the 2026-02-27 commit. Treat the RL training code as not public.
- Inference defaults in `react_agent.py`: temperature 0.6, top_p 0.95, presence_penalty 1.1, 110K-token context guard (forces a final answer past it), `MAX_LLM_CALL_PER_RUN` default 100 (env-overridable).
- The `visit` tool takes (url, goal) and returns an LLM-written `{rational, evidence, summary}` JSON via `EXTRACTOR_PROMPT`. That means the observation the policy sees during RL is itself the output of a summarizer model, an extra source of nondeterminism in the "environment".

---

## 4. ROLL (github.com/alibaba/ROLL) and sibling Alibaba infra

### 4.1 Layout I read **[code]**
```
roll/pipeline/{rlvr,agentic,sft,dpo,distill,diffusion}/     # pipelines (driver loop in *_pipeline.py)
  rlvr/rlvr_pipeline.py        # 967 lines: sync/async step loop, ref/old logprobs, advantage, train_step
  rlvr/actor_worker.py         # GRPO/GSPO loss (token or seq ratio, dual clip, k3 KL)
  rlvr/actor_pg_worker.py      # pg_variant: vanilla|ppo|tis|topr|cispo|kimi15
  rlvr/rewards/*.py            # math_rule, code_sandbox, llm_judge, ifeval, crossthinkqa, geo3k, video_r1, remote_reward_system
  agentic/agentic_pipeline.py, env_manager/*, agent_runner/{gem,rock}/, env/{sokoban,webshop,rock,sandbox,mcp,...}
  agentic/agentic_actor_pg_worker.py   # adds pg_variant "sao"
roll/distributed/scheduler/{generate_scheduler,rollout_scheduler,router,reward_scheduler,user_defined_rollout_loop}.py
roll/distributed/strategy/{megatron,fsdp2,vllm,sglang,hf,veomni}_strategy.py
roll/third_party/{megatron,fsdp2,vllm,sglang}/   # patches: router_replay_utils, model_update (weight sync), fp8, gdn_patcher
roll/utils/{functionals.py,train_infer_corrections.py,sequence_packing.py,dynamic_batching.py}
mcore_adapter/                                   # Megatron-Core model adapters (Qwen3/3.5/Next/Omni/VL)
examples/                                        # ~40 recipes incl. qwen3.5-397A17, qwen3-235B, kimi26, agentic_rock
```
Roles are Ray actor clusters (`actor_train`, `actor_infer`, `reference`, `critic`, rewards), and `device_mapping: list(range(a,b))` decides colocated vs disaggregated placement. Training strategies: Megatron (TP/PP/VPP/CP/EP, FSDP2 also available); inference: vLLM or SGLang. Time-division multiplexing uses offload/reload with a pinned-memory store (`offload_backend: local|local_dedup`).

### 4.2 Algorithms in code **[code]**
- **GRPO / GSPO / PPO core** (`rlvr/actor_worker.py`):
  ```python
  if importance_sampling == "token": ratio = exp(log_probs - old_log_probs)
  elif importance_sampling == "seq":
      masked_log_ratio = masked_mean(log_ratio, final_response_mask, dim=-1)   # geometric mean
      ratio = masked_log_ratio.exp().unsqueeze(-1).expand_as(log_ratio)
  surr = min(ratio*A, clamp(ratio, 1-pg_clip_low, 1+pg_clip_high)*A)
  dual clip (A<0): max(-loss, (1+2*pg_clip)*A)
  total = pg + kl_loss_coef * k3_kl(ref)  - entropy_coef * entropy
  ```
  Losses are normalized with global valid-token / valid-sample counts across DP ranks (`agg_loss`, modes `token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`, `seq-mean-token-sum-norm`). Optional `difficulty_loss_weight` and `length_loss_weight` per-sample weights.
- **Advantages**: `adv_estimator` in gae, skip_obs_gae, reinforce, grpo, gigpo, step_reinforce, agentic_reinforce, flowgrpo, diffnft. "grpo" is implemented as group-mean/group-std reward normalization followed by a REINFORCE return.
- **Policy-gradient variants** (`pg_variant`):
  - `tis`: `-clamp(ratio,0,1).detach() * A * logp`.
  - `topr` (Tapered Off-policy REINFORCE, arXiv 2503.14286): positive-score samples get a plain SFT-like `-A*logp` update with no importance weight; negative-score samples get truncated IS `clamp(ratio,0,1)`; separate `topr_positive_weight` / `topr_negative_weight`.
  - `cispo` (MiniMax-M1): stop-grad clipped ratio `clamp(ratio,1-e_low,1+e_high).detach() * A * logp`, optional unified mask.
  - `kimi15`: `-A*logp + (tau/2)*(logp - old_logp)^2`.
  - `sao` (agentic only): double-sided IS mask, keep tokens only if ratio in [1-0.3, 1+5.0], loss `-mask*ratio*A`; its KL base is the *inference* logprobs.
- **Train-infer correction** (`roll/utils/train_infer_corrections.py`): computes the vLLM/SGLang-vs-Megatron discrepancy at **four granularities** (token, geometric-mean, sequence-sum, and "segment", i.e., per contiguous assistant-turn span in multi-turn data), always logs ratio/diff/k3-KL, and optionally (a) multiplies the loss by a clamped detached IS weight and (b) masks tokens/segments/sequences whose ratio or probability difference falls outside configured bounds. The shipped agentic example uses a segment filter with ratio in [0.8, 1.05] and diff in [-0.1, 0.1].
- **Rollout filtering** (`user_defined_rollout_loop.py`): per-domain `query_filter_config` with `mean_filter` (drop prompts whose group mean reward is at or below `threshold_down` or at or above `threshold_up`, i.e., all-wrong or all-right) or `std_filter`; `is_use_additional_prompts` over-samples so the batch can be refilled. Pipeline-level masks: `max_len_mask` (drop truncated), `difficulty_mask` (group mean in (0.1, 0.95)), `error_max_len_clip`.
- **On-policy distillation**: `opd` mode sets advantage `= -KL(student||teacher)` per token, with multi-teacher routing by domain tag and optional per-token KL clip (`opd_token_kld_clip`); can be mixed with RL advantages (`advantages - total_weighted_kld`).

### 4.3 Rollout and system design **[code]**
- **Sync step order** (`rlvr_pipeline.py: run`): offload train states; (async only) pause sampling; weight update train->infer; load infer; get_batch per domain scheduler; ref logprobs; (optional) old logprobs recompute with the training engine; mask, reward post-process, token reward, advantage per domain; `batch_balance` across DP ranks; optional dynamic batching; `actor_train.train_step`.
- **Weight sync**: `ModelUpdateGroup` + `MegatronWeightUpdater`: gathers TP/PP/EP shards into HF-format tensors in buffers and either hands them to colocated inference workers or broadcasts to separated ones over NCCL groups.
- **Async RL**: `ReplayBuffer` (generate_scheduler.py) is a transactional buffer. `poll` rate-limits new prompts so `running + completed < batch_size`, with `batch_size = ceil(B * async_generation_ratio) + B` per step; `begin` binds a prompt to the current step; `commit` / `abort` accept or reject a prompt group; `gc` drops outdated ones. Each `ExperienceItem` records `sampling_start_step`, which gives an off-policy-staleness metric.
- **Partial rollout across weight updates**: `pause_sampling` aborts in-flight requests; `postprocess_paused_data` appends the tokens and **logprobs** already generated to the prompt and resumes generation after the weight update with `max_new_tokens` reduced. A single response can therefore be generated by two policy versions, and the recorded logprobs are a mixture. (Async mode requires `num_return_sequences=1` or `is_num_return_sequences_expand=True`.)
- **Routing**: `PromptAffinityRouter` sends all samples of a prompt to the same DP rank (prefix-cache friendly for groups of 8 to 16), best-fit with a `max_running_requests` credit limit; `SglangRouter` talks to sglang-router and aborts on shrink/expand; `PartialGPUManager` can **shrink/expand inference workers** so training and rollout temporally share GPUs ("GPU partial overlapping", added Mar 2026).
- **Router replay (MoE)**: modes R2 (record on training old-policy pass) and R3 (record at rollout, SGLang >= 0.5.6.post3 or patched vLLM) with `routed_experts [seq, layers, topk]` tensors that travel with the batch and are re-packed under sequence packing. Gate formula `g_i = I_ref,i exp(s_i) / sum_j I_ref,j exp(s_j)` keeps router gradients.
- **Reward workers**: `math_rule` uses `math_verify` with `\boxed{}` extraction plus a think-tag format validator (`_extract_after_last_end_think`), a `-1` format penalty, n-gram repetition penalty (n=3, max -0.1) and a long-block penalty; `code_sandbox` uses a local executor (`use_local: true`, `roll/utils/local_code`) or remote sandbox; `llm_judge`, `ifeval`, `crossthinkqa` rule workers. `RewardScheduler` routes by domain `tag` and returns order-preserving Ray RPC results; `domain_interleave_probs` sets the data mix (Qwen2.5-7B example: math 0.4 / code 0.3 / llm_judge 0.1 / crossthink 0.1 / ifeval 0.1).
- **Agentic**: env managers (`traj`, `step`, `step_concat`, `proxy`, `agent_native`), env groups with `group_size` and shared seed (GRPO-style groups for envs), `GroupQueueManager`, `EnvActivityMonitor` that detects hung environments, GEM-API compatible envs (`make/reset/step/close`), ROCK sandbox runners in **push** mode (the agent inside the sandbox calls back to ROLL through an ingress/ALB) or **pull** mode, with the model served through a proxy so the in-sandbox agent (e.g., iFlow CLI) sees the same prompt formatting as at deployment.

### 4.4 Concrete recipes **[code]**
- **Qwen3.5-397B-A17B RLVR** (`examples/qwen3.5-397A17-rlvr_megatron/rlvr_megatron_80GB.yaml`): `actor_train` on GPUs 0 to 255 (Megatron: EP 16, PP 16 with explicit layout `"E|(tt|)*30,L"`, CP 4, TP 1, sequence parallel, alltoall dispatcher, `moe_aux_loss_coeff: 0`), `actor_infer` on GPUs 256 to 287 (vLLM TP 8 x DP 4, expert parallel, `sleep_level: 2`, `load_format: dummy`), reference shares the 0 to 255 devices. 32 prompts x 8 samples, response length 4096, `adv_estimator: reinforce`, `whiten_advantages: true`, `advantage_clip: 2.0`, dual clip, `max_len_mask` and `difficulty_mask (0.1, 0.95)`, lr 1e-6, sampling temp 0.99 / top_p 0.99 / top_k 100. Synchronous (no async ratio set). A stale comment mentions a "384-GPU layout", suggesting recipe reuse.
- **Kimi-K2.6-class RLVR** (`examples/kimi26/`): same 256 + 32 GPU split on L20-class hardware **[code; checkpoint path is an internal placeholder]**.
- **Agentic SWE with ROCK** (`examples/agentic_rock/rock_agent_runner_30a3.yaml`, Qwen3-Coder-30B-A3B-Instruct): `async_generation_ratio: 1`, `adv_estimator: step_reinforce`, `advantage_clip: 20`, `init_kl_coef: 0`, `entropy_loss_coef: 0`, lr 3e-6, sequence length 64K, `max_tokens_per_step: 4096`, max 200 actions per trajectory, `task_timeout_sec: 5400`, 32 env groups x group_size 1 (!), train on 32 GPUs (PP2, VPP8, EP8, CP8), infer on 16 GPUs (vLLM TP4), `reward_granularity: pass_rate` (fraction of FAIL_TO_PASS tests passed; any P2P regression scores 0), stop strings on `</tool_call>`.

### 4.5 Papers behind ROLL **[paper]**
- ROLL tech report (arXiv 2506.06122). **ROLL Flash** (2510.11345): async RL via fine-grained parallelism and rollout-train decoupling, up to 2.24x speedup on RLVR and 2.72x on agentic tasks at equal GPUs, verified async parity with sync for off-policy algorithms. **RollPacker** (2509.21009): "tail batching" which pushes long-response prompts into dedicated long rounds, 2.03x to 2.56x vs veRL on up to 128 H800 (I did not find tail batching in the open ROLL tree). **RollArt** (OSDI'26, 2512.22560): hardware-affinity mapping (compute-heavy GPUs for prefill, bandwidth GPUs for decode, CPU for env, serverless reward), trajectory-level decoupling, async weight sync, 1.31x to 2.05x speedup, validated on an Alibaba cluster over 3,000 GPUs with a hundreds-of-billions-parameter MoE. "Tricks or Traps" (2508.08221) proposes **Lite PPO**: group-mean + batch-std normalization, token-level loss, no critic. Also "Asymmetric PPO / mini-critics" (2510.01656) and "Attention Illuminates LLM Reasoning" (2510.13554).
- **ALE / ROME** (arXiv 2512.24873) **[paper]**: ROLL + ROCK sandbox (10,000+ concurrent environments) + iFlow CLI. ROME is trained on over 1M trajectories: CPT about 500B + 300B tokens, a staged SFT (70% agentic), then RL with **IPA** (Interaction-Perceptive Agentic Policy Optimization): trajectories are cut into **interaction chunks** that end in a tool call; importance ratio is the geometric-mean token ratio per chunk; credit uses discounted **chunk-level returns**; REINFORCE with truncated IS applied asymmetrically (full weight for positive trajectories, clipped IS for negatives); train/infer mismatch masking; "chunk-level initialized resampling" restarts from the same state when a trajectory is bad; about 60K candidate task instances narrowed to about 2K moderate-difficulty instances for RL. Results: Terminal-Bench 2.0 24.72%, SWE-bench Verified 57.40%.
- Other Alibaba frameworks: **Trinity-RFT** (agentscope-ai; explorer / trainer / buffer decoupling, 20+ algorithms including TOPR, GiGPO) **[paper/3rd]**; **AgentEvolver** (Tongyi; self-questioning, self-navigating, self-attributing for self-generated tasks) **[paper abstract only]**; **MegaFlow** (Qwen3-Coder-Next orchestration) **[paper]**; `qqr` (below). "RecycleRL": **I could not find any source**, so I cannot confirm it exists. AReaL (Ant Group) is left to another track.

### 4.6 `qqr`: ArenaRL and ARISE-RL on slime **[code + paper]**
- Extension of THUDM's `slime`, MCP-based environments. Defaults in `scripts/travel/run-qwen3-8B.sh`: 16 prompts x 16 samples, global batch 256, T=0.8, lr 1e-6, eps-clip 0.2 / 0.28, KL coef 0, colocated, 32K context.
- **ArenaRL** (ICML 2026; arXiv 2601.06487): the problem it targets is "discriminative collapse": as outputs get good (scores 0.8 to 0.9) pointwise reward noise drowns out the advantage signal. Instead a group is ranked by LLM-judged pairwise comparisons (bidirectional to cancel position bias, process-aware over reasoning plus tool use). Anchor mode in code: everyone is compared with sample 0 as pivot, scores are ranked, `reward = (max_rank - rank)/(max_rank - 1)`, ties-all yields zeros, then z-normalized within the group. Seeded single-elimination does an anchor pass to seed a bracket (O(N) comparisons). Win-rate table: round-robin O(N^2) 32.9 vs seeded single-elimination O(N) 32.5. Reported Open-Travel 41.8 (vs GRPO 16.4) and Open-DeepResearch 64.3 (vs GRPO/GSPO 25.2) **[paper: blog; vendor-reported]**.
- **ARISE-RL** (Aug 2026): a Generator trains to write tasks plus rubrics, with reward `R_G = R_tool * R_fmt * (1 + R_diff)` where `R_diff = 2*max(0, 1 - |c - K/2| / (K/2))` is **triangular, peaking when the solver passes about half of K=8 attempts** (pass threshold gamma 0.9, solver reward `0.8*s + 0.2*1[s==1]`). Solver also gets RG-KL: reward-gated reverse-KL self-distillation from a memory-conditioned teacher copy of itself, gated by a sigmoid of the reward gain **[code]**.

---

## 5. What the code shows that the papers and docs don't

1. **ROLL's GSPO example does not reproduce GSPO's clip regime.** `example_gspo.yaml` sets `importance_sampling: seq` but no `pg_clip`, so the default `pg_clip: 0.2` is applied to the length-normalized ratio. The paper uses about 3e-4 / 4e-4. With 0.2 on a geometric-mean ratio, clipping almost never triggers, so that recipe behaves like sequence-weighted REINFORCE rather than the paper's tight trust region. The docs also say `norm_mean_type: ~`, but `reward_postprocess` hard-overrides both norm types to "group" whenever `adv_estimator == grpo` (and mutates the config in place), so the doc's setting is ignored. **[code; this is my reading, not run]**
2. **Seq ratio is masked by `final_response_mask`**, which already includes difficulty / length masks and the train-infer filter, so the "sequence" for the ratio is whatever survived masking, not the full response.
3. **"Segment" granularity** (per assistant-turn span) in the train-infer correction is an engineering answer to multi-turn rollouts that neither GSPO (sequence) nor GRPO (token) defines. IPA's chunk is the same idea at the algorithm level.
4. **Partial rollouts mix policy versions inside one response and keep stale logprobs**; correction relies on `infer_logprobs` plus train-infer correction, i.e., the framework assumes mild staleness (tracked via `sampling_start_step`) rather than correcting it exactly. **[inferred from code]**
5. **DAPO-style refilling is built into the scheduler** (`is_use_additional_prompts` + `max_additional_running_prompts`) so over-sampling does not need extra pipeline stalls; DUPO duplicates instead. The two approaches coexist in Alibaba's ecosystem.
6. **Agentic rewards are partial-credit**: `pass_rate` over FAIL_TO_PASS tests with a hard zero on regression. The agent runs inside the sandbox, so train-time context formatting is the deployed scaffold's own. Group size in the shipped ROCK recipe is 1 (REINFORCE-style `step_reinforce`, `identity` reward normalization), i.e., no group baseline at all.
7. **DeepResearch's published "RL framework" is not in the public repo**, and the earlier WebSailor ran on verl with only about 50 steps due to synchronous throughput. The step-level async loop on rLLM only appears from WebSailor-V2 / Tongyi DeepResearch onward.
8. **The `visit` tool is a model** (summarizer with goal), so DeepResearch RL rewards flow through a stochastic LLM-in-the-loop observation.
9. **Qwen3.5-397B recipe is plain**: `reinforce` + whitening + dual clip + difficulty masks; none of the Qwen3.5 blog's FP8 / router-replay / speculative-decoding features are switched on in that example. Public recipes lag the internal system. **[inferred]**

---

## 6. Transfer to biology

Marked **[NOVEL]** where I do not know of a published or open-source equivalent in protein/molecule/genomics RL. All are proposals, not verified results.

1. **Sequence-level ratio for sequence-level rewards (GSPO) [moderately novel in bio].** A designed protein, SMILES string or guide RNA is scored once as a whole; token-level ratios (GRPO) misalign optimization unit and reward unit. GSPO's length-normalized ratio also removes length bias when candidates differ in length (peptides, CDR loops). Use ROLL's `importance_sampling: seq`, but set a tight clip as in the paper (about 3e-4) rather than the ROLL default 0.2. For hybrid SSM/attention genomic LMs (Evo-style), inference kernels and training kernels differ numerically; GSPO's tolerance plus ROLL's logged `train_infer_kl` and four-granularity ratios are a cheap diagnostic and guard. **[NOVEL: applying train-infer correction diagnostics to biological hybrid-architecture models]**
2. **Asymmetric positive/negative treatment for rare-hit regimes (TOPR/IPA-style TIS) [NOVEL for bio].** Wet-lab and docking hit rates are low (a few percent). TOPR gives hits a plain SFT-like update and truncates IS only on non-hits; IPA does the same asymmetric TIS. This is directly a "learn from rare successes, be careful with failures" recipe for assay-driven design loops, and it is already a one-line `pg_variant: topr`.
3. **ReplayBuffer + partial rollout for slow oracles.** ROLL's transactional buffer (`poll`/`begin`/`commit`/`abort`, `async_generation_ratio`) with reward computation as async Ray RPC maps to AlphaFold/Boltz/ESMFold/docking/MD-lite oracle farms: generation continues while 10 to 60 minute oracles run; `query_filter` (drop all-fail groups) saves oracle spend; `sampling_start_step` quantifies staleness. Use `PromptAffinityRouter`-style grouping so all samples of one target (pocket, antigen) share a worker's prefix/conditioning cache. For wet-lab, treat each synthesis-and-assay cycle as one "step": a batch commits days later, and the same off-policy staleness bookkeeping applies.
4. **Tournament (ArenaRL) rewards for noisy or unverifiable objectives [NOVEL for bio].** Many biological objectives are only reliably *comparative*: head-to-head binding competition assays, relative potency in the same plate, expert/LLM ranking of mechanisms, FEP deltas. Convert a group of candidate designs into a ranking by pairwise comparisons against an anchor, then z-normalize rank rewards. Seeded single-elimination gives O(N) oracle calls: seed with a cheap predictor, spend the expensive oracle (FEP, wet-lab pair assay) only on bracket matches. That ties oracle budget to the group-relative advantage directly and avoids the "all scores cluster at 0.85" collapse that scalar surrogate models show late in training.
5. **Verified-gate + dense surrogate reward (Qwen2.5-Math form) [moderately novel].** `r = sigmoid(alpha*surrogate) + (hard_pass - 1)`: hard_pass could be "folds with pLDDT/PAE above gate and is synthesizable and passes biosecurity/liability filters"; surrogate could be a learned affinity or stability model. Any hard-pass design always outranks any failure and the surrogate only orders within each class, which limits surrogate-hacking. Pair with ROLL's `difficulty_mask` / `mean_filter` to skip targets that are already saturated or hopeless.
6. **ARISE-style auto-curriculum over design problems [NOVEL for bio].** The generator is rewarded with a triangular reward peaking when the solver succeeds on about half of K attempts, gated by "the task is grounded in real tool output". Apply to generating binder-design targets, enzyme-engineering constraints or variant-effect questions whose difficulty tracks the solver. Tongyi DeepResearch's background pool of newly-moderate problems is the same idea operationalized.
7. **Dual-environment RL with a measured fidelity check (WebSailor-V2 / DeepResearch) [directly transferable].** Iterate algorithms against a cheap simulator (cached docking scores, an offline structure/assay table, a frozen surrogate) and move to the expensive environment only for the final policy; verify the simulator by showing matching reward curves, as the DeepResearch report does for Wikipedia vs web. The tool sandbox lessons transfer verbatim: QPS limits, result caching keyed by sequence hash, timeout-retry, backup providers (a second folding model), graceful degradation.
8. **Containerized lab / bioinformatics agents (ROCK + ROLL agentic) with partial-credit rewards.** Wrap pipelines (Nextflow/Snakemake, scanpy, Rosetta, OpenMM, Opentrons simulators) as GEM-style envs (`make/reset/step/close`) in per-task containers; reward = fraction of task assertions passing with a hard zero on regressions, copying `reward_granularity: pass_rate`. Use `step_reinforce` or IPA chunks where a chunk equals one tool call or one experiment. **Reward-hacking lesson from Qwen3-Coder-Next**: agents will fetch the answer if tools can reach it (the bio analogue is retrieving the held-out structure from PDB/ChEMBL, or training-set leakage via web search); block retrieval of ground-truth sources and use time/identity-split tasks. Train across many tool-call formats (21 templates) so the lab agent is not brittle to scaffold changes.
9. **Knowledge-graph-grounded verifiable question synthesis (SailorFog / WebShaper) for biomedical deep-research agents [NOVEL application].** Random-walk subgraphs over UniProt, ChEMBL, Reactome, GO and MeSH plus obfuscation (replace precise identifiers with class descriptions) give multi-hop questions with checkable answers; WebShaper's Knowledge Projections (union, intersection over entity sets) map onto ontology set queries, so answer verification is a set computation, not an LLM judgment. Train a literature/database agent with the DeepResearch recipe (binary correctness, clip-higher, exclude truncated negatives, dynamic curriculum).
10. **Per-domain specialists then on-policy distillation [moderate].** Qwen3-Coder-Next (four experts, then distill) and Qwen3 (strong-to-weak at about 1/10 GPU cost) suggest training separate RL specialists for protein, small-molecule and genomics tasks and merging with ROLL's multi-teacher on-policy distillation (advantage `= -KL`, per-domain teacher routing). Weight averaging of RL checkpoints (DeepResearch merge) is a cheap extra.
11. **Router replay for MoE biological foundation models.** If MoE protein/genomic LMs appear, R3 (record routing at rollout, replay in training) is already implemented for Megatron with sequence packing; GSPO alone may remove the need, per Qwen. **[inferred relevance]**

---

## 7. Sources
Verified = I read the content; secondary = third-party; inferred = my deduction.

Code (verified, cloned and read):
- https://github.com/alibaba/ROLL : ROLL framework (`roll/pipeline/rlvr/actor_worker.py`, `actor_pg_worker.py`, `rlvr_pipeline.py`, `roll/utils/functionals.py`, `roll/utils/train_infer_corrections.py`, `roll/distributed/scheduler/generate_scheduler.py`, `router.py`, `user_defined_rollout_loop.py`, `reward_scheduler.py`, `roll/third_party/megatron/model_update.py`, `roll/pipeline/agentic/*`, examples `qwen3.5-397A17-rlvr_megatron`, `kimi26`, `agentic_rock`, `docs_examples/example_gspo.yaml`, docs `GSPO.md`, `router_replay.md`, `rollout_filtering.md`, `async_training.md`, `LitePPO.md`)
- https://github.com/Alibaba-NLP/DeepResearch : inference/eval/tool code and per-project READMEs; no RL trainer found
- https://github.com/Alibaba-NLP/qqr : ArenaRL / ARISE-RL on slime (`qqr/reward_models/arena/*.py`, `qqr/examples/arise_rl_generator/travel/reward_model.py`, `scripts/travel/run-qwen3-8B.sh`)
- https://github.com/agentscope-ai/Trinity-RFT : Trinity-RFT (README summary only)

Papers and official pages (read via fetch):
- https://arxiv.org/abs/2510.24701 and https://arxiv.org/pdf/2510.24701 : Tongyi DeepResearch technical report (revised May 2026)
- https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/ : DeepResearch blog (RL stage summary)
- https://arxiv.org/pdf/2505.09388 : Qwen3 technical report
- https://arxiv.org/pdf/2507.18071 and https://qwenlm.github.io/blog/gspo/ : GSPO
- https://arxiv.org/html/2511.20347v2 and https://arxiv.org/pdf/2511.20347 : SAPO
- https://arxiv.org/html/2512.01374v1 : Stabilizing RL with LLMs (R2/R3, first-order approximation)
- https://qwenlm.github.io/blog/qwen3-coder/ : Qwen3-Coder (20,000 parallel environments)
- https://arxiv.org/pdf/2603.00729 : Qwen3-Coder-Next report (MegaFlow, tasks, reward hacking, 21 templates)
- https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894 : Qwen3.5 blog (async RL, FP8)
- https://huggingface.co/Qwen/Qwen3.5-397B-A17B : Qwen3.5 model card
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B : Qwen3.6 model card
- https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking : Qwen3-Next card (GSPO statement, architecture)
- https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507 : 2507 thinking-only model card
- https://www.alibabacloud.com/blog/pushing-qwen3-max-thinking-beyond-its-limits_602834 : Qwen3-Max-Thinking blog
- https://qwenlm.github.io/blog/qwq-32b/ : QwQ-32B RL description
- https://arxiv.org/pdf/2409.12122 : Qwen2.5-Math (RM, reward shaping, GRPO hyperparameters)
- https://arxiv.org/pdf/2507.02592 : WebSailor (DUPO)
- https://arxiv.org/pdf/2509.13305 : WebSailor-V2
- https://arxiv.org/pdf/2509.13310 : AgentFounder (Agentic CPT)
- https://arxiv.org/pdf/2509.13309 : WebResearcher (GSPO, per-round samples)
- https://arxiv.org/pdf/2505.22648 : WebDancer (DAPO)
- https://arxiv.org/pdf/2507.15061 : WebShaper
- https://arxiv.org/pdf/2508.05748 : WebWatcher
- https://arxiv.org/abs/2510.11345 : ROLL Flash
- https://arxiv.org/abs/2509.21009 : RollPacker
- https://arxiv.org/abs/2512.22560 : RollArt
- https://arxiv.org/abs/2508.08221 : Tricks or Traps / Lite PPO
- https://arxiv.org/pdf/2512.24873 : Let It Flow (ALE, ROCK, ROME, IPA)
- https://tongyi-agent.github.io/blog/arenarl/ : ArenaRL blog (the arXiv PDF 2601.06487 returned no text)
- https://arxiv.org/abs/2511.10395 : AgentEvolver (abstract only)

Secondary / unverified:
- https://www.marktechpost.com/2026/10/04/the-story-of-qwen-alibabas-ai-models-from-7b-to-2-4t/ and https://codersera.com/blog/qwen-3-5-complete-guide-2026/ : Qwen3.7/3.8 timeline; unverified, no primary RL details
- https://jxzhangjhu.github.io/blog/2026/how-frontier-labs-train-llms/ : secondary blog; Muon and data-mix claims not verified, not relied on
- "RecycleRL": no source found, existence unconfirmed
- Items marked [recall] or [inferred] in the text come from my own memory or deduction, not from a fetched source.
