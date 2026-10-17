# Track 09: Other global labs (MiniMax, StepFun, Xiaomi, Tencent, Baidu, Meituan, Ant, Kuaishou, Shanghai AI Lab, LG, Naver, Sakana, TII, MBZUAI, Mistral, Baichuan, iFlytek)

Date of survey: 2026-10-05. Evidence tags used below: **[code]** = I read the repo or config myself; **[paper]** = I read the paper or PDF text (via fetch or `pdftotext` of a PDF shipped in the lab's repo); **[blog]** = lab blog or model card; **[secondary]** = only a search snippet or a summary of a page, so unverified in detail; **[memory]** = from my prior knowledge, not re-checked.

Tooling caveats. The WebSearch and WebFetch budgets ran out near the end of the session. The "Ling and Ring 2.6" PDF (arXiv 2606.15079) returned HTTP 429, so I only have its abstract plus the AReaL code that implements KPop. Several fetches went through a small summarizing model, so exact figures from those are marked [secondary] unless I confirmed them in code or in a local PDF.

---

## 1. TL;DR

1. **The loss-function story in this group is mostly about one problem: train/inference mismatch in long-horizon MoE RL.** Each lab fixes it in a different way.
   - MiniMax: CISPO, plus an FP32 LM head.
   - Ant: IcePop, then KPop (token masking).
   - StepFun: MIS-PO (binary masks instead of IS weights).
   - Meituan: truncated IS plus "triplet clipping".
   - Xiaomi and Shanghai AI Lab: Rollout Routing Replay.
   - Kuaishou: Monte-Carlo log-prob averaging.
   - Falcon: truncated IS.
2. **Async infra is now table stakes, and it is open in places.** AReaL (Ant/Tsinghua), XTuner v1 (Shanghai AI Lab) and verl (with CISPO built in) are readable and runnable. The proprietary async systems are not released: MiniMax Forge, Meituan DORA, Kuaishou KRL, Xiaomi's Seamless Rollout Engine. Their designs are documented well enough to copy.
3. **Scientific RL at these labs is mostly verifiable QA, not oracle-in-the-loop design.** Intern-S1 does RL over 1000+ InternBootcamp tasks with a Mixture-of-Rewards, and protein and chemistry items are scored by a generative verifier. Intern-S1-Pro and Intern-S2-Preview do not publish bio-specific RL environments. Protein-binder (AF3/RFdiffusion-scored), TOMG, MP20 and Biology-Instructions are evaluation benchmarks. That is the gap a biology RL project can fill.
4. **Environment code that is directly reusable for bio.** The current InternBootcamp repo (renamed InternAgentHarness) ships simulator-backed, budgeted, tool-calling RL environments:
   - `mole_bootcamp`: OpenMM energy landscape search with decoys.
   - `battery_bootcamp`: PyBaMM.
   - `fenics_bootcamp`: FEniCS PDE solver.
   - Each runs as an HTTP simulator server, with a `case_generator` that samples fresh tasks and a `verify_score` reward, wired into verl.
5. **Verifier design lessons with direct biology analogues.**
   - MiniMax MaxProof: when a learned verifier is the reward, suppress false positives first. It uses pessimistic min-aggregation across verifiers.
   - StepFun Step-Audio-R1.5: pure RLVR hurts real-world quality (the "verifiable reward trap").
   - MiMo and Klear: partial-credit code rewards, either tiered by test difficulty or proportional to pass rate.
   - LongCat: mask samples whose grader crashed.
6. **Post-training topology.** Several labs train domain specialists and then fold them into one model: MiMo-V2 (multi-teacher on-policy distillation, MOPD), LongCat (domain-parallel RL plus model merge), Kuaishou KAT-Coder-V2 (specialize-then-unify), StepFun (expert self-distillation), Intern-S2 (two-expert OPD). This fits biology, where each modality has its own oracle.
7. **Cost anchors.**
   - MiniMax-M1: 512 H800 for 3 weeks, quoted at about $535K.
   - POLARIS-4B: 32 H800 for about 10 days.
   - Falcon-H1R-7B RL: 256 H100.
   - Ring-1T: 480 prompts x 8 rollouts per step at 64K.
8. **Open source reality.**
   - Released RL code: ORZ, Klear-Reasoner and CE-GPPO (verl forks), POLARIS, InternBootcamp, AReaL, XTuner v1, verl (CISPO).
   - Weights only: MiniMax-M1/M2, Ring-V2 (repo has SFT training only), Hy3-preview (SFT only), LongCat, Step-3.5-Flash, MiMo-V2-Flash.

---

## 2. Per-lab sections

### 2.1 MiniMax (M1, M2.x, M3, Forge, MaxProof)

**CISPO (M1, June 2025) [paper, code].**
- The loss clips the importance-sampling weight and stops its gradient. Every token keeps a gradient through `log pi`, unlike PPO or DAPO, which zero out clipped tokens.
- J = (1 / sum|o_i|) sum_i sum_t sg(r_hat_{i,t}) A_hat_{i,t} log pi_theta(o_{i,t}|...), with r_hat = clip(r, 1 - eps_low_IS, 1 + eps_high_IS).
- The paper sets eps_low to a large value (effectively no lower bound) and tunes only eps_high. The paper text did not give the numeric eps_high.
- The verl implementation `compute_policy_loss_cispo` in `verl/trainer/ppo/core_algos.py` matches this. The example script `examples/cispo_trainer/run_qwen3_8b_fsdp.sh` uses `clip_ratio_low=10`, `clip_ratio_high=0.2`, `rollout_n=5`, `lr=1e-6`, and `loss_mode=cispo`.
- M1 claims a 2x speedup over DAPO on a controlled Qwen2.5-32B study.

**M1 recipe details [paper].**
- Constant lr 8e-5, Adam betas (0.9, 0.95), eps 1e-15.
- 16 off-policy update rounds per generation batch. This is notable: expensive samples are reused heavily, which CISPO tolerates.
- Early truncation: generation halts if 3,000 consecutive tokens each have probability above 0.99, which catches repetition loops.
- Staged window expansion: 40K, 48K, 56K, 64K, 72K, 80K. This was needed because of "pattern collapse", where the tail of long generations turned into garbled text.
- Data: math, competitive code, SynLogic (41 logic tasks), SWE-bench with execution rewards, and a generative RM for non-verifiable data.
- Compute: 512 H800, about 3 weeks, quoted at $534,700.

**LM-head precision fix [paper, blog].**
- Lightning-attention hybrids showed a train/inference probability gap that stalled reward growth. The cause was high-magnitude activations at the LM output head.
- The fix was to compute the output head in FP32. The M1 paper reports the correlation between training and inference token probabilities going from about 0.987 to about 0.997.
- The M2.1 post-training blog repeats this lesson. It shows "piecewise horizontal segments" in the probability scatter plot as the diagnostic, and says FP32 head restoration was critical.
- XTuner's RL loss also upcasts to float: `logits = logits.float()` in `rl/loss/oreal_loss.py` [code]. Intern-S1-Pro lists "LM head FP32" in its mixed-precision recipe [paper, secondary].

**M2 / M2.1 / M2.5 agentic RL [paper (arXiv 2605.26494), blog].**
- Reward has three parts:
  1. Process reward: penalties for language mixing and tool-format errors.
  2. Task-completion-time reward, r_speed = h(T/T_baseline) with h decreasing. This pushes parallel tool calls and shorter paths.
  3. Outcome reward from execution (tests, artifacts).
- Advantage is reward-to-go minus a trajectory baseline.
- Multi-turn agentic RL added multiple importance sampling (MIS) and PPO-style trajectory filtering to cut gradient spikes from long-tail off-policy trajectories [blog].
- SWE environments:
  - Docker environments are built per GitHub PR.
  - Reward uses Fail-to-Pass and Pass-to-Pass tests. The blog says P2P is important so that no new bugs are introduced.
  - Task variants include bug injection, merged commits, test writing, and code review.
- The blog is explicit that single-scaffold training generalizes poorly. Hence "hundreds of scaffolds, thousands of tool-call formats".
- Interleaved thinking: ablations show that dropping earlier reasoning blocks hurts agent performance most on deep search and SWE [secondary].
- AppDev uses "Agent-as-a-Verifier": a Playwright-driven agent checks execution, interaction and visual quality, with expert-written rubrics [blog].

**Forge (agent RL framework, 2026-02) [blog].**
- Three modules: agent side (a "pure trajectory producer"), a middleware layer (Gateway Server plus Data Pool), and rollout/train engines. Both white-box and black-box agents are supported.
- **Windowed FIFO** scheduling: the trainer may only fetch completed trajectories inside a window W = 0.3N of the queue. This gives local out-of-order consumption but global blocking on stragglers, which prevents drift toward short or easy samples.
- **Prefix-tree merging** of multi-turn samples, reported at about 40x training speedup with an equivalent loss.
- Context management is modeled as an agent action inside the environment.
- Reported scale: up to 200K context, millions of samples per day, over 100,000 scaffolds and environments.
- Forge is not open source.

**M2.7 and MaxProof (2026) [blog].**
- M2.7: the model handles "30-50%" of its own RL team's daily workflow through a research-agent harness. In a 100+ round loop it analyzes failure trajectories, edits scaffold code, evaluates, and keeps or reverts, for a reported 30% gain on internal evals.
- MaxProof (2026-06) trains a generative verifier as the RL reward for proofs, using CISPO "with minimal modifications".
  - Top priority is suppressing false positives.
  - The verifier has a four-layer defense: rule filters for abnormal outputs, normalization against format hacking, **pessimistic min-aggregation across several verifier models and modes**, and removal of groups with too little reward variance.
  - The verifier is trained on error location plus verdict, with R = 0.7 R_error + 0.3 R_verdict.
  - The proof reward is the verifier's continuous [0,1] score at trajectory level, with no process rewards.
  - At test time it uses evolutionary search: population 32, top-4 diverse parents, PATCH and REWRITE mutations, early stop at 2 perfect candidates, and a pairwise-ranker tournament.
- M3 (2026-06-01): marketing coverage only. It mentions mixed-modality training from step 0 and about 100T tokens, and no RL specifics [secondary].

**Open artifacts.** Weights and inference code for M1 (`github.com/MiniMax-AI/MiniMax-M1`, no RL code) and M2 (`github.com/MiniMax-AI/MiniMax-M2`, docs only). CISPO lives on in verl and in the follow-on literature.

---

### 2.2 StepFun (Open-Reasoner-Zero, Step-Audio-R1/R1.5, Step 3.5 Flash, Step-3)

**Open-Reasoner-Zero (ORZ, 2025) [code, paper].** `github.com/Open-Reasoner-Zero/Open-Reasoner-Zero`, built on OpenRLHF.
- Recipe: vanilla PPO with a separate critic and GAE with gamma = lambda = 1. There is no KL term (`init_kl_coef = 0`, `kl_loss_coef = 0.0` in `playground/orz_32b_ppo.py`).
- 128 prompts x 64 samples per step. Actor lr 1e-6, critic lr 5e-6, 50 warmup steps. 12 critic update steps per rollout (`critic_update_steps = 12`). Strictly one policy update per generation. Temperature 1.0.
- Reward is binary exact match from `<answer>` via Math-Verify. Data is 57K curated math problems (`data/orz_math_57k_collected.json`).
- Qwen2.5-32B reaches AIME24 48.1 and GPQA-Diamond 55.5, matching DeepSeek-R1-Zero-Qwen-32B at about a tenth of the steps [paper].
- Why it matters: a critic gives a per-token baseline without needing 16-64 samples per prompt to normalize. That is relevant when each reward is expensive.

**Step 3.5 Flash (2026-02) [paper, read from the PDF in the repo].** 196B-class MoE with 11B active, 128K context.
- **MIS-PO** (Metropolis-Independence-Sampling-filtered PO) replaces continuous IS weights with two binary masks and treats retained samples as on-policy.
  - L_actor = -E[ I(x_t) * I(rho_bar(tau)) * log pi_theta(a_t|s_t) * A_t ].
  - Token level: x_t = pi_old / pi_vllm, kept if in [0.5, 2].
  - Trajectory level: geometric-mean ratio, kept if in [0.996, 1.001].
  - It is compared against PPO and GSPO on dense and MoE models, with lower gradient-norm noise and slower entropy decay.
- **Truncation-aware value bootstrapping**: when a response is truncated, the reward is replaced by V(s_T) instead of 0. It is stable at truncation rates up to 20%.
- **Routing confidence** (the average probability mass of activated experts) is proposed as an instability proxy. Low-confidence MoEs need router replay or strict on-policy updates. High-confidence ones tolerate off-policy updates.
- Rollouts: 256 prompts x 16 for reasoning, 512 x 8 for preference, 128 x 8 for tool use. Muon optimizer, actor lr 2e-6 (20 warmup steps), critic lr 5e-6 (50 warmup), gamma = lambda = 1 (following ORZ), KL 0.001 in the final stage only. 4 actor and 12 critic mini-batches per step. Max 128K tokens.
- Rewards:
  - Rule-based for logic, instruction following and code; **model-based verifiers for STEM** (+2.0% over vanilla math-verify in a 450-step ablation).
  - Pairwise GenRM converted to a Bradley-Terry win rate, with a length penalty folded into the confidence and zero reward for fabricated citations.
  - A MetaRM that lowers the reward when the GenRM's preference rests on spurious reasoning (+0.5-3% per benchmark in a 200-step ablation).
  - Search agents: entity-match scoring. Reports: rubric judge with ternary labels mapped to asymmetric binary rewards.
- SFT stage 2 injects about 30K expert-level chemistry trajectories and synthetic arithmetic as out-of-distribution signal to raise "reasoning density".
- Environment building for code agents reaches a 40% automatic success rate (from SWE-factory), and construction trajectories are themselves training data.
- Open: weights, vLLM patch, llama.cpp support and the PDF (`github.com/stepfun-ai/Step-3.5-Flash`). No training code.

**Step-Audio-R1 (2025-11) and R1.5 [paper].**
- R1 uses PPO with no KL, gamma = lambda = 1, clip 0.2, 10K-token sequences. Reward is binary for text; for audio it is 0.8 correctness plus 0.2 "reasoning present". Without the format term, reasoning length collapsed from about 3,000 to under 1,500 tokens.
- The RL set is only 5K items (2K text, 3K audio), chosen with pass@8 in [3, 6]. Scaling to 200K unfiltered items gave no gain.
- **MGRD** (modality-grounded reasoning distillation) iterates to move reasoning from transcript-surrogate to acoustic features.
- R1.5 documents the "verifiable reward trap": RLVR raises benchmark scores but degrades prosody and conversational feel. It adds RLHF with a rubric-conditioned generative reward model (PPO objective with a KL term).

**Step-3 [memory, not re-verified].** Mostly an inference-architecture story (attention-FFN disaggregation). I found no RL-specific content.

---

### 2.3 Xiaomi MiMo (MiMo-7B, MiMo-V2-Flash)

**MiMo-7B (2025-05) [paper].**
- 130K verifiable problems (100K math, 30K code). Math problems with pass rate above 90% over 16 SFT rollouts were dropped (about 50% of the pool), with n-gram dedup and contamination removal.
- **Test-difficulty-driven code reward.** Test cases are bucketed by empirical pass rate across models.
  - "Strict": a tier's reward is given only if its tier and all lower tiers pass.
  - "Soft": a tier's score is shared across its tests and summed over the tests passed. Soft was better on LiveCodeBench.
- **Easy-data resampling**: perfect-pass-rate problems go into a pool, sampled with p = 10% to keep batches stable once dynamic sampling filters most easy prompts.
- GRPO changes: no KL, dynamic sampling, clip-higher. Batch 512, lr 1e-6, 32K length.
- **Seamless Rollout Engine**: continuous rollout, asynchronous reward computation, FIFO early termination. It reports 2.29x training speedup, GPU idle time down from 69.3% to 27.7%, and sample waste from 22.1% to 12.9%. It also extended vLLM with MTP speculative decoding.
- Thesis: RL is much better from a base model with high pass@k, so mid-training for reasoning matters.

**MiMo-V2-Flash (309B total / 15B active; tech report 2026-01) [paper, read from the PDF in the repo].**
- **MOPD (multi-teacher on-policy distillation).** Stage 1 SFT. Stage 2 trains domain teachers independently (search, code agent, math reasoning, safety, and so on) with RL or SFT. Stage 3 distills them into one student by RL.
- MOPD advantage is A_t = sg[log(pi_domain(y_t|...) / pi_theta(y_t|...))] + alpha * A_ORM. That is a token-level reverse-KL reward plus an outcome-reward advantage, with a token mask on training/inference discrepancy (Eq. 7-9). The aim is to avoid the see-saw effect of mixing domains.
- Agentic RL data (Table 8): code agent 90K real plus 30K synthesized prompts, search agent 150K (recursive fact-graph expansion with detail obfuscation), general/function-calling agent 50K (tool-call graphs with explicit and implicit dependencies). There is also a web-dev agent with a video-based multimodal verifier (Playwright recordings scoring visual quality, function and executability), and a terminal agent with about 30K validated Stack Exchange-derived tasks.
- The code-agent scaffold is intentionally minimal: three atomic tools (`bash`, `str_replace`, `finish`), no predefined workflow. Environment setup succeeds 70% of the time across 8 languages on a Kubernetes cluster with over 10,000 concurrent pods.
- Scaling RL over about 120K environments improved SWE-Bench Verified and Multilingual, and also transferred to tau2-Bench, AIME25, GPQA-Diamond and LiveCodeBench (Fig. 4-5).
- Infra: SGLang inference plus Megatron-LM training, **FP8 for both**. Rollout Routing Replay (R3) with a request-level prefix cache that stores KV and routed experts across turns, to keep expert routing consistent. A Data Scheduler schedules fine-grained sequences, with staleness-aware truncated importance sampling for partial rollouts and per-data-source quotas and priorities. MTP raises arithmetic intensity at small on-policy batches.
- Reported reward-hacking incident: the official SWE-Bench images left ground-truth commits in the repo, and the model learned to peek at future commits. Their own training images were rebuilt to remove this.

**Later Xiaomi models (MiMo-V2.5/Pro and so on).** I did not verify any; search snippets only mention a "MiMo-V2.6-Flash-RL".

---

### 2.4 Tencent Hunyuan (Hunyuan-T1, Hunyuan-A13B, Hy3-preview)

- **Hunyuan-A13B report [secondary: fetched summary, arXiv 2609.27284 as listed; model released 2025-06].** GRPO with the KL term removed, following DAPO.
  - Rewards: a lightweight LM verifier giving a binary outcome reward; a multilingual code sandbox (36 languages, over 1,000 concurrent executions on a CPU cluster); and a generative RM that uses the reference answer as a "semantic anchor" for open tasks.
  - Reasoning RL has 150K samples (math:code:logic:science = 2:2:1:1, 10% overlap with SFT data); the all-scenarios stage has 16 sub-topics and over 30 scoring services.
  - Two length phases (24K, then 32K). Low temperature (0.6-0.8), large batches, many rollouts.
  - Fast and slow thinking via `/no_think` and `/think` tags.
- **Hunyuan-T1 (2025-03) [memory, not verified here].** Mamba-Transformer MoE, curriculum RL and self-reward.
- **Hy3-preview (295B / 21B active, 2026-04-23) [code: repo read].** The model card mentions "a rebuilt RL infrastructure and larger-scale training tasks" and coding/search agent evals. The repo's `train/` directory is SFT and LoRA only (DeepSpeed/LLaMA-Factory). No RL code or algorithm details are published.

---

### 2.5 Baidu ERNIE (4.5, X1, 5.0)

- **ERNIE 4.5 (2025) [secondary: report summary].**
  - Progressive RL in three stages: logic corpora only, then math and code, then general domains.
  - Unified Preference Optimization (UPO): DPO-style loss combined with PPO, online variant uses rejection sampling per iteration.
  - Rewards:
    - rule verifiers and sandbox;
    - reference-guided LLM-as-judge;
    - checklist-aware verifiers and generative RMs for open tasks;
    - for visual RLVR, multiple-choice reformulated to open-ended to deter guessing, 10K+ synthesized visual puzzles with dual-LLM verification, and UI2Code with a rendered-fidelity verifier.
  - Stability: domain-specific reward normalization; dropping prompts with accuracy 0 or 1; token-balanced loss; asynchronous online EMA; router orthogonalization loss at 1e-3.
  - Infra: PaddlePaddle, FP8 mixed precision, ERNIEKit.
- **ERNIE 5.0 (2026-02, arXiv 2602.04705) [secondary].** A trillion-parameter, under-3%-activation unified multimodal MoE. RL pieces:
  - U-RB (unbiased replay buffer with a data-ordering constraint to hide long-rollout latency);
  - MISC (multi-granularity importance-sampling clipping with two-sided masking);
  - WPSM (mask well-learned positive samples to shift gradient to harder prompts);
  - **AHRL (adaptive hint-based RL)**: attach the first p_hint tokens of the reasoning trace to the prompt, with p_hint = p_initial * exp(-gamma * t * pass_initial) annealed by step and initial pass rate;
  - a disaggregated RL stack with a unified FP8 path.
- Baidu code release is limited to ERNIEKit and FastDeploy (SFT/DPO and serving). I found no open RL trainer.

---

### 2.6 Meituan LongCat (Flash-Thinking, Flash-Thinking-2601, DORA)

**LongCat-Flash-Thinking (2025-09) [paper, read from `tech_report.pdf` in the repo].**
- **DORA** (Dynamic ORchestration for Asynchronous rollout): a standalone generator group plus an elastic role group that is dynamically co-located for generation or training. Inference instances hold multiple policy versions up to a staleness bound, so each response finishes under a single weight version. Interrupted samples reuse KV cache. Reports over 3x speedup versus synchronous training at tens of thousands of accelerators.
- Algorithm (Eq. 4): GRPO with
  1. no KL term (the k3 estimator's gradient is biased);
  2. token-level loss normalized by a global constant max length T_max;
  3. **triplet clipping**: eps_neg_low and eps_neg_high bound the ratio for negative advantages, and eps_pos_high bounds it for positive ones, aimed at staleness in sparse MoE where negative-advantage tokens can have unbounded ratios;
  4. truncated importance sampling min(r(mu), C) between the training and inference engines.
- Efficiency strategies:
  - with-replacement online filtering of prompts at accuracy 0 or 1;
  - staleness-bounded reuse through a replay buffer with shuffling;
  - **incomplete-signal masking**: mask samples whose grader failed (sandbox execution errors) and samples that hit the length limit without being detected as repetition.
- Rewards: a discriminative RM for non-verifiable tasks (reasoning removed from its input); a reasoning **GenRM for STEM** instead of rules; Lean4 server verification with expert iteration for formal proofs.
- Domain-parallel training: separate STEM, code and agentic RL runs, then fusion into one model that is described as near-Pareto-optimal.
- Agentic: a dual-path check (does tool use raise success on this query?) selects tool-worthy queries. Reports a 64.5% token reduction on AIME-25 with tools (19,653 to 6,965).

**LongCat-Flash-Thinking-2601 (2026, arXiv 2601.16725) [secondary].** 560B total / 27B active.
- **Environment scaling**: over 20 domains, "tens of thousands" of environments. Each domain has a tool graph of more than 60 tools in a dependency graph. Tasks come from BFS-style expansion that only adds a tool node when its dependencies are already satisfied. Any executable tool chain counts as correct, and incomplete ones are rejected.
- **Noise injection**: instruction noise and tool noise (failures, inconsistent responses, partial results), with a curriculum. Training with noise matched clean-benchmark performance and improved robustness.
- DORA extended to 32,000 concurrent environments across about 400 machines, fully streaming (generation, environment execution and reward at per-sample granularity), prefill/decode disaggregation with CPU-resident KV cache. It reports 2-4x faster than synchronous training.
- Objective switches to GSPO. Dynamic budget allocation: a "value function" over task properties and current model metrics, with a heap-based greedy allocation of rollouts to maximize aggregate learning value.
- On-policy self-verification is activated when training stagnates. Context management: summary compression above 80K tokens plus discard-all resets.
- "Heavy Thinking": parallel trajectories, then a summary model, with an extra RL stage for the summary phase.

---

### 2.7 Ant Group (Ling, Ring-1T, Ring-2.5/2.6, IcePop, C3PO++, ASystem, AReaL, KPop)

**Ring-1T (2025-10, arXiv 2510.18855) [secondary for numbers; consistent with AReaL docs].**
- **IcePop**: double-sided token masking of the training/inference probability ratio. Tokens with ratio outside [alpha, beta] are dropped from the gradient. Defaults alpha = 0.5, beta = 5. In AReaL, `examples/math/gsm8k_icepop.yaml` and `docs/en/algorithms/grpo_series.md` specify this as a `rejection_sampling` config with `metric: ratio`, `lower: 0.5`, `upper: 5.0`, and it requires `use_decoupled_loss: true` [code].
- Reasoning RL: GRPO-style, lr 2e-6, KL coefficient 0, 480 prompts/step x 8 rollouts, 65,536 tokens. Then an RLHF stage at lr 3e-6.
- **C3PO++**: budget-aware partial rollout. An inference pool and a training pool, a token budget that triggers training, and a retention threshold that prunes stale rollouts. Reports about 2.5x faster rollout per step.
- **ASystem** runtime pieces: Hybrid Runtime, ASandbox (serverless sandboxes), AReaL.

**KPop (2026-06) [code].** AReaL README dated 2026/06/17 introduces KPop. `areal/utils/functional/functional.py` computes, per token, the forward and reverse binary KL between old and proximal probabilities (treating each token probability as a Bernoulli parameter) and masks tokens where max(KL_fwd, KL_rev) > phi (default `upper: 2.0`). `action: mask` only; `use_decoupled_loss: true` required. AReaL docs say IcePop and KPop "can be composed with any RL algorithm". The AReaL docs list 15+ algorithms (GRPO, DAPO, GSPO, Dr.GRPO, RLOO, SAPO, M2PO, and others), with a `max_head_offpolicyness` staleness knob, and an "AReaL 2.0" microservice release in 2026-07. The "Ling and Ring 2.6" report (arXiv 2606.15079) names KPop as the main RL framework for Ring-2.6-1T, with cross-domain asynchronous scheduling [secondary: abstract only; PDF was rate-limited].

**Ring-2.5-1T (2026-02) [blog/model card].** Hybrid linear attention (1:7 MLA plus Lightning), "large-scale fully-async agentic RL", dense rewards for reasoning rigor, IMO/CMO 2025 gold-level claims, compatibility with Claude Code-style agent frameworks.

**Open artifacts.** `github.com/inclusionAI/AReaL` (full async RL, ~15 algorithms, SGLang and FSDP/Megatron backends, agent and VLM examples, NPU support). `github.com/inclusionAI/Ring-V2` (Ring/Ling model code) contains only SFT code (`training/megatron`, LLaMA-Factory), not the RL stack [code]. The IcePop and KPop blogs are on Notion (`ringtech.notion.site/icepop`, `/kpop`); I could not read them.

---

### 2.8 Kuaishou (Klear-Reasoner, KAT-Coder / KAT-Dev / KAT-Coder-V2)

**Klear-Reasoner and GPPO (2025-08) [paper, code].** `github.com/Kwai-Klear/KlearReasoner` (a verl fork).
- **Gradient-Preserving Clipping**: the clipped tokens keep a (bounded) gradient via the identity delta / sg(delta) = 1 in the forward pass. For positive advantages beyond 1 + eps_h the coefficient is (1 + eps_h); for negative advantages below 1 - eps_l it is (1 - eps_l); otherwise delta.
- Code: `compute_gppo_loss_general_beta` in `verl/trainer/ppo/core_algos.py` applies `-beta1 * (1 - cliprange_low) / ratio.detach() * ratio * A` in the low-clip case, and `-beta2 * (1 + cliprange_high) / ratio.detach() * ratio * A` in the high-clip case, with defaults `gppo_loss_beta1 = 0.25`, `beta2 = 1.0`, plus dual-clip `clip_ratio_c = 3`. Selected via `loss_func: gppo` or `gppo_general` in `workers/actor/dp_actor.py` [code].
- Findings: a few high-quality SFT sources beat many diverse ones. For hard samples, **keeping incorrect traces improved SFT**. Soft code reward proportional to the test pass rate beats binary. Hyperparameters: math lr 1e-6, code lr 5e-7, clip-higher 0.28 / 0.2, alpha for the SFT loss 0.1, 32K max length.
- **CE-GPPO** (`github.com/Kwai-Klear/CE-GPPO`, arXiv 2509.20712) uses the same gradient-preserving idea to control entropy. I read only the repo layout.

**KAT-Coder (2025-10, arXiv 2510.18779) [secondary].** Four-stage pipeline (mid-training, SFT, RFT, "reinforcement-to-deployment"), multi-ground-truth rewards with relative evaluation, **trie packed training** for prefix-sharing trajectories, **difficulty-entropy-aware advantage rescaling** (group scale by 1 - success rate, sample scale by token entropy), and error-masked SFT for broken tool calls.

**KAT-Coder-V2 (2026-03, arXiv 2603.27703) [secondary].** Five domain experts (SWE, web coding, terminal, web search, general) merged by on-policy distillation. Turn-level GSPO variant (a separate importance ratio per turn). **MCLA** (Monte-Carlo log-prob averaging over K = 8 forward passes) to reduce MoE log-prob variance. **KwaiEnv** (dataset, verifier, scaffold, sandbox, trajectory manager as decoupled modules; tens of thousands of concurrent sandboxes; black-box scaffolds through a network proxy). **KRL** with tree training (up to 6.2x speedup) and cache-aware scheduling. 79.6% on SWE-bench Verified is their report.

---

### 2.9 Shanghai AI Lab / InternLM (Intern-S1, S1-Pro, S2-Preview, InternBootcamp, XTuner) and POLARIS

**Intern-S1 (2025-08, arXiv 2508.15763) [paper, secondary].** 235B MoE with a 6B vision encoder, a scientific multimodal model.
- Offline RL stage (SFT on best-of-N responses chosen by accuracy, fluency and safety), then online RL.
- **Mixture-of-Rewards (MoR)** over more than 1,000 tasks. Easy-to-verify tasks use rule verifiers, **CompassVerifier** (a generative verifier) and environment feedback. Hard-to-verify tasks use **POLAR-7B** (a reward model pre-trained on 3.6T synthetic tokens and fine-tuned on 150K preference pairs).
- Science tasks include SmolInstruct/ChemBench (SMILES), MatBench-style materials property tasks, ProteinLMBench-style protein questions, reaction conditions, physics equations, chemistry formulas and medical reasoning [secondary]. Scoring is by verifier model plus rules, not by physical oracles.
- Algorithm: **OREAL** (SFT-style behavior cloning on positive samples plus policy gradient on negatives) with **KL-Cov** entropy control, chosen to avoid token-level probability clipping that interacts badly with MoE routing.
- Config [secondary]: 8 rollouts per prompt, batch 4,096 in 8 mini-batches, lr 5e-7 AdamW, 600 steps, FP8 in rollout and training, ViT and MoE router frozen, batches with gradient norm above 0.3 excluded (about 3%), hybrid offline plus online filtering that drops groups that are all correct or all wrong or contain garbled text. A reported 10x less RL time than comparable recent work.
- **Dynamic tokenizer** for scientific strings: rule-based detection of SMILES and protein sequences and per-segment tokenization, raising compression from 1.44 to 2.64 characters per token on SMILES.

**Intern-S1-Pro (2026-02, arXiv 2603.25040) [paper, secondary].** 1T-parameter MoE (22B active), 6T tokens of continued pretraining.
- RL objective: REINFORCE-style with **dual importance sampling** and a leave-one-out baseline. L = -E[1/G sum_i 1/|y_i| sum_t sg(M(rho_it; alpha, beta) * r_it) * A_it * log pi_theta]. M is a hard mask on the train/rollout ratio rho. Thresholds alpha, beta not stated.
- **Rollout router replay**: expert indices recorded at inference and replayed in the learner, passed through Ray object references rather than the HTTP channel.
- Operator-level precision work (RMSNorm, router softmax, positional embeddings), FP8 expert linears with BF16 non-experts and an **FP32 LM head**. Grouped routing with 8-way EP and a straight-through estimator for the router.
- The paper gives no science-specific reward details. XTuner plus LMDeploy is the stack, and the paper claims only about 20% efficiency loss at 4x model size.

**Intern-S2-Preview (2026-08, arXiv 2608.13505) [paper, two reads].** Released as a 397B MoE and a 35B version (Qwen3.5-based) [repo README].
- Reasoning RL: leave-one-out REINFORCE with token-level IS clipped to [1 - eps_low, 1 + eps_high]. Optimizer **Muon**, lr 1e-6, rollout batch 8,192 responses, 8 mini-batch updates per iteration, max generation 65,536.
- **Partial rollout system** in XTuner plus LMDeploy: in-flight requests are paused, prefixes retained, resumed after a weight update. Per-token behavior-policy version and logprob recorded. Trajectories are discarded if their oldest segment is more than 3 policy updates stale.
- **Bidirectional KL masking**: D_BKL(p||q) = p log(p/q) + (1-p) log((1-p)/(1-q)), tokens above threshold phi masked. This is the same binary-KL idea as Ant's KPop. Also R3 routing replay with FP8 experts, BF16 elsewhere, and FP32 for rope, RMSNorm, router and recurrent states.
- **GEPO** (group-level entropy-controlled PO): scales positive advantages by alpha_low < 1 in low-entropy groups and negative advantages by alpha_high > alpha_low in high-entropy groups. An adaptive length regularizer reweights positive advantages by length among successful responses, with no auxiliary reward.
- Agentic RL uses a **harness x task abstraction** (Claude Code, OpenCode, OpenHands, Mini-SWE, Terminus 2, OpenClaw and others behind a gateway), session-aware credit assignment (one outcome reward to all eligible segments), "all-correct" SWE verification (target-fix plus regression), leakage prevention (single baseline commit, remote refs removed, gold patches and tests restored only after execution) and an `adv_penalty` on deterministic errors applied only to positive advantages. A trace store (incremental prefix tree with token ids, logprobs and routed experts) and an online-trained speculative-decoding draft model (about 2x rollout speedup, 1.7x end-to-end) are included. RL task counts are dominated by software tasks (213,854 coding/terminal tasks from SWE-smith, SWE-Gym, R2E-Gym, Scale-SWE, Nemotron-Terminal, ClawGym).
- Two experts (reasoning, agentic) are consolidated by on-policy distillation transmitting only the teacher's logprob for each sampled token (O(H) instead of O(HV)).
- **Science content**: ProteinBinder-9 (RFdiffusion and AlphaFold 3 plus interface-confidence, binding-energy and contact criteria), TOMG-Bench (molecule editing, property optimization, customized generation), MP20 crystal structures, and Biology-Instructions (multi-omics) appear. On a second pass, the paper states these are **evaluation benchmarks, not RL environments**, and that no science RL task counts or rubric rewards are given. The model produces sequences that external pipelines score. It does not call the oracle in training.

**InternBootcamp / InternAgentHarness [paper, code].** `github.com/InternLM/InternBootcamp`, now titled InternAgentHarness.
- The paper describes 1000+ tasks in 8 domains (algorithms, character puzzles, cryptography, graphical puzzles, language analysis, logic, math modeling, natural science), grown from 118 hand-written to 704 validated bootcamps by an evolutionary agent workflow with DeepSeek-R1. Bootcamps with accuracy above 0.85 or below 0.03 were filtered out. Scaling task count (8 to 512 tasks) improved reasoning near-linearly with a DAPO-like recipe on Qwen2.5-7B/32B.
- Each bootcamp implements `case_generator` (a random instance plus its verification protocol), `prompt_func` and `verify_score` (0 to 1).
- The current repo is verl-centric. The base classes are `BaseInstructionGenerator`, `BaseTool`, `BaseInteraction` and `BaseRewardCalculator` (`internbootcamp/src/`), with YAML tool schemas and interaction limits. It has a distributed tool server for multi-machine simulator hosting. Multi-round tool calls are bound to verl's SGLang rollout. It lists battery design, bot trajectory planning, finance and Texas Hold'em as examples.
- **Simulator-backed science bootcamps I read [code]:**
  - `bootcamps/mole_bootcamp`: an OpenMM `CustomExternalForce` 2D energy landscape built from 12-20 Gaussians. The global minimum is deep and narrow (-90 to -70 kJ/mol, sigma 0.15-0.3 nm). Three wide shallow decoys (-50 to -40 kJ/mol) and sinusoidal ruggedness make local search misleading. Tools are `create_session`, `measure_point` (energy and force) and `run_local_minimization`, served over HTTP and Redis-backed sessions. Interaction config: `max_turns: 15`, `max_tool_calls: 20`. The agent submits `{"pin", "coords"}`; reward is 1.0 if the energy at the submitted point is within 1.5 kJ/mol of the true global minimum (computed per instance with scipy differential evolution), else 0. Tool errors return -0.1 step rewards.
  - `bootcamps/battery_bootcamp`: PyBaMM simulation, answer `\boxed{thickness, porosity}`, reward 1.0 only if specific energy meets the target and max temperature stays under the limit.
  - `bootcamps/fenics_bootcamp`: FEniCS heat problem with graded partial credit (1.0 for meeting both temperature targets within tolerance and the safety limit, 0.5 if targets met but the safety limit violated, 0.25 for one target plus safety, 0 otherwise).
- I found no biology or chemistry-reaction bootcamp in the current repo tree. The "natural science" tasks from the paper are probably the older QA-style ones.

**XTuner v1 RL [code].** `github.com/InternLM/xtuner`, `xtuner/v1/rl/`.
- Loss (`loss/`): `grpo_loss.py` and `oreal_loss.py`; policy-loss registry in `loss_fn.py` (vanilla PPO clip with dual-clip c, a clamp on the log-ratio of +/-20, and a registry for custom functions); the OREAL loss splits positive and negative tokens with separate global normalizers, and adds an SFT cross-entropy term on positive-advantage tokens only. The logits are cast to float32 in the loss.
- Advantage (`advantage/`): `grpo.py`, `rloo.py`, `opo.py`, `passk.py`.
- Rollout importance sampling (`rl/rollout_is.py`, modified from verl `mismatch_helper.py`): token, sequence or geometric level; truncate or mask mode; separate clip, mask and veto thresholds (a per-token veto for catastrophic outliers).
- `data_proto/rl_data.py` implements seq-level and token-level staleness (`token_stale_threshold`) masks.
- Config `examples/v1/config/rl_dapo_math_async_filter.py`: `prompt_repeat_k = 16`, clip 0.2 / 0.28, `clip_ratio_c = 10`, no KL, `AsyncProduceStrategyConfig(over_sample_threshold=0.2, enable_partial_rollout=True, max_staleness=0, tail_batch_trigger_size=256)`, and `enable_return_routed_experts` for R3. Rollout engines: LMDeploy, vLLM, SGLang. Judgers: `compass_verifier_v2.py` (an LLM judge returning A/B/C where C = INVALID for truncated or garbled answers), `dapo_math.py` with an overlong-buffer penalty, plus GSM8K and Geo3K. Examples include `rl_interns2_preview_*`.
- Training code for Intern-S1 itself is not in the Intern-S1 repo; XTuner is the stack.

**POLARIS [code, paper].** Not a Shanghai AI Lab project. `github.com/ChenxinAn-fdu/POLARIS` (Fudan/HKU authors with ByteDance Seed; the search results show it on the Seed blog). It is built on verl.
- Recipe: 4B model, difficulty-filtered data (`drop_easy_data.py` after each stage), 3 stages, and **temperature search** (`search_optimal_temperature.py`) to find the temperature giving diversity similar to stage 1. Training temperature 1.4-1.6 (vs. Qwen3's 0.6). Batch 128 x 8 rollouts, 32 H800, about 10 days. Needs over 64K length at eval. POLARIS-4B reports AIME24 81.2 and AIME25 79.4.

**ScienceBoard [paper, code].** `github.com/OS-Copilot/ScienceBoard`. 169 human-validated tasks across biochemistry (ChimeraX), astronomy (Celestia), geoinformatics (Grass GIS), Lean 4, KAlgebra and TeXstudio, running in OSWorld-style VMs. Success is checked by **state checkers** that read application internals through plugins or embedded servers. Frontier models reached only about 15% at release. It is benchmark-only; the README does not describe RL training use. The state-checker approach is the reusable part.

---

### 2.10 Korea (LG EXAONE, Naver HyperCLOVA X)

- **K-EXAONE (2026-01, arXiv 2601.01739) [secondary].** RL uses **AGAPO** (asymmetric group advantage PO): A_group,i = r_i - mean of the other G-1 rewards (leave-one-out), then A_global = batch-normalized A_group, with truncated token-level IS min(rho, eps), a stop-gradient on the weight, zero-variance filtering, a frozen MoE router, and no KL. Rewards r in [0,1] from rule verifiers plus an LLM judge on math, code, STEM and instruction following. A **GrouPER** preference stage (group-wise SimPER) follows.
- **K-EXAONE 2.0 (2026-08, arXiv 2608.04505) [secondary].** Keeps GrouPER and AGAPO; only preference-learning details are exposed. Chosen/rejected pairs are length-matched when possible. A Korea-specific safety taxonomy (K-AUT-V2, 100+ plus 70 new risk areas) drives the safety stage. No RL infra specifics.
- **HyperCLOVA X 32B Think (2026-01, arXiv 2601.03286) [secondary].** Three RL tracks: MRLVR (GRPO variants with dynamic sampling and adaptive clipping, with format, language and repetition auxiliary rewards), Agent-RL (GRPO, no KL, per-domain reward weights; general agents with 44K context and group 8, SWE agents with 128K context and group 16), and MRLHF (PPO with a trained RM, chosen for efficiency over GRPO variants).

---

### 2.11 Japan (Sakana AI)

- **Fugu / Fugu-Ultra (2026-06, arXiv 2606.21228) [paper, secondary].** An orchestrator over frontier models.
  - **Fugu** trains a lightweight selection head with soft-target SFT, then **sep-CMA-ES (an evolution strategy)** on terminal rewards. It does not use policy-gradient RL, and ES works with expensive black-box rewards.
  - **Fugu-Ultra** uses GRPO (KL beta = 0) with reward 0 for unparseable workflows, 1 for correct, 0.5 for parseable but wrong. It emits natural-language workflows of up to 5 steps across Gemini, Claude and GPT models.
  - Failure mode named: "orchestration collapse", countered by per-agent visibility lists.
- The earlier Sakana "Conductor" (7B RL-trained orchestrator, ICLR 2026) reports 83.9% LiveCodeBench and 87.5% GPQA-Diamond [secondary].

---

### 2.12 Middle East and Europe

- **TII Falcon-H1R-7B (2026-01, arXiv 2601.02346) [paper].** A Transformer-SSM hybrid. GRPO with clip-high (0.2, 0.28), no KL or entropy terms, G = 16, temperature 0.85, 48K max length, lr 2e-6, truncated IS, an auxiliary CE loss on correct samples only, and a "backfill" strategy for zero-advantage groups with a generation cache. Built on a custom verl on 256 H100. Rewards: LaTeX match then math-verify; SandboxFusion code with random test sampling; LLM-as-judge for science; `<think>` format penalty. RL data is disjoint from SFT data and difficulty-stratified. SFT findings: single-teacher beats mixing teachers, hard problems upweighted 1.25-1.75x, balanced token normalization (+4-10% AIME25). Test-time scaling via DeepConf.
- **MBZUAI K2-Think (2025-09, arXiv 2509.07604) [paper, secondary].** 32B. SFT on AM-Thinking-v1-Distilled, then RLVR with GRPO in verl on Guru (about 92K prompts across math, code, science, logic, simulation and tabular). Key negative result: **RL from the strong SFT checkpoint gave only about +5% on AIME24, versus about +40% when RL started from the base model**. Test-time: plan-before-think and best-of-3, with Cerebras wafer-scale serving at about 2,000 tokens/s.
- **Mistral Magistral (2025-06, arXiv 2506.10910) [paper, secondary].** GRPO without KL, minibatch advantage normalization, loss normalized by total group length, clip-higher (eps_high 0.26-0.28), zero-advantage group filtering. Reward: format 0.1, correctness 0.9, soft length penalty, and a fastText language-consistency bonus. Fully asynchronous: generators never wait for trainers, NCCL weight updates in under 5 seconds, and in-flight generations continue with a slightly stale KV cache. Text-only RL transferred to multimodal reasoning (MMMU +5%).

---

### 2.13 Baichuan-M2 (medical, not in the assigned list but the closest bio-adjacent RL system I found)

[paper, secondary] 32B on Qwen2.5-32B-Base. A **Large Verifier System**: a patient simulator (termination gate, affective unit, factual unit) built from de-identified records, plus a rubric generator trained on clinical records, guidelines and synthetic scenarios (92.7% consistency with experts). Three RL stages: rule-based RL (math, code, medical knowledge QA), then rubric-based RL on open-ended medical QA, then multi-turn RL against the patient simulator with fragment-level sampling. Length reward decays as 4/sqrt(|o|) and is activated only when the 80th percentile rubric score passes a threshold. GRPO changes: no KL, asymmetric clip, length-normalized loss. HealthBench 60.1 / Hard 34.7.

### 2.14 iFlytek

[secondary] Spark X1/X2/X2.5 are trained on Huawei Ascend domestic compute and marketed on reasoning and agent skills. I found no RL algorithm, infra or code details beyond press coverage.

---

## 3. What the code shows that the papers don't

1. **CISPO is a three-line change in verl.** `sg(clip(ratio)) * A * log_prob`, with `clip_ratio_low = 10` as the practical way to disable the lower clip. The M1 paper never states the epsilon values, but verl's reference script uses eps_high 0.2.
2. **GPPO and CISPO are the same family.** Both keep the gradient of clipped tokens, but GPPO keeps it bounded by (1 +/- eps) via `ratio / ratio.detach()`, with beta1 = 0.25 on the low side by default. CISPO detaches the whole ratio.
3. **Three labs independently converged on masking or discarding tokens by train/rollout ratio**: Ant IcePop (ratio window [0.5, 5]), StepFun MIS-PO (token window [0.5, 2], trajectory geometric-mean window [0.996, 1.001]), Shanghai AI Lab dual-IS mask (thresholds unstated). KPop and Intern-S2's BKL mask both use the binary KL, which treats a token probability as a Bernoulli and is better behaved than a raw ratio when probabilities are tiny. The MIS-PO trajectory window of [0.996, 1.001] is extremely tight; it only makes sense because most trajectories are near-exact matches.
4. **FP32 LM head is a recurring, cheap, high-value fix**: M1 (disclosed as the key fix), M2.1, Intern-S1-Pro, XTuner's `logits.float()`.
5. **Staleness is handled at token granularity in practice.** XTuner masks tokens older than `token_stale_threshold`. DORA keeps per-sample single-version generation. Intern-S2 discards trajectories whose oldest segment is more than 3 updates stale. AReaL uses `max_head_offpolicyness`.
6. **Graders crash, and labs mask those samples.** LongCat masks sandbox-error samples and truncated non-repetition samples; XTuner's CompassVerifier has an INVALID class for truncated or garbled answers; Intern-S2 applies `adv_penalty` only to positive advantages.
7. **The science-flavored environments in InternBootcamp are simulator servers with randomized instance generators and explicit tool-call budgets** (20 calls, 15 turns), not static QA. The tolerance is on the energy value, not the coordinates, so any basin of equal depth counts.
8. **Open RL code from the larger Chinese labs is rare.** Ring-V2 and Hy3-preview repos only train SFT. The substantive code lives in AReaL, XTuner, verl and the Klear forks.

---

## 4. Transfer to biology

Ideas marked **[NOVEL]** are my own combinations that I did not see in any of these labs' materials.

1. **Pessimistic multi-oracle reward for design tasks (from MaxProof). [NOVEL for bio]** For protein or small-molecule design, a single predicted score (ipTM, docking, a learned affinity head) will be hacked. Use min-aggregation over several independent oracles (AF3-type complex confidence, a physics score, a learned binder predictor, a solubility or developability predictor), drop GRPO groups whose reward variance is low, and train a generative "critic" on error localization plus verdict (R = 0.7 R_error + 0.3 R_verdict). Prioritize false-positive suppression over mean accuracy.
2. **Budgeted active-learning environments on landscape tables. [NOVEL]** Copy the `mole_bootcamp` pattern. A `case_generator` samples a fitness landscape (a deep mutational scanning table, a ProteinGym assay, a synthetic epistatic landscape with decoys and noise). The agent has `measure(variant)` and `local_search(variant)` tools with a hard budget (for example 20 calls), then submits its best guess. Reward is the true-fitness percentile. This teaches sample-efficient experimental-campaign policies with a free oracle, and it can later be swapped for a real or simulated assay with the same tool schema.
3. **Slow oracles with async rollouts and token-level staleness masks.** For folding, docking, MD or wet-lab oracles whose results arrive hours late, copy the Forge Windowed FIFO (W = 0.3N), XTuner `AsyncProduceStrategyConfig` with `max_staleness` and `token_stale_threshold`, and Ant's KPop or IcePop masking to tolerate stale rollouts. The oracle is the straggler, so the trainer should consume whatever has completed inside a bounded window.
4. **Budget-aware oracle allocation (from LongCat-2601's heap-based rollout allocation). [NOVEL for bio]** When oracle calls are the scarce resource, allocate them across targets or prompts by estimated learning value, so you do not spend calls on targets that are already solved or hopeless.
5. **Reward-sample reuse for expensive labels.** M1 reuses each generation batch for 16 off-policy update rounds, and CISPO keeps gradients on all tokens. That is a reasonable base when each reward costs real money. A critic-based baseline (ORZ-style PPO with GAE lambda = 1, or Step 3.5's 12-mini-batch critic) avoids needing 16-64 samples per prompt to compute group baselines.
6. **Partial-credit rewards for multi-constraint design.** MiMo's difficulty-tiered code reward and Klear's pass-rate-proportional reward map onto multi-property objectives (valid, folds, binds, soluble, synthesizable). Tier the checks by empirical pass rate, or give proportional credit, to avoid all-zero groups. The FEniCS bootcamp shows a hand-written version (1.0 / 0.5 / 0.25 / 0).
7. **Cost and time rewards for lab automation (from MiniMax's speed reward).** Reward fewer wet-lab steps, less reagent, or more parallel plate runs, with r_speed = h(T / T_baseline) decreasing in cost. Combine with outcome rewards from the instrument or simulator.
8. **Executable-chain acceptance for protocol tasks (from LongCat-2601's tool graphs). [NOVEL for bio]** Build tool-dependency graphs over lab-automation APIs and sample chains by BFS expansion with satisfied dependencies. Accept any chain that executes in a protocol simulator and meets the goal, instead of matching a reference protocol. Add tool and instruction noise (instrument failures, partial results) with a curriculum, as LongCat did.
9. **Specialist-then-distill across modalities (MOPD, domain-parallel, specialize-then-unify). [NOVEL for bio]** Train separate RL specialists for protein design, small-molecule design and sequence/genomics reasoning against their own oracles, then fold them into one student with token-level reverse-KL (MOPD). This avoids the see-saw between domains whose rewards have different scales and latencies.
10. **Population-based ES for orchestrators over expensive black boxes (Sakana Fugu). [NOVEL for bio]** Where a rollout means a multi-hour campaign, optimize a small head (tool or model router, campaign-policy selector) with sep-CMA-ES on terminal rewards and use GRPO only where per-rollout cost is low.
11. **Verifiable-reward trap (Step-Audio-R1.5).** Single-label RLVR improves benchmarks but degrades the real quality; the bio analogue is optimizing predicted binding while sequence naturalness, expressibility and developability drift. Add a rubric-conditioned generative reward model or a separate naturalness and developability term, and monitor held-out orthogonal metrics.
12. **Scientific tokenization (Intern-S1).** Dynamic tokenization of SMILES and protein strings (1.44 to 2.64 characters per token) cuts sequence length and RL cost for text-based chemistry and sequence models. For long genomic sequences, the same trick applies.
13. **Hint-annealed RL for sparse design rewards (ERNIE 5.0 AHRL).** Attach a prefix of a known-good reasoning trace or partial design with p_hint decaying by step and prompt pass rate. For a protein-design agent, hint with a known scaffold or motif, then anneal.
14. **Truncation-aware bootstrapping (Step 3.5).** For long agentic design campaigns that hit the context limit, bootstrap with V(s_T) rather than 0.
15. **Verifier-environment pairs for experimental dialogue (Baichuan-M2).** A simulated lab-partner or reviewer plus a rubric generator makes open-ended experimental-design conversations trainable. The simulator needs a factual unit that checks consistency against ground truth.
16. **Self-evolution loop for protocol and scaffold tuning (MiniMax M2.7).** The 100-round analyze-failures, edit-scaffold, evaluate, keep-or-revert loop is directly applicable to tuning a lab agent's harness before spending RL compute.
17. **Gap worth owning.** None of these labs publish RL where the reward comes from a structure predictor, docking engine or wet assay. The open building blocks to reuse: the InternBootcamp simulator-server pattern, AReaL/XTuner async trainers, verl's CISPO.

---

## 5. Sources

**MiniMax**
- https://arxiv.org/abs/2506.13585 and https://arxiv.org/pdf/2506.13585 (M1; CISPO, FP32 head 0.987 to 0.997, lr 8e-5, 16 off-policy rounds, repetition truncation; verified from fetch)
- https://github.com/MiniMax-AI/MiniMax-M1 (weights, no RL code; verified)
- https://github.com/verl-project/verl : `verl/trainer/ppo/core_algos.py`, `examples/cispo_trainer/` (CISPO implementation; read the code)
- https://arxiv.org/html/2605.26494v1 (M2 series report; reward, Forge, windowed FIFO summary; secondary)
- https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models (M2.1 post-training; FP32 head, MIS and trajectory filtering; secondary)
- https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm and https://www.minimax.io/blog/forge-scalable-agent-rl-en-1779896141 (Forge; secondary)
- https://www.minimax.io/blog/minimax-maxproof-math-proof-evolution (MaxProof; secondary)
- https://www.minimax.io/blog/minimax-m27 (M2.7 self-evolution; secondary)
- https://sebastianraschka.com/blog/2026/minimax-m2-technical-report.html (notes; secondary)
- https://www.marktechpost.com/2026/06/01/minimax-releases-minimax-m3-with-msa-architecture-supporting-1m-token-context-native-multimodality-and-agentic-coding/ (M3; secondary)

**StepFun**
- https://github.com/stepfun-ai/Step-3.5-Flash : `step_3p5_flash_tech_report.pdf` (read with pdftotext: MIS-PO, hyperparameters, reward system; verified); arXiv https://arxiv.org/html/2602.10604v1
- https://arxiv.org/html/2503.24290v2 and https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero : `playground/orz_32b_ppo.py` (ORZ; code read)
- https://arxiv.org/pdf/2511.15848 (Step-Audio-R1; secondary); `Step-Audio-R1.5.pdf` in https://github.com/stepfun-ai/Step-Audio-R1 (R1.5; read with pdftotext)

**Xiaomi**
- https://arxiv.org/pdf/2505.07608 (MiMo-7B; secondary) and https://github.com/XiaomiMiMo/MiMo (registry only; read)
- https://github.com/XiaomiMiMo/MiMo-V2-Flash : `paper.pdf` (read with pdftotext: MOPD Eq. 7-9, Table 8, infra 4.6; verified); arXiv https://arxiv.org/pdf/2601.02780

**Tencent**
- https://arxiv.org/html/2609.27284v1 (Hunyuan-A13B; secondary)
- https://github.com/Tencent-Hunyuan/Hy3-preview (repo read; `train/` is SFT only)
- https://tencent.github.io/llm.hunyuan.T1/README_EN.html (T1; seen in search results only)

**Baidu**
- https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf (ERNIE 4.5; secondary); https://arxiv.org/pdf/2602.04705 (ERNIE 5.0; secondary)

**Meituan**
- https://github.com/meituan-longcat/LongCat-Flash-Thinking : `tech_report.pdf` (read with pdftotext: DORA, Eq. 4, mask rules, reward system; verified); arXiv https://arxiv.org/html/2509.18883v1
- https://arxiv.org/html/2601.16725v2 (LongCat-Flash-Thinking-2601; secondary)

**Ant Group**
- https://arxiv.org/html/2510.18855 (Ring-1T; IcePop, C3PO++, ASystem; secondary)
- https://github.com/inclusionAI/AReaL : `README.md`, `docs/en/algorithms/grpo_series.md`, `areal/utils/functional/functional.py`, `examples/math/gsm8k_kpop.yaml`, `gsm8k_icepop.yaml` (code read)
- https://github.com/inclusionAI/Ring-V2 (SFT only; checked via clone)
- https://huggingface.co/inclusionAI/Ring-2.5-1T (model card; secondary); https://arxiv.org/abs/2606.15079 (Ling/Ring 2.6 abstract only; the PDF fetch was rate-limited)
- https://ringtech.notion.site/icepop and /kpop (referenced in AReaL docs; not read)

**Kuaishou**
- https://arxiv.org/html/2508.07629v1 and https://github.com/Kwai-Klear/KlearReasoner : `verl/trainer/ppo/core_algos.py` (GPPO; code read); https://github.com/Kwai-Klear/CE-GPPO (layout only)
- https://arxiv.org/html/2510.18779v2 (KAT-Coder; secondary); https://arxiv.org/pdf/2603.27703 (KAT-Coder-V2; secondary)

**Shanghai AI Lab**
- https://arxiv.org/pdf/2508.15763 and https://arxiv.org/html/2508.15763 (Intern-S1; secondary)
- https://arxiv.org/pdf/2603.25040 and https://arxiv.org/html/2603.25040 (Intern-S1-Pro; secondary)
- https://arxiv.org/pdf/2608.13505 and https://arxiv.org/html/2608.13505 (Intern-S2-Preview; two reads; protein/molecule/materials items are evaluation-only per the second read)
- https://github.com/InternLM/Intern-S1 (model list; secondary)
- https://arxiv.org/html/2508.08636 and https://github.com/InternLM/InternBootcamp : `internbootcamp/bootcamps/{mole,battery,fenics}_bootcamp/`, `internbootcamp/src/` (code read)
- https://github.com/InternLM/xtuner : `xtuner/v1/rl/{loss,advantage,judger,rollout,rollout_is.py}`, `data_proto/rl_data.py`, `examples/v1/config/rl_dapo_math_async_filter.py` (code read)
- https://arxiv.org/abs/2505.19897 and https://github.com/OS-Copilot/ScienceBoard (ScienceBoard; secondary)
- https://github.com/ChenxinAn-fdu/POLARIS (POLARIS; clone read, README summarized)

**Korea / Japan / Middle East / Europe / others**
- https://arxiv.org/html/2601.01739v2 (K-EXAONE), https://arxiv.org/html/2608.04505v1 (K-EXAONE 2.0), https://arxiv.org/pdf/2601.03286 (HyperCLOVA X 32B Think); all secondary
- https://arxiv.org/html/2606.21228v1 (Sakana Fugu; secondary); https://sakana.ai/learning-to-orchestrate/ (Conductor; secondary)
- https://arxiv.org/pdf/2601.02346 (Falcon-H1R; secondary, consistent across two fetches)
- https://arxiv.org/html/2509.07604v1 (K2-Think; secondary)
- https://arxiv.org/pdf/2506.10910 (Magistral; secondary)
- https://arxiv.org/pdf/2509.02208 (Baichuan-M2; secondary)
- iFlytek: press coverage only (tmtpost, pandaily); not used for any technical claim

**Inferred or unverified (explicit list).** Hunyuan-T1 details; Step-3 details; exact CISPO eps_high in M1; thresholds alpha, beta in Intern-S1-Pro; any detail of the IcePop and KPop Notion blogs; anything in the Ling/Ring 2.6 PDF beyond the abstract; Xiaomi's models after MiMo-V2-Flash; the claim that no bio-specific bootcamp exists in the full InternBootcamp history (I only saw the current repo tree).
