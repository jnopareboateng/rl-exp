# Track 08: Deep research agents and agentic / tool-use RL (with a biology lens)

Date of survey: 2026-10-05. Kimi (Kimi-Researcher, K2/K3) and Tongyi (WebSailor, Tongyi DeepResearch) are covered in depth by other tracks and are only referenced here.

Evidence labels: **[code]** = I read the file in a cloned repo; **[paper]** = fetched the arXiv/blog page (the fetch tool summarizes pages, so numbers are as reported by the page, not re-derived); **[secondary]** = news/blog/model card only; **[inferred]** = my own inference. Anything unlabeled in the transfer section is my proposal, not a reported result.

---

## 1. TL;DR

1. **The open recipe is converging.** A small-model recipe that works: (a) a text-tag or JSON tool protocol, (b) group-relative RL (GRPO, RLOO or REINFORCE) with no critic, (c) **observation tokens masked out of the policy loss**, (d) a binary or F1 outcome reward, ideally judged by an LLM and not by token overlap, (e) a pass-rate filter on training questions (keep prompts with 1 to 7 correct of 8), and (f) a deterministic, cached tool backend.
2. **The reward is moving away from string matching.** Search-R1 and ReCall use EM or F1 and R1-Searcher and DeepResearcher use F1 with format penalties. ASearcher, MiroThinker and SFR-DeepResearch use LLM-judge correctness. DR Tulu uses rubrics that evolve during training, and CaRR-style work adds citation-grounded rubrics. The 2026 papers I read all argue that F1 is exploitable (answer avoidance, paraphrase penalties).
3. **Environment cost is the main engineering constraint, and there are three answers.** Live web with rate-limit handling (DeepResearcher, ASearcher-Web, MiroThinker). A frozen local corpus (LiteResearcher: about 32M pages, 73.2M tool calls at $0). An LLM-simulated search engine (ZeroSearch, SSRL). Biology can use all three, and databases like UniProt and ChEMBL are far easier to snapshot than the web.
4. **Long-horizon technique is where papers and code diverge most.** Context management is hand-coded in the agent in the open code I read (ASearcher summarizes pages with the policy itself and truncates history; MiroThinker keeps only the last 5 tool results). The learned variants (SFR-DR `clean_memory`, FoldGRPO, ReSum-GRPO) exist only as papers. Credit assignment beyond trajectory-level advantage (IGPO, GiGPO, AT2PO, TRACE, Dr.Credit) is mostly 2025-2026 paper work with little released infrastructure.
5. **Synthetic hard questions matter as much as the algorithm.** Tongyi reports that "data and stability of the training environment" matter more than algorithm choice. The recurring recipe is knowledge-graph random walks plus entity obfuscation (DeepDive, MiroVerse), "fuzzing" and context injection (ASearcher), and a difficulty filter against a strong LLM.
6. **Open training code is rarer than open weights.** Search-R1, ReCall/ReSearch, R1-Searcher, DeepResearcher, ZeroSearch, ASearcher and DR Tulu ship trainers. WebThinker, MiroThinker, II-Researcher and ether0 ship inference or reward code only. I checked each repo.
7. **Biology already has the pieces, but nobody has combined them.** ether0 (RL with chemistry oracles), Biomni-R0 (multi-turn agent RL on 10 bio tasks, [secondary] details), BixBench and Kosmos/Robin (agents with programmatic or expert verification). A biology deep-research agent RL-trained against **versioned database snapshots, programmatic ground truth from structured queries, identifier-existence checks and rubric rewards** is a clear gap (section 7).

---

## 2. Open recipes with code

### 2.1 Search-R1 (UIUC; PeterGriffinJin/Search-R1)
- **What:** The reference "R1-style search" repo, built on veRL (older fork). Supports PPO, GRPO and REINFORCE, local sparse and dense retrievers, and online engines. It has since been ported to veRL upstream, SkyRL (2025-07) and the Thinking Machines Tinker cookbook (2025-10) [code: README news]. Last commit in my clone: 2025-11-13.
- **Rollout [code `search_r1/llm_agent/generation.py`]:** Synchronous, lock-step multi-turn loop (`run_llm_loop`). Each turn generates for the still-active rows only, parses `<search>q</search>` or `<answer>a</answer>` with a regex, runs a batched HTTP retrieval call, and appends `<information>...</information>`. An invalid action gets a fixed corrective message appended and the loop continues. `max_turns=2` in the shipped GRPO script (3 to 4 in the paper). Observations are truncated to `max_obs_length=500`. The whole context is re-fed and re-prefilled each turn. There is no async and no partial rollout.
- **Observation masking [code]:** `_info_masked_concatenate_with_padding` builds a second copy of the response in which every retrieved block is overwritten with `pad_token_id`. The `info_mask` and then the `loss_mask` are derived from `token != pad`. In `ray_trainer.py`, `_create_loss_mask` runs **after** `compute_advantage`, immediately before `update_actor`. Advantages and GAE values still see the observation tokens, and only the actor loss ignores them. `state_masking` defaults to `False` in `ppo_trainer.yaml` and the training scripts turn it on. The KL penalty uses `info_mask`.
- **Loss [code `core_algos.py`]:** Standard clipped PPO surrogate (`compute_policy_loss`), with GRPO group normalization when `adv_estimator=grpo`.
- **Reward [code `verl/utils/reward_score/qa_em.py`]:** Pure exact match after SQuAD-style normalization, 1 or 0, with `format_score=0`. Nothing for intermediate steps.
- **Hyperparameters [code `train_grpo.sh`]:** 8 GPUs, batch 512, mini-batch 256, 5 samples per prompt (`n_agent=5`), lr 1e-6 with 0.285 warmup ratio, `kl_loss_coef=0.001` (`low_var_kl`), temperature 1, response length 500, FSDP with full offload, vLLM at 0.6 GPU utilization, 1005 steps, Qwen2.5-3B base on NQ+HotpotQA.
- **Second paper [paper arXiv 2505.15117]:** Format rewards help, intermediate retrieval rewards have limited effect, reasoning-specialized LLMs beat general ones, and the search engine used in training affects robustness.
- **2026 re-study [paper arXiv 2602.19526]:** On the Search-R1 setup, a "fast thinking" prompt beats long reasoning. F1 reward shows **answer avoidance** (the model withholds answers), and F1 plus action penalties beats EM (0.429 vs 0.422). REINFORCE (0.437) was the most stable of REINFORCE, PPO (0.422) and GRPO (0.433, "poor" stability). Their "Search-R1++" reaches 0.442 on Qwen2.5-7B.

### 2.2 ReSearch and ReCall (Agent-RL/ReCall; ReSearch is a branch)
- **What:** ReSearch (2025-03) trains search-in-reasoning with RL and no supervised trajectories. ReCall (2025-04) generalizes the tool to any JSON function call and adds a synthetic-environment data idea. The original ReSearch code is archived on the `re-search` branch [code: README]. Built on veRL 0.3.0 plus vLLM 0.8.4. Last commit 2025-05-16.
- **Reward [code `src/verl/utils/reward_score/re_call.py`]:** Gated: 0 if the response does not end in EOS, 0 if the template is invalid (balanced `<think>`, valid JSON in every `<tool_call>` with `name` and `arguments`, `\boxed` in the last turn), otherwise **token F1 against the ground truth if above 0, else a floor of 0.1 for a wrong answer with good format**. Truncated or overlong rollouts get nothing.
- **Training [code `scripts/train/*.sh`]:** `ROLLOUT_NAME=vllm_with_tool`, `MAX_TURNS=5`, 5 rollouts per prompt, max response 6656, batch 256, lr 1e-6, tool sandbox and search over HTTP (`SANDBOX_URL`, `SEARCH_URL`), 8 GPUs per node.

### 2.3 R1-Searcher (RUCAIBox/R1-Searcher)
- **Algorithm:** Two-stage outcome RL on OpenRLHF-RAG (a fork with a `retrieve_mask`, see `OpenRLHF-RAG/openrlhf/trainer/ppo_trainer.py`). Stage 1 teaches the retrieval format and stage 2 teaches using results. Reinforce++ or GRPO via `--advantage_estimator group_norm`. Last commit 2025-08-05.
- **Reward [code `train/reward_server_*.py`]:** A separate FastAPI reward server (`/get_reward`) called by the trainer. Stage 1 uses a cover-exact-match style correctness reward of +1/-1 plus +0.5 bonuses for retrieval and format behaviours, and a separate length score. Stage 2 uses answer F1 and **subtracts 2 for any format violation**. The violations checked are unbalanced document or query tags, "Assistant" leaking into the text, a missing or duplicated `<answer>` tag, an answer over 10 words, and Chinese characters inside a non-document span.
- **Related by the same group:** R1-Searcher++ (SFT cold start plus RL that rewards internal knowledge use and a memorization mechanism) and SimpleDeepSearcher (distillation plus self-distillation with few samples) [README].

### 2.4 DeepResearcher (GAIR-NLP/DeepResearcher)
- **What:** End-to-end RL against the **real web**. Last commit 2026-05-10.
- **Environment [code `scrl/handler/`]:** A server handler fronts Serper, Bing or DuckDuckGo search, fetches pages with paging (`page_view_port_size`), and uses a **reading agent** (`qwen-plus` through an API) with an "extract only new information given context so far" prompt, which also decides whether to page down. DuckDuckGo gets explicit throttling (5 concurrent, 1.0 s interval, 5 retries). Search results are cached to disk, and several handler nodes can be listed (`server_url_list`).
- **Training [code `train_grpo.sh`, `scrl/llm_agent/generation.py`]:** Qwen2.5-7B-Instruct, `agent_grpo.n=16`, `max_turns=10`, batch 256, lr 1e-6, KL 0.001, `max_prompt_length=30767` and `max_response_length=2000` (the prompt grows with each tool turn), Ulysses sequence parallel 4. Rollouts use the chat template with `tools=` and a forced `<think>` prefix, and the trajectory is rebuilt as a string and **re-tokenized** after generation.
- **Reward [code `verl/utils/reward_score/format_and_f1.py`]:** Set-based word F1 (sets, not multisets) on the first `<answer>`, and **-1 for unbalanced `tool_call` / `think` / `answer` tags or a missing answer**.
- **Finding from the code:** I found no `loss_mask` or `info_mask` anywhere in the agent loop or the actor. Tool responses are inside `responses` and are therefore **trained on**. This differs from Search-R1 and is worth knowing if you reuse the code. [code, grep-verified; I did not run it]
- Paper claims up to +28.9 points over prompted baselines and +7.2 over RAG-based RL agents, plus emergent behaviours (planning, cross-validation, reflection) [paper/README].

### 2.5 ZeroSearch (Alibaba-NLP/ZeroSearch; Tongyi Lab)
- **Idea:** Replace the search engine during RL with an **LLM that writes the documents**. Last commit 2025-08-17.
- **Mechanics [code `llm_agent/generation.py`]:** `search_simulate_sft` and `search_simulate_prompt` ask a simulator LLM (served through SGLang, e.g. a 14B model with `--tp 2 --dp 2`) to produce five roughly 30-word "Google results" for the query. With probability `gt_threshold` it is told to produce **noisy** documents, otherwise **useful** documents. **The simulator prompt contains the question and the ground-truth answer** ("The user is trying to answer the question X whose answer is Y"). So the simulator is a privileged-information environment.
- **Curriculum [code `dynamic_threshold`]:** Noise probability goes from `START_THRESHOLD` (0 in the README runs) to `END_THRESHOLD` (0.5) along an exponential schedule with base 4: `start + (4^p - 1)/3 * (end - start)` with `p = step/total_steps`.
- **Training:** veRL; PPO (GAE), GRPO and REINFORCE are all supported, `state_masking=True`, EM reward, 4 GPUs for the policy plus separate simulator GPUs. A lightly SFT-tuned simulator (released as `Simulation_LLM_*`) beats a prompted one in their comparison.
- **Related:** SSRL (self-search RL) uses the policy's own knowledge as the simulator with format and rule rewards, and claims sim-to-real transfer [paper arXiv 2508.10874].

### 2.6 ASearcher (inclusionAI/ASearcher; built on AReaL)
- **What:** Large-scale, **fully asynchronous** agentic RL for long-horizon search. Last commit 2025-10-08. The paper reports tool calls above 100 turns and above 400k generated tokens in a trajectory during RL [README].
- **Config [code `ASearcher/configs/asearcher_web_qwq.yaml`]:** QwQ-32B, `n_nodes: 6 x 8 GPUs`, `allocation_mode: sglang.d2t8+d4t8` (SGLang rollout on one pool, FSDP training on the other), `async_training: true`, `max_concurrent_rollouts: 160`, `max_head_offpolicyness: 4`, 16 samples per prompt, `max_new_tokens: 30000`, lr 5e-5 constant, `eps_clip: 0.4`, `kl_ctl: 0.0`, `reward_scaling: 10.0`, `reward_bias: -0.5`, `use_decoupled_loss: true`, `behav_imp_weight_cap: 5.0`, `recompute_logprob: true`, group size 16 with `adv_norm`.
- **Reward [code `train/reasoning_agent.py`, `asearcher_reasoning.py`]:** LLM-as-judge ("correct" / "incorrect" JSON, answer truncated to 200 characters) run on a **separate SGLang judge engine**. Advantage is `(r - mean) / std`, and **groups with no reward variance are dropped** (`return None`).
- **Credit assignment [code]:** Every LLM call in a trajectory is a separate training sample. The group-normalized trajectory advantage is set on every turn's completion (`client.set_reward(comp.id, advantage)`, `export_completions(turn_discount=0.0)`), so each turn is trained with its own prompt and its own response tokens and the observation never enters the loss.
- **Context management [code `reasoning_agent.py`, `prepare_queries`]:** Hard-coded in the agent. Raw pages and search results are read by the **policy itself** in a separate "READ_PAGE" or "READ_SEARCH_RESULTS" call, whose summary is stored as `short_info_str`. The history string is capped to the **last 25,000 characters**. A forced-answer phase triggers after `force_turns` actions or 20 document reads, and generation stops if a prompt would exceed about 32k tokens.
- **Data synthesis [README, `qa_synthesis/qa_synthesis_agent.py`]:** An LLM agent that starts from seed questions and applies **fuzzing** (blur key details) and **context injection** (add retrieved facts to deepen the question), then three checks: quality, difficulty (a strong LRM must fail against the ground truth), and answer uniqueness.
- **Results [README]:** Avg@4 of 58.7 (GAIA), 51.1 (xBench-DeepSearch), 74.5 (Frames) for the 32B model, with RL adding +15.0 / +22.4 / +14.6.

### 2.7 WebThinker (RUC-NLPIR/WebThinker)
- **Repo has no RL trainer [code: only `scripts/run_web_thinker*.py`, prompts, evaluation]. Last commit 2025-12-08.** The "Deep Web Explorer" and "Think-Search-and-Draft" report writing are inference-time strategies. The paper builds preference pairs from reasoning, tool usage and final output for **online DPO** (README says so; I found no training code). Same group's DeepAgent (2510.21618) adds "memory folding" [README].

### 2.8 MiroThinker / MiroMind (MiroMindAI/MiroThinker)
- **Repo is inference and evaluation only [code]**, but with a useful reference implementation of context management. In `apps/miroflow-agent/conf/agent/mirothinker_1.7_keep5_max200.yaml`: `max_turns: 200`, `keep_tool_result: 5`, `context_compress_limit: 5`. `src/llm/base_client.py` filters older tool results out of the message list while keeping all thoughts and actions. Tools are Serper search, Jina scraping with an LLM summarizer (`scrape_and_extract_info`), and an E2B Python sandbox. Last commit 2026-03-23.
- **Training [paper arXiv 2511.11793, v1.0]:** SFT on thought-action-observation trajectories, then DPO on correct vs incorrect trajectories (plus an SFT term), then **online GRPO** ("rollout trajectories update the policy exactly once"), temperature 1.0, top-p 0.95. Reward `R = alpha_c R_correct - alpha_f R_format`. Trajectories with more than 5 consecutive API failures or excessive timeouts are filtered. Context rule: keep only the last K=5 tool responses in the context while keeping all reasoning, which allows up to 600 tool calls in a 256K window. Infra: streaming task queue, "thousands of concurrent agentic rollouts", an **LLM grading system for noisy agent predictions**, Linux sandbox tools. Data: **MiroVerse** (MultiDocQA from hyperlink knowledge graphs with constraint obfuscation, plus agentic trajectories synthesized with several strong LLMs). v1.0 72B: GAIA 81.9, HLE 37.7, BrowseComp 47.1, BrowseComp-ZH 55.6, xBench-DS 77.8 [paper].
- **2026 releases [README, secondary]:** v1.5 (2026-01-05, 30B/235B, up to 400 tool calls, "interactive scaling") and 1.7 (2026-03-11, 300 tool calls, BrowseComp 74.0, with a proprietary MiroThinker-H1 described as "step-verifiable and globally verifiable" reasoning). The RL details for 1.5/1.7/H1 are not in the repo.

### 2.9 II-Researcher / II-Search (Intelligent-Internet)
- **II-Researcher is an agent framework, not an RL trainer [code: `ii_researcher/`, `api.py`, `benchs/`]**, last commit 2026-07-02, with a reported 84.12% on Frames using DeepSeek-R1-0528. The trained model **II-Search-4B** (and the CIR 4B variant with code-integrated reasoning) used distillation from Qwen3-235B, synthetic reasoning problems, STORM-style report data, and a final RL stage on MuSiQue plus proprietary Wikipedia / Fineweb / ArXiv search data. The model card gives no algorithm or reward details [secondary].

### 2.10 Jan (Menlo Research) and other small open agents
- **Jan-nano (4B, Qwen3-4B base) [paper arXiv 2506.22760]:** Described as multi-stage RLVR with **no SFT**, using DAPO in three stages (8K context with tool-execution, correctness and XML-format rewards; 8K with tool-execution reward removed; then 40K context). 83.2% on SimpleQA with MCP tools. A 128K variant adds only about 2.5 points. (The page summary I fetched expanded DAPO wrongly; DAPO is Decoupled Clip and Dynamic sAmpling Policy Optimization.) Jan-v1-4B (Qwen3-4B-thinking base) reports 91.1% on SimpleQA, with no training recipe published on the model card [secondary].

### 2.11 DR Tulu (Ai2 and collaborators; rlresearch/dr-tulu)
This is the most complete open recipe for **open-ended** deep research, and the one most relevant to biology. Last commit 2026-03-30.
- **Stack [code]:** SFT with LLaMA-Factory, RL with a fork of **open-instruct** (`grpo_fast.py`, `grpo_fast_rubric.py`), tools through an **MCP server** (`snippet_search`, `google_search`, `browse_webpage`; Semantic Scholar, Serper, Jina, Crawl4AI).
- **RL config [code `train_dr_tulu.sh`]:** start from `DR-Tulu-SFT-8B`; 32 unique prompts x 8 samples; lr 5e-7 constant; `beta 0.001` with `kl3`; response length 16,384; `max_tool_calls 10`; DeepSpeed ZeRO-3; 4 learners + 4 vLLM engines per node (the Ai2 command uses 2 nodes x 8 GPUs); `MCP_MAX_CONCURRENT_CALLS=512`; judge `gpt-4.1-mini`.
- **Tool masking [code]:** `mask_tool_use: bool = True` in `grpo_fast.py`. In `tool_utils/tool_vllm.py`, tool outputs are appended to the response token ids with mask `0` and model tokens with mask `1`. Tool calls are executed **asynchronously inside the vLLM engine loop**: pending tool futures are polled without blocking and a new generation request is submitted when each returns, with clipping against the model context and the per-request token budget.
- **Rewards [code `search_rewards/longform_averaged_outcome_rewards.py`]:** Weighted sum of **rubric reward 0.5, citation reward 0.2, format reward 0.2, number-of-search-turns reward 0.1** (a no-citation variant uses 0.6 / 0.2 / 0.2). The fine-grained variant (`longform_finegrained_rewards_v2.py`, `fgrpo_fast.py`) weights rubric 0.7 and attaches each reward to **character spans** of the response for span-level credit.
- **Evolving rubrics (RLER) [code `search_rewards/utils/rubric_utils.py`, `grpo_fast_rubric.py`]:** Each prompt has persistent rubrics (from the dataset) and an `active_rubrics` buffer. After every rollout group, a rubric-generation LLM (`gpt-4.1` by default) sees the question, all 8 responses and the existing rubrics, and writes new **discriminative** positive and negative rubrics ("distinguish otherwise similar responses", "never duplicate existing"). They are appended to the prompt's buffer (`max_active_rubrics: 5`) and the judge scores every response against persistent plus active rubrics. Zero-variance groups are discarded. There is a TODO to keep only rubrics with the highest score std.
- **Reported result [paper arXiv 2511.19399]:** 8B model, beats Tongyi DR by 15.6% on average and matches OpenAI's research tool on four long-form benchmarks at very low per-query cost.

### 2.12 Other 2025-2026 recipes worth knowing (paper-level only unless stated)
- **SFR-DeepResearch (Salesforce, arXiv 2509.06283):** single-agent RL with an internal `clean_memory()` tool the model is told to call when the context nears the limit. Modifications: length-normalized advantage (without it "failing-but-long trajectories" grow and the model learns repeated tool calling), filtering of invalid or unbalanced trajectories, partial rollouts reusing intermediate states. Tools: search, static page fetch, stateless Python (5 min timeout). SFR-DR-20B: FRAMES 82.8, GAIA 66.0, HLE 28.7. [paper]
- **Context-Folding / FoldGRPO (arXiv 2510.11967):** branch and return actions with process rewards so sub-trajectories are folded away; about 10x smaller active context. **ReSum / ReSum-GRPO (arXiv 2509.13313):** periodic summarization tool with advantage broadcast across segments, +4.5% training-free and +8.2% more with RL. [paper]
- **DeepDive (arXiv 2509.10446):** KG random walks with entity obfuscation, an LLM-difficulty filter, multi-turn GRPO with a strict binary reward and early exit on format errors; 14.8% BrowseComp at 32B, 20.8% with extra data; tool calls grew about 30% during RL. [paper]
- **LiteResearcher (arXiv 2604.17931):** a 4B agent trained in a **deterministic local web** of about 32M pages (BGE-M3 + DiskANN, about 0.15 s per search; PostgreSQL page store, about 0.17 s per page). Strictly on-policy GRPO, 8 rollouts, batch 128, lr 1e-6, no KL or entropy term, keeps prompts with 1 to 7 of 8 correct, context grows 16K per stage from 32K, 700+ stable steps and 73.2M tool calls (45.8M searches and 27.4M browses; the paper estimates $59K to $243K if done online). Reports GAIA-text 71.3, xBench-DS 78.0, Frames 83.1, BrowseComp 27.5 with summarization. [paper]
- **OffSeeker (arXiv 2601.18467):** argues online RL is not needed: 33K SFT trajectories plus 21K DPO pairs from synthesized tasks; claims results competitive with online-RL agents at zero API cost. **OpenResearcher (arXiv 2603.20278):** offline corpus pipeline for trajectory synthesis (I could not read its page). [paper/secondary]
- **Rethinking RL deep-research design (arXiv 2510.15862):** LLM judge (Gemini-2.5-Flash-Lite, 97.6% agreement with a stronger judge) beats F1; **RLOO more sample-efficient than GRPO**; remove 62% of low-quality samples, keep questions with correctness rate 2.5% to 50%; error-tolerant rollouts raise average reward 0.699 to 0.711. [paper]
- **2026 rubric-based work:** CaRR / C-GRPO (arXiv 2601.06021) gates a rubric reward on correct outcomes: `R = (1-a) R_outcome + a R_outcome R_rubric`, with rubrics = atomic facts with hidden entities, identified in the answer, supported by cited pages, and connected by a BFS evidence chain to the answer. DeepRubric (2606.17029) builds query-rubric pairs from evidence trees, about 13x fewer RL GPU-hours. DR-Rubric (2606.01091) generates rubrics by running deep research itself. Dr.Credit (2609.34296) does rubric-grounded process credit without a gold answer. [paper, fetched summaries only]

---

## 3. Long-horizon agent RL technique, as seen in the code and papers

| Topic | What the open code does | What papers add |
|---|---|---|
| **Observation masking** | Search-R1/ZeroSearch: pad-id trick, `loss_mask` created after advantages. DR Tulu: per-token 0/1 mask from the vLLM tool loop. ASearcher: each turn a separate sample, observation only in the prompt. ReCall: veRL tool rollout. DeepResearcher: **no masking found** | Masking is assumed standard in MiroThinker/Tongyi/Kimi writeups |
| **Advantage / credit** | Trajectory-level group-normalized, broadcast to every token (all repos above). ASearcher broadcasts to every turn | IGPO: per-turn reward = increase in log-prob of the gold answer, normalized separately from the outcome reward, then discounted-summed. TRACE (2607.13988): frozen reference model probes prefixes, `V(S_k)=log(d0/d_k)` TD over tool boundaries, BrowseComp-Plus Qwen3-4B 7.2 to 35.6. GiGPO: group actions from the **same anchor state** across trajectories. AT2PO: entropy-guided tree expansion and child-weighted value aggregation. ARPO: entropy spikes after tool calls, branch there. Dr.Credit: rubric-based process credit |
| **Context management** | ASearcher: policy summarizes pages, last 25K chars. MiroThinker: keep last K=5 tool results, all thoughts | SFR-DR `clean_memory` (learned), FoldGRPO, ReSum-GRPO, Tongyi IterResearch, Kimi-Researcher (retain important info, drop documents) |
| **Turn-level rewards** | Only outcome + format (+0.1 floor in ReCall, -2/-1 penalties in R1-Searcher/DeepResearcher, 0.1 search-turn term in DR Tulu) | IGPO, TRACE, StepSearch-style info gain; the Search-R1 paper itself found intermediate retrieval rewards of limited value |
| **LLM-judge rewards** | ASearcher (separate SGLang judge, JSON correct/incorrect), MiroThinker (LLM grader), DR Tulu (gpt-4.1-mini rubric judge) | 2510.15862: small-judge agreement of 97.6% at a fifth of the cost |
| **Rubric rewards** | DR Tulu (static plus evolving; weights above) | CaRR, DeepRubric, DR-Rubric, Dr.Credit, RaR-style data (the DR Tulu script can mix `RaR-Medicine-20k` and `RaR-Science-20k`) |
| **Async / partial rollouts** | ASearcher (AReaL, staleness 4, decoupled PPO loss, importance cap 5), DR Tulu (async tool calls in vLLM) | Kimi-Researcher turn-level partial rollouts reported 1.5x speedup; MiroThinker streaming queue |
| **Sandboxing at scale** | DR Tulu MCP server (512 concurrent), ReCall sandbox URL, MiroThinker E2B | Practice survey (secondary) says labs now use one sandbox per rollout, microVM snapshot/resume for long tasks (Kimi K3 is claimed to use resumable microVM sandboxes), pooled CPU sandboxes |
| **Hard question synthesis** | ASearcher (fuzzing, context injection, 3 checks), MiroThinker repo has no data code | DeepDive KG walks, MiroVerse, LiteResearcher (mask the source), Tongyi SailorFog, OffSeeker DeepForge |
| **Difficulty filtering** | Not in the open trainers I read | Keep 1 to 7 correct of 8 (LiteResearcher), 2.5% to 50% (2510.15862), drop zero-variance groups (ASearcher, DR Tulu code) |

Failure modes reported: format collapse after negative-sample-heavy updates (Tongyi, "conservative filtering of negative samples"), answer avoidance under F1, longer-and-longer failing trajectories without length normalization (SFR-DR), training on poor-quality synthetic questions, retriever or API instability (DeepResearcher throttling, LiteResearcher's motivation).

### Benchmarks (what each is useful for)
- **BrowseComp** (OpenAI): 1,266 questions, "hard to find, easy to verify". Human trainers solved only 29.2%. Deep research scored 51.5% and GPT-4o with browsing 1.9%. Best-of-64 aggregation added 15 to 25 points. [paper/blog]
- **BrowseComp-Plus**: 830 BrowseComp queries over a fixed corpus of about 100K documents, so retriever and agent effects can be separated. Useful as a deterministic RL environment and test set (TRACE trains on it). [paper]
- **HLE**, **GAIA** (text subset commonly 103 or 165 validation items), **xBench-DeepSearch**, **Frames**, **WebWalkerQA**, **SimpleQA**, **FutureX** (live future-event prediction, harder to contaminate), **Seal-0**. The 2026 long-form set: Deep Research Bench, HealthBench, ResearchQA. Bio side: **BixBench** (53 scenarios, 296 open-answer questions; Claude 3.5 Sonnet 17%, GPT-4o 9% at launch) and its Edison follow-up BixBench3 (20 study-scale tasks, graded programmatically against 138 reference artifacts with dependency graph; page-reported top scores about 0.48 and run times of about 6.8 hours and 102M tokens per task). [paper/secondary]
- **Contamination note [inferred]:** benchmark questions leak into web search results; exclude benchmark-hosting domains from training search and use time-split or live benchmarks.

---

## 4. Scientific-discovery agents (high level)

- **Google AI co-scientist (arXiv 2502.18864):** multi-agent generate / debate / evolve with a tournament (Elo-style) over hypotheses; the paper credits **test-time compute scaling** and experimentally validated AML drug-repurposing candidates. It reports no gradient-based RL; the tournament is a self-play-like selection signal at inference time. [paper]
- **Sakana AI Scientist-v2 (arXiv 2504.08066):** agentic tree search with an experiment manager and VLM figure feedback; one of three fully AI-generated papers passed an ICLR workshop review bar. No RL. [paper]
- **FutureHouse / Edison:** PaperQA2 and Robin (multi-agent drug repurposing, not RL); **ether0** (24B, Mistral-Small-24B base) is the RL-trained one: SFT on reasoning traces, per-task GRPO "specialists" with verifiable rewards, rejection-sampled distillation into a generalist, then GRPO on all tasks. The abstract says 640,730 problems over 375 tasks; the model card lists 12 task groups. **Kosmos** (arXiv 2511.02824) runs up to 12 hours, about 200 agent rollouts, about 42K lines of code and about 1,500 papers per run, with 79.4% of statements judged accurate by scientists. A structured **world model** shares state between a data-analysis agent and a literature agent. No RL reported. [paper]
- **Biomni (Stanford; A1 agent, CodeAct, `biomni_e1` environment with a large data lake) and Biomni-R0 (8B and 32B; Qwen3-32B base):** per the MarkTechPost write-up (secondary), SFT on rejection-sampled Claude-4-Sonnet trajectories and then multi-turn RL on correctness and `<think>/<execute>/<solution>` format, async rollout scheduling, 64K context, evaluated on 10 tasks (rare disease diagnosis, GWAS variant prioritization, CRISPR delivery and others; Biomni-Eval1 has 433 instances) with 0.669 overall. I could not open the primary technical report (URL 404), so treat training specifics as unverified.
- **Pattern [inferred]:** The discovery systems that exist mostly rely on **inference-time search, tournaments and verification loops**. The ones that do RL (ether0, Biomni-R0) use closed-form or programmatic rewards. Nobody has yet shown RL on top of a long-horizon (hours, hundreds of tool calls) data-analysis agent such as Kosmos, because the outcome signal is expensive and the environment is not deterministic. BixBench3-style programmatic artifact grading is the closest ingredient.

---

## 5. What the code shows that the papers do not

1. **Search-R1's reward silently needs two `<answer>` tags.** `extract_solution` returns `None` unless there are at least two `<answer>` matches, because the decoded sequence includes the prompt, whose instruction contains `<answer> Beijing </answer>` as an example. If you change the prompt, the reward becomes 0 everywhere without an error. [code `qa_em.py`, `scripts/data_process/nq_search.py`]
2. **Search-R1 masks only the actor loss.** The loss mask is built after advantage computation, so GAE (PPO mode) and the critic see retrieved text. [code `ray_trainer.py` lines ~818 and ~854]
3. **DeepResearcher does not mask tool output (in the released code) and re-tokenizes the trajectory string.** Retokenization can change token boundaries versus what the policy sampled, which makes the training log-probs slightly off-policy. [code; I grep-verified and did not run it]
4. **ReCall gives zero reward to anything not ending in EOS**, so a length-truncated rollout is indistinguishable from a wrong answer, with a 0.1 floor only for well-formed wrong answers. [code `re_call.py`]
5. **R1-Searcher's reward is a separate HTTP service** that also logs every rollout to JSONL. Format rules there are very specific (answers over 10 words are penalized by -2, non-document Chinese characters are penalized). [code]
6. **ZeroSearch's "simulator" is told the gold answer.** It is not a faithful search engine; it is a teacher generating helpful or distracting documents. The curriculum variable named threshold is the *noise probability*. [code]
7. **ASearcher uses a very permissive trust region** (`eps_clip 0.4`, `behav_imp_weight_cap 5`, decoupled loss) to tolerate up to 4 policy versions of staleness, and removes the KL term entirely. Advantage broadcast makes each of up to 128 turns a short independent sequence, which is why 30K-token generations and 100+ turns are feasible without a 400K-token sequence. [code]
8. **ASearcher's context management is a hand-written prompt pipeline** (policy-written summaries, 25K-char history cap), not a learned policy. MiroThinker's `keep_tool_result: 5` is also a fixed rule. Learned context management exists in papers (SFR-DR, FoldGRPO), not in these repos. [code]
9. **DR Tulu's rubric RL discards zero-variance groups and keeps a per-prompt rubric buffer**, and its tool loop is non-blocking inside vLLM. Weights of the composite reward (0.5 / 0.2 / 0.2 / 0.1) and the existence of a span-level reward variant are not obvious from the abstract. [code]
10. **MiroThinker, WebThinker, II-Researcher and ether0 repos contain no RL trainer.** ether0 releases only reward functions and says to use NeMo-RL or TRL. [code]
11. **ether0's reward has anti-hacking machinery:** ring-system and "reasonable fingerprint" bloom filters, a counter-ion check (at most one fragment of 5 or fewer heavy atoms), bad-substructure screening with an optional `good_molecule_bonus`, "soft" training vs strict test scoring, and remote oracle servers (solubility, reaction predictors) behind `ether0.remotes`. [code `rewards.py`]

---

## 6. Infrastructure summary

| Project | Trainer | Rollout engine | Sync? | Environment | Scale |
|---|---|---|---|---|---|
| Search-R1 | veRL fork (FSDP) | vLLM | sync, lock-step turns | local retriever HTTP (BM25 / E5+ANN) | 8 GPUs, 3B-7B (README mentions 30B+ multinode) |
| ReCall | veRL 0.3 | vLLM | sync | search HTTP + code sandbox | 8 GPUs/node |
| R1-Searcher | OpenRLHF-RAG (Ray) | vLLM | sync | local Wikipedia index, remote reward server | 4 train + 4 ref GPUs, vLLM TP 2 |
| DeepResearcher | veRL fork (Ray) | vLLM | sync | live web via Serper/Bing + API reading agent | 8 GPUs (+ API cost) |
| ZeroSearch | veRL fork | vLLM + SGLang simulator | sync | LLM-simulated search | 4 GPUs + simulator GPUs |
| ASearcher | AReaL (FSDP, SGLang) | SGLang | **async** | live web (Serper + page access) or local | 6 nodes x 8 GPUs for 32B |
| DR Tulu | open-instruct (DeepSpeed ZeRO-3) | vLLM | async tool calls, sync steps | MCP server (live) | 1 to 2 nodes x 8 GPUs, 8B |
| MiroThinker | not released | n/a | streaming queue (paper) | Serper, Jina, E2B | "thousands" concurrent rollouts (paper) |
| LiteResearcher | not released in my reading | n/a | on-policy | local 32M-page web | 73.2M tool calls total |

---

## 7. Transfer to biology: RL-training a deep research agent for literature + databases + code + lab tools

### 7.1 Where the analogy to web deep research is strong
Web agents need hard-to-find, easy-to-verify questions, a cheap deterministic tool backend and an outcome check. Biology has **more structured verifiers than the web**: accessions, SMILES, sequences, numeric assay values and curated relationships can all be checked by a program. The main differences are (a) tool results can be very long (FASTA, PDB, tables), (b) some tools are compute oracles (folding, docking, MD, simulators) rather than lookups, and (c) the true oracle (wet lab) is slow, noisy and delayed.

### 7.2 A concrete design (proposal)

**Tools (MCP-style, like DR Tulu).** PubMed / Europe PMC / bioRxiv full text, UniProt, RCSB PDB and AlphaFold DB, ChEMBL / PubChem / BindingDB, ClinVar / gnomAD / GWAS Catalog, GEO / SRA metadata, STRING / Reactome / KEGG, BLAST / MMseqs2 / HMMER, a Python sandbox with RDKit / Biopython / scanpy / statsmodels (as in Biomni's CodeAct and SFR-DR's stateless Python), then expensive oracles: ESMFold / Boltz / AlphaFold-style folding, Vina / GNINA docking, ADMET predictors. Lab tools: protocol generator, plate-layout checker, a liquid-handler or cloud-lab API (dry-run in training).

**Environment = versioned snapshots (LiteResearcher idea, inferred transfer).** Freeze UniProt, ChEMBL, PDB metadata and a PubMed/PMC subset at a date, index with a dense retriever, and cache every tool call by content hash. This makes rollouts deterministic, free per call and replayable, and it lets you **time-split**: ask about facts that appear only after the snapshot or after the base model's cutoff for evaluation, and block benchmark domains. Add API fault injection (timeouts, empty results, rate limits) because DeepResearcher and Tongyi both report that a stable sandbox with retries is critical.

**Data: programmatic hard-question synthesis [partly novel].**
- *Cross-database KG walks with obfuscation (DeepDive / MiroVerse transfer):* walk UniProt to PDB to ChEMBL to Reactome to disease edges and hide entity names behind constraints ("a human kinase with a co-crystal structure released after 2019 whose ligand series has a reported IC50 below 10 nM and which is annotated in a pathway also containing X"). The **ground truth is the result of the structured query run at synthesis time**, so no LLM labeling error. [Novel as a bio-specific deep-research RL environment; I am not aware of an open one.]
- *Paper-reproduction tasks (BixBench / BixBench3 transfer):* take a GEO dataset and the published analysis, ask for a specific statistic, count or ranked gene list, and grade numerically with tolerance. Use the dependency-graph partial credit idea from BixBench3.
- *Historical-campaign replay [novel]:* build environments from time-ordered medicinal-chemistry or protein-engineering campaigns (ChEMBL by document, deep mutational scans). The agent proposes the next round; the reward comes from the recorded outcomes. This is a "counterfactual lab" that costs nothing and has real labels (caveat: only for compounds that were actually made).
- Filter with the pass-rate rule from LiteResearcher (keep 1 to 7 correct of 8) and drop zero-variance groups (ASearcher, DR Tulu).

**Rewards, in tiers.**
1. *Exact / set-F1 on identifiers* (UniProt accessions, PDB IDs, InChIKey equality, Reactome IDs) and *numeric with tolerance*. Cheap, deterministic, no judge.
2. *Code-checked oracle rewards* (ether0-style): RDKit validity, substructure and property constraints, folding TM-score or pLDDT thresholds against a known structure, docking-score thresholds. Copy ether0's anti-hacking: ring / fingerprint sanity filters, a "soft" shaping mode in training and strict scoring at test, and a bonus only for passing a "good molecule" screen.
3. *Open-ended literature synthesis*: rubric reward with **evolving rubrics (RLER)** seeded by expert-written rubrics from review articles, plus CaRR-style gating so rubric credit is only given when the outcome is right and each claim is supported by a cited, fetched passage.
4. *Process shaping*: IGPO-like information-gain reward is natural here because the gold answer is a short structured object (a gene, an accession, a SMILES): reward turns that raise the reference model's likelihood of the gold answer (IGPO) or the TD-version in TRACE. Both need only a gold answer, no judge.
5. *Cost penalties*: a small per-call price on expensive oracles and a penalty for redundant calls, analogous to DR Tulu's search-turn term and Kimi's gamma decay.

**Ideas I believe are new in this form [novel, unverified against all literature].**
- **Identifier-existence and record-entailment verifier.** Every UniProt / PDB / PMID / DOI / gene symbol / compound ID in the final report is checked against the snapshot (does it exist, does the record say what the claim says, e.g. organism, resolution, assay type, value within tolerance). Hallucinated identifiers get a negative reward. It is a cheap, dense, hard-to-game version of CaRR's citation check, and biology makes it possible because the "web" is a set of curated registries.
- **Tool-trace provenance reward**: a claim only counts if the cited ID was actually retrieved in the trajectory (checked from the logged tool results, not from the model's text). This closes the "cite from memory" loophole.
- **Privileged-information simulators for slow assays (ZeroSearch transfer).** Train a model (or a regressor trained on historical assay data) that returns an assay-like result conditioned on the hidden true value, with a noise curriculum (`start=0 -> end=0.5`, base-4 exponential) so the agent first learns the protocol of using lab tools and then learns to cope with noisy, conflicting results. Treat it as a pre-training environment and finish against real or replayed data to avoid sim bias (ZeroSearch itself needs a real-engine eval; SSRL shows sim-to-real transfer exists in the search case).
- **Oracle-cost gating inside the group.** Score all 8 rollouts with a cheap proxy (a surrogate predictor, a fast docking mode) and spend the expensive oracle (full docking, folding, MD or a lab call) only on the top and bottom ones, since only ordering inside the group matters for GRPO advantages. Reward caches keyed by sequence / SMILES amortize repeats. [Inferred design; I did not find a paper doing exactly this.]
- **Tree-structured branching at "lab-call" decision points** (ARPO / AT2PO transfer): branch rollouts from high-entropy turns right after a tool result, so expensive shared prefixes (retrieval, folding, long analyses) are computed once and only the divergent decision is re-sampled.

**Long-horizon machinery for bio specifics.**
- Keep a *large-artifact-by-reference* convention: sequences, structures, count matrices stay in the sandbox filesystem, and the model sees a handle plus a head or summary. This pairs with MiroThinker's "keep last K tool results" rule and with SFR-DR's learned `clean_memory`; learn it end to end (FoldGRPO / ReSum-GRPO) rather than hard-coding it as ASearcher does. [partly inferred]
- Mask all tool-result tokens, including files and code outputs, in the loss. Use per-turn samples with advantage broadcast (ASearcher) if sequences exceed the training context.
- Length-normalize advantage and filter malformed trajectories (SFR-DR), and keep an error-tolerant rollout mode so a malformed call does not end the episode (2510.15862).
- Use an async trainer (AReaL / slime-style / open-instruct) with a bounded staleness and a decoupled importance-weighted loss (ASearcher's `behav_imp_weight_cap`), because oracle latency in biology is far more variable than web latency.

**Training schedule (inferred from what worked).** (1) Cold-start SFT on trajectories from a strong model with rejection sampling on the verifiers (Biomni-R0, ether0 and MiroThinker all do some version). (2) Per-skill RL "specialists" (database retrieval, sequence-analysis code, chemistry oracle design), then rejection-sample and distil into one generalist, then RL on the mix (ether0's loop). (3) Final stage with rubric-rewarded long-form questions. Use a small judge or rubric model (gpt-4.1-mini in DR Tulu; Gemini-Flash-Lite in 2510.15862) so judge cost does not dominate.

**Risks specific to biology [inferred].** Reward hacking on docking or surrogate scores (the model finds adversarial inputs); literature that is wrong or retracted (add retraction metadata and a "conflicting evidence" rubric); data leakage between training questions and benchmarks like BixBench / LAB-Bench-style sets; dual-use review for generative design tools; and judge bias toward fluent but unsupported claims, which the provenance reward is meant to counter.

---

## 8. Sources

Code actually cloned and read (shallow clones, now deleted):
- https://github.com/PeterGriffinJin/Search-R1 : `search_r1/llm_agent/generation.py`, `verl/trainer/ppo/ray_trainer.py`, `verl/utils/reward_score/qa_em.py`, `train_grpo.sh` [verified]
- https://github.com/Agent-RL/ReCall (ReSearch archived branch) : `src/verl/utils/reward_score/re_call.py`, `scripts/train/*.sh` [verified]
- https://github.com/RUCAIBox/R1-Searcher : `train/reward_server_qwen_zero.py`, `train/reward_server_llama_stage1.py`, `OpenRLHF-RAG/`, `scripts/qwen_grpo.sh` [verified]
- https://github.com/GAIR-NLP/DeepResearcher : `train_grpo.sh`, `scrl/llm_agent/generation.py`, `scrl/handler/`, `verl/utils/reward_score/format_and_f1.py` [verified]
- https://github.com/Alibaba-NLP/ZeroSearch : `llm_agent/generation.py`, `train_grpo.sh`, `README.md` [verified]
- https://github.com/inclusionAI/ASearcher : `ASearcher/train/asearcher_reasoning.py`, `ASearcher/train/reasoning_agent.py`, `ASearcher/configs/asearcher_web_qwq.yaml`, `README.md` [verified]
- https://github.com/RUC-NLPIR/WebThinker : README, `scripts/` (no RL code) [verified]
- https://github.com/MiroMindAI/MiroThinker : README, `apps/miroflow-agent/conf/agent/mirothinker_1.7_keep5_max200.yaml`, `src/llm/base_client.py` [verified; no RL code]
- https://github.com/Intelligent-Internet/ii-researcher : README, tree (no RL code) [verified]
- https://github.com/rlresearch/dr-tulu : `rl/open-instruct/train_dr_tulu.sh`, `open_instruct/grpo_fast_rubric.py`, `open_instruct/search_rewards/*`, `open_instruct/tool_utils/tool_vllm.py` [verified]
- https://github.com/Future-House/ether0 : `src/ether0/rewards.py`, `models.py`, README [verified; no trainer]

Papers, blogs and model cards (fetched pages; summaries are tool-generated, numbers as reported):
- https://arxiv.org/pdf/2511.11793 : MiroThinker v1.0 training, reward, context rule, infra [paper]
- https://arxiv.org/abs/2511.19399 : DR Tulu / RLER [paper]
- https://arxiv.org/html/2509.06283v1 : SFR-DeepResearch [paper]
- https://arxiv.org/abs/2510.11967 : Context-Folding [paper]
- https://arxiv.org/abs/2509.13313 : ReSum [paper]
- https://arxiv.org/html/2509.10446v1 : DeepDive [paper]
- https://arxiv.org/html/2604.17931v1 : LiteResearcher [paper]
- https://arxiv.org/html/2601.18467v2 : OffSeeker [paper]
- https://arxiv.org/abs/2603.20278 : OpenResearcher (page unreadable; title only)
- https://arxiv.org/html/2510.15862v4 : Rethinking RL deep research design [paper]
- https://arxiv.org/html/2602.19526 : How to train your deep research agent (Search-R1++) [paper]
- https://arxiv.org/abs/2505.15117 : Search-R1 empirical study [paper]
- https://arxiv.org/abs/2508.10874 : SSRL [paper]
- https://arxiv.org/abs/2507.19849 : ARPO [paper]
- https://arxiv.org/abs/2505.10978 : GiGPO [paper]
- https://arxiv.org/html/2510.14967 : IGPO [paper]
- https://arxiv.org/pdf/2601.04767 : AT2PO [paper]
- https://arxiv.org/html/2607.13988v1 : TRACE [paper; only the fetched summary read]
- https://arxiv.org/html/2601.06021 : CaRR / C-GRPO [paper]
- https://arxiv.org/abs/2606.17029 : DeepRubric [paper abstract]
- https://arxiv.org/html/2606.01091 : DR-Rubric [paper]
- https://arxiv.org/abs/2609.34296 : Dr.Credit [paper abstract]
- https://arxiv.org/html/2606.15367v1 : S1-DeepResearch (fetch rate-limited; not read)
- https://arxiv.org/html/2506.22760v1 and https://huggingface.co/janhq/Jan-v1-4B : Jan-nano / Jan-v1 [paper/model card]
- https://huggingface.co/Intelligent-Internet/II-Search-4B : II-Search recipe [model card]
- https://moonshotai.github.io/Kimi-Researcher/ : Kimi-Researcher (REINFORCE, gamma-decay, partial rollouts 1.5x, context management) [blog]
- https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/ : Tongyi DeepResearch (CPT, SFT, GRPO changes, simulated Wikipedia env) [blog]
- https://www.kimi.ai/blog/kimi-k3 : Kimi K3 (few RL details) [blog]
- https://openai.com/index/introducing-deep-research/ and https://openai.com/index/browsecomp/ : deep research (end-to-end RL on browsing tasks), BrowseComp [blog]
- https://texttron.github.io/BrowseComp-Plus/ : BrowseComp-Plus [benchmark page]
- https://arxiv.org/html/2503.00096v1 and https://advances.edisonscientific.com/research/bixbench3-benchmarking-study-scale-computational-biology/ : BixBench / BixBench3 [paper/blog]
- https://arxiv.org/abs/2506.17238 and https://huggingface.co/futurehouse/ether0 : ether0 [paper/model card]
- https://arxiv.org/abs/2511.02824 : Kosmos [paper]
- https://arxiv.org/abs/2502.18864 : Google AI co-scientist [paper]
- https://arxiv.org/abs/2504.08066 : AI Scientist-v2 [paper]
- https://github.com/snap-stanford/Biomni , https://huggingface.co/biomni/Biomni-R0-32B-Preview , https://www.marktechpost.com/2025/09/04/biomni-r0-new-agentic-llms-trained-end-to-end-with-multi-turn-reinforcement-learning-for-expert-level-intelligence-in-biomedical-research/ : Biomni / Biomni-R0 [repo / model card / secondary news; training details unverified]
- https://arxiv.org/html/2607.19044v1 : RLVR for molecular generation (GRPO, RDKit rewards) [paper]
- https://huggingface.co/blog/sergiopaniego/rl-environments-2026 : sandbox-per-rollout survey [secondary]
- https://arxiv.org/abs/2501.10120 : PaSa (RL paper-search agent; found via search, not read in detail)

Not verified / gaps: S1-DeepResearch, OpenResearcher, WebExplorer and StepSearch were not read; MiroThinker 1.5/1.7/H1 RL details; Biomni-R0 primary technical report; Tongyi/Kimi internals (other tracks); any wet-lab-in-the-loop RL for LLM agents (I found none, which is itself a finding, but my search was not exhaustive).
