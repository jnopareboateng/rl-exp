# Track 12: RL for biology and chemistry with LLMs and generative models, state of the art (as of 2026-10-05)

Scope: what has already been done with RL (RLVR, GRPO/PPO/REINFORCE, DPO, self-play, active-learning loops) on LLMs and generative models in biology and chemistry, what is open-source and how it is built, and which frontier-lab RL techniques have not yet reached which biology domains.

Evidence legend: **[code]** means I read the file in a cloned repo. **[paper]** means I read the paper or preprint through the fetch tool, which returns a model-written summary of the page, so numbers are only as good as that summary. **[secondary]** means a blog, news item or search snippet. **[memory]** means from prior knowledge and not re-verified here. **[title-only]** means I saw the title or abstract line and nothing more.

Limits of this survey: the shared web-search budget (200 calls) and the fetch budget ran out partway through. Several 2026 arXiv items (for example 2609.39048, 2604.07669, 2604.23061) are therefore title-only. The Biomni-R0 technical report URL returned 404, so Biomni-R0 details rest on secondary reports. A few leads were never read at all: Genome-Bench (the arXiv ID I guessed was wrong), CellDuality (PMC page blocked), the ProteinDPO paper text, and the Saturn and GFlowNet papers.

---

## 1. TL;DR

- **RLVR works in biology and chemistry wherever a cheap deterministic or learned verifier exists.** The clearest case is ether0 (24B, chemistry) [paper][code]. Others: Cell-o1 (cell-type puzzles), BioReason-Pro (GO-term F-max reward), Pro-1 (Rosetta energy reward), LLMol and RePO (RDKit/Tanimoto rewards), C2S-Scale (negative MSE and Kendall-tau rewards). Almost all of it is single-turn or short-horizon, uses GRPO or a GRPO variant, and runs on small models (0.5B to 32B; ether0 is the exception with a 384-H100 generalist stage).
- **"Soft verification" is the genuinely new biology-specific idea.** rbio (CZI) uses a learned virtual-cell model as the reward source. Its MLP/Transformer-based verifier gives a confidence in [0,1], which is used directly as the answer reward. This lets RL run where wet-lab labels are scarce [paper][code]. Nothing equivalent has yet been done for proteins or genomes with LLM policies.
- **Protein design RL is mature but small-scale and mostly not LLM-based.** Policies are ProGen2 (via InstructPLM), ProtGPT3/ZymCTRL, ESM-IF, ProteinMPNN or diffusion inverse-folding models, trained with GRPO, weighted DPO, REINFORCE, DDPO or RAFT against ESMFold TM-score, a fast ΔΔG proxy, or experimental assays. Wet-lab closure exists for one case only: ProtRL, with an EGFR binder hit-rate of 3/6 then 9/20, including a 27 nM binder [secondary].
- **Genomics RL is the thinnest area.** RL on regulatory DNA exists (TACO uses REINFORCE, Ctrl-DNA uses constrained GRPO-style with an Enformer reward). Evo 1/2 themselves are used with SFT, prompting and filtering, not RL: the Evo-designed phage work explicitly uses no RL [paper]. BioReason applies GRPO only to a DNA-encoder plus small-LLM hybrid.
- **Agents and labs are not RL-trained in the headline systems.** Kosmos and Robin are prompted multi-agent systems. I could find no RL in either [paper]. The OpenAI/Ginkgo 36,000-reaction closed loop used no retraining [secondary]. The only RL-trained biomedical agents are Biomni-R0 (multi-turn RL with async tool rollouts, 64k context, algorithm unpublished), ATHENA-R1 and Edison's NeMo-RL/Gym step-level GRPO on BixBench-style Jupyter tasks [secondary].
- **A recurring empirical finding: RL sharpens more than it discovers.** "From Supervision to Exploration" [paper] finds pass@k gains with reduced diversity/novelty across AMP, kinase, antibody and inverse-folding tasks. Gains are capped by reward accuracy and task headroom. ProteinZero needed embedding-space diversity regularization to avoid mode collapse [paper].
- **Biggest gaps (details in section 6):** no on-policy distillation anywhere in bio/chem; no async or off-policy RL for slow structure/docking oracles; generative/rubric reward models only for medicine and ML research plans, not experimental design or protocol critique; no RL on lab-automation code even though a simulator verifier already exists in Biomni; no long-horizon "deep research" RL for computational biology; self-play only for inverse folding.

---

## 2. Per project sections

### 2.1 FutureHouse / Edison Scientific

**ether0** (chemistry reasoning, Mistral-Small-24B base) [paper][code]
- Repo: https://github.com/Future-House/ether0. Weights: https://huggingface.co/futurehouse/ether0. Benchmark: https://huggingface.co/datasets/futurehouse/ether0-benchmark. Paper: arXiv 2506.17238.
- The repo contains reward functions, prompts and data models only. The README says plainly it does **not** contain training code and points to NeMo-RL or TRL.
- Pipeline (README and paper): (1) long-CoT SFT warm start on 14,021 filtered DeepSeek-R1 traces (summarized by Mistral-Small); the model card says these were "mostly incorrect" traces, i.e. the SFT is only to elicit format and reasoning style. (2) Seven task-group "specialist" GRPO runs on 24 to 72 H100s each. (3) Rejection-sample specialist outputs, filtering for correctness and quality, giving about 186k sequences. (4) SFT the base model again on the pooled traces to make a generalist. (5) Final all-task GRPO on 384 H100s for about 4 days. Then safety/refusal post-training.
- Data: 640,730 experimentally grounded problems across 375 task types (ChEMBL, COCONUT, ORD, PubChem and others). Tasks include reaction prediction, retrosynthesis, solubility edit, formula-constrained design, functional-group design, scent, BBB, receptor activity.
- Objective: GRPO with group-normalised advantage, PPO clipping and a KL term. Reward is `format_reward × accuracy_reward`, mostly binary, with soft variants (0.5 partial credit for formula; Tanimoto for reaction products; KDESol regression for solubility).
- Advantage-based curriculum: about 90% of groups initially have identical rewards (zero advantage), so a buffer of recently non-trivial problems is resampled to cut wasted rollouts.
- Known failure modes (paper): loses chat, instruction-following and tool-calling after intensive RL; weak on inorganic/crystal; reasons poorly from common names (model card).
- Code facts:
  - `src/ether0/rewards.py`: `EVAL_FUNCTIONS` maps task type to verifier. Open-ended design tasks go through `is_reasonable_molecule` (sanitize, counter-ion rule of at most one fragment of 5 or fewer heavy atoms, `is_reasonable_ring_system`, `is_reasonable_fp`, backed by `rings.bloom` and `fingerprints.bloom`).
  - `accuracy_reward(..., good_molecule_bonus=...)` adds a bonus only when the molecule contains none of the `BAD_SMARTS_PATTERNS` (peroxide, nitro, nitroso, hydrazine, 3 or more adjacent N, 7 or more aliphatic carbon chain, and similar). This is the anti-reward-hacking patch, applied in the last RL stage.
  - The `soft` flag exists for training only; `accuracy_reward` raises if `soft and test`.
  - `packages/remotes/src/ether0/server.py` is a FastAPI reward server with bearer-token auth. It wraps three exotic oracles: a Molecular Transformer (OpenNMT, USPTO480k checkpoint) as a forward-reaction oracle, KDESol for solubility, and molbloom for purchasability of reactants. The `rxn_forward` reward demands that every reactant be purchasable (or at most 4 heavy atoms), that the product not appear among reactants, and that the model's own predicted product match the target.
  - Note: Edison, which spun out of FutureHouse, has since used NeMo-RL/Gym for scientific agents (see below), which suggests that is the stack, but I did not see ether0 itself tied to NeMo-RL beyond the README mention.

**Aviary** (arXiv 2412.21154) [paper summary]: an open gymnasium for language agents with five environments, three biology ones (DNA construct manipulation for molecular cloning, literature QA, protein stability engineering). It treats agents as language-grounded POMDPs and shows that open-source non-frontier LLM agents can match or exceed frontier agents and humans on several tasks at up to 100x lower inference cost, using online training and scaled inference compute.

**Edison + NVIDIA NeMo-RL/Gym** [secondary: NVIDIA developer blog "How to Train Scientific Agents with Reinforcement Learning"]: Edison trains scientific agents with GRPO modified to operate on individual steps instead of full trajectories. Rewards are verifiable checks such as code execution. Test bed is a Jupyter-based bioinformatics data-analysis environment (BixBench-like). The blog lists async RL, on-policy distillation and end-to-end FP8 RL as supported by the latest NeMo-RL release, but gives no results or hardware for Edison's runs.

**Kosmos** (arXiv 2511.02824; https://advances.edisonscientific.com) [paper summary]: structured world model shared by a data-analysis agent and a literature agent; about 200 agent rollouts per 12-hour run (20 cycles), 42,000 lines of code, about 1,500 papers; 79.4% of statements judged accurate; seven discoveries. The "How we built Kosmos" post covers sandboxing only (Kubernetes, gVisor, JuiceFS). **No RL or model training is described for Kosmos** [paper]. Stated limitations: "rabbit holes" chasing statistically significant but irrelevant findings, and likely inversion of the scaling trend at excessive depth.

**Robin** (arXiv 2505.13400): multi-agent LLM system for hypothesis generation, experiment planning and analysis (ripasudil for dry AMD). Prompted agents (Crow/Falcon/Finch), no RL training found.

### 2.2 CZI rbio1 (soft verification with biology world models)
- Repo: https://github.com/czi-ai/rbio. Files read: `training/training.py`, `training/rewards.py`, `training/README.md`. Paper: bioRxiv 10.1101/2025.08.18.670981 (NeurIPS 2025).
- Base model Qwen2.5-3B-Instruct, GRPO. The paper reports 100k steps on 8 H100s for about 10 days, batch 4, 4 generations, lr 5e-6 [paper]. Task: PerturbQA (CRISPRi differential expression, K562/RPE1/HepG2/Jurkat) as binary classification.
- Three verifier families: **rbio-VCM** (virtual cell models: Transcriptformer PMI co-expression; 2-layer MLP with ESM gene embeddings trained on three cell lines and tested on the fourth), **rbio-Prior** (Gene Ontology: ROUGE, keyword overlap, or model likelihood), **rbio-Hard** (experimental labels). Verifier outputs are min-max normalised around a threshold into [0, 0.5] and [0.5, 1]. Multiple verifiers compose additively.
- Headline: soft-verified models (F1 about 0.65 to 0.68) approach models trained on experimental labels (0.67 to 0.75), then reach SOTA on PerturbQA with chain-of-thought additions (0.74 to 0.79).
- Code facts [code]: the released training script is explicitly "a different implementation released for illustration" than the paper. It uses TRL `GRPOTrainer` with `reward_funcs=compute_simple_reward`. The reward is `format + 2.0*answer + mention`:
  - `answer` is the MLP verifier's **class confidence** for the matching label (`reward_answer_against_label` returns the confidence, not 0/1);
  - `format` is a mean of 11 binary structural checks (tag closure, single answer, at most 3 think blocks, 200-word limit, and similar);
  - `mention` is the fraction of target gene keywords that appear inside `<think>`.
  The dataset is generated by a streaming generator that labels examples with the MLP (`create_mlp_labeled_dataset_generator`). Note the mention reward could be gamed by keyword dumping, which the 200-word cap partly limits (inferred).

### 2.3 BioReason (DNA-LLM + reasoning RL) and BioReason-Pro
- Repo: https://github.com/bowang-lab/BioReason (NeurIPS 2025; arXiv 2505.23579). Files read: `sh_grpo.sh`, `train_grpo.py`, `bioreason/trainer/grpo_trainer.py`, `bioreason/trainer/grpo_config.py`, `bioreason/dna_modules/nucleotide_module.py`.
- Architecture: a DNA foundation model (Nucleotide Transformer or Evo 2, selectable via `--dna_is_evo2`) is projected into Qwen3 (1.7B and 4B) as DNA pad tokens. SFT then GRPO.
- RL config in `sh_grpo.sh` [code]: Qwen3-1.7B + NT-500M, LoRA r16/alpha32, 2 GPUs (SLURM, A100), vLLM in `colocate` mode at 0.3 memory utilisation, 8 generations, max completion 800 tokens, lr 1e-5 cosine, **beta 0.0** (no reference model), temperature 1.0, 1000 steps, DeepSpeed ZeRO-2.
- The trainer is a fork of TRL's GRPOTrainer; the config default `loss_type="dr_grpo"` and `scale_rewards="group"`; `epsilon_high` supports DAPO-style clip-higher.
- Rewards: `xmlcount`, `soft_format`, `strict_format`, `concise` (0.5 if the answer is 10 words or fewer), and `correctness` (2.0 if the gold answer string is a **substring** of the extracted answer). The substring match is a weak verifier. The `concise` reward offsets it, but this is a likely hacking surface (inferred).
- Tasks: KEGG-derived disease pathway reasoning (reported 86% to 98% accuracy) and ClinVar variant-effect prediction.
- **BioReason-Pro** (bioRxiv 2026.03.19.712954; https://github.com/bowang-lab/BioReason-Pro) [paper summary]: ESM3-1B encoder + GO graph encoder + Qwen3-4B. RL is **GSPO** with Dr.GRPO/DAPO modifications, LoRA r16, lr 3e-5, small KL, 24 rollouts per protein, 8 H100s. Reward is the weighted F_max over GO terms parsed from the reasoning trace by regex and propagated through the GO hierarchy (CAFA5 style), i.e. the evaluation metric is the reward. About 133k proteins, 130k GPT-5 synthetic reasoning traces for SFT. Experts preferred its output over UniProt entries 79% of the time (SFT variant). This is the first clean "evaluation metric as reward" case for protein function.

### 2.4 Biomni (Stanford) and Biomni-R0
- Repo: https://github.com/snap-stanford/Biomni (Science paper 2026 [secondary]). Files read: `biomni/eval/biomni_eval1.py`, `biomni/tool/lab_automation.py`, README.
- Biomni-A1 is a ReAct/CodeAct-style agent over about 150 tools, about 59 databases and many software packages. It contains **no RL training code**. A grep for GRPO, verl and reward finds only unrelated tool code.
- **Biomni-R0** (8B and 32B; 32B on Qwen3-32B) [secondary: HF card, MarkTechPost; the technical report URL 404'd]: two phases. (1) SFT on Claude-4-Sonnet trajectories obtained by rejection sampling. (2) End-to-end multi-turn RL "using the Biomni-E1 environment scaffolding", with correctness and format rewards, **asynchronous rollout scheduling** to hide tool-execution latency, and 64k context. Scores reported 0.669 (32B) vs 0.346 baseline, 0.588 (8B); rare-disease diagnosis 0.03 to 0.67; GWAS variant prioritisation 0.16 to 0.74. The RL algorithm is not named in anything I could read.
- Code facts [code]: `BiomniEval1.evaluate()` returns a scalar named `reward` computed by `_compute_reward`: exact match on letters or gene symbols, OMIM ID equality for rare disease, set intersection for patient causal gene. The public evaluator is effectively a ready-made RLVR reward for 10 tasks (433 instances). It is small and exact-match only.
- `tool/lab_automation.py::test_pylabrobot_script` validates a generated PyLabRobot script in simulation (ChatterboxBackend, optional tip and volume tracking, import validation, timeout, error list). It is a simulator-based pass/fail verifier for liquid-handler code and is not being used as an RL reward anywhere I found.

### 2.5 Evo 2 and genomic LMs
- Evo 2 (Nature 2026, s41586-026-10176-5; 40B, 9.3T bp) is a pretrained model; I found no RL post-training of Evo 2 in the paper or follow-ups. From memory [unverified here], the Evo 2 paper demonstrates design by guided beam-search against Enformer/Borzoi scoring, which is inference-time search and not RL.
- The Evo-designed phage genomes work (bioRxiv 2025.09.12.675911) [paper summary]: SFT of Evo 1/2 on about 15k Microviridae sequences, prompt-length and temperature tuning (0.7 to 0.9), then **tiered rule filters** (length 4 to 6 kb, GC 30 to 65%, homopolymer, at least 60% spike identity to ΦX174, less than 95% AA identity to natural proteins, architecture constraints). 285 of 302 synthesized, 16 viable. Explicitly no RL. This filter cascade is a ready-made verifier.
- RL on regulatory DNA:
  - **TACO** (ICLR 2025, arXiv 2503.07981) [paper]: REINFORCE on an autoregressive DNA model with TFBS-aware reward shaping (SHAP-derived), hill-climb replay buffer and entropy regularization. 100 rounds of 256 sequences, lr 5e-4 (yeast) and 1e-4 (human).
  - **Ctrl-DNA** (NeurIPS 2025, arXiv 2505.20578) [paper]: constrained MDP, Lagrangian primal-dual, GRPO-inspired batch advantages, clipped surrogate with KL, Enformer reward models per cell type, HyenaDNA policy, TFBS-correlation regulariser.
  - **DRAKES** (ICLR 2025, arXiv 2410.13643; https://github.com/ChenyuWang-Monica/DRAKES): reward backpropagation through discrete diffusion for DNA enhancers and protein stability.
- Note: none of these use frontier-scale genomic LMs (Evo 2, GenomeOcean) as the policy.

### 2.6 Protein LMs with RL or DPO
- **ProtRL / DPO_pLM** (Dec 2024, arXiv 2412.12979; https://github.com/AI4PDLab/ProtRL). Files read: `README.md`, `example/tiny-llama/GRPO/src/pLM_GRPO.py`, `example/tiny-llama/wDPO/src/pLM_weigtedDPO.py`, `src/ProtRL_Trainer/*`.
  - Algorithms: GRPO, weighted DPO (optional IRPO regulariser) and REINFORCE with entropy bonus, built as subclasses of TRL trainers. The loop is generate → score externally → offline update → repeat (`ProtRL.sh`: `seq_gen.py`, `dataset_gen.py`, `train.py`). Models: ProtGPT3-112M, ZymCTRL and others. The README warns that without the direction token and BOS the model emits `MMMKKK...` junk.
  - **wDPO loss** [code]: `pi_ratio = beta * (mean-token logp − mean-token ref logp)` per sequence; `weights = softmax(advantages)` over the group; `loss = cross_entropy(pi_ratio, weights)`. This is a listwise preference loss against a reward-softmax target.
  - Code oddity [code]: in the tiny-llama GRPO copy, `advantages` is computed with per-completion std, sliced per process, and then **overwritten** by `rewards - mean_grouped_rewards` and a second normalisation. The sliced version is dead code. I did not check whether the main `src/ProtRL_Trainer` copy has the same issue.
  - Results [secondary: Adaptyv blog]: carbonic anhydrase fold-shift to over 95% alpha-topology in 6 iterations; EGFR binders 3/6 (fine-tune round) then 9/20 (ProtRL round), best 27.4 nM; sequence identity fell from 88.6% to 73.1% (diversity increased).
- **ProteinZero** (arXiv 2506.07459, v5 in 2026; https://github.com/ziwenwang28/ProteinZero). Files read: `run_proteinzero_grpo.sh`, `train_proteinzero_grpo.py`.
  - Policy: InstructPLM (ProGen2-xlarge decoder, 6.57B, conditioned on backbone via ProteinMPNN encoder); LoRA r16, about 0.5% of parameters trained.
  - Online GRPO and a RAFT variant. Rewards: **designability** (ESMFold → TM-align TM-score to the target backbone) and **stability** (a fast self-derived ΔΔG proxy from backbone-conditioned likelihood normalised by an unconditional prior; PCC about 0.6 vs experiment, 236 to 760x faster than physics). Plus an embedding-space diversity penalty (`alpha_diversity` 0.05).
  - Launcher defaults [code]: 64 generations per group, KL beta 0.1, clip epsilon 0.2, lr 1e-5, grad-accumulation 1024/num_GPUs, `--use_vllm False` (HF generate with a `ForceExactLength` logits processor), strict fp32, `scale_rewards` selectable between group-std (GRPO) and none (Dr.GRPO). Rewards are **min-max normalised within each group per component** before weighted summation (`_compute_combined_reward_for_group`). The README announces **PARPO**, a Pareto-targeting algorithm for ProteinZero V2 (paper "in final stages", teaser checkpoints only).
  - Reported: 90.13% design success for 0 to 150 residues vs 84.45% baseline; a full RL run takes under three days on one 8-GPU node. The authors found sequence-level diversity penalties destabilised training, and embedding-level penalties fixed mode collapse.
- **ProteinDPO** (Nature Methods 2026; https://github.com/evo-design/protein-dpo). Preference-optimised ESM-IF1 on megascale stability data. The repo contains inference only (`sample.py`, `score.py`, `score_complex.py`, weights on Zenodo). No training script is released despite the README's "training code" phrase [code].
- **RL-DIF** (arXiv 2410.17173) [paper]: DDPO on a D3PM/PiFold inverse-folding model with sc-TM reward; ESMFold on 20 GPUs on Kubernetes; "foldable diversity" 29% vs 23%.
- **Symmetric self-play online preference optimisation** (SSP, bioRxiv 2026.03.26) [paper]: two policies, one rewarded on self-consistency (scTM) and one on predicted confidence (pTM), sharing a sample pool, with separate preference models; ESM3-based SSP gets pTM 0.727 and scTM 0.838 on BoltzGen-419.
- **MAProt** (bioRxiv 2026.01) [paper]: DPO per agent (ProteinMPNN, ESM, SaProt) on experimental fitness, then consensus-built preference pairs and logit summation. Megascale stability success +14.1%, GFP 0.88, antibody design 32.1% success.
- **Pro-1** (Michael Hla, 2025, https://huggingface.co/mhla/pro-1) [secondary]: Llama-3.1-8B reasoning model; GRPO against Rosetta REF2015 stability (binary improved/not reward, found more stable than delta rewards), plus an LLM-judge "creativity" reward and a specificity reward. QLoRA; 70B GRPO abandoned because FSDP all-gather during generation was the bottleneck. No wet-lab validation. The author flags overfit to REF2015.
- **Proteo-R1** (arXiv 2605.02937, 2026) [paper]: LLM "understanding expert" feeding an AlphaFold3-style generator for antibody CDR design, trained with a three-stage supervised curriculum. No RL.
- **"From Supervision to Exploration"** (arXiv 2510.01571) [paper]: DPO, PPO and GRPO across InstructPLM, Amphion-SFT, ESM-2 and ProGen2 on inverse folding, AMP, kinase and antibody mutation tasks. GRPO is best; RL mainly sharpens; gains depend on task headroom, reward accuracy and policy capacity.
- Other (title-only): "Structure-aware RL for Protein Directed Evolution" (arXiv 2609.39048, the fetch was rate-limited), ThermoRL (2507.18816), "Accelerating virtual directed evolution via RL" (bioRxiv 2025), EvoPlay/self-play protein engineering (Nature MI 2023 [memory]), RLXF for CreiLOV brightness [secondary: review 2511.21476].
- Review (arXiv 2511.21476) [paper summary] names the gap directly: "the biggest limiting factor is the difficulty of selecting good scoring metrics"; unclear whether RL finds novel solutions or recombines training patterns.

### 2.7 Structure-based design: diffusion and AlphaFold-style models
- **No public RL post-training of RFdiffusion, Boltz or Chai that I could find.** Search for DPO/RL on Boltz/Chai/AF3 returned design-by-inversion work (BoltzDesign1) and BoltzGen (a generator), not RL.
- Search-based alternative: **Proteina-Complexa** (NVIDIA, ICLR 2026; https://github.com/NVIDIA-BioNeMo/Proteina-Complexa) [paper/README]: flow-based generator plus inference-time optimisation (beam search, best-of-N, MCTS, FK steering under `src/proteinfoundation/search/`) using composable reward models (AF2 iPAE-style, RoseTTAFold3 refolding, TMOL force field, ProteinMPNN) configured in YAML such as `binder_generate.yaml`. The docs describe no RL fine-tuning. This is test-time compute as a substitute for RL.
- Discrete-diffusion reward fine-tuning (DRAKES), DDPO for inverse folding (RL-DIF) and GRPO-style fine-tuning of molecular diffusion (**FTDiff**, arXiv 2606.01220: double-head clipped loss for coordinates and atom types, threshold-aware sigmoid rewards to prevent over-optimisation, time-free 10-step sampling; CrossDocked2020 Vina −7.18) [paper] show the diffusion-RL toolkit exists. It has not yet been applied to structure generators like RFdiffusion or Boltz at scale.
- Boltz-2 appears as an RL reward: "REINFORCE-ing Chemical Language Models" (arXiv 2501.15971) shows improved efficiency with Boltz-2 as the scoring function (ACEGEN repo) [paper].

### 2.8 Small-molecule RL
- **REINVENT 4** (https://github.com/MolecularAI/REINVENT4; J. Cheminf. 2024). Files read: `reinvent/runmodes/RL/reward.py`, `learning.py`, `NEWS.md` [code]. The default loss is **DAP**: `augmented_lls = prior_lls + sigma * scores; loss = (augmented_lls − agent_lls)^2`, with sigma 120 to 128 by default. This regresses the agent log-likelihood onto the KL-regularised optimum (policy ∝ prior·exp(sigma·score)). It is the same closed-form target that DPO-style methods use, in a squared-error form. MAULI/MASCOF/SDAP are deprecated; `dap_reinforce_strategy` and `mauli_reinforce_strategy` are REINFORCE variants. The staged-learning runner has diversity filters (`memories/diversity_filter.py`, scaffold and Murcko variants), inception replay and intrinsic penalties. Scoring plugins include Chemprop, DockStream, ICOLOS, AiZynthFinder (CAZP), SynthSense, custom alerts and a generic REST/external-process component. Version 4.8 adds peptide enumeration.
- **Saturn** (arXiv 2405.17066) and **Augmented Memory** [secondary/memory]: sample-efficient generative design with SMILES augmentation plus replay, a Mamba policy, and 1,000-oracle-call budgets. I could not read the paper this session.
- **REINFORCE-ing chemical LMs** (arXiv 2501.15971) [paper]: systematic ablation of replay, hill-climb, baselines and reward shaping; improved regulariser; ACEGEN.
- LLM-based small-molecule RL (2025 to 2026):
  - **Mol-R1** (arXiv 2508.08401) [paper]: Llama-3.1-8B-Instruct, GRPO; PRID (one expert-written trace used for in-context distillation) plus MoIA (alternating SFT, RL, rejection sampling). ChEBI-20; exact-match, format, length and similarity rewards; exact match 0.234.
  - **ChemDFM-R** (arXiv 2507.21990) [paper summary]: mid-training on atomised knowledge, SFT, then domain RL.
  - **ChemCRAFT** (arXiv 2601.17687; https://github.com/HowardLi1984/ChemCraft) [paper summary]: Qwen-2.5/3 7 to 14B, SMILES-GRPO inside a tool sandbox (RDKit, RDChiral, DL predictors, templates) with a Hypothesis-Action-Observation loop; multi-part reward (format, scaffold similarity, functional-group fidelity, ΔLogP/ΔQED, reaction validity); retrosynthesis top-1 above 40%.
  - **LLMol** (arXiv 2607.19044, July 2026) [paper summary]: Qwen3-4B + LoRA, SFT then RLVR/GRPO with RDKit verifiers; logP, QED and similarity-constrained tasks.
  - **RePO** (arXiv 2603.05900) [paper summary]: GRPO exploration term + answer-level reference-likelihood term + KL, on Qwen2.5-3B, TOMG-Bench; up to +17.4% success over GRPO.
  - Title-only: C-MORAL (2604.23061), RL with LLM-guided action spaces for synthesizable lead optimisation (2604.07669), DrugR, MolReasoner.
- **Intern-S1** (Shanghai AI Lab, arXiv 2508.15763) [paper summary]: scientific multimodal MoE with a **Mixture-of-Rewards** RL stage (1,000+ InternBootCamp tasks, about 20k mixed RL samples). Verifiers are the rule-based plus CompassVerifier (generative verifier) and POLAR-7B (preference). Infra: XTuner FSDP training, LMDeploy inference, FP8 training and inference, colocated design, claimed 10x less RL time. The paper reports no chemistry- or biology-specific verifier.
- GFlowNets: I did not read primary sources this session; treat them as [memory] (RGFN, SynFlowNet, and similar, used for diverse synthesizable molecule sampling).

### 2.9 Single-cell and genomics reasoning
- **Cell2Sentence-Scale (C2S-Scale)** (bioRxiv 2025.04.14.648850 v3) [paper summary]: models from 410M to 27B. GRPO for perturbation prediction (reward is negative MSE on the cytokine dataset; Kendall tau on L1000 because it handles ties) and for QA (BioBERT similarity reward, β 0.03, lr 5e-7, 32 candidates per prompt). The perturbation GRPO used 4 H100s, about 3,000 steps, batch 8, lr 1e-6. Gains: scFID −16% on interferon genes, Kendall tau +9.2% (410M).
- **Cell-o1** (arXiv 2506.02911) [paper]: 7B; SFT on 3,912 o1-distilled traces (38.5% rejection-sampling acceptance), then GRPO. Batch-level reward = format validity × all-cells-correct, with −1 for malformed output. Batch accuracy 32.9% (cell-level 68.5%), reported to beat o1 by more than 73% relative.
- **CellDuality** (PMC13441433, "self-supervised RLVR for biological reasoning") [title-only; page blocked]. VCR-Agent (virtual-cell mechanistic reasoning agents) [title-only].
- **Genome-Bench** ("Toward scientific reasoning in LLMs: training from expert discussions via RL", May 2025) [secondary: listing]: RL on genome-engineering forum Q&A, reported gains above 15%.
- scGPT and similar single-cell FMs have no LLM-style RL that I found.

### 2.10 Lab automation, self-driving labs and closed-loop experimentation
- **OpenAI GPT-5 + Ginkgo cloud lab** (Feb 2026) [secondary: openai.com/index/gpt-5-lowers-protein-synthesis-cost]: six closed-loop rounds, more than 36,000 unique cell-free protein-synthesis compositions across 580 plates on Ginkgo reconfigurable automation carts; GPT-5 designed 384-well batches; "strict programmatic validation" gated every experiment; 40% CFPS cost reduction, with reagent cost down 57%. **No model retraining or RL**; humans handled protocol improvements and reagent handling.
- **Lab-in-the-loop in-context learning** (arXiv 2603.26177) [paper]: JUMP Cell Painting, 10 iterations of batch size 100 gene selections, no parameter updates. In-context feedback gives +53% discoveries with Sonnet 4.6; random-feedback control erases the gain; Sonnet 4.5 hallucinated non-existent genes 45% of the time vs 9%. The learning signal exists only when the model is reliable enough to use it.
- **Coscientist** (Nature 2023), **BioAutoMATED**, Emerald Cloud Lab and Ginkgo APIs [memory/secondary]: LLM-planned, GPT-4-driven automation (Opentrons + cloud lab docs search) or AutoML pipelines. They are not RL-trained policies. Self-driving-lab work in the RL sense is mostly Bayesian optimisation or bandits [memory].
- **ProtRL's EGFR campaign** is the only RL-policy-updated-by-wet-lab-feedback example I found.
- **Robotics/protocol simulators as verifiers:** PyLabRobot simulation (Biomni, above) exists as a pass/fail tool.

### 2.11 Benchmarks and environments
| Name | What it is | Verifier type | RL-ready? |
|---|---|---|---|
| ether0-benchmark (375 task types) | chemistry, open-answer | deterministic RDKit/oracle | yes (reward code released) |
| Biomni-Eval1 (433 instances, 10 tasks) | GWAS, rare disease, CRISPR delivery, LAB-Bench, HLE | exact-match rules in `biomni_eval1.py` | yes, but small |
| BixBench (53 capsules, 296 open-answer questions) | notebook-based bioinformatics | LLM judge (open) or MCQ | Edison uses it as an RL test bed [secondary] |
| LAB-Bench (FutureHouse, 2024) | multiple-choice across literature, protocols, sequences, cloning | exact match | yes (MCQ, easy to hack by guessing) [memory] |
| BioProBench (ICML 2026) | biological protocol understanding and reasoning | mixed | corpus, not an environment [secondary] |
| Aviary environments | cloning, protein stability, literature QA | programmatic | yes (gym API) |
| ProteinGym | DMS fitness landscapes | Spearman versus experiment | reward for fitness policies (e.g., ProtRL's CSV mode takes experimental rewards) [memory/README] |
| ChemBench, ScienceAgentBench, SciGym | knowledge Q&A / data-driven discovery tasks / simulator-based discovery | rubric / execution | partially [memory, not read] |
| TOMG-Bench | text-based molecular generation and optimisation | RDKit | used by RePO |
| PerturbQA | CRISPRi differential expression | experimental labels | used by rbio |
| CellPuzzles | batch cell-type annotation | exact | used by Cell-o1 |
| P-Bench (425 tasks incl. biology) | hypothesis testing with statistically valid p-values | statistical validity | Fisher-R1 (arXiv 2608.07437) RLVR [secondary] |

---

## 3. What the code shows that the papers do not

1. **ether0's repo is a reward-server architecture, not a trainer.** The expensive rewards (Molecular Transformer forward model, KDESol, purchasability bloom filter) sit behind an authenticated FastAPI service, so rollout workers call them over HTTP. This decoupling is what lets a 24B GRPO run use a 2019-era OpenNMT model as an oracle. Also, the "good molecule" bonus is applied only when reward already equals 1.0 (`accuracy_reward`), so it acts as a tie-breaker among correct answers rather than a competing objective.
2. **rbio's released code is deliberately not the paper code**, and its verifier signal is the MLP's confidence value, not a hard label. The `2.0*answer + mention + format` mix shows that the paper's "soft verification" is a smooth reward plus auxiliary shaping.
3. **BioReason trains with beta = 0.0 and Dr.GRPO loss by default**, which means no reference model, and its correctness reward is a substring test. The repo is an easily reproducible 2-GPU GRPO recipe. It is not a scaled system.
4. **ProteinDPO ships no training code.** ProtRL's own example GRPO trainer overwrites its computed advantages. When reproducing protein-RL results, expect to re-implement the loss.
5. **ProteinZero's reward normalisation is per-group min-max per component**, not a fixed scale. That makes the reward relative within a backbone's rollouts, which matters for multi-objective stability. It also runs without vLLM (HF generate with an exact-length logits processor) and in strict fp32, indicating the oracle (ESMFold plus TM-align) and not generation is the throughput limit.
6. **REINVENT 4's default "RL" is not policy gradient** but squared-error regression onto prior·exp(sigma·score). It is stable and sample-hungry, and sigma, not a KL coefficient, sets the exploitation level.
7. **Biomni ships the reward function but not the RL.** `BiomniEval1.evaluate` is effectively the Biomni-R0 reward interface, and `test_pylabrobot_script` is an unused verifier for protocol code.
8. **Proteina-Complexa implements RL-like optimisation as search** (beam, MCTS, FK steering) over the same reward models, a signal that for expensive structure oracles the field currently prefers test-time compute to weight updates.
9. **Kosmos and the Ginkgo loop are both "no weight update" systems.** All adaptation is in context and in a world-model memory; their open problem (rabbit holes, hallucinated genes) is exactly what RL on verified outcomes could target.

---

## 4. Transfer to biology: concrete ideas (NOVEL = I found no prior work)

1. **Soft-verifier RL for proteins and genomes (extends rbio).** Use ProteinGym-calibrated ensembles (PLM likelihood, ESM/ProteinNPT-style regressors) as a confidence-weighted reward for an LLM or pLM policy, as rbio did with perturbation MLPs. The reward is the verifier's calibrated probability that a variant improves a property. Include an uncertainty penalty and periodically recalibrate on new assay data. **NOVEL for proteins/genomes.**
2. **Simulator-verified lab-protocol RLVR.** Train a coder LLM with GRPO where reward = PyLabRobot simulation pass + tip/volume tracking + constraint checks, and BioProBench/LAB-Bench ProtocolQA for curriculum. Everything needed exists in Biomni's `test_pylabrobot_script`. **NOVEL.**
3. **On-policy distillation to merge domain specialists.** ether0 merges specialists by SFT on rejection-sampled traces. Replace this with on-policy distillation (student rollouts, per-token reverse-KL to the specialist teacher for that task) across chemistry, protein, genomics and single-cell specialists. NeMo-RL already supports it. I found no bio/chem OPD paper in a Hugging Face papers search. **NOVEL.**
4. **Rubric rewards for experimental design, built from Methods sections.** Meta's "Training AI Co-Scientists Using Rubric Rewards" extracts rubrics from papers and self-grades with a frozen policy copy (domain: ML research plans, with medical transfer). Apply it to biology: rubrics = controls, replicates, readouts, power and orthogonal validation extracted from papers; use it to train agents on assay design and Kosmos-style hypothesis vetting. **NOVEL for biology.** Pair it with a hacking audit (see 9).
5. **Multi-fidelity async RL for slow oracles.** Reward every rollout with a cheap surrogate (like ProteinZero's fast ΔΔG proxy), send only the top-k per group to the expensive oracle (Boltz-2 affinity, docking, AF3 refolding), and run rollouts asynchronously with importance-weight corrections and a cache. REINVENT/Saturn-style oracle budgets are the baseline to beat. Edison and Biomni-R0 have async for tool latency but not for expensive scientific oracles. **NOVEL as a combination.**
6. **Genome LM RL with filter-cascade verifiers.** Use the phage paper's tiered constraints (length, GC, homopolymer, tropism identity, novelty vs natural proteins, ORF architecture) as hard-gate rewards, Enformer/Borzoi/Evo-likelihood as soft rewards and the base Evo 2 as KL anchor. Constrained-GRPO (Ctrl-DNA's Lagrangian) handles the hard constraints. Evo 2 as an RL policy is unreported. **NOVEL.**
7. **Self-play with an adversarial structure critic.** A designer policy against a discriminator that tries to distinguish designed from natural binders (trained on PDB plus held-out wet-lab hits) to guard against reward hacking of the structure predictor. SSP's two-policy design (scTM vs pTM) is the nearest prior art, and it is not adversarial. **NOVEL.**
8. **Wet-lab rounds as offline RL batches.** Treat each plate as a batch of delayed, noisy rewards. Use ProtRL's wDPO or a conservative offline variant with a lab noise model, and replay public closed-loop datasets such as the Ginkgo 36k-reaction logs (if released) to pre-train an experimental-design policy before any new wet-lab spend. ProtRL's CSV-reward mode is the starting point. **Partly novel (offline replay of agent-generated lab logs).**
9. **Reward-hacking audits for biology verifiers.** Build adversarial prompts for common structure/property rewards (low-complexity or repeat sequences with high pLDDT, ESMFold-fooling designs, SMILES that pass RDKit but not REOS). ether0's `BAD_SMARTS_PATTERNS` and `good_molecule_bonus` are the template to generalise to proteins and DNA. Fisher-R1 and the "CHERRL / reward hacking in rubric RL" lines of work are the generic precedent. **Gap in biology.**
10. **Step-level GRPO for Kosmos-class agents.** Take Edison's step-level GRPO and add a reward that penalises the "significant but irrelevant" rabbit-hole analyses (pre-registered hypothesis vs post-hoc findings; permutation-test sanity checks). **NOVEL.**

---

## 5. Sources (V = verified by reading this session; S = secondary; M = memory; T = title/abstract only)

Code (V, cloned and read):
- https://github.com/Future-House/ether0 (`src/ether0/rewards.py`, `packages/remotes/src/ether0/server.py`, README)
- https://github.com/czi-ai/rbio (`training/training.py`, `rewards.py`, README)
- https://github.com/bowang-lab/BioReason (`sh_grpo.sh`, `train_grpo.py`, `bioreason/trainer/grpo_config.py`, `grpo_trainer.py`, `dna_modules/nucleotide_module.py`)
- https://github.com/snap-stanford/Biomni (`biomni/eval/biomni_eval1.py`, `biomni/tool/lab_automation.py`, README)
- https://github.com/AI4PDLab/ProtRL (README, example GRPO and wDPO trainers)
- https://github.com/ziwenwang28/ProteinZero (`run_proteinzero_grpo.sh`, `train_proteinzero_grpo.py`, README)
- https://github.com/evo-design/protein-dpo (README, file list)
- https://github.com/MolecularAI/REINVENT4 (`reinvent/runmodes/RL/reward.py`, NEWS.md)
- https://github.com/NVIDIA-BioNeMo/Proteina-Complexa (README via fetch, S)

Papers and pages (read via fetch summaries unless noted):
- ether0: https://arxiv.org/abs/2506.17238, https://arxiv.org/html/2506.17238v1, https://huggingface.co/futurehouse/ether0
- rbio: https://www.biorxiv.org/content/10.1101/2025.08.18.670981v2
- BioReason: https://arxiv.org/abs/2505.23579 (T); BioReason-Pro: https://www.biorxiv.org/content/10.64898/2026.03.19.712954v2.full.pdf
- Biomni-R0: https://huggingface.co/biomni/Biomni-R0-32B-Preview, https://www.marktechpost.com/2025/09/04/biomni-r0-new-agentic-llms-trained-end-to-end-with-multi-turn-reinforcement-learning-for-expert-level-intelligence-in-biomedical-research/ (S). Technical report biomni.stanford.edu/blog/biomni-r0-technical-report returned 404.
- Kosmos: https://arxiv.org/abs/2511.02824, https://advances.edisonscientific.com/research/announcing-kosmos/, https://advances.edisonscientific.com/research/how-we-built-kosmos/
- Edison/NVIDIA: https://developer.nvidia.com/blog/how-to-train-scientific-agents-with-reinforcement-learning/ (S); https://developer.nvidia.com/case-studies/scientific-literature-ai-nvidia-nemotron (no RL, confirmed)
- Aviary: https://arxiv.org/abs/2412.21154; BixBench: https://arxiv.org/html/2503.00096v2; Robin: search results only (S, arXiv 2505.13400)
- C2S-Scale: https://www.biorxiv.org/content/10.1101/2025.04.14.648850v3.full.pdf; Cell-o1: https://arxiv.org/html/2506.02911v1
- Evo 2: https://www.nature.com/articles/s41586-026-10176-5 (T); Evo phages: https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1
- TACO: https://arxiv.org/html/2503.07981v1; Ctrl-DNA: https://arxiv.org/html/2505.20578v1; DRAKES: https://arxiv.org/abs/2410.13643 (T) and https://github.com/ChenyuWang-Monica/DRAKES (T)
- ProtRL paper: https://arxiv.org/abs/2412.12979 (T); Adaptyvbio blog: https://www.adaptyvbio.com/blog/protrl/ (S)
- ProteinZero: https://arxiv.org/html/2506.07459v5; Supervision-to-Exploration: https://arxiv.org/html/2510.01571v1; RL-DIF: https://arxiv.org/html/2410.17173v1; SSP: https://www.biorxiv.org/content/10.64898/2026.03.26.714453v1.full; MAProt: https://www.biorxiv.org/content/10.64898/2026.01.13.699365v1.full.pdf; Pro-1: https://michaelhla.com/blog/pro1.html; Proteo-R1: https://arxiv.org/html/2605.02937v1; review: https://arxiv.org/html/2511.21476v2
- Unread / title-only: https://arxiv.org/abs/2609.39048 (rate-limited), https://arxiv.org/abs/2507.18816, ProteinDPO Nature Methods (PMC page blocked)
- Small molecules: https://arxiv.org/html/2508.08401v1 (Mol-R1), https://arxiv.org/abs/2507.21990v2 (ChemDFM-R), https://arxiv.org/html/2601.17687v2 (ChemCRAFT), https://arxiv.org/html/2607.19044v1 (LLMol), https://arxiv.org/html/2603.05900v1 (RePO), https://arxiv.org/html/2606.01220 (FTDiff), https://arxiv.org/abs/2501.15971 (REINFORCE-ing CLMs), https://arxiv.org/html/2508.15763 (Intern-S1). Saturn (arXiv 2405.17066) and GFlowNets not read.
- Lab/agents: https://openai.com/index/gpt-5-lowers-protein-synthesis-cost/ ; https://arxiv.org/html/2603.26177 ; Coscientist https://www.nature.com/articles/s41586-023-06792-0 (T/M).
- Rubric/OPD/deep research landscape (Hugging Face paper-search listings, S/T): https://huggingface.co/papers?q=rubric%20biology%20reinforcement%20learning ; https://huggingface.co/papers?q=biomedical%20deep%20research%20reinforcement%20learning ; https://huggingface.co/papers?q=bioinformatics%20agent%20reinforcement%20learning ; https://huggingface.co/papers?q=on-policy%20distillation%20chemistry ; Meta rubric co-scientist https://arxiv.org/abs/2512.23707 (V summary)

Inference flags: the "mention reward is gameable", "substring match is weak", "dead-code advantage" and "no OPD in bio" statements are my own inferences. The last rests on a single Hugging Face paper-search listing, so absence is not proven.

---

## 6. Gap analysis: frontier-lab RL techniques vs biology domains

Legend: **DONE** (published, at least one system), **PARTIAL** (adjacent or small-scale), **GAP** (nothing found), **?** (not verified).

| Technique \ Domain | Small-molecule design | Chemistry reasoning (LLM) | Protein design (pLM / diffusion) | Protein function / annotation | Genomics / regulatory DNA | Single-cell / perturbation | Bioinformatics agents | Lab automation / wet lab |
|---|---|---|---|---|---|---|---|---|
| **Verifier-based RLVR (GRPO/GSPO)** | DONE (LLMol, RePO, Mol-R1, ChemCRAFT; REINVENT DAP/REINFORCE) | DONE (ether0) | DONE (ProteinZero, ProtRL, RL-DIF) | DONE (BioReason-Pro, F_max reward) | PARTIAL (Ctrl-DNA, TACO on small policies) | DONE (Cell-o1, C2S-Scale, rbio) | PARTIAL (Biomni-R0, Edison; exact-match rewards) | GAP (no simulator or lab reward) |
| **Async / agentic multi-turn RL** | GAP | GAP (ether0 is single-turn; ChemCRAFT tool loop is the nearest, async not stated) | GAP | GAP | GAP | GAP | PARTIAL (Biomni-R0 async rollouts; Edison step-level GRPO; both with little published detail) | GAP |
| **Generative / rubric reward models** | GAP | GAP | PARTIAL (Pro-1 LLM-judge "creativity") | PARTIAL (LLM-judge eval only) | GAP | GAP (BioBERT similarity reward in C2S is a learned scorer, not rubric) | GAP | GAP |
| **Self-play / adversarial co-training** | GAP | GAP | PARTIAL (SSP two-policy; EvoPlay [M]) | GAP | GAP | GAP | GAP | GAP |
| **On-policy distillation** | GAP | PARTIAL (ether0 offline specialist→generalist SFT only) | GAP | GAP | GAP | GAP | GAP (supported in NeMo-RL, no bio use reported) | GAP |
| **Long-horizon deep-research RL** | GAP | GAP | GAP | GAP | GAP | GAP | PARTIAL (general or medical DR agents such as DR Tulu, MedResearcher-R1; Kosmos is prompted not trained) | GAP |
| **Soft / learned world-model verifiers** | PARTIAL (Boltz-2 reward, ML property oracles) | PARTIAL (ether0 oracles: reaction model, solubility) | PARTIAL (ProteinZero ΔΔG proxy; SSP uses ESMFold confidence) | GAP | PARTIAL (Enformer rewards) | DONE (rbio-VCM) | GAP | GAP |
| **Wet-lab / experimental feedback in the policy loop** | GAP (published SMILES RL is computational) | GAP | PARTIAL (ProtRL EGFR rounds; DPO on assay data) | GAP | GAP | GAP | GAP | GAP (closed loop with ICL only: GPT-5/Ginkgo, 2603.26177) |
| **Test-time search as RL substitute** | PARTIAL | GAP | DONE (Proteina-Complexa beam/MCTS/FK) | GAP | PARTIAL (Evo 2 guided search [M]) | GAP | GAP | GAP |

Key readings of the table:

1. **On-policy distillation is absent everywhere**, even though ether0's specialist→generalist recipe is the most natural fit in biology (many narrow verifiable task groups, each trained separately). NeMo-RL supports OPD today.
2. **Async/off-policy RL for expensive oracles is absent.** Protein RL papers mention the oracle as the bottleneck (ESMFold on 20 GPUs for RL-DIF; Pro-1's generation bottleneck) yet use synchronous loops. Chemistry hits the same wall with docking.
3. **Rubric / generative reward models** have so far been applied to medicine and ML research plans. Experimental design, protocol critique and hypothesis quality are unaddressed in biology, and these are exactly where no deterministic verifier exists.
4. **Self-play** exists only in inverse folding and not in the form (adversarial discriminators, proposer-solver) that frontier labs use.
5. **Long-horizon agentic RL for computational biology** (Kosmos-class, 12-hour, 200-rollout runs) has not been attempted. The inputs exist: BixBench, Biomni-Eval1, the Aviary Jupyter environment, and step-level GRPO from Edison.
6. **Genomic foundation models (Evo 2, NT) as RL policies** are essentially untouched. The two RL papers on DNA use small custom policies, and the Evo phage paper explicitly avoids RL in favour of filters.
7. **RL on structure generators (RFdiffusion, Boltz, Chai, AF3-style)** is not public. The field is using inference-time search (Proteina-Complexa) and inversion (BoltzDesign1), which marks an opening for GRPO-style post-training with confidence rewards. Caveat: the diversity-collapse and reward-hacking evidence (ProteinZero, "From Supervision to Exploration") suggests this needs Pareto, constrained or diversity-regularised objectives from the start.
8. **Closed-loop wet-lab RL is essentially absent.** The best large closed loops (Ginkgo/GPT-5, lab-in-the-loop ICL) learn only in context, and the one RL-updated campaign (ProtRL/EGFR) is small (26 designs over two rounds).
