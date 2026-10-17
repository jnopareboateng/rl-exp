# Related work and sources

Status column: verified = the ID or content was fetched in a session on 2026-10-17; recalled = from memory, check before citing.

## RL for chemistry and verifiable rewards

| Work | Why it matters here | ID | Status |
|---|---|---|---|
| ether0: Training a Scientific Reasoning Model for Chemistry (FutureHouse) | Closest prior art: 24B model, SFT + GRPO on 640,730 verifiable chemistry problems (375 tasks); we have about 240 training molecules, a different regime | arXiv 2506.17238 | verified (search) |
| The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL) | Sigmoid compute-performance form; recipe choices (CISPO, FP32 head, prompt-level aggregation) | arXiv 2510.13786 | verified (search) |
| Does RL really incentivize reasoning capacity beyond the base model? (Yue et al.) | RL mostly sharpens pass@1 toward pass@k; motivates reporting pass@8 and maj@8 | arXiv 2504.13837 | verified (advisor search) |
| A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce | Rejection-sampling FT as a strong cheap comparator (arm A3) | arXiv 2504.11343 | verified (link in review) |
| LoRA Without Regret (Thinking Machines, 2025) | LoRA matches full fine-tuning in RL; all-layer LoRA, higher learning rate | blog post | recalled |
| DAPO; MiniMax-M1 (CISPO); Dr. GRPO | Loss variants used in the recipe | arXiv 2503.14476; 2506.13585; 2503.20783 | recalled |
| Scaling laws for transfer (Hernandez et al.) | Effective data transferred grows with model size; loss in unsupervised fine-tuning, not RL | arXiv 2102.01293 | recalled |

## Resources considered and parked (2026-10-17)

| Resource | What it is | Relevance |
|---|---|---|
| microsoft/msr-acc (dataset msr-acc-tae25; paper arXiv 2506.14492) | Coupled-cluster total atomization energies for broad chemical space, built for density-functional development | None for structure recovery (different target: energies). Cite only if a physics-based property reward is ever added. |
| HuggingFaceBio (org) | DNA language models (Carbon family) and genomics benchmarks; includes `seqqa-reward-reference`, a 39-case hand-checked regression fixture for `trl.internal.seqqa.reward` | Different modality. Two precedents: TRL ships verifiable-reward code for biology; hand-checked reward fixtures (our T1-T5 and 77 format fixtures) |
| Anthropic/BioMysteryBench-full | 90 anonymised bioinformatics analysis problems needing tool-based analysis (alignment, variant calling, ...), CC-BY-4.0, gated | Agentic data-analysis benchmark, not pharmacology to structure. Design precedent: anonymise so the source cannot be looked up. Revisit only if tool use is added. |

## Data and tooling sources

ChEMBL REST API (structures, first approval); openFDA Drugs@FDA (original approval dates); PubChem PUG REST and NCBI E-utilities (exposure covariates); RDKit (standardisation, InChIKey, Murcko scaffolds, Morgan fingerprints); vLLM, TRL, ms-swift (see `gpu/STACK.md` for versions and sources).
