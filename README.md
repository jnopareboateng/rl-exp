# rl-exp

How much of a drug's structure can a language model recover from a description of its pharmacology, and does online reinforcement learning add anything beyond cheaper alternatives?
A controls-first study on Qwen3.5-4B, 9B and 27B-FP8 (and Qwen3.8-27B-FP8 as a reference), with exact parse-free scoring against known structures, a temporal held-out cohort (drugs first approved 2025-2026), a profile-association metric, and a pre-registered analysis plan.
The aim is a reproducible record that can be written up as a paper whether or not RL wins.

## Status (2026-10-17)

| Phase | State |
|---|---|
| Data foundation (references, approval dates, leakage-safe splits) | built and audited |
| Reward v2, metrics v3 (first-valid policy, profile-specific lift, component-level sign-flip), independent reviews | built; review fixes landing |
| CPU controls (constant, nearest-profile copy, reward-optimal constant), class-information ceiling (ChEMBL) | scored by the CLI on every split before the lock, reported on Val only |
| Zero-RL baselines on Val (4B, 9B, Qwen3.5-27B-FP8, Qwen3.8-27B-FP8), identification and name-given diagnostics | done on Val |
| Arms A1 (trace SFT), A2 (answer-only SFT), A4 (in-context retrieval) on Val | done |
| A5 RL (own HF-generate GRPO loop, reward modes full / contrastive / validity-only) | gates passed on the A100; pilot running |
| Locked Test evaluation | not started; the protocol is [docs/prereg-v1.md](docs/prereg-v1.md), frozen by git tag `prereg-v1` before any Test sampling |

Headline so far (Val, 20 drugs): no arm recovers a structure (exact@1 is 0, core@1 at most 5%); the best constant answer scores mean Tanimoto 0.176; only the untrained 27B with first-valid-of-8 decoding clears it (0.186). Details: [docs/plan.md](docs/plan.md) section 5 and `results/val_baselines.json`.

Plan and review log: [docs/plan.md](docs/plan.md). Protocol: [docs/prereg-v1.md](docs/prereg-v1.md). File and schema contracts: [docs/contracts.md](docs/contracts.md). Durable project facts: [docs/memory/MEMORY.md](docs/memory/MEMORY.md). Paper scaffold: [docs/paper/README.md](docs/paper/README.md).

## Layout

    src/rlexp/     chem, reward, data_prep, metrics, controls, arms, ident, shuffle, grpo   (import as `from rlexp import chem`)
    scripts/       data audit, ChEMBL/openFDA resolution, offline reward checks, ChEMBL pool and class ceiling, Val table
    gpu/           Colab/Kaggle packs: vLLM sampler, LoRA SFT + merge, answer-only RL (RL.md, SFT.md, STACK.md)
    tests/         pytest (about 900 tests, including 77 answer-format fixtures); torch-dependent tests skip without torch
    data/raw/      original row files and the ChEMBL cache (not in git)      data/curated/   hand-resolved inputs (in git)
    data/processed/ refs.jsonl and split_report.json (in git); prompt files (not in git)
    runs/          sampling and scoring outputs (not in git)      results/   small tracked summaries used by the paper
    docs/          plan, protocol, contracts, scaling notes, memory, paper scaffold, research survey, archive of the original handoff

## Setup

    uv venv && uv pip install -e ".[dev]"
    .venv/bin/python -m pytest

## Reproduce

    # data products (needs data/raw/train_think.jsonl and val_think.jsonl)
    .venv/bin/python scripts/resolve_refs.py data/curated/chembl_refs.json data/raw/train_think.jsonl data/raw/val_think.jsonl   # network: ChEMBL, openFDA
    .venv/bin/python -m rlexp.data_prep                       # references, splits, prompt files -> data/processed
    .venv/bin/python -m rlexp.controls --data data/processed --out runs
    .venv/bin/python -m rlexp.arms build --data data/processed --out data/processed/arms     # A1/A2 SFT files, A4 neighbours

    # GPU (Colab A100): see gpu/SFT.md and gpu/RL.md
    sh gpu/run_sft.sh                                          # ARM=A2 or A1; merges the best epoch
    GATE=smoke sh gpu/run_rl.sh; sh gpu/run_rl.sh              # RL gates, then a run (REWARD_MODE=full|contrastive|validity)
    gpu/.venv/bin/python gpu/sample_vllm.py --model <path> --data <dir> --splits val --k 8 --out runs/<name>

    # scoring and tables (CPU)
    .venv/bin/python -m rlexp.metrics score --samples runs/X/samples.jsonl --data data/processed --out runs/X/scored.jsonl
    .venv/bin/python -m rlexp.metrics report --scored runs/X/scored.jsonl --data data/processed [--vs runs/Y/scored.jsonl]
    .venv/bin/python scripts/val_table.py                      # results/val_baselines.json
    .venv/bin/python scripts/class_ceiling.py                  # results/class_ceiling.json (needs the ChEMBL cache)

## Provenance

Every run records its git revision, the sha256 of the data files, seed, library versions and hardware (`runs/<id>/meta.json`). Decisions and their reasons live in `docs/memory/`; the analysis plan is frozen by git tag before any held-out data is touched.
The raw rows are a private dataset and are never committed. Hand-resolved facts (structures, approval years) carry their sources in `data/curated/*_notes.md`.

## License

Not yet chosen.
