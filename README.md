# rl-exp

Does online reinforcement learning help a small language model recover a drug's structure from a description of its pharmacology, beyond cheaper alternatives?
A controls-first study on Qwen3.5-4B and 9B (and Qwen3.8-27B if earned), with exact, parse-free scoring against known structures, a temporal held-out cohort (drugs first approved 2025-2026), and a pre-registered analysis plan.
The aim is a reproducible record that can be written up as a paper whether or not RL wins.

## Status

| Phase | State |
|---|---|
| Data foundation (references, approval dates, leakage-safe splits) | built; hand-resolution and exposure covariates in progress |
| Reward v2 and offline tests | in progress |
| CPU controls (constant, copy-nearest) | built, to be scored on the final splits |
| First GPU session (stack check, 4B sampling probe) | not run; Kaggle ready, Colab needs a browser accept |
| Arms A0-A5, locked Test evaluation | not started; design frozen at tag `prereg-v1` before any Test sampling |

Plan and review log: [docs/plan.md](docs/plan.md). File and schema contracts: [docs/contracts.md](docs/contracts.md). Durable project facts: [docs/memory/MEMORY.md](docs/memory/MEMORY.md). Paper scaffold: [docs/paper/README.md](docs/paper/README.md).

## Layout

    src/rlexp/     chem, reward, data_prep, metrics, controls     (the library; import as `from rlexp import chem`)
    scripts/       data audit, prompt keyword survey, ChEMBL/openFDA resolution, structure statistics, offline reward checks
    gpu/           stack memo (STACK.md) and the standalone Colab/Kaggle sampling pack
    tests/         pytest, including 77 answer-format fixtures
    data/raw/      original row files (not in git)       data/curated/    hand-resolved inputs and lookup caches (in git)
    data/processed/ refs.jsonl and split_report.json (in git); prompt files (not in git)
    runs/          sampling and scoring outputs (not in git)       results/   small tracked summaries used by the paper
    docs/          plan, contracts, scaling notes, memory, paper scaffold, research survey, archive of the original handoff

## Setup

    uv venv && uv pip install -e ".[dev]"
    .venv/bin/python -m pytest

## Reproduce the data products (needs data/raw/train_think.jsonl and val_think.jsonl)

    .venv/bin/python scripts/resolve_refs.py data/curated/chembl_refs.json data/raw/train_think.jsonl data/raw/val_think.jsonl   # network: ChEMBL, openFDA
    .venv/bin/python -m rlexp.data_prep                       # references, splits, prompt files -> data/processed
    .venv/bin/python -m rlexp.controls --data data/processed --out runs

## Provenance

Every run records its git revision, the sha256 of the data files, seed, library versions and hardware (`runs/<id>/meta.json`). Decisions and their reasons live in `docs/memory/`; the analysis plan is frozen by git tag before any held-out data is touched.
The raw rows are a private dataset and are never committed. Hand-resolved facts (structures, approval years) carry their sources in `data/curated/*_notes.md`.

## License

Not yet chosen.
