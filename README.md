# rl-exp

Can a language model write an approved drug's structure from a written description of its pharmacology, and does reinforcement learning (RL) with a fingerprint-similarity reward add anything over cheaper alternatives?
A controls-first, pre-registered study on Qwen3.5-4B (the 9B and 27B models are validation-only references), scored against known structures, with a temporal held-out cohort (41 drugs first approved in 2025 and 2026).

## Result in one paragraph

No system recovers a structure (exact match 0 on every split). A single reward-optimal constant answer, chosen from training molecules and a public drug pool at zero generated tokens, delivers 0.199 similarity on the temporal cohort (the best of 1,828 candidate constants); the validation-selected RL checkpoint delivers 0.186
and the one pre-registered contrast does not reject (difference -0.013, 10 wins and 31 losses over 41 drugs, p = 0.061). RL beats the supervised and untrained models (+0.054 and +0.113) by converging on a generic structure, and RL trained with the profile deleted reaches the same reward.
The profile carries a small drug-specific signal on random held-out drugs and in larger untrained models, which we could not detect on the temporal cohort. Report a tuned constant, a calibrated scale, the cost, a shuffled-input run and a constant-proof readout beside any similarity-rewarded result.

## Where to read

| What | Where |
|---|---|
| The paper (draft with all results) | [docs/paper/draft.md](docs/paper/draft.md) |
| Protocol, frozen by git tag `prereg-v1`; addendum with the secondary analyses, tag `prereg-v1-a1` | [docs/prereg-v1.md](docs/prereg-v1.md), [docs/prereg-v1-addendum1.md](docs/prereg-v1-addendum1.md) |
| Benchmark card, controls checklist and the release tables they describe (splits, control answers, reference values) | [docs/benchmark_card.md](docs/benchmark_card.md), [docs/controls_checklist.md](docs/controls_checklist.md), [release/](release/) |
| Aggregate results behind every number (JSON, tables, figures) | [results/](results/) |
| Plan and review log, file and schema contracts, durable project facts | [docs/plan.md](docs/plan.md), [docs/contracts.md](docs/contracts.md), [docs/memory/MEMORY.md](docs/memory/MEMORY.md) |
| Compute and time per stage | [docs/paper/compute_budget.md](docs/paper/compute_budget.md) |
| The locked Test procedure | [gpu/FINAL.md](gpu/FINAL.md) |

## Status (2026-10-18)

| Phase | State |
|---|---|
| Data (references, approval dates, leakage-safe splits), reward v2, metrics v3 and independent reviews | done |
| CPU controls C0 to C3, class-information ceiling, zero-RL baselines at 4B, 9B and 27B (Val), identification and name-given diagnostics | done |
| Arms: trace SFT (A1, Val only), answer-only SFT (A2), retrieval (A4), RL (R1, R5; validity-only R2, contrastive R3 cut short, profile-free R4) | done on Val |
| Locked Test evaluation (4B arms; protocol frozen before any Test sample) | done once; scored; results in `results/` and the paper |
| Secondary analyses (rank readout, random-pair calibration, shuffled-profile contrasts), registered before any model sample was scored | done |
| Release tables (splits, control answers, values), benchmark card, checklist, licences | done and published (`release/`, `data/`, `docs/`); the private profile texts and traces are not |

## Layout

    src/rlexp/     chem, reward, data_prep, metrics, controls, arms, ident, shuffle, grpo   (import as `from rlexp import chem`)
    scripts/       data audit, ChEMBL/openFDA resolution, offline reward checks, ChEMBL pool and class ceiling, Val table, arm selection, locked-run scoring and tables,
                   rank readout, random-pair calibration, cohort diagnostics, figures, release builder
    gpu/           Colab/Kaggle packs: vLLM sampler, LoRA SFT + merge, answer-only RL (RL.md, SFT.md, STACK.md), the locked-run driver (FINAL.md)
    tests/         pytest (about 1,700 tests, including 77 answer-format fixtures); torch-dependent tests skip without torch
    data/          tracked: hand-resolved facts and lookups (curated/), the reference and split tables (processed/refs.jsonl, split_report.json), MANIFEST.sha256 and a data card; not published: the private raw rows (raw/) and the prompt files in processed/
    runs/          sampling and scoring outputs (not in git)      results/   small tracked summaries used by the paper      release/   release tables (splits, control answers, reference values)
    docs/          plan, protocol and addendum, contracts, paper, memory, benchmark card, controls checklist

## Setup

    uv venv && uv pip install -e ".[dev]"
    .venv/bin/python -m pytest

## Reproduce

Without the private rows you can still reproduce the control values and score your own answers against the published splits: `docs/benchmark_card.md`, section 11. The full pipeline needs the rows.

    # data products (needs data/raw/train_think.jsonl and val_think.jsonl)
    .venv/bin/python scripts/resolve_refs.py data/curated/chembl_refs.json data/raw/train_think.jsonl data/raw/val_think.jsonl   # network: ChEMBL, openFDA
    .venv/bin/python -m rlexp.data_prep                       # references, splits, prompt files -> data/processed
    .venv/bin/python -m rlexp.controls --data data/processed --out runs
    .venv/bin/python -m rlexp.arms build --data data/processed --out data/processed/arms     # A1/A2 SFT files, A4 neighbours

    # GPU (Colab A100): see gpu/SFT.md and gpu/RL.md
    sh gpu/run_sft.sh                                          # ARM=A2 or A1; merges the best epoch
    GATE=smoke sh gpu/run_rl.sh; sh gpu/run_rl.sh              # RL gates, then a run (REWARD_MODE=full|contrastive|validity, PROFILE_MODE=real|none)
    gpu/.venv/bin/python gpu/sample_vllm.py --model <path> --data <dir> --splits val --k 8 --out runs/<name>

    # scoring and tables (CPU)
    .venv/bin/python -m rlexp.metrics score --samples runs/X/samples.jsonl --data data/processed --out runs/X/scored.jsonl
    .venv/bin/python -m rlexp.metrics report --scored runs/X/scored.jsonl --data data/processed [--vs runs/Y/scored.jsonl]
    .venv/bin/python scripts/val_table.py                      # results/val_baselines.json
    .venv/bin/python scripts/class_ceiling.py                  # results/class_ceiling.json (needs the ChEMBL cache)
    .venv/bin/python scripts/rank_readout.py --final runs/colab-s1 --split val --arms a2-val,rlfull5-s50-t0.3 --out results/rank_val.json   # the constant-proof readout
    .venv/bin/python scripts/random_pair_calibration.py --out results/random_pair_calibration.json                                        # the scale of the metric
    .venv/bin/python scripts/make_figures.py --split val       # results/figures: F1 to F4 and the Val table
    .venv/bin/python scripts/answer_diversity.py --constant val8-control_constant_pool --names a2-val,rlfull5-s50-t0.3 --json results/answer_diversity.json
    .venv/bin/python scripts/compute_budget.py                 # results/compute_budget.json, docs/paper/compute_budget.md
    .venv/bin/python scripts/make_release.py                   # release/ (public tables), refuses anything prompt-like

    # locked evaluation (after the tag prereg-v1): gpu/FINAL.md has the whole procedure
    .venv/bin/python scripts/select_arms.py --candidates results/arm_candidates.json --runs runs/colab-s1 --out results/selection.json   # Val only
    python gpu/run_final.py --drive <drive dir>                # Colab A100: merge, Val sanity gate, sample every frozen arm
    .venv/bin/python scripts/final_controls.py --final runs/final --manifest data/MANIFEST.sha256   # C0 to C3 on the Test rows
    .venv/bin/python scripts/confirmatory.py --selection results/selection.json --final runs/final --manifest data/MANIFEST.sha256 --out results/confirmatory.json
    .venv/bin/python scripts/final_table.py --final runs/final --splits test_t,test_r --spec results/frozen_arms.json --out results/test_baselines.json --allow-test

## Provenance

Every run records its git revision, the sha256 of the data files, seed, library versions and hardware (`runs/<id>/meta.json`). Decisions and their reasons live in `docs/memory/`; the analysis plan is frozen by git tag before any held-out data is touched.
The raw rows are a private dataset: they and the prompt files built from them (profile texts and reasoning traces) are not published. The tables built from public databases (references, splits, hand-resolved facts, release tables) are, with the terms of their sources in [data/README.md](data/README.md).

## License

| What | Licence |
|---|---|
| Code (`src/`, `scripts/`, `gpu/`, `tests/`) | MIT, [LICENSE](LICENSE) |
| Paper, protocol, documentation, figures and result summaries (`docs/`, `results/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Data tables (`data/curated/`, `data/processed/refs.jsonl`, `data/processed/split_report.json`, `release/`) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): they adapt ChEMBL (CC BY-SA 3.0, attribution and share-alike); openFDA is CC0; NCBI restricts nothing (terms checked 2026-10-18, see [data/README.md](data/README.md)) |
| The private dataset (profile texts, reasoning traces, prompt files), model samples and adapters | not included, and not covered by these licences |

Cite the current ChEMBL paper when you use the tables.
