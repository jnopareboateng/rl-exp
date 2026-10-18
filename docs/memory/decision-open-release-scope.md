---
name: decision-open-release-scope
description: 'Public repo: code, docs, results and public-database tables; private rows and runs stay local'
type: decision
updated: 2026-10-18
---
The owner asked on 2026-10-18 to open-source everything, and `jnopareboateng/rl-exp` is public. Public: the code (MIT), the paper, protocol and docs (CC BY 4.0), `results/`, and the tables built from public databases (CC BY-SA 4.0, because ChEMBL is share-alike): `data/curated/`, `data/processed/refs.jsonl`, `data/processed/split_report.json` and `release/`, with the checked source terms in `data/README.md`. Not public: the private dataset (`data/raw/`), every prompt file in `data/processed/` (train, val, test_*, arms/), the run outputs in `runs/` (they can echo profile text) and the adapters. `.gitignore` and the local `.git/hooks/pre-push` enforce this.

**Why:** the private dataset's ownership and licence are not recorded (`docs/benchmark_card.md`, section 3); the owner first said not to publish datasets and then to open-source everything, and publication cannot be undone, so the held-back files wait for their statement that they hold the rights.
**How to apply:** to publish prompt files, raw rows, runs or adapters, get that statement first, then in one commit change `.gitignore`, the hook, the repo AGENTS.md rule on prompt files, the README, the card (section 7) and the paper's Section 9. The history was rewritten once before the repo went public (the original, with the derived tables, is in `tmp/backup/rl-exp-full-history-before-dataset-removal.bundle`); commit ids in older docs such as `1f44e4d` refer to it.
Supersedes: decision-datasets-not-published (2026-10-18).
