---
name: decision-open-release-scope
description: 'Public repo: code, docs, results, public-database tables; in-house rows and runs stay local'
type: decision
updated: 2026-10-18
---
The owner asked on 2026-10-18 to open-source everything, and `jnopareboateng/rl-exp` is public. Public: the code (MIT), the paper, protocol and docs (CC BY 4.0), `results/`, and the tables built from public databases (CC BY-SA 4.0, because ChEMBL is share-alike): `data/curated/`, `data/processed/refs.jsonl`, `data/processed/split_report.json` and `release/`, with the checked source terms in `data/README.md`. Not public: the in-house dataset (`data/raw/`), every prompt file in `data/processed/` (train, val, test_*, arms/), the run outputs in `runs/` (they can echo profile text) and the adapters. `.gitignore` and the local `.git/hooks/pre-push` enforce this.

**Why:** the dataset's provenance, ownership and licence are not documented. Asked, the owner had no answer and chose to call it an in-house dataset: the paper and the public docs say so, state that its provenance is undocumented and list that as a limitation. Publication cannot be undone, so it stays unpublished until its rights are established.
**How to apply:** to publish prompt files, raw rows, runs or adapters, first get a statement that the owner holds the rights, then in one commit change `.gitignore`, the hook, the repo AGENTS.md rule on prompt files, the README, the card (section 7) and the paper's Section 8. The history was rewritten once before the repo went public (the original, with the derived tables, is in `tmp/backup/rl-exp-full-history-before-dataset-removal.bundle`); commit ids in older docs such as `1f44e4d` refer to it.
Supersedes: decision-datasets-not-published (2026-10-18).
