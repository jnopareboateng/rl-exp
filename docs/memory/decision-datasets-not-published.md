---
name: decision-datasets-not-published
description: 'Datasets stay local: no raw rows or derived tables are pushed; remote history was rewritten'
type: decision
updated: 2026-10-18
---
The project owner said not to publish the datasets. Nothing derived from the private rows is tracked or pushed: `data/raw/`, `data/processed/*` (prompt files, `refs.jsonl`, `split_report.json`), `data/curated/` and `release/` are ignored; only `data/MANIFEST.sha256` (file hashes) stays tracked because the pre-registration tags hash it. The private GitHub repo `jnopareboateng/rl-exp` (origin, SSH alias `github-jnopareboateng`) was first pushed with those tables inside (the prompt rows were never in it); on 2026-10-18 the history was rewritten with `git filter-branch --prune-empty` to drop them from every commit and force-pushed, the tag messages and hashes unchanged. A local `.git/hooks/pre-push` refuses any push whose history contains dataset paths.

**Why:** an explicit instruction from the owner; the first push preceded their message, and GitHub still serves the pre-rewrite commits by id (private repo, owner only) until the repo is deleted (the token lacks `delete_repo`).
**How to apply:** never `git add -f` dataset files or run `git push` without the hook; the full pre-rewrite history is in `tmp/backup/rl-exp-full-history-before-dataset-removal.bundle`, and commit ids cited in older docs (for example `1f44e4d` in `docs/prereg-v1.md`) refer to it. To publish the release tables later, build them with `scripts/make_release.py` after the licence and terms are settled and remove the ignore rule on purpose.
