# Repo rules for agents (adds to the global rules; may only strengthen them)

- Read `docs/plan.md` (what and why), `docs/contracts.md` (exact schemas and signatures) and `docs/memory/MEMORY.md` (durable facts) before changing anything. Code wins over docs: when they disagree, fix the doc in the same commit.
- Layout: library in `src/rlexp`, runnable analyses in `scripts/`, GPU pack in `gpu/`, tests in `tests/`. Run from the repo root with `.venv/bin/python`; the package is installed editable.
- The data is drug pharmacology and structures. Print counts, lengths, rates and short structural facts only; never print or log whole prompts or completions. Raw and processed prompt files are never committed.
- Test splits (`test_t`, `test_r`) are touched only by the locked evaluation after tag `prereg-v1`. Reward weights, decoding, model selection and gates use Train rollouts of untrained models and Val only.
- A contract changes only in `docs/contracts.md` plus the code and tests that implement it, in one commit. New dependencies need a reason in the commit message.
- Workers write results to disk after every item (a session restart wipes scratch). Scratch lives in `.agent-runs/` (ignored). Durable facts go to `docs/memory/` via the chronicle skill; results that feed the paper go to `results/` with the run's `meta.json` fields.
- Review: non-trivial code (reward, splits, metrics, training) gets an independent cross-vendor review before it is relied on; findings and responses are summarised in `docs/plan.md`'s review log.
- Git: commit by concept; no AI trailers; identity is the personal account (`jnopareboateng`) set in the repo config; push only when asked.
