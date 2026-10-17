# SFT → RL for the inverse-design dataset

Turning `train.jsonl` (1,087 inverse molecular-design SFT examples) into RL.

## What the data is

Each SFT row is a drug-design task: a prompt describing a protein target and a
desired property profile, and a completion that reasons and then gives a
molecule (task C explicitly asks for a **SMILES string**). Fields:
`id, corpus, task_type (A/B/C), drug, group, prompt, completion, text, n_*tokens`.

The key fact for RL: **the answer is a molecule, which is machine-checkable.**
That makes this RLVR, not a judge problem — at least for Tiers 0–3 below. The
`drug` field is the real approved drug for each target, i.e. a known-good anchor.

## Files

- `convert.py` — `train.jsonl` → `rl_train.jsonl`. Keeps `prompt` verbatim
  (it already ends in `<|im_start|>assistant\n<think>\n`, so **don't** re-apply a
  chat template), carries `task_type`/`drug` as `extra_info`, and stashes any
  SMILES it could parse from the SFT answer as a hint.
  Run: `python convert.py train.jsonl rl_train.jsonl`
- `reward.py` — the reward, verl signature
  `compute_score(data_source, solution_str, ground_truth, extra_info) -> [0,1]`.
  Modelled on ether0's reward server.

## The reward, tiered cheap→expensive

| Tier | Checks | Cost | Status |
|---|---|---|---|
| 0 format | is there a parseable SMILES? | µs | done |
| 1 validity | RDKit-sane, drug-like-sized, no junk substructures | µs | done |
| 2 property | fits the requested MW/logP/HBD/HBA/TPSA/QED profile | ms | **placeholder targets — wire to real profile** |
| 3 novelty | not a copy of the reference drug (fingerprint distance) | ms | done |
| 4 binding | does it actually hit the target? (docking / learned model) | s–min | stub, off by default |

Tiers 0–3 run on CPU in milliseconds and are enough to start training today.
Tier 4 is the slow oracle — run it behind an async reward service with a
staleness budget (prime-rl + verifiers, or AReaL online mode), not inline.

## The one thing to finish before training

Tier 2's property targets are a generic oral-drug band. Replace them with the
**real per-target profile**, two options:
1. compute the reference `drug`'s RDKit descriptors (needs a name→SMILES map you
   build once) and score distance to that vector, or
2. regex the numeric constraints out of each prompt and check against them.

Option 1 is more robust; option 2 needs no external lookup. Start with 1.

## Don't throw away the completions

The SFT completions are a free teacher signal. Two good uses:
- **SFT warm-start** then RL (standard; stabilises early RL).
- **On-policy distillation** if you have a stronger model — 9–30× cheaper than
  RL to the same score in the survey's numbers. See `../frontier-rl/`.

Status: `convert.py` and `reward.py` run; reward smoke-tested with RDKit
installed. Tier-2 targets still placeholder.
