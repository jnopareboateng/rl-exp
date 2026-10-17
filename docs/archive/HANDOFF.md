# Handoff: continuing the SFT-to-RL work locally

Written 2026-10-16 for whoever continues this in the user's local folder (Claude
Code on their machine, or the user themselves). The work so far happened in a
cloud session that could not read the user's data, so nothing below has been run
against the real file yet.

## The goal and how the user wants to work

Start real RL on the user's own data, drawing on the survey of frontier-lab RL
stacks (`../frontier-rl/`, start with its `README.md`). The user wants the process
to be mechanical and wants to understand every part of it, so explain each stage
and check it before moving on. `JOURNEY.md` is the stage-by-stage plan.

## The data

`train.jsonl` (the user's path: `~/projects/justjosh/rl-exp/train.jsonl/train.jsonl`
in WSL, note the doubled name), 1,087 rows of inverse molecular design.
Fields: `id, corpus, task_type (A/B/C), drug, group, n_tokens, n_prompt_tokens,
n_completion_tokens, prompt, completion, text`.
- Prompts are ChatML and end with `<|im_start|>assistant\n<think>\n`. Do not apply
  a chat template on top.
- The answer is a molecule. Task C prompts ask for a SMILES string. Whether A and B
  end in a SMILES is unconfirmed.
- `drug` is the approved drug the row is about; `group` equals it in the sample
  seen. Split by `group`.
- Sample completions were 1,585 and 5,262 tokens. Memory will be set by this.
- Do not read the whole file or paste whole prompts and completions into chat.
  Work with counts, lengths and short tails. One earlier response was stopped by a
  safety classifier after the first sample of this data was shown, so keep the
  discussion at the level of method and statistics.

## What exists in this folder

| File | State |
|---|---|
| `JOURNEY.md` | The eight-stage plan. Read first. |
| `audit.py` | Stage 1. Tested on synthetic data only. Run it on the real file. |
| `split.py` | Stage 2, split by drug. Tested on synthetic data only. |
| `convert.py` | SFT rows to an RL prompt file (verl-style fields). Untested on real data. |
| `reward.py` | Tiered reward, verl signature. Smoke-tested on one aspirin string. See below. |

## Known problems in `reward.py`, in the order to fix them

1. **Tier 2 ignores the target.** It scores generic drug-likeness for every prompt,
   so a policy trained on it would converge on one generic molecule. It must be
   conditioned on each prompt's target or profile. `JOURNEY.md` Stage 3 has the
   offline test that proves it (score answers against their own prompt vs another
   drug's prompt; the distributions must differ).
2. **The validity gate is a stub.** Replace `_is_reasonable` with ether0's
   `is_reasonable_molecule` and `contains_bad_substruct` (Apache-2.0, install:
   `pip install git+https://github.com/Future-House/ether0.git`). Verified present
   in `src/ether0/rewards.py` at commit `b77e0d8`. It was not installed or run in
   the cloud session, so the integration is untested.
3. **The SMILES extractor is a guess** (`_extract_smiles`, regex on "SMILES:" or a
   code fence). `audit.py` reports how often it succeeds per task type; fix it
   from the printed completion tails.

## Where a model helps and where it doesn't

User suggestion, agreed with a boundary. A small model (Haiku-class) is a good fit
for one-off offline steps: turning each prompt's stated requirements into
structured fields, and extracting answers the regex misses. Spot-check a sample
against the prompts and report the agreement rate. It should not compute
molecular properties inside the reward loop (RDKit is exact; model estimates are
noisy and exploitable) and an LLM judge in the loop should wait until there is a
part of the task with no checker.

## Next actions

1. Run `python audit.py <path to train.jsonl>` and review the output with the user.
2. Ask the user: base model, GPU (`nvidia-smi`), whether A and B end in a SMILES.
3. Run `split.py`.
4. Stage 3: install ether0, swap the validity gate, build the target-conditioned
   score (likely via model-parsed requirements), run the three offline reward
   tests.
5. Stage 5 baseline sampling before any training.

Framework default for the first run is TRL's `GRPOTrainer` (reward is a plain
callable; one GPU). Check argument names against the installed version. See
`../frontier-rl/tracks/06-open-frameworks.md` and `10-algorithms-rewards.md` for
why, and for the point at which to move to prime-rl or verl (an expensive,
asynchronous reward such as docking). `../frontier-rl/tracks/13-xiaomi-mimo-verl.md`
has the reward-hygiene ideas worth borrowing then: treat infrastructure failures
as neutral rather than as wrong answers, and keep ground truth out of the
agent's reach.
