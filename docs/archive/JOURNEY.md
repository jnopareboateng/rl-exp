# From SFT file to RL run: the journey, stage by stage

Every stage has the same four parts: **what** we do, **why** it exists, **how to
check it worked**, and **what it looks like when it didn't**. Nothing moves on
until its check passes. Scripts live next to this file.

Data: `train.jsonl`, 1,087 rows of inverse molecular design. Fields: `id, corpus,
task_type (A/B/C), drug, group, prompt, completion, text, n_*tokens`. Prompts are
ChatML and end with `<|im_start|>assistant\n<think>\n`, so generation starts right
at the end of the prompt.

## The shape of the whole thing

```
SFT file ──audit──▶ split by drug ──▶ reward + offline reward tests
                                          │
         SFT model ──▶ baseline sampling ─┴▶ go / no-go on the pass rate
                                          │
                              tiny GRPO run ──▶ read rollouts ──▶ scale
```

RL here means: sample several answers per prompt from the model, score each with
the reward, nudge the model toward answers that scored above their group's
average. Three things must be true or nothing is learned: the model sometimes
gets reward and sometimes doesn't on the same prompt (variance), the reward
measures what we care about (validity), and the loop is stable (the numbers in
Stage 6). The stages below test those three things one at a time, cheapest first.

---

## Stage 0. Fix the three facts everything else depends on

| Fact | Why it decides things | Status |
|---|---|---|
| Base model (which weights, how big) | sets memory, speed, and the chat template | **need from you** |
| Hardware (GPU model, VRAM, how many) | rollouts of thousands of tokens are the main cost | **need from you** |
| What the final answer looks like for tasks A, B, C | the reward has to parse it | Stage 1 tells us |

Until the model and GPU are known, the framework choice below is a default, not a
decision.

## Stage 1. Audit the data (`audit.py`)

**What.** Run `python audit.py train.jsonl` on your machine and paste the block it
prints. It reports counts per task type, how many distinct drugs there are, token
length percentiles, whether the format is what we think it is, and whether a
molecule can be extracted from each SFT completion.

**Why.** Every later number depends on these facts. Two matter most. Completion
length: one sample you sent was 5,262 completion tokens, and RL generates several
answers per prompt, so length sets how much memory and time each step costs.
Extraction rate: if we can't parse the SFT answers, we can't parse the model's.

**Check.** `text starts with prompt` is all rows; `prompt ends with '<think>'` is
all rows; `extracted` is close to `rows` for every task type; `rdkit_valid` is
close to `extracted`.

**If not.** Low extraction means the answer format varies. The printed completion
tails show how; we adapt the extractor before anything else.

## Stage 2. Split by drug (`split.py`)

**What.** `python split.py train.jsonl splits/` writes train/val/test files with
whole drugs held out.

**Why.** Rows about the same drug are near-duplicates. If one is in train and one
in test, a model that memorised the drug scores well on test and we conclude it
generalises. Splitting by `group` makes every test drug genuinely unseen.

**Check.** The script asserts no drug appears in two files. Also look at the
per-file row counts: with few distinct drugs the test file is small, so numbers
from it are noisy. Say so in any result you report from it.

## Stage 3. Build the reward and test it without any model

**What.** Tiers 0 to 3 of `reward.py` exist. Before using them, three offline
tests on the SFT data:

1. **Reference answers score high.** Run the reward on every SFT completion. If the
   known-good answers score low, the reward is wrong.
2. **Junk scores low.** Score deliberately bad answers: invalid text, a single
   atom, a huge molecule. They must get 0 or near it.
3. **The reward must depend on the prompt.** Score each SFT answer against its own
   prompt, then against a prompt from a different drug. If the two score
   distributions match, the reward ignores the target.

**Why test 3 matters, and a correction to what I sent earlier.** The Tier 2
property score I wrote is target-independent: it rewards any drug-like molecule
whatever the prompt asks. A policy trained on it learns to emit one generic
drug-like structure for every prompt. The reward would climb and the model would
get worse at the real task. I labelled Tier 2 a placeholder but did not say it
would be exploited; it would. The fix is to condition the score on the target:
compare the molecule's properties to the reference drug's, or to the constraints
stated in the prompt, and cap similarity to the reference so copying it is not the
best strategy. We decide which after the audit shows what the prompts contain.

**Check.** Test 1 high, test 2 near zero, test 3 clearly lower for mismatched
prompts.

**Reuse before writing (checked against the ether0 source, Apache-2.0).** The
validity gate should not be my two-line stub. ether0's `src/ether0/rewards.py`
has `is_reasonable_molecule` (valence check, a rule that allows one small
counter-ion, a ring-system check against a bloom filter of known rings, and a
REOS-style fingerprint filter) and `contains_bad_substruct`. Install per its
README: `pip install git+https://github.com/Future-House/ether0.git`, then
replace `_is_reasonable` in `reward.py` with calls to these. What ether0 does
not have is the target-conditioned part: its task verifiers (`formula_eval`,
`functional_group_eval`, `valid_mol_eval`, `sol_eval`, reaction evals) each check
a molecule against a specific stated requirement, and your prompts state a target
and a property profile instead. So the validity gate is reused, and the
profile-matching reward has to be built for your prompts.

**Where a model helps (Stage 3).** A small model such as Haiku can read each of
the 1,087 prompts once, offline, and write the stated requirements as structured
fields (numeric ranges, yes/no properties). We spot-check a sample against the
prompts, then score molecules against those fields with RDKit. That replaces
hand-written regexes with something we can measure. It does not belong inside the
RL reward loop for computing properties: a model's logP or TPSA is an estimate,
RDKit's is exact, and a judge in the loop adds cost, noise and something for the
policy to exploit.

**What the reward does and doesn't mean.** Tiers 0 to 3 teach "produce a valid
molecule whose properties fit this target's profile". That is a real stepping
stone and not the same as "this molecule binds the target". Only Tier 4 (docking
or a learned affinity model) measures binding, and it is slow, so it comes after
the cheap tiers work.

## Stage 4. Start from the SFT model

**What.** RL starts from a model that already follows the format. If the model
you have is already fine-tuned on this file, use it. If not, fine-tune on the
training split first.

**Why.** From a base model that doesn't produce the format, almost every sample
scores 0, nothing differs within a group, and RL gets no signal.

**Check.** Sample a few prompts by hand: does the output have a reasoning section
and then a parseable molecule?

## Stage 5. Baseline sampling: the go / no-go test

**What.** For each prompt in a sample of the training split, draw 8 answers from
the SFT model, score them, and record:
- mean reward, and the share of answers with a parseable molecule
- for each prompt, whether all 8 scored the same. These are **dead groups**: they
  give GRPO zero learning signal (every answer equals the group average).
- how many answers were cut off at the length limit

**Why.** GRPO learns only from differences inside a group. If most groups are all
0 or all identical, the run will sit still no matter how long it trains. This test
costs a few GPU-minutes and tells us whether the full run is worth starting.

**Check.** A healthy start has a mean reward well above 0 and well below the
maximum, and most groups containing mixed scores. Truncation under a few percent;
more means raise the length limit or the model is rambling.

**If not.** All zeros: fix extraction or the format, or warm-start with more SFT.
All high and identical: the reward is too easy, tighten it.

## Stage 6. The smallest possible RL run

**What.** GRPO for around 50 steps with TRL's `GRPOTrainer`, passing `reward.py`
as a plain Python function. TRL is the default because the reward is an ordinary
callable, the loop is short enough to read, and it runs on one GPU. The survey
(tracks 06 and 10) recommends it for prototyping and `prime-rl` or `verl` once an
expensive reward needs asynchronous scoring. Settings the survey found converge
across labs: no KL penalty (`beta=0`), a Dr. GRPO or DAPO style loss, drop groups
with zero variance. Check the option names against your installed TRL version;
they have changed between releases.

**Why small.** To find bugs while a mistake costs minutes.

**The five numbers to watch, every step:**
1. mean reward: should rise, noisily
2. within-group reward spread: should stay above zero. Falling to zero means
   collapse onto one answer
3. completion length: sudden growth or shrinkage means the model found a shortcut
4. entropy of its outputs: a steady fall to near zero means it stopped exploring
5. share of parseable answers: must not drop

**Check.** Reward up, spread above zero, length stable, parse share stable.

## Stage 7. Read the rollouts yourself

**What.** Open 20 sampled answers from step 0 and 20 from the last step. Read the
molecules and reasoning.

**Why.** The numbers can improve while the behaviour gets worse. This is how
reward hacking shows up: the same molecule for every prompt, a molecule that
games one descriptor, reasoning that gets shorter and emptier. No dashboard
replaces reading.

**Check.** Different prompts produce different, sensible molecules, and the
reasoning still addresses the target.

## Stage 8. Scale, one change at a time

Longer run; then the held-out val split to check generalisation to unseen drugs;
then Tier 4 binding score behind an asynchronous scoring service; then, if the
single-turn version is solid, tools the model can call and revise with. Change
one thing per run so any regression has one cause.

---

## What I need from you to move

1. Run `audit.py` and paste the output (Stage 1).
2. Which model are you training, and what GPU do you have (`nvidia-smi` output)?
3. Is the `rl-exp` folder set up for training already (an environment, a model
   download), or are we starting from nothing?
