# Answer-only RL pilot (arm A5)

GRPO-family RL from the A2 LoRA adapter in answer-only mode (about 50-100 completion tokens per rollout). One process, Hugging Face
`generate`, no vLLM, no TRL. `docs/plan.md` section 6 says what and why (section 2, "Revised before the lock", the controls); this file says
what the code does. GPU evidence recorded here: the parity gate on an A100 (Gates below); the rest of the GPU path is **untested** (last
section). Files: `src/rlexp/grpo.py` (numpy only: maths, schedule, diagnostics, first-valid policy, stop rules), `gpu/rl_grpo.py` (trainer and
gates; reuses `gpu/sft_lora.py`), `gpu/run_rl.sh` (Colab runner), `tests/test_grpo.py` and `tests/test_rl_pack.py` (CPU; torch tests skip without torch).

## One step
1. Draw `--prompts-per-step` (32) distinct Train drugs, one random view each, from a stream of seeded permutations (one per pass). The
   cursor is the whole schedule state. Sample `--group-size` (8) completions each: bf16 autocast, left padding, `do_sample`, T=1.0,
   `top_k=0`, `top_p=1.0`, 256 tokens, stop at `<|im_end|>`, torch seeded `seed * 1_000_003 + step`. The model's own generation config
   is replaced by a clean one (the checkpoint's sets top_k, top_p and temperature).
2. Score the decoded tokens with `rlexp.reward.score`. A completion that hit the length cap is a failure: reward 0, no valid answer, no match
   (`finished_score` is the one seam until metrics v3 takes the rule). It stays in its group and in the update, never masked.
3. `A_i = r_i - mean(group)`; groups with spread <= 1e-9 are dropped; every retained A is divided by the std of all retained A (+1e-6).
   Under `--min-groups` (16) retained groups: one refill of up to 2x prompts at the observed survival rate. Under half of it, rounded up
   (8 of 16, 2 of 3): skip the update, log it, the step still counts (with no retained group the update is skipped, never NaN).
4. Every token of completion i gets `A_i / (T_g * G)` (T_g: tokens of group g, G: retained groups), loss `-sum w * log pi`, summed over
   micro-batches of `--max-tokens-per-batch` (8192), so the split cannot change the gradient. One AdamW step (lr 1e-5, weight decay 0, warmup
   5 steps, clip 1.0). The LM head is an fp32 copy in generation and training; log-probs are taken at the sampling temperature in chunks.
5. Every `--eval-every` (10) steps, and at the last step: all Val rows x `--eval-k` (4), drug-balanced means; the adapter goes to
   `adapters/step-N/` (adapter files plus `revision.json`; never pruned, swapped in so a failed rename keeps the old copy). A step-0 baseline
   eval uses the same sampler. Resumable checkpoint every `--save-every`. The base loads at the revision the init adapter's `revision.json` records.

## Controls and diagnostics
- `--reward-mode validity`: validity-only RL, the training reward is 1.0 for a valid, closed, finished answer and 0.0 otherwise.
  `--reward-mode contrastive`: for such an answer `max(0, reward_v2 - 0.6 * mean T(answer, reference))` over the OTHER drugs of the step
  (0.6 = the reward's similarity weight; one reference per drug; same count-vector Tanimoto), else 0. The floor stops abstention from paying
  (an answer closer to the other drugs than to its own would score below a failure): a bad answer ties with failure, the gradient separates
  the answers better than the generic one from the rest. Before the floor a constant answer's similarity terms cancel over a batch, the others
  do not: it averages `0.05 + 0.3 * (share of drugs with its scaffold)` (plus `1 - 0.05 - 0.6 T - 0.3 s` per drug it copies exactly); the
  floor only adds. Computed over all attempted drugs before the zero-variance filter, recomputed after a refill merge; Val uses the other
  Val drugs. Advantages use the training reward in every mode; `reward_full` is logged. `--profile-mode none`: every Train and Val user turn
  becomes `Drug: unknown.` (rewards still use the reference); one constant answer is then optimal, so the collapse stop is off (logged).
- Per step, on all groups before the zero-variance filter: `top_answer_share`, `distinct_answer_share`, `modal_pairwise_tanimoto`,
  `train_copy_share` (valid answers equal to a Train reference), `reward_validity_term`, `reward_tanimoto_term`, `reward_scaffold_term`,
  `reward_identity` (batch means), `lift_batch` (plan section 2 lift against the other drugs of the batch; exactly 0 for a constant answer),
  `collapse_all` (distinct modal answers over all groups; `collapse_score`, which the stop rule reads, divides by groups with a valid answer).
- Val eval: `fv_tanimoto` (Tanimoto of the first valid, closed, finished answer among a row's `--eval-k` samples, 0 when none; metrics v3's
  endpoint, drug-balanced; **`val` and the patience rule are this**), `fv_lift`, `fv_rate`, `fv_samples` (draws to it, k when none), `val_reward`
  = `reward` (the training reward), `reward_full`, `validity`, `tanimoto`, `tanimoto_valid`, `lift` (the other Val drugs are the others),
  core, exact, truncation, length. The endpoint takes 8 samples: `EVAL_K=8` matches it. Checkpoint selection stays manual.
- Known scorer behaviour: `chem.extract_answer` strips a leading or trailing `*` as markdown, so `*<smiles>` scores like `<smiles>`. Harmless: a
  dummy-atom SMILES is invalid anyway. Views of one drug must share `ref_smiles` and alternates (lift and the contrastive pool take one
  reference per drug); rows that disagree are refused at load, by count.

## Deviations from plan section 6, on purpose
- TRL first: an own loop on HF `generate` and PEFT (short rollouts, training dominates; no engine or weight-sync mismatch). CISPO's clipped
  ratio: the ratio is 1, REINFORCE with a group baseline (one update per rollout batch; `ponytail:` in `Policy.train`). Cap 256, not 8k.
- Overlong samples are not masked: they score 0 and stay in the update (a deployed system cannot use an answer that never stops; 7 of 29
  capped 27B and 25 of 71 capped retrieval samples held a SMILES the scorer accepted). Val also at step 0, from HF `generate`, not vLLM.

## Run
```sh
.venv/bin/python -m pytest -q; PYTHONPATH=src .agent-runs/venv-torch/bin/python -m pytest -q tests/test_rl_pack.py tests/test_grpo.py  # without, with torch
DRY_RUN=1 INIT_ADAPTER=none PLAN_PYTHON=.venv/bin/python sh gpu/run_rl.sh   # counts only, no GPU; needs numpy and rdkit
# Colab A100, repo root. Gates with the A2 adapter (a fresh LoRA has B = 0 and passes parity trivially):
GATE=parity sh gpu/run_rl.sh      # 8 prompts x 4: sampling log-probs vs the training forward, batched / mixed lengths / alone
GATE=smoke  sh gpu/run_rl.sh      # 3 steps, then steps 1-2 + resume + step 3 against an uninterrupted run; PASS/FAIL per check
# The pilot runs. The file keeps its defaults; the pilot overrides them by environment (run a line again to resume it):
export INIT_ADAPTER=<A2 adapter dir> PROMPTS_PER_STEP=16 MIN_GROUPS=8 STEPS=60 LR=2e-5 EVAL_EVERY=10 STOP_AFTER_SECONDS=10000
OUT=.agent-runs/rl_full      sh gpu/run_rl.sh                        # 1. full reward, real profile
OUT=.agent-runs/rl_validity  REWARD_MODE=validity sh gpu/run_rl.sh   # 2. validity-only RL
OUT=.agent-runs/rl_noprofile PROFILE_MODE=none sh gpu/run_rl.sh      # 3. profile-free RL
OUT=.agent-runs/rl_zero      INIT_ADAPTER=none sh gpu/run_rl.sh      # 4. zero-RL ablation: a fresh LoRA, full reward, real profile
OUT=.agent-runs/rl_contrastive REWARD_MODE=contrastive sh gpu/run_rl.sh  # 5. contrastive reward (needs PROMPTS_PER_STEP >= 2)
```
Variables are the flag names in capitals (`LR`, `GEN_BATCH`, `GEN_TOKEN_BUDGET`, ...; `TRAIN_FILE`, `VAL_FILE`, `OUT`, `MODEL`, `ALLOW_SLOW_KERNELS=1`
if the kernels fail). Live view: `tail -f .agent-runs/rl-runner/train.log` and `<out>/metrics.jsonl`. Needs compute capability 8.0+ (bf16).

## Gates
- Parity passes a case at mean |diff| <= 0.03, at most 15% of tokens differing by more than 0.05, and a mean ratio within 2% of 1; the gate
  exits 0 only if the judged cases (batch, mixed lengths, alone) all pass (`PARITY PASS`; else `PARITY FAIL`, exit 1); the mixed batch's shortest
  prompt is printed, not judged. Measured, then chosen, not derived: the accepted A100 runs gave a mean |diff| of 0.015-0.018, 8-11% above
  0.05 and a mean ratio of 0.994-1.000.
- Smoke adds the third update: the resumed run's parameters and AdamW state (moments, step) against the uninterrupted run's, as a relative norm of
  difference: PASS at <= 1e-6 (CPU-equivalent paths reach it exactly, per tests), else only an INFO line with the numbers, like rollout equality: GPU
  nondeterminism changes the rollouts and with them the update, so read the two lines together.

## Outputs and stops
`<out>/metrics.jsonl` (one line per step), `evals.jsonl`, `adapters/step-N/`, `checkpoints/` (latest only), `summary.json` (state). The logs
are rebuilt from the committed state on every save. Final `status`: `complete`; `stopped_time_limit` and `stopped_signal` (resume continues);
`stopped_collapse` (`collapse_score` < 0.7 for 3 steps; off under `--profile-mode none`), `stopped_truncation` (> 30% of a step's samples),
`stopped_patience` (Val `fv_tanimoto` not improved for 3 evals while the train reward rose) are final: a resume refuses them. Resume
identity: options (incl. both modes; not paths, `--steps`, `--save-every`, `--gen-batch`, `--gen-token-budget`, `--max-tokens-per-batch`, time
limit), model revision, Train/Val and init-adapter hashes, dtype, torch/transformers/peft versions, and `objective_sha256` (hash of `reward.py`,
`chem.py`, `grpo.py`, the system turn and the prompt-building source). Another or unrecorded objective refuses the resume ("objective changed:
use a new output directory"; a checkpoint from before the hash resumes only with the files that wrote it). `trainer_sha256` (`rl_grpo.py`,
`sft_lora.py`) is recorded, not pinned: a resume warns if they changed. The smoke gate writes under `<out>/gate-smoke`. Logs hold aggregates only.

## Tested and untested
CPU (`tests/`, mutation-tested, see the report): the maths, loop, resume, diagnostics, stop rules, and the real `Policy` on a tiny torch model
(clipping, the AdamW step, checkpoints restoring optimizer moments and RNG states: a disk resume is bit-exact; sampling arguments; seeding). A
scratch run (`.agent-runs/rl_pack/hf_check.py`) on a tiny random Qwen3.5 (transformers 5.3.0, peft 0.18.1): log-probs match to 1e-7 (fp32). UNTESTED (GPU):
- HF `generate` with left padding through the fla/causal-conv1d GatedDeltaNet kernels on transformers 5.17.0 beyond the parity cases; PEFT 0.21.2
  `from_pretrained(is_trainable=True)` of the real A2 adapter; `torch.load(weights_only=True)` of the optimizer state on torch 2.10 (2.6, 2.14 ran).
- Memory at 128 sequences or 98,304 prompt tokens per `generate` call and 8192-token micro-batches on 40 GB (lower `--gen-batch` or
  `--gen-token-budget` on OOM; a resume may change them); step time; the fp32 head inside `generate` under CUDA autocast; the runner's installs,
  driver check and `timeout` on Colab; `gpu/merge_lora.py` on an RL `adapters/step-N/`.
