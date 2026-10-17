# Answer-only RL pilot (arm A5)

GRPO-family RL from the A2 LoRA adapter in answer-only mode (about 50-100 completion tokens per rollout). One process, Hugging Face
`generate`, no vLLM, no TRL. `docs/plan.md` section 6 says what and why (section 2, the controls); this file says what the code does
(`src/rlexp/grpo.py`, numpy only; `gpu/rl_grpo.py`; `gpu/run_rl.sh`; `tests/test_grpo.py`, `tests/test_rl_pack.py`). GPU evidence recorded here:
the parity gate on an A100 (Gates below); the rest of the GPU path is **untested** (last section).

## One step
1. Draw `--prompts-per-step` (32) distinct Train drugs, one random view each, from a stream of seeded permutations (one per pass). The
   cursor is the whole schedule state. Sample `--group-size` (8) completions each: bf16 autocast, left padding, `do_sample`, T=1.0,
   `top_k=0`, `top_p=1.0`, 256 tokens, torch seeded `seed * 1_000_003 + step`. A rollout stops at every token vLLM stops at: `<|im_end|>`, the
   ids in the checkpoint's `generation_config` eos list and the tokenizer's eos (recorded as `stop_token_ids`). The model's own generation
   config is replaced by a clean one (the checkpoint's sets top_k, top_p and temperature).
2. Score the decoded tokens with `rlexp.reward.score`. A completion that hit the length cap is a failure: reward 0, no valid answer, no match
   (`finished_score` is the one seam until metrics v3 takes the rule). It stays in its group and in the update, never masked.
3. `A_i = r_i - mean(group)`; groups with spread <= 1e-9 are dropped; every retained A is divided by the std of all retained A (+1e-6).
   Under `--min-groups` (16) retained groups: one refill of up to 2x prompts at the observed survival rate. Under half of it, rounded up
   (8 of 16, 2 of 3): skip the update, log it, the step still counts (with no retained group the update is skipped, never NaN).
4. Every token of completion i gets `A_i / (T_g * G)` (T_g: tokens of group g, G: retained groups), loss `-sum w * log pi`, summed over
   micro-batches of `--max-tokens-per-batch` (8192), so the split cannot change the gradient. One AdamW step (lr 1e-5, weight decay 0, warmup
   5 steps, clip 1.0). The LM head is an fp32 copy in generation and training; log-probs are taken at the sampling temperature in chunks.
5. Every `--eval-every` (10) steps, and at the last step: all Val rows x `--eval-k` (8, the endpoint's), drug-balanced means; the adapter goes
   to `adapters/step-N/` (adapter files plus `revision.json`; never pruned, swapped in so a failed rename keeps the old copy). A step-0 baseline eval
   uses the same sampler. Resumable checkpoint every `--save-every`. The base loads at the revision the init adapter's `revision.json` records;
   `--revision` names it (required with `--init-adapter none`, which has no adapter to read it from).

## Controls and diagnostics
- `--reward-mode validity`: validity-only RL, the training reward is 1.0 for a valid, closed, finished answer and 0.0 otherwise.
  `--reward-mode contrastive`: for such an answer `max(0, reward_v2 - 0.6 * mean T(answer, reference))` over the OTHER drugs of the step
  (0.6 = the reward's similarity weight; one reference per drug), else 0. The floor stops abstention from paying (an answer closer to the
  other drugs than to its own would score below a failure): a bad answer ties with failure, the gradient separates the answers better than the
  generic one from the rest. Before the floor a constant answer's similarity terms cancel over a batch, the others do not: it averages
  `0.05 + 0.3 * (share of drugs with its scaffold)` (plus `1 - 0.05 - 0.6 T - 0.3 s` per drug it copies exactly); the floor only adds. Computed
  over all attempted drugs before the zero-variance filter, recomputed after a refill merge; Val uses the other Val drugs. Advantages use the
  training reward in every mode; `reward_full` is logged. `--profile-mode none`: every Train and Val user turn becomes `Drug: unknown.`
  (rewards still use the reference); one constant answer is then optimal, so the collapse stop is off (logged).
- Per step, on all groups (before the zero-variance filter): `top_answer_share`, `distinct_answer_share`, `modal_pairwise_tanimoto`, `train_copy_share`
  (valid answers equal to the reference of ANOTHER Train drug; an exact hit on the drug's own is recall), the four `reward_*` term means, `lift_batch`
  (plan section 2; exactly 0 for a constant answer), `collapse_all` (over all groups; `collapse_score`, which the stop rule reads, divides by groups
  with a valid answer). Both count exact structures, so near-duplicate answers (a molecule and its methyl or halogen variants) evade them:
  `modal_pairwise_tanimoto` above 0.5 for 3 steps in a row prints a WARNING (`similar_streak`), never a stop.
- Val eval: `fv_tanimoto` (Tanimoto of the first valid, closed, finished answer among a row's `--eval-k` samples, 0 when none; metrics v3's
  endpoint, drug-balanced; **`val` and the patience rule are this**), `fv_lift`, `fv_rate`, `fv_samples` (draws to it, k when none), `val_reward`
  = `reward` (the training reward), `reward_full`, `validity`, `tanimoto`, `tanimoto_valid`, `lift` (the other Val drugs are the others),
  core, exact, truncation, length. Checkpoint selection stays manual.
- Known scorer behaviour: `chem.extract_answer` strips markdown emphasis (`*`, backticks) at the ends of a token, so `*<smiles>` scores like
  `<smiles>`: accepted leniency, not exploitable (it gains nothing over writing the stripped string, and Train and Val share the extractor); a wildcard
  atom is invalid anyway. Views of one drug must share `ref_smiles` and alternates; rows that disagree are refused at load, by count.
- Deviations from plan section 6, on purpose: an own loop on HF `generate` and PEFT, not TRL (short rollouts, no engine or weight-sync mismatch);
  ratio 1, REINFORCE with a group baseline, not CISPO's clipped ratio (one update per rollout batch; `ponytail:` in `Policy.train`); cap 256, not 8k;
  Val also at step 0, from HF `generate`; overlong samples are not masked but score 0 and stay in the update (a deployed system cannot use an
  answer that never stops; 7 of 29 capped 27B and 25 of 71 capped retrieval samples held a SMILES the scorer accepted).

## Run
```sh
DRY_RUN=1 INIT_ADAPTER=none REVISION=<commit> PLAN_PYTHON=.venv/bin/python sh gpu/run_rl.sh   # counts only, no GPU; needs numpy and rdkit
# Colab A100, repo root. Gates with the A2 adapter (a fresh LoRA has B = 0 and passes parity trivially):
GATE=parity sh gpu/run_rl.sh      # 8 prompts x 4: sampling log-probs vs the training forward, batched / mixed lengths / alone
GATE=smoke  sh gpu/run_rl.sh      # 3 steps, then steps 1-2 + resume + step 3 against an uninterrupted run; PASS/FAIL per check
# The pilot runs. The file keeps its defaults; the pilot overrides them by environment (run a line again to resume it):
export INIT_ADAPTER=<A2 adapter dir> PROMPTS_PER_STEP=16 MIN_GROUPS=8 STEPS=60 LR=2e-5 EVAL_EVERY=10 EVAL_K=8 STOP_AFTER_SECONDS=10000
OUT=.agent-runs/rl_full      sh gpu/run_rl.sh                        # 1. full reward, real profile
OUT=.agent-runs/rl_validity  REWARD_MODE=validity sh gpu/run_rl.sh   # 2. validity-only RL
OUT=.agent-runs/rl_noprofile PROFILE_MODE=none sh gpu/run_rl.sh      # 3. profile-free RL
OUT=.agent-runs/rl_zero      INIT_ADAPTER=none REVISION=<A2 revision.json> sh gpu/run_rl.sh   # 4. zero-RL ablation: fresh LoRA, full reward, real profile
OUT=.agent-runs/rl_contrastive REWARD_MODE=contrastive sh gpu/run_rl.sh  # 5. contrastive reward (needs PROMPTS_PER_STEP >= 2)
```
Variables are the flag names in capitals (`LR`, `GEN_BATCH`, `GEN_TOKEN_BUDGET`, `TRAIN_FILE`, `OUT`, `ALLOW_SLOW_KERNELS=1` if the kernels fail). Live view:
`tail -f .agent-runs/rl-runner/train.log` and `<out>/metrics.jsonl`. Needs compute capability 8.0+ (bf16).

## Gates
- Parity passes a case at mean |diff| <= 0.03, at most 15% of tokens differing by more than 0.05, and a mean ratio within 2% of 1; the gate exits 0
  only if the judged cases (batch, mixed lengths, alone) all pass (`PARITY PASS`; else `PARITY FAIL`, exit 1); the mixed batch's shortest prompt is
  printed, not judged. Measured, then chosen: the accepted A100 runs gave a mean |diff| of 0.015-0.018, 8-11% above 0.05 and a mean ratio of 0.994-1.000.
- Smoke FAILS unless the resumed process, right after the checkpoint loads and before any step, holds exactly the weights, parameter order, AdamW
  state and RNG states the interrupted run ended with (max abs difference 0; no sampling in between, so no GPU noise). After the third update the
  parameters and AdamW state are compared with the uninterrupted run's (relative norm): PASS at <= 1e-6, else only INFO, like rollout equality
  (GPU nondeterminism changes the rollouts and the update): read the two lines together.

## Outputs and stops
`<out>/metrics.jsonl` (a line per step), `evals.jsonl`, `adapters/step-N/`, `checkpoints/` (latest), `summary.json` (state); the logs are
rebuilt from the committed state on every save. Final `status`: `complete`; `stopped_time_limit` and `stopped_signal` (resume continues);
`stopped_collapse` (`collapse_score` < 0.7 for 3 steps; off under `--profile-mode none`), `stopped_truncation` (> 30% of a step's samples),
`stopped_patience` (Val `fv_tanimoto` not improved for 3 evals while the train reward rose) are final: a resume refuses them. Resume
identity: options (incl. both modes and `--revision`; not paths, `--steps`, `--save-every`, `--gen-batch`, `--gen-token-budget`,
`--max-tokens-per-batch`, time limit), model revision, Train/Val and init-adapter hashes, dtype, the torch, transformers, peft, rdkit and numpy
versions (rdkit is pinned in the runner), and `objective_sha256`: `reward.py`, `chem.py`, `grpo.py`, the system turn, the prompt building and the
reward code of `rl_grpo.py` (`OBJECTIVE_FUNCTIONS`: stop tokens, length rule, validity reward, contrastive assembly). Another or unrecorded objective
refuses the resume ("objective changed: use a new output directory"); so does a version drift. `trainer_sha256` (`rl_grpo.py`, `sft_lora.py`) is recorded, not pinned.

## Tested and untested
CPU (`tests/`, mutation-tested, see the report): the maths, loop, resume, diagnostics, stop rules and the real `Policy` on a tiny torch model (clipping,
AdamW step, checkpoints restoring moments and RNG states, sampling arguments, seeding); a scratch run on a tiny random Qwen3.5 (`hf_check.py`; transformers
5.3.0, peft 0.18.1) matches log-probs to 1e-7 in fp32. UNTESTED (GPU):
- HF `generate` with left padding through the fla/causal-conv1d GatedDeltaNet kernels on transformers 5.17.0 beyond the parity cases; PEFT 0.21.2
  `from_pretrained(is_trainable=True)` of the real A2 adapter; `torch.load(weights_only=True)` of the optimizer state on torch 2.10 (2.6, 2.14 ran).
- Memory: prefill grows faster than `--gen-token-budget` (rows x longest prompt; the attention mask is rows x length^2): 81920 ran out of memory at
  step 27 on a 40 GB card, 49152 (the default) ran cleanly. `--gen-batch` and the budget are operational but set the sub-batches, so the sampled
  tokens: a resume may change them and then continues the run, it does not replay its rollouts. Also untested: step time, the fp32 head under CUDA
  autocast, the runner's Colab installs and `timeout`, `gpu/merge_lora.py` on an RL adapter.
