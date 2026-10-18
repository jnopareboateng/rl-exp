# Track 13 (addendum, 2026-10-05): XiaomiMiMo/verl

Added after the main survey. Track 09 covered Xiaomi's MiMo models but not this repo.
Read directly from a clone at the fork commit; evidence tag = read-in-code unless noted.

## What it is

`github.com/XiaomiMiMo/verl` is the **agentic RL training code for MiMo-V2.6**. It is a fork of
verl at `2781c1e4` (0.9.0.dev) with **one** commit, `e2b9fc0` (a Xiaomi author, 2026-09-21):
**191 files changed, +30,672 / -88**. No upstream merges since. The recipe it reproduces is
Section 7 of the MiMo-V2.6 technical report, *Scaling Reinforcement Learning Towards
Self-Improvement* (`MiMo_V2_6_technical_report.pdf`, HF).

Five runnable RL environments, each with a launch script and an `env.example`:

| Domain | Task family | Verifier | Script |
|---|---|---|---|
| Code | Software engineering | Executable tests | `scripts/code/train.sh` |
| Cyber | Vulnerability reproduction | Rule checks | `scripts/arvo/arvo.sh` |
| General | Knowledge work | **Rubric-based judging** | `scripts/general/general.sh` |
| Visual | Web development | Visual grading | `scripts/design/webdev.sh` |
| Music | Symbolic composition | Rule checks | `scripts/design/music.sh` |

Also released: dataset `XiaomiMiMo/MiMo-V2.6-RL-oss`, model
`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, docker `xiaomimimo/mimo-v2.6-rl-oss`.

Two pinned submodules, **not populated by a plain `git clone`**:
- `third_party/mimoagent-osr` — fork of mini-swe-agent; harnesses, tools, execution
  environments and the graders themselves (`run_verify.py`, `verify.py`, `rubrics.json`,
  `verifier_meta.json`). Used by Code / Cyber / General / Visual, driven from verl AgentLoops
  in `recipes/{arvo,general,design}/agent_loop.py`.
- `third_party/uni_agent` — fork of `verl-project/uni-agent`; model gateway plus TransferQueue
  trajectory capture. Used by Code only, via `recipes/code/mimoagent_runner.py`.

The two submodules do not import each other; all glue is in `recipes/`. Music uses neither.

## The four trainer-side contributions

These are the genuinely new material relative to tracks 01-12, which covered objectives and
infrastructure but not the bookkeeping for *messy* agentic rollouts.

### 1. Infrastructure failure is not a wrong answer

`algorithm.invalid_reward_value` (General recipe sets `-999`). In
`compute_grpo_outcome_advantage` (`verl/trainer/ppo/core_algos.py`), a trajectory whose summed
score equals the sentinel is marked invalid and its score **replaced by the mean of the valid
scores in its own `index[i]` group** (`valid_by_uid: dict[Any, list] = defaultdict(list)`), so it
receives ~zero advantage instead of being trained as a failure.

The environment layer supplies the sentinel with a named cause. In
`recipes/general/general_agent/environment.py`, `_mask()` returns
`error_category = REWARD_TESTBED_CORRUPTED` plus a `reward_error` tag, for:
`judge_key_missing`, `mcp_backend_down`, `missing_run_verify`,
`missing_or_invalid_reward_json`, `reward_out_of_range` (reward must be finite and in [0,1]).

### 2. Sign-aware, mass-conserving token penalties (`adv_signed`)

`ToolCallErrorPenaltyConfig` (`verl/trainer/config/algorithm.py`) takes a per-token mask of
model-attributable tool-call errors and one of
`MARKED_TOKEN_PENALTY_STRATEGIES = {"monitor", "mask", "adv_reduction", "adv_set", "adv_signed"}`.
`monitor` records only; `mask` drops the tokens from the response loss; `adv_reduction` /
`adv_set` subtract from or overwrite their post-normalization advantages.

`adv_signed` (`verl/trainer/ppo/signed_rebalance.py`, 215 lines) is the interesting one:
- marked token in a **positive** sample -> advantage set to 0 (stop reinforcing the span)
- marked token in a **negative** sample -> advantage x kappa (>= 1)
- each sign then conserves its total mass over the whole train batch: removed positive mass is
  returned to clean positive tokens by one factor `alpha >= 1`; added negative mass is taken back
  from clean negative tokens by one factor `beta <= 1`. Clamped to
  `algorithm.signed_min_scale: 0.5` / `signed_max_scale: 2.0`; a clamped side is not conserved.
- optional per-row `token_weights` (the `prompt-mean` loss weights) so "mass" means gradient mass
  under that aggregation — the one documented extension over the internal framework.

The docstring states the failure it replaces: a whole-turn `adv_set -1` *injects net negative mass
into the batch, which with no KL and no entropy bonus flattens the policy and lengthens outputs*.

### 3. In-group length penalty that only touches correct rollouts

`verl/utils/length_penalty.py` (364 lines). For a rollout with `reward >= pass_threshold`,
penalize it against the **quantile of the passed rollouts in the same prompt group**:
- metrics: `turns`, `input_tokens`, `output_tokens`; excess per metric
  `max(0, (value - group_anchor) / group_anchor)`; combined by `max` / `mean` / `weighted`
- deadzone `excess_threshold` suppresses penalty, then a convex ramp `t**penalty_exponent` to
  `max_penalty` at `excess_saturate`
- failed rollouts untouched; `min_pass_rate` skips a group whose pass fraction is too low
- pure scalar functions over control-plane metadata, computed one uid group at a time at the
  accept boundary; additive counters finalized once per step so dashboards match the old
  whole-batch implementation

General recipe values: `max_penalty 0.1`, `deadzone 0.3`, `saturate 1.0`, `pass_threshold 0.5`,
`penalty_exponent 1.5`, `anchor_quantile 0.5`, `min_pass_rate 0.5`.

### 4. Hidden verifiers enforced by container topology

In the General environment, the pod is split into a **main** container (agent, sees
`/work/workspace` RW only) and a **sidecar** that serves the environment's MCP tools over
streamable-http on the shared network namespace, owns the backing SQLite state
(`/work/system/<mcp>/state.db`) and runs the verifier. Main never mounts `/work/system`, so
"DB isolation is physical — the agent cannot bypass the MCP tools with direct SQL."

`VERIFIER_FILES = (run_verify.py, verify.py, rubrics.json, answer_key.json, verifier_meta.json,
_helpers.py)` are uploaded **to the sidecar only, at reward time**, so the agent can never read
the rubric or the answer key during the rollout. The verifier writes `reward.json` and
`reward_detail.json`; `_annotate_rubric_kinds` joins each rubric result back to
`verifier_meta.json` to recover `method` (rule | llm), `weight`, `gate`, `dim` (display only).
`src_protect` is verify.py's own source-conservation gate. `reward_binary_fullscore` optionally
binarizes a continuous rubric score while keeping `raw_reward` for logging.

## Also noted

- **Experimental and honestly labelled.** `DeepFailureMaskConfig` computes
  `k = ceil(alpha * median(num_turns of successful trajectories))` per group and clears
  `response_mask` from turn k for negative-advantage trajectories (`mask_failure`) or for all
  (`mask_both`); advantages untouched, groups with no success untouched, fails closed and reports
  `penalty/deep_failure_mask_status`. Its own docstring calls it "a biased estimator (removes
  negative gradient mass) used only to test where deep-turn entropy growth comes from".
  `RepetitionPenaltyConfig` likewise has an EXPERIMENTAL `early_stop` strategy.
- **Repetition detection runs in the agent framework at scoring time**, not in the trainer: a
  port of `dirty_repetition` (an `ngram_size`-token n-gram repeated `min_repeat` times inside a
  `window_size` window) writes a `repetition_mask` and may zero the session reward; the trainer
  then applies monitor / mask / adv_reduction / adv_set / early_stop.
- **General recipe scale** (`recipes/general/config/general.yaml`): `max_prompt_length 49152`,
  `max_response_length 212992`; `adv_estimator: grpo` with `norm_adv_by_std_in_grpo: false`
  (i.e. Dr. GRPO), `use_kl_in_reward: false`, `use_kl_loss: false`, `clip_ratio: 0.2` flat (no
  clip-higher), `entropy_coeff: 0`, `loss_agg_mode: prompt-mean`, `lr 2e-6`,
  `train_batch_size 64`, `ppo_mini_batch_size 64`, temperature 1.0 / top_p 0.95,
  `filter_groups.enable: true` on `reward` with `max_inflight_gen_batches: 2`, 4 nodes x 8 GPUs,
  Megatron TP8 with full offload.
- ~20 new `*_on_cpu.py` tests ship with the fork, incl.
  `tests/recipes/code/test_code_recipe_on_cpu.py` (653 lines) — the penalties and the
  rebalance are testable without GPUs.

## Transfer to biology

1. **`invalid_reward_value` plus named `reward_error` tags is the most directly reusable thing in
   the repo.** A docking job that OOMs, a folding server that times out, a licence check that
   fails, an assay API that 500s — none of those is a bad molecule, and training them as negatives
   is a silent, systematic bias toward whatever the oracle happens to compute cheaply. Group-mean
   imputation gives such a trajectory zero advantage, which is the honest answer.
2. **The General arm is the template for rubric-graded experimental design**, and it answers the
   gap matrix's "no rubric/generative reward models" cell with runnable code: rubric items with
   `method` rule-or-llm, `weight`, `gate` (a conjunctive veto), `dim`, graded by a script in a
   sidecar the agent cannot read, with the answer key uploaded only at reward time.
3. **The sidecar topology generalizes to any oracle that must not be probed.** Holding the
   ground truth and the state DB in a container the agent has no mount for is a stronger
   guarantee than prompt-level instructions, and it is the pattern to use if a reward oracle is
   itself a queryable service (a structure predictor, a property model, a LIMS).
4. **The length penalty is the cleanest open fix for the turn-count bloat** that long-horizon
   bio-agent RL will hit: penalize only the runs that already succeeded, relative to their own
   group's median turn count, with a deadzone so ordinary variation is free.
5. `adv_signed` matters once the agent calls real instruments or expensive tools: it lets a
   malformed tool call be punished at the token level without changing the batch's net gradient
   mass, which is exactly the regime where the naive fix was observed to flatten the policy.

## Caveats

- Single frozen commit on a 0.9.0.dev base, no upstream merges; expect to rebase yourself.
- The graders are in `third_party/mimoagent-osr`, which a plain clone leaves empty
  (`git submodule update --init --recursive` needed); the trainer-side code above was read in
  the main tree.
- No SAO / no GSPO / no CISPO here — the objective is plain GRPO (Dr. GRPO normalization). The
  contribution is credit assignment and reward hygiene, not the objective.
- The MiMo-V2.6 report itself was not re-read for this addendum; the recipe claims above come
  from the configs and code, not the paper.
