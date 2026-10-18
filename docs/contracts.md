# Contracts: layout, schemas, function signatures

Every worker codes against this file. `docs/plan.md` says what and why; this file says exactly how the pieces connect.
Change a contract only through the orchestrator, and update this file in the same commit as the code.

## Layout and conventions

    src/rlexp/    chem.py  reward.py  data_prep.py  metrics.py  controls.py      (import as `from rlexp import chem`; installed editable)
    scripts/      audit_data.py  prompt_keyword_survey.py  resolve_refs.py  structure_stats.py  offline_reward_checks.py  data_manifest.py
    gpu/          STACK.md  env_check.py  sample_vllm.py  (training scripts later)      standalone, runs on Colab and Kaggle without installing rlexp
    tests/        pytest; tests/fixtures/answer_formats.json
    data/raw/         original row files (ignored by git): train_think.jsonl, val_think.jsonl
    data/curated/     tracked inputs and lookup caches: chembl_refs.json, manual_refs.json, approval_audit.json, exposure.json (+ *_notes.md)
    data/processed/   outputs of data_prep: refs.jsonl, split_report.json (tracked); train.jsonl, val.jsonl, test_r.jsonl, test_t.jsonl (ignored, hold prompt text)
    runs/<run_id>/    samples.jsonl, scored.jsonl, meta.json (ignored)         results/  small tracked summaries of runs (JSON, CSV) used by the paper
    docs/         plan.md  contracts.md  scaling.md  memory/  paper/  research/  archive/

- Python: `.venv/bin/python` at the repo root (`uv venv`, `uv pip install -e ".[dev]"`). Run from the repo root. Tests: `.venv/bin/python -m pytest`.
- Style: lean scripts, a short module docstring, type hints where they help, no frameworks, no classes unless state demands it. Standard library, numpy and rdkit only; no new dependency without the orchestrator.
- Data is drug pharmacology and structures. Print counts, lengths, rates and short structural facts only. Never print or log whole prompts or completions (at most 200 characters of a tail when debugging).
- Raw rows (`data/raw/*.jsonl`, never modify): id, corpus, task_type A/B/C/MO/PD, drug, group, n_tokens, n_prompt_tokens, n_completion_tokens, prompt, completion, text.
  `data/curated/chembl_refs.json`: one entry per drug component: name, status, match, chembl_id, pref_name, first_approval, max_phase, fda_first_approval, approval_year, parent_smiles.
- `prompt` ends with `<|im_start|>assistant\n`; `completion` starts with `<think>\n` and contains exactly one `</think>`. The RL/sampling prompt is `prompt + "<think>\n"`; the SFT target is `completion` without its leading `<think>\n`.
  Native Qwen3.5/3.8 generation prompt (thinking on) is `<|im_start|>assistant\n<think>\n`; thinking off is `<|im_start|>assistant\n<think>\n\n</think>\n\n`.

## Chemistry definitions (one implementation, `rlexp.chem`, imported everywhere)

- standardize(smiles) -> Mol | None: parse, keep the largest fragment (`LargestFragmentChooser`), neutralise (`Uncharger`). None if unparseable, if the molecule has more than `chem.MAX_ATOMS` (400) atoms in total, or if it has a dummy atom; both checks run right after parsing so a runaway answer stays cheap.
- canonical = `Chem.MolToSmiles(mol)` (isomeric). conn = identity key: first block of the standard InChIKey (formula, connectivity, hydrogen placement; no stereo). inchikey = the full standard InChIKey, used for stereo_match only.
  A tautomer written differently is a different conn. Five reference drugs differ from their ChEMBL entry only in hydrogen placement, so references may carry `ref_alt_smiles`: other human-curated registry SMILES of the same compound with an equal `chem.skeleton`
  (formula plus heavy-atom skeleton hash, never used to match answers: a looser key equated delta-8 and delta-9 THC and norethindrone and norethynodrel). An answer matches when its conn equals the conn of the reference or of any alternate.
- scaffold = canonical non-isomeric SMILES of the Murcko scaffold (stereo is scored separately); empty string for acyclic molecules. A scaffold match requires both non-empty and equal.
- Cost guards for generated text: answers over `chem.MAX_LEN` (1,500) characters are never extracted; the atom and dummy-atom limits above apply (longest reference 715 characters, 295 heavy atoms).
- tanimoto = Morgan radius 2 count-vector Tanimoto (repeated units change the score; the bit version scored a 16- and a 108-heavy-atom phenyl fatty acid as 1.0). The reward uses it. tanimoto_bits = the 2048-bit version, used for the leakage graph (threshold 0.4) and the controls.
- extract_answer(text) -> str | None: text after the last `</think>`; scan whitespace tokens from the end, stripping `` ` * . , ; : `` from each token, after replacing the literal `<|im_end|>` with a space;
  first token with length >= 8 containing one of `=()[]@` or a digit that RDKit parses. None if the text has no `</think>`.
  Known limits: (a) tests/fixtures/answer_formats.json lists 11 cases that return a wrong molecule instead of None (for example a SMILES-like token in a closing sentence);
  (b) a molecule whose SMILES is under 8 characters or has none of those characters cannot be extracted, so a reference with that property is excluded from the data (`unextractable_reference`).
  Kept as is until real rollouts show how often (a) matters; if it does, prefer the last SMILES-only line before falling back to the scan.

## reward.py (v2)

    score(solution_str: str, ref: dict, finished: bool = True) -> dict                            # finished: v3, 2026-10-17
    compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float    # verl-style; ground_truth is a dict or JSON string; no completion status
    ref keys: ref_smiles (required) and ref_alt_smiles (optional list); ref_conn, ref_scaffold and ref_inchikey are re-derived in the scoring process (stored copies are informational)
    score keys: closed, answer, valid, tanimoto (0..1, 0 if invalid), scaffold_match, conn_match, stereo_match (full InChIKey equal; reporting only, never rewarded), reward

    reward = 0                                              if not closed, or answer is None, or answer does not standardize
    reward = 1.0                                            if conn_match
    reward = 0.05 + 0.6 * tanimoto + 0.3 * scaffold_match   otherwise (max below 1.0)

`closed` is True when `</think>` appears. Weights are module constants (W_VALID, W_TANI, W_SCAF); provisional, tuned only on Train rollouts and Val, never on test splits.

**Finished rule (v3, 2026-10-17).** `finished=False` marks a sample that did not end by itself: cut by the length cap (`finish_reason == "length"`) or aborted (`"abort"`). Such a sample scores 0, because a deployed system cannot use text that never ended:
`valid=False`, `reward=0.0`, `tanimoto=0.0`, `conn_match=scaffold_match=stereo_match=False`; `answer` is still extracted and `closed` still computed, for diagnostics. A bad reference still raises. The default `True` changes no existing call.
`compute_score` is unchanged and has no completion status (the verl signature carries none; `extra_info` is ignored): it scores the text as if finished, so an adapter that knows a sample was cut must call `score(..., finished=False)` or zero the reward itself (the RL trainer does), or the old training signal comes back.

## Data products (written by `rlexp.data_prep`)

`data/processed/refs.jsonl`, one row per single-drug group:

    group, components[list], ref_smiles, ref_inchikey, ref_conn, ref_scaffold, ref_source ("chembl"|"sft_answer"|"manual"),
    chembl_id|null, approval_year|null, approval_source ("chembl"|"openfda"|"audit"|"manual"|null), sft_agrees (bool|null),
    ref_alt_smiles (list, usually empty), cluster_id (int; component of the graph on included drugs with an edge when bit Tanimoto >= 0.4 OR the two drugs share an identical non-empty Murcko scaffold),
    split ("train"|"val"|"test_r"|"test_t"|"buffer"|"excluded"), exclude_reason (str|null), n_rows, n_heavy (heavy atoms of ref_smiles), convention ("ligand_only"|null), pubchem_create_year|null, pubmed_count|null   (last two from exposure.json when present)

`data/processed/{train,val,test_r,test_t}.jsonl`, one row per prompt row:

    id, group, task_type, split, cluster_id, prompt (= original prompt + "<think>\n"), completion (original minus leading "<think>\n"; train and val only),
    ref_smiles, ref_alt_smiles, ref_conn, ref_scaffold, ref_inchikey, n_heavy, approval_year, n_prompt_tokens, n_completion_tokens

`data/processed/split_report.json`: counts per split (drugs, rows, task type, approval year), dropped sets with reasons (PD, combinations, each exclude_reason with group names), leakage numbers, component sizes, scaffold overlap between splits, seed, input hashes.
Hand files in `data/curated/` (read by `data_prep`, never written by it):
`manual_refs.json` `{group: {"smiles": str|null, "approval_year": int|null, "source": str, "note": str, "convention": "ligand_only" (optional)}}`;
a manual entry whose `smiles` is null means the group is unresolvable and is excluded (`manual_unresolvable`), never falling back to ChEMBL or the SFT answer; `convention: ligand_only` marks metal or radiolabel complexes whose reference is the organic ligand (carried to refs rows);
`approval_audit.json` `{group: {"fda_year": int|null, "earliest_global_year": int|null, "earliest_regulator": str, "evidence": [url], "confidence": "high"|"medium"|"low"}}`;
`exposure.json` `{group: {"pubchem_cid": int|null, "pubchem_create_year": int|null, "pubmed_count": int|null}}` (a proxy for public exposure before approval, a covariate only).

Split rules: single-drug groups only (no `" | "`); PD dropped. Final approval year precedence: manual_refs > approval_audit (`earliest_global_year`) > chembl_refs (ChEMBL first_approval, else openFDA).
Test-T = final year >= 2025 (a temporal cohort, not proof that a model has not seen the molecule). Buffer = other drugs in a Test-T drug's component (unused anywhere).
From the rest, whole components go to Test-R (30 drugs), Val (20 drugs), Train (the remainder) by a seeded shuffle (seed 0), greedy, skipping a component that would overshoot a target.
Excluded: no resolvable structure (`unresolved_disagreement` = ChEMBL and the SFT answers disagree without a manual entry), no final year (`no_approval_year`),
a null manual entry (`manual_unresolvable`), or a reference the extractor cannot recover from itself or from 20 random SMILES variants of it (`unextractable_reference`). Assert and report that no pair across Train, Val, Test-R, Test-T has Tanimoto >= 0.4 or an identical non-empty scaffold.

## Sampling and scoring products

`runs/<run_id>/samples.jsonl`, one row per generated sample:

    id, group, split, sample_idx, text (generated text only; in nothink mode prefixed with `</think>\n\n` so the normal scorer applies), finish_reason, n_tokens, model, mode ("think"|"nothink"), temperature, top_p, max_tokens, seed

`scored.jsonl`: each sample row (it keeps `finish_reason`) plus the `score()` keys and, since v3 (2026-10-17), `lift` (a finite float; defined under metrics.py; files scored before v3 lack it and stay readable, the report then shows lift as n/a and says why). `meta.json`: model, library versions, GPU name and memory, wall time, tokens per second, command line, git revision, data manifest hash (`scripts/data_manifest.py`).

## metrics.py

    python -m rlexp.metrics score  --samples runs/X/samples.jsonl --data data/processed --out runs/X/scored.jsonl
    python -m rlexp.metrics report --scored runs/X/scored.jsonl [--vs runs/Y/scored.jsonl] [--split test_t,test_r,val] [--boot 10000] [--seed 0] [--data data/processed] [--json out.json]
    score_samples(samples, data_rows, scorer=None) -> list of scored rows            # what `score` runs

Changes since v2 are marked "v3, 2026-10-17".

Unit of analysis is the drug (`group`); within a drug average over prompt rows and samples; across drugs equal weight. Intervals: percentile bootstrap resampling clusters (`cluster_id` components), 95%, descriptive.

**Confirmatory endpoint (decision, v3, 2026-10-17): `fv_tanimoto`.** The mean count-vector Tanimoto of the FIRST valid, closed, finished answer among the first 8 samples of each prompt row (0 when none qualifies), drug-balanced, with its cost `fv_samples` (generations per answer) and `fv_tokens` (generated tokens per answer) reported beside it.
It is the deployable resample-until-RDKit-accepts policy: it removes the validity confound without conditioning on validity (unlike `tanimoto_valid`, which lets an arm drop hard cases). **Its cost is stated, not matched**: only the cap of 8 draws is common to the arms (`fv_samples` was 1.5 to 4.9 across the Val baselines), and an arm that makes its weak answers invalid can raise `fv_tanimoto` by paying in draws.
So `fv_tanimoto` must always be read together with `fv_samples` and `fv_tokens` (and `prompt_tokens`, below). `lift` and `fv_lift` are diagnostics of profile association, not the endpoint (see the condition below); so are all other metrics.
The tested set (wins, losses, ties, a sign-flip p-value and a studentized line each) is `fv_tanimoto`, `mean_tanimoto`, `lift`, `fv_lift`, `core@1`, `exact@1`; only `fv_tanimoto` is confirmatory and no multiplicity adjustment is claimed for the rest.

**Finished (v3, 2026-10-17).** A sample is finished only if its `finish_reason` is `"stop"`; one without `finish_reason` counts as finished, while `"length"` (cut by the cap), `"abort"` and any other value are failures, as in dense RFT, which keeps only `"stop"`. `score_samples` passes `finished = (finish_reason == "stop" or absent)` to a scorer whose signature accepts it
(it names `finished` or takes `**kwargs`; `rlexp.reward.score` does, and then such a sample is a failed answer, see reward.py); a scorer that does not is called as before. `truncation` stays the share cut by the length cap. `rlexp.ident` applies the same rule: a sample that did not end by itself or whose think block never closed is neither a name match nor answered,
and its `name_pass@k` names the k that every row supplies (up to 8) in the key and in `pass_k`, so a run of 2 samples per row reports `name_pass@2`, never a pass@8.

**Lift (v3, 2026-10-17), a diagnostic.** For a sample of drug d on split S with a valid, closed and finished answer a: `lift = T(a, ref_d) - mean over the OTHER drugs d' of S of T(a, ref_d')`, T = `chem.tanimoto` (count vectors, the reward's).
The references are one `ref_smiles` per drug of S in the data files given to `score_samples` (the rows of a drug share it, `InputError` if they differ; `InputError` too if a reference of S does not standardize when a valid answer needs it), so a run that sampled few drugs is still judged against all of S.
Any other sample (invalid, unclosed, length-terminated), and every sample of a split with a single drug, has lift 0.0; the report notes a single-drug split. Stored in the scored row as `lift`; averaged like the other sample metrics. `fv_lift` is the lift of the first valid answer (0 when none).
**When the zero holds.** A constant answer that is valid and finished on every row has mean lift exactly 0 over a complete panel (the terms cancel), and so has `fv_lift` of a constant when every row has a valid answer in its first 8 samples. An answer that is independent of the profile has expected lift 0 only if the completion process (valid, closed, finished) is also independent of the drug.
If completion depends on the drug, or on a nuisance cue that correlates with it such as prompt length, the mean is the completion-weighted sum of the drugs' lifts and need not vanish: one fixed molecule scores clearly above 0 when only one drug finishes (+0.18 in the independent review's synthetic example, and +0.71 for a policy that used prompt length alone; the tests pin the first).
Lift therefore measures the association between the drug and everything the policy does, including nuisance cues, not the content of the answer alone; a claim about the profile needs a comparison, at a stated cost, against the shuffled-profile control (`rlexp.shuffle`, below: it keeps task type and length, not format beyond that) or against the profile-removed control (`rlexp.arms blank`, below). The value also depends on the panel (the same answer scores differently against two drugs and against four).
Each answer's molecule and each reference's fingerprint are built once; at most samples x drugs-in-S similarities are computed.

| Metric (per split) | Definition | Notes |
|---|---|---|
| `fv_tanimoto` | the Tanimoto of the first valid, closed, finished answer among the first 8 samples of a row (0 when none), drug-balanced | **confirmatory endpoint**; deployable with resampling, cost `fv_samples` and `fv_tokens`; tested |
| `exact@1`, `core@1`, `stereo_exact@1` | conn_match; conn_match or scaffold_match; stereo_match | exact@1 and core@1 are tested diagnostics |
| `lift` | as above, invalid, unclosed or length-terminated = 0 | v3; diagnostic, tested |
| `mean_tanimoto` | `tanimoto` over all attempts, invalid = 0 | tested diagnostic |
| `tanimoto_valid` | per row the mean `tanimoto` over its valid, closed, finished samples; a row without one has no value; a drug without one drops out (the `drugs` column counts the rest) | v3; descriptive, conditions on validity |
| `mean_reward`, `truncation`, `n_tokens_mean`, `n_tokens_p90` | mean reward; share with `finish_reason == "length"`; tokens (p90 is nearest-rank over the drug-balanced distribution) | |
| `valid_closed` | share that is valid and closed and finished ("validity"); a length-terminated or aborted sample is a failure | v3 wording |
| `exact@8`, `core@8` | pass@8 over all n samples of a row with n >= 8: oracle coverage that needs the answer to select, never a deployable result | labelled as such |
| `maj@8` | the plurality connectivity among the first 8 samples equals the reference's or an alternate's; unparseable answers cast no vote; ties earn nothing | deployable |
| `fv_reward`, `fv_lift` | the reward and the lift of the same first valid answer (0 when none of the 8 qualifies) | v3; "deployable with resampling: costs fv_samples generations per answer"; `fv_lift` tested |
| `fv_rate`, `fv_samples`, `fv_tokens` | share of rows with such a sample; mean number of samples drawn up to and including it (8 when none); mean generated tokens of those samples (all 8 samples' when none) | v3; the cost per answer, never tested |

`prompt_tokens` (one number per split beside the table, not a bootstrapped metric) is the drug-balanced mean `n_prompt_tokens` of the run's prompt rows in the data files, None without `--data` or when a row lacks it. That count belongs to the row's original prompt, which arm files keep (`arms.py`), so for a retrieval arm it undercounts:
the tokens per answer are `fv_tokens + prompt_tokens` if the prompt is read once and `fv_tokens + fv_samples * prompt_tokens` if it is read again for every draw.

The @8 and `fv_*` metrics use prompt rows with at least 8 samples; shorter rows are skipped and counted (`skipped_rows`), and a drug with no usable row has no value (a run with fewer than 8 samples per row has no `fv_tanimoto`; for a deterministic control that always parses it equals `mean_tanimoto`).

**Input checks (v3, 2026-10-17).** `score_samples` and `report` raise `InputError` instead of reading a bad value as missing (a NaN would otherwise be dropped by the mean and silently remove a drug from the estimand):
any NaN, inf, null, text or bool in a sample's or scored row's `sample_idx`, `n_tokens`, `reward`, `tanimoto` or `lift` (the error names the field and the number of rows; an absent `lift` is not an error), and a scorer result whose `tanimoto` or `reward` is not finite. A bool is not a number here. A flag (`valid`, `closed`, `conn_match`, `scaffold_match`, `stereo_match`) must be a real boolean, so the text "false" never counts as true, and `id`, `group` and `split` must be strings (`cluster_id` a string or integer, `answer` and `finish_reason` a string or null):
a list, dict or other value of the wrong type is an `InputError`, never a traceback. `sign_flip_p` raises on a NaN or inf difference instead of dropping it as a tie.
Component metadata is all or nothing: rows of a run that carry `cluster_id` only in part, a drug with two `cluster_id`s, runs A and B that disagree on a drug's `cluster_id` (or only one of them carries it) and a scored row whose `cluster_id` differs from the data row's all raise; `--data` fills a missing `cluster_id` from the data rows.

**Data files read (v3, 2026-10-17).** `score` opens only the `<split>.jsonl` files of the splits that its samples name (every sample needs a `split`). `report` opens those of the splits in the scored file that it reports (`--split` narrows them; rows of the `--vs` run in other splits are left out) and `train.jsonl` for the constant-answer bar when `--data` is given.
A Val run never opens `test_r.jsonl` or `test_t.jsonl`.

`--vs B`: paired per-drug differences A minus B with a bootstrap interval for every metric. For the tested set `fv_tanimoto`, `mean_tanimoto`, `lift`, `fv_lift`, `core@1`, `exact@1` (v3; was `exact@1`, `core@1`) also the wins, losses and ties (per drug) and an exact two-sided sign-flip permutation p-value that flips whole `cluster_id` components
(the drug differences are summed within a cluster and the cluster totals are flipped; drugs when rows carry no cluster_id): exact enumeration when there are at most 20 non-tied units, otherwise 100,000 seeded Monte Carlo flips,
with the number of clusters reported; a difference of at least 1e-9 is a win or a loss (1e-9 only absorbs float noise in drug means). The p-value is None when no drug has a value in both runs (an `fv_*` metric of runs with fewer than 8 samples per row). The p-value, not the bootstrap interval, is the decision statistic for the confirmatory contrast.
The report prints once a plain note that the test is not assumption-free (v3): the components are independent exchangeable units and a sign flip is valid under a null in which each component-summed difference is symmetric about zero (a zero mean alone is not enough).
Beside the p-value (v3) a studentized line: the paired mean difference divided by its component-bootstrap standard error (`boot_se`, the standard deviation of the `--boot` resampled differences; `studentized` in the JSON is None when the standard error is undefined or zero). It is a descriptive size in bootstrap standard errors, not a calibrated test: five wins among 41 drugs with 36 ties give about 2.4 with an exact p of 0.0625.
maj@8 counts the plurality answer as correct when its conn equals that of the reference or of any `ref_alt_smiles` alternate.
**Constant-answer bar (v3, 2026-10-17).** With `--data` the report scores the best constant Train reference on the run's own prompt rows, each answered 8 times so that `fv_*` exist, on the rows that have 8 samples in the run (the run's own `fv_*` rows; none when the run has none) (the constant always parses, so its `fv_tanimoto` is its Tanimoto): `exact@1`, `core@1`, `lift`, `mean_tanimoto`, `mean_reward`, `fv_tanimoto`.
Its lift uses the same reference panel as the samples' lift, every drug of the split in the data. A constant has lift 0 over a complete panel, so the bar prints 0.000 when the run covers every drug of the split; on a partial run it is the mean over the sampled drugs of their lifts against the whole panel and is not forced to 0.

## controls.py (CPU baselines, same scorer)

    python -m rlexp.controls --data data/processed --out runs
Writes `runs/control_copy_nearest/`, `runs/control_constant/` and `runs/control_constant_scaffold/` (the shortest Train reference among those with the most common non-empty scaffold: a do-nothing bar for core@1) for val, test_r, test_t, then scores them.
Copy-nearest: TF-IDF cosine over the user part of the prompt, nearest Train prompt, answer = that row's `ref_smiles` (an answer-copying control, not retrieval-augmented prompting). Constant: the Train reference with the highest mean Tanimoto to the other Train references.

## arms.py (profile-removed control, a diagnostic)

    python -m rlexp.arms blank --data data/processed --out data/processed/arms/blank [--system-text TEXT] [--splits val]

Writes `<split>.jsonl` and `<split>.meta.json` for the splits named by `--splits` (a comma-separated subset of `val,test_r,test_t`; default `val`, so no Test file is opened unless it is named, and an unknown or empty list is an error before anything is written). A row is the processed row unchanged except `prompt`, so `gpu/sample_vllm.py --data out` reads the files unchanged and the scorer sees the original ids and references. The new prompt is the system turn (`--system-text`, default `arms.DIRECT_SYSTEM`) and the user turn `Drug: unknown.` (`arms.BLANK_USER`, the text `gpu/rl_grpo.py` uses for `--profile-mode none`; a test pins the two strings), with the generation prefix kept: no profile and no drug name, whatever the row. `<split>.meta.json`: `file`, `rows`, `system_source`, `system_text`, `control` (`"blank_profile"`). `n_prompt_tokens` keeps the row's original count.
It is the other side of the shuffled-profile control: shuffled keeps a profile, wrong for this drug; blank keeps none. An arm that uses the pharmacology loses `fv_tanimoto` and `lift` under both. `name` writes the name-given oracle files (val, test_r, test_t) in the same schema.

## shuffle.py (shuffled-profile control, a diagnostic)

    python -m rlexp.shuffle write --data data/processed --out data/processed/arms/shuffle [--seed 0] [--splits val]

Writes `<split>.jsonl` and `<split>.meta.json` per split (the Test files were built before the tag as inputs only: prompts, no model output; they are in `data/MANIFEST.sha256` and reach the GPU runtime only through the locked-evaluation step). Every row is shown the prompt of a donor row and keeps every other field, in the input order, so `gpu/sample_vllm.py --data out` reads the files unchanged and the scorer sees the original ids and references (`n_prompt_tokens` keeps the row's own count).
**Donors.** A row-level derangement within strata: the rows of a `task_type`, cut into terciles of prompt characters (rank terciles, ties by id); each stratum is a bijection, so no prompt is shown twice and none to its own row. A donor has the same `task_type` (always), is another drug (always) and from another `cluster_id` component whenever the stratum allows it (a null `cluster_id` makes the drug its own component),
and comes from the same length tercile. A tercile that cannot give every row such a donor is merged with the adjacent tercile, then the length bin is dropped (the task type is one stratum); where even that leaves rows with a donor from their own component their number is the minimum possible and is recorded; a task type that cannot show every row another drug's prompt (one row, or one drug owns more than half of its rows) makes the write raise.
Method: an exact min-cost assignment (Hungarian algorithm; a donor from the own drug is forbidden, one from the own component costs one unit) with seeded random tie-breaking, so a valid derangement is found whenever one exists, with no random search; the same seed gives the same mapping and another seed another valid one.
`<split>.meta.json`: `file`, `rows`, `control`, `seed`, `design`; `strata` (per task type: the tercile sizes and the groups solved, each with its terciles, rows, level `tercile|merged|task_type` and `component_overlap_rows`); `fallbacks` (the groups that needed a merge or still share a component);
`balance` (`mean_abs_length_gap` between the own and the shown prompt against `baseline_gap_any_other_drug` and `baseline_gap_same_task_type`, `length_correlation`, `same_task_type_share` which is 1.0, `same_tercile_share`, `same_component_rows`, `same_drug_rows` which is 0, `max_rows_per_donor` which is 1); `mapping` (drug to its donor drugs) and `row_mapping` (row id to donor row id). No prompt text.
**What it controls.** The control keeps task type and length (within a tercile), not format beyond that: section order and wording still differ between two prompts of one task type, so a policy keyed on the exact length or on the format keeps part of its signal here. Used as the nuisance-matched comparison for a claim about the profile: an arm that uses the pharmacology loses `fv_tanimoto` and `lift` under it, one that keys on task type or length bin does not.

## rank_readout.py (addendum 1)

`scripts/rank_readout.py --final <dir> --split <split> [--arms a,b,...] [--a A --b B] [--pairs A:B,...] --out <json> [--md <md>] [--data data/processed] [--allow-test]`

Candidate set: the N distinct drugs of `--split`, each by the `ref_smiles` of `<data>/<split>.jsonl`, ordered by group name. Fixed by the split, never by the arm. A reference that does not parse is an `InputError`.
Per prompt row: the first valid, closed, finished answer among the first `metrics.K` = 8 samples by `sample_idx` (the endpoint's policy; `metrics.finished` decides "finished"). Rank = 1 + #{candidates strictly nearer than the own reference} + (#tied with it, excluding itself) / 2, by `chem.tanimoto` (count-vector Morgan, r=2, 2048 positions, the endpoint's similarity), with `metrics.TIE` as the equality tolerance.
A row with no usable answer scores rank N (percentile 1.0) and contributes 0 to `fv_rate`.
Measures, each the mean over a drug's rows and then over drugs (`rlexp.metrics`' unit): `hit@1` (rank <= 1, so a tie for nearest is not a hit), `pct_rank` (rank / N, lower is better) and `fv_rate` (share of rows with a usable answer). Chance, for any policy whose output does not depend on the drug: `hit@1` = 1/N and `pct_rank` = (N+1)/2N, exactly; both are written to the output as `chance` and printed in every table.
Contrasts: per-drug A minus B on the drugs the two arms share, summed within `cluster_id` and tested by `metrics.sign_flip_p` (the endpoint's test, the same exchangeable units, seed 0, `metrics.FLIPS` above `metrics.EXACT_MAX` units). Wins count drugs for which the first arm is better, which is a negative difference for `pct_rank` and a positive one for `hit@1`. A Test split needs `--allow-test` (`final_table.check_splits`).
Output: `{split, n_candidates, chance, confirmatory: false, sign, arms: [...], contrasts: [...]}`. No answer text, prompt, structure or drug name is printed or written. Nothing it produces is confirmatory (`docs/prereg-v1-addendum1.md` section 5).

## random_pair_calibration.py (addendum 1)

`scripts/random_pair_calibration.py [--refs data/processed/refs.jsonl] [--splits train,val,test_r,test_t] [--at 0.176,...] --out <json> [--md <md>]`

Population: the distinct reference structures of the named splits, one per connectivity key (`ref_conn`, else `chem.conn`), so a duplicated drug cannot inject a self-pair. Every unordered pair once, scored by `chem.tanimoto`. Reports n, the number of pairs, mean, standard deviation, min, max, the percentiles 5 to 99.9, and for each `--at` value its percentile and the share of pairs at or above it.
Reference structures only: no model output, no prompt file and no sample is read, so it is computable on every split before or after the freeze (`docs/prereg-v1.md` section 3). Only aggregates are written; no structure or drug name reaches the output. Repository values (the four used splits): 332 structures, 54,946 pairs, mean 0.141, median 0.141, p75 0.180, p90 0.213, p99 0.274; 0.1949 is at the 83rd percentile.

## GPU pack (`gpu/`)

- `gpu/sample_vllm.py --model M --data data/processed --splits val,test_t,test_r --mode think|nothink --k 8 --max-tokens N --temperature 1.0 --top-p 1.0 --seed 0 --out runs/X [--limit-drugs n]`: resumable, flushes after every prompt, writes `samples.jsonl` and `meta.json`, prints tokens per second.
- `gpu/env_check.py`: prints GPU, VRAM, CUDA, python, torch, transformers, vllm, peft, trl, flash-linear-attention versions as JSON.
- Everything under `gpu/` runs unmodified on Colab and Kaggle, detects the GPU, and assumes no more than 16 GB unless it detects more.
