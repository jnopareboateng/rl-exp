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

    score(solution_str: str, ref: dict) -> dict
    compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float    # verl-style; ground_truth is a dict or JSON string
    ref keys: ref_smiles (required) and ref_alt_smiles (optional list); ref_conn, ref_scaffold and ref_inchikey are re-derived in the scoring process (stored copies are informational)
    score keys: closed, answer, valid, tanimoto (0..1, 0 if invalid), scaffold_match, conn_match, stereo_match (full InChIKey equal; reporting only, never rewarded), reward

    reward = 0                                              if not closed, or answer is None, or answer does not standardize
    reward = 1.0                                            if conn_match
    reward = 0.05 + 0.6 * tanimoto + 0.3 * scaffold_match   otherwise (max below 1.0)

`closed` is True when `</think>` appears. Weights are module constants (W_VALID, W_TANI, W_SCAF); provisional, tuned only on Train rollouts and Val, never on test splits.

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

`scored.jsonl`: each sample row plus the `score()` keys. `meta.json`: model, library versions, GPU name and memory, wall time, tokens per second, command line, git revision, data manifest hash (`scripts/data_manifest.py`).

## metrics.py

    python -m rlexp.metrics score  --samples runs/X/samples.jsonl --data data/processed --out runs/X/scored.jsonl
    python -m rlexp.metrics report --scored runs/X/scored.jsonl [--vs runs/Y/scored.jsonl] [--split test_t,test_r,val] [--boot 10000] [--seed 0] [--json out.json]

Unit of analysis is the drug (`group`); within a drug average over prompt rows and samples; across drugs equal weight. Intervals: percentile bootstrap resampling clusters (`cluster_id` components), 95%, descriptive.
Metrics per split: exact@1 (conn_match), core@1 (conn_match or scaffold_match), stereo_exact@1, mean_tanimoto, mean_reward, valid_closed rate, truncation rate, n_tokens mean and p90,
exact@8 and core@8 (pass@8, an oracle-coverage upper bound that needs the answer to select, never a deployable result), maj@8 (the plurality connectivity among the samples equals the reference; ties earn nothing).
`--vs B`: paired per-drug differences A minus B with wins, losses and ties (per drug), a bootstrap interval, and an exact two-sided sign-flip permutation p-value that flips whole `cluster_id` components
(the drug differences are summed within a cluster and the cluster totals are flipped; drugs when rows carry no cluster_id): exact enumeration when there are at most 20 non-tied units, otherwise 100,000 seeded Monte Carlo flips,
with the number of clusters reported. The p-value, not the bootstrap interval, is the decision statistic for the confirmatory contrast.
maj@8 counts the plurality answer as correct when its conn equals that of the reference or of any `ref_alt_smiles` alternate.
Also report the constant-answer bar when `--data` is given.

## controls.py (CPU baselines, same scorer)

    python -m rlexp.controls --data data/processed --out runs
Writes `runs/control_copy_nearest/`, `runs/control_constant/` and `runs/control_constant_scaffold/` (the shortest Train reference among those with the most common non-empty scaffold: a do-nothing bar for core@1) for val, test_r, test_t, then scores them.
Copy-nearest: TF-IDF cosine over the user part of the prompt, nearest Train prompt, answer = that row's `ref_smiles` (an answer-copying control, not retrieval-augmented prompting). Constant: the Train reference with the highest mean Tanimoto to the other Train references.

## GPU pack (`gpu/`)

- `gpu/sample_vllm.py --model M --data data/processed --splits val,test_t,test_r --mode think|nothink --k 8 --max-tokens N --temperature 1.0 --top-p 1.0 --seed 0 --out runs/X [--limit-drugs n]`: resumable, flushes after every prompt, writes `samples.jsonl` and `meta.json`, prints tokens per second.
- `gpu/env_check.py`: prints GPU, VRAM, CUDA, python, torch, transformers, vllm, peft, trl, flash-linear-attention versions as JSON.
- Everything under `gpu/` runs unmodified on Colab and Kaggle, detects the GPU, and assumes no more than 16 GB unless it detects more.
