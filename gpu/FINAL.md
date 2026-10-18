# Locked evaluation: the procedure (docs/prereg-v1.md sections 3 to 6)

Run only after the git tag `prereg-v1` exists. Nothing here chooses an arm, a decoding setting, a checkpoint or a comparator: those are in `results/selection.json` (Val only) and
`results/frozen_arms.json` (hash in `results/frozen_arms.sha256` and in the tag message). Every step is either a CPU check or a GPU run on the Colab A100.

## 0. Before the tag (CPU, in the repo)
1. `.venv/bin/python scripts/select_arms.py --candidates results/arm_candidates.json --runs runs/colab-s1 --out results/selection.json` (Val reports only).
2. Write `results/frozen_arms.json`; `sha256sum results/frozen_arms.json > results/frozen_arms.sha256`; check it with
   `python gpu/final_eval.py --spec results/frozen_arms.json --spec-sha256 $(cut -d' ' -f1 results/frozen_arms.sha256) --selection results/selection.json --manifest data/MANIFEST.sha256 --dry-run`
   (this validates the spec against the selection and the manifest, and writes nothing).
3. `python3 scripts/data_manifest.py` (the manifest lists the Test files and the arm prompt files; the Test and variant files exist as inputs only: they hold prompts, no model output).
4. Full test suite green; commit; annotated tag whose message is exactly `prereg-v1 spec=<sha256> selection=<sha256> manifest=<sha256>` (the spec's sha256 is in `results/frozen_arms.sha256`;
   `scripts/confirmatory.py` reads the expected spec hash from this message with `git tag -n99 prereg-v1`).

## 1. Runtime (GPU)
1. Colab runtime A100-40GB, Drive mounted. Build the bundle at the tagged commit only: `.venv/bin/python scripts/make_bootstrap_bundle.py --out runs/bootstrap.tgz` refuses unless HEAD is the commit of the annotated tag `prereg-v1`
   and no code file differs (`--allow-untagged` is for Val-only bundles before the tag), and refuses every file whose hash is a Test file's. Upload it, extract it, `sh gpu/bootstrap_colab.sh` (fails without CUDA in both venvs; refuses any `test_` file under data/).
   Do not extract it over a running RL job (`gpu/run_rl.sh` is read by the shell while it runs).
2. Upload, after the bootstrap and never inside the bundle (one archive: `tar czf runs/final_inputs.tgz data/MANIFEST.sha256 results/frozen_arms.json results/frozen_arms.sha256 results/selection.json data/processed/test_t.jsonl data/processed/test_r.jsonl data/processed/arms/{a4,name,ident,shuffle,blank}/test_{t,r}.jsonl`):
   the frozen spec, its committed sha256 file (never regenerate it on the runtime), the selection, the manifest and the 12 Test prompt files; `final_eval` hashes every file it reads against the manifest.
3. Adapters from Drive to the paths the spec names: plain copies `.agent-runs/adapters_plain/{A2-epoch1,R5-step-50}` with THREE files each (`adapter_model.safetensors`, `adapter_config.json`, `revision.json` with the base revision pin of the spec),
   not the run directories (`gpu/merge_lora.py` treats `<run>/adapters/step-N` as owned by a run and needs that run's checkpoint state). `gpu/prepare_final.py` checks the files against the frozen hashes and revision before merging.
4. `python gpu/run_final.py --drive /content/drive/MyDrive/rl-exp` (run it in the background with nohup and poll its log; `--dry-run` prints the plan). It: restores `runs/final` from Drive first if an earlier runtime left a partial run there;
   merges the adapters of the stages still to run (A2, RL-full) with `gpu/prepare_final.py`; runs the Val gate for each (the merged model sampled on Val at its frozen settings, seed 0, must reproduce the selection's score within
   `final_eval.SANITY_TOLERANCE` = 0.005; every attempt is appended to `runs/final/sanity.json`, the latest of an arm decides, a failure stops the run and stays on record); then samples the stage of the protocol's list
   with `gpu/final_eval.py` (the 4B arms, the confirmatory pair first; the larger models are Val-only references and have no stage in the spec), and copies `runs/final` to Drive every 3 minutes and at the end.
   A stage that stops is resumed by the same command plus `--skip-sanity` (it needs passed gate records already in `sanity.json`), `--from-stage <stage name>` and, inside a stage, `--keep-cache`.
   Arms that could not be sampled are reported as omissions in the paper.
5. `tar czf runs_final.tgz runs/final` on the runtime (samples.jsonl, meta.json, manifest.json, sanity.json) and download it; the Drive copy is the second one.

## 2. Scoring (CPU, repo)
1. Extract the archive into `runs/final/`. `rlexp.metrics score --samples runs/final/<id>/samples.jsonl --data data/processed --out runs/final/<id>/scored.v3.jsonl` for every arm (ident probes through `rlexp.ident`).
2. `.venv/bin/python scripts/final_controls.py --final runs/final --manifest data/MANIFEST.sha256` (C0 to C3, eight copies per row, scored; refuses Test files that are not the manifest's).
3. `.venv/bin/python scripts/confirmatory.py --selection results/selection.json --final runs/final --manifest data/MANIFEST.sha256 --out results/confirmatory.json` (the one confirmatory contrast on test_t,
   test_r exploratory; the spec hash comes from the tag message; refuses a run not made under the frozen spec, scored files that do not belong to their arm or samples, a cohort of the wrong size, a missing or failed Val sanity check).
4. Tables and figures (a Test split needs `--allow-test` in each; run `final_table.py` first, the others read what it leaves in `runs/final/<id>/`):
   `scripts/final_table.py --final runs/final --splits test_t,test_r --spec results/frozen_arms.json --out results/test_baselines.json --md results/test_table.md --allow-test`,
   `scripts/paired_table.py --final runs/final --split test_t --selection results/selection.json --auto --out results/paired_test_t.json --md results/paired_test_t.md --allow-test` (and `--split test_r`),
   `scripts/exposure_terciles.py --final runs/final --split test_t --a <RL arm> --b <comparator> --out results/exposure_test_t.json --md results/exposure_test_t.md --allow-test`,
   `scripts/make_figures.py --split test_t --allow-test`, `scripts/strict_rescore.py` (sensitivity), `scripts/answer_diversity.py`, `scripts/compute_budget.py`.

Every deviation from this list is reported in the paper next to the result it touches (protocol section 6).
