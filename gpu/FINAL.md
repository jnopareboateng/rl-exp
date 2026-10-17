# Locked evaluation: the procedure (docs/prereg-v1.md sections 3 to 6)

Run only after the git tag `prereg-v1` exists. Nothing here chooses an arm, a decoding setting, a checkpoint or a comparator: those are in `results/selection.json` (Val only) and
`results/frozen_arms.json` (hash in `results/frozen_arms.sha256` and in the tag message). Every step is either a CPU check or a GPU run on the Colab A100.

## 0. Before the tag (CPU, in the repo)
1. `.venv/bin/python scripts/select_arms.py --candidates results/arm_candidates.json --runs runs/colab-s1 --out results/selection.json` (Val reports only).
2. Write `results/frozen_arms.json`; `sha256sum results/frozen_arms.json > results/frozen_arms.sha256`; check it with
   `python gpu/final_eval.py --spec results/frozen_arms.json --spec-sha256 $(cut -d' ' -f1 results/frozen_arms.sha256) --selection results/selection.json --manifest data/MANIFEST.sha256 --dry-run`
   (this validates the spec against the selection and the manifest, and writes nothing).
3. `python3 scripts/data_manifest.py` (the manifest lists the Test files and the arm prompt files; the Test and variant files exist as inputs only: they hold prompts, no model output).
4. Full test suite green; commit; `git tag -a prereg-v1 -m "<spec sha256> <selection sha256> <manifest sha256>"`.

## 1. Runtime (GPU)
1. Colab runtime A100-40GB, Drive mounted. Upload `runs/bootstrap.tgz` (`scripts/make_bootstrap_bundle.py`: refuses every file whose hash is a Test file's) and run `sh gpu/bootstrap_colab.sh`
   (fails without CUDA in both venvs; refuses any `test_` file under data/).
2. Upload, after the bootstrap and never inside the bundle: `results/frozen_arms.json`, `results/frozen_arms.sha256` (the committed one, never regenerated on the runtime), `results/selection.json`, `data/MANIFEST.sha256`, `data/processed/test_t.jsonl`, `test_r.jsonl`
   and `data/processed/arms/{a4,name,ident,shuffle,blank}/test_{t,r}.jsonl`; `cd data && sha256sum -c` on those paths against the manifest must pass.
Steps 3 to 5 are scripted: `python gpu/run_final.py --drive /content/drive/MyDrive/rl-exp` (`--dry-run` prints the plan; to resume after a stop add `--skip-sanity --from-stage <stage name> [--keep-cache]`).
It stops before anything runs if a spec arm belongs to no stage, and stops after the sanity step if a merged model does not reproduce its Val score. What it does:

3. Adapters from Drive to the paths the spec names (`adapter.path`); `gpu/prepare_final.py` checks both adapter files against the frozen hashes before merging:
   `python gpu/prepare_final.py --spec results/frozen_arms.json --spec-sha256 <hex> --selection results/selection.json --models final_models --only A2,RL-full`.
   A merged 4B model is 9 GB: delete it after its Test samples exist.
4. Pre-Test sanity on Val (the driver refuses Val, so call the sampler): sample each merged adapter arm on Val at its frozen temperature and settings with `gpu/sample_vllm.py --splits val`
   and compare fv_tanimoto with the selection's score (a sampler-seed replicate moves it by about 0.003). Stop if it is off by more than 0.01: the model on disk is not the selected one.
5. Sampling: `python gpu/final_eval.py --spec ... --spec-sha256 <hex> --selection ... --manifest data/MANIFEST.sha256 --data data --out runs/final --only <ids>`; resumable, so a killed
   run is started again with the same command. Order: the confirmatory pair first (the Val-selected A5, then A2, A0n, A4), then 9B, the profile diagnostics, the name and identification probes
   of 4B and 9B, then the 27B references (each needs the HF cache cleared first: `rm -rf ~/.cache/huggingface/hub`, 31 GB per model, `gpu_memory_utilization` 0.93 is in the spec). Omissions
   from running out of compute are listed, in the order of the protocol's Test run list, in the paper.
6. `tar czf runs_final.tgz runs/final` on the runtime (samples.jsonl, meta.json, manifest.json per arm) and download it; copy to Drive as well.

## 2. Scoring (CPU, repo)
1. Extract the archive into `runs/final/`. `rlexp.metrics score --samples runs/final/<id>/samples.jsonl --data data/processed --out runs/final/<id>/scored.v3.jsonl` for every arm (ident probes through `rlexp.ident`).
2. `.venv/bin/python scripts/final_controls.py --final runs/final` (C0 to C3, eight copies per row, scored).
3. `.venv/bin/python scripts/confirmatory.py --selection results/selection.json --spec-sha256 <hex> --final runs/final --out results/confirmatory.json` (the one confirmatory contrast on test_t,
   test_r exploratory; refuses a run not made under the frozen spec or a cohort of the wrong size).
4. Tables and figures (a Test split needs `--allow-test` in each; run `final_table.py` first, the others read what it leaves in `runs/final/<id>/`):
   `scripts/final_table.py --final runs/final --splits test_t,test_r --spec results/frozen_arms.json --out results/test_baselines.json --md results/test_table.md --allow-test`,
   `scripts/paired_table.py --final runs/final --split test_t --selection results/selection.json --auto --out results/paired_test_t.json --md results/paired_test_t.md --allow-test` (and `--split test_r`),
   `scripts/exposure_terciles.py --final runs/final --split test_t --a <RL arm> --b <comparator> --out results/exposure_test_t.json --md results/exposure_test_t.md --allow-test`,
   `scripts/make_figures.py --split test_t --allow-test`, `scripts/strict_rescore.py` (sensitivity), `scripts/answer_diversity.py`, `scripts/compute_budget.py`.

Every deviation from this list is reported in the paper next to the result it touches (protocol section 6).
