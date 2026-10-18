#!/usr/bin/env python3
"""The one confirmatory contrast and its exploratory siblings (docs/prereg-v1.md section 4), run ONLY after the lock.

  .venv/bin/python scripts/confirmatory.py --selection results/selection.json --manifest data/MANIFEST.sha256 --final runs/final --out results/confirmatory.json \
      [--spec-sha256 <hex>] [--tag prereg-v1] [--data data/processed]

For the pair chosen on Val (selection.json: "rl" and "comparator" arm ids) this scores nothing new: it reads the scored Test samples
(runs/final/<arm>/scored.v3.jsonl, made by `rlexp.metrics score`) and calls `rlexp.metrics report --scored RL --vs COMPARATOR` on test_t (the
confirmatory split) and test_r (exploratory), 10,000 bootstrap resamples, seed 0, then copies the paired fv_tanimoto block (wins, losses, ties, the cluster-level
sign-flip p-value) into one JSON with the command lines, the sha256 of the input files and of the spec, selection, manifest and run manifest. No arm, decoding or comparator is chosen here.

Checks that stand between the files and a result (any failure: exit 1, no result file):
 - The git tag (--tag, default prereg-v1) must exist, be an ANNOTATED tag (`git cat-file -t`), and its message must carry spec=<sha256> selection=<sha256> manifest=<sha256>
   (`prereg-v1 spec=... selection=... manifest=...`); the spec hash read from it is the one the run is held to. --spec-sha256 may repeat it and must then agree; a typed hash with no tag is refused.
   selection= and manifest= must be the hashes of --selection and --manifest.
 - --data's test_t.jsonl and test_r.jsonl are the files data/MANIFEST.sha256 lists (processed/test_t.jsonl, processed/test_r.jsonl).
 - <final>/manifest.json (written by gpu/final_eval.py) carries the spec hash (itself and every arm entry), the sha256 of --selection and the sha256 of the five gpu/ files that sampled
   (final_eval, sample_vllm, run_final, prepare_final, merge_lora), each equal to `git show <tag>:<path>`; each sampled arm of the pair has an entry whose settings (model, mode,
   temperature, max_tokens, k and, when recorded, data_sha256) equal the sampler's own <final>/<arm>/meta.json.
 - The scored file of each arm is that arm's: its (id, sample_idx) pairs are those of its samples.jsonl, and its rows carry the manifest entry's model, mode and temperature.
 - A CPU control (null temperature in the selection) has no sampler: its <final>/<arm>/meta.json, as scripts/final_controls.py writes it, says control == the arm id, k == 8,
   splits test_t and test_r, rows == 180, a source_sha256 (C3 also a pool_sha256) and a model named control_*, which every scored row carries.
 - Every adapter arm of the pair has a Val reproduction in <final>/sanity.json (gpu/run_final.py): the file's tolerance is gpu/final_eval.py's SANITY_TOLERANCE and the latest attempt of
   the arm passed. The attempts of each arm are printed.
 - The pair covers the whole cohort, test_t 41 drugs and 100 prompt rows, test_r 30 and 80, with fv_tanimoto defined for every drug.
The per-split paired reports are written beside --out as confirmatory_paired_<split>.json and .txt (paired_table.py has its own names).
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY, EXPLORATORY = "test_t", "test_r"
COHORT = {CONFIRMATORY: (41, 100), EXPLORATORY: (30, 80)}           # (drugs, prompt rows) of the frozen Test splits
SAMPLER = ("model", "mode", "temperature", "max_tokens", "k")        # the sampler fields of a run manifest entry that meta.json must repeat
CONTROL_ROWS = sum(rows for _, rows in COHORT.values())              # 180 prompt rows with one answer each, replicated to k=8 by scripts/final_controls.py
TEST_FILES = {split: f"processed/{split}.jsonl" for split in COHORT}  # the manifest's names of the Test files that `rlexp.metrics --data` reads


_FINAL_EVAL = []


def final_eval():
    """gpu/final_eval.py loaded by path (it imports the standard library only): the SANITY_TOLERANCE the gate used and the CODE_FILES the run manifest records."""
    if not _FINAL_EVAL:
        spec = importlib.util.spec_from_file_location("final_eval_for_confirmatory", ROOT / "gpu" / "final_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FINAL_EVAL.append(module)
    return _FINAL_EVAL[0]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    """{relative path: sha256} of a data/MANIFEST.sha256 (lines 'sha256  path')."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, sep, name = line.strip().partition("  ")
            if not sep:
                raise ValueError(f"{path} is not a sha256 manifest")
            out[name.strip()] = digest
    return out


def check_test_files(data, manifest_path):
    """{split: sha256} of <data>/test_t.jsonl and test_r.jsonl once they match the entries of data/MANIFEST.sha256 (processed/test_t.jsonl, processed/test_r.jsonl); ValueError if
    an entry or a file is missing or a hash differs. `rlexp.metrics report --data` reads these files: they must be the frozen ones."""
    entries, out = read_manifest(manifest_path), {}
    for split, rel in TEST_FILES.items():
        path = Path(data) / f"{split}.jsonl"
        if rel not in entries:
            raise ValueError(f"{rel} has no entry in {manifest_path}")
        if not path.is_file():
            raise ValueError(f"{path} is missing")
        out[split] = sha256(path)
        if out[split] != entries[rel]:
            raise ValueError(f"{path} differs from its sha256 in {manifest_path}")
    return out


def read_tag(tag, cwd=ROOT):
    """The `git tag -n99 <tag>` listing of the repository, or None when there is no such tag (or no git)."""
    try:
        done = subprocess.run(["git", "-C", str(cwd), "tag", "-n99", tag], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 and done.stdout.strip() else None


def tag_object_type(tag, cwd=ROOT):
    """`git cat-file -t refs/tags/<tag>`: 'tag' for an annotated tag, 'commit' for a lightweight one, None when there is no such tag (or no git)."""
    try:
        done = subprocess.run(["git", "-C", str(cwd), "cat-file", "-t", f"refs/tags/{tag}"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def tag_file_sha256(tag, path, cwd=ROOT):
    """sha256 of `git show <tag>:<path>`, or None when the tag has no such file (or no git)."""
    try:
        done = subprocess.run(["git", "-C", str(cwd), "show", f"{tag}:{path}"], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return hashlib.sha256(done.stdout).hexdigest() if done.returncode == 0 else None


def tag_hashes(message):
    """{'spec': sha256, 'selection': ..., 'manifest': ...} from the `key=<64 hex digits>` tokens of a tag message (the first of each key; absent keys are left out)."""
    return {key: m.group(1) for key in ("spec", "selection", "manifest") if (m := re.search(rf"\b{key}=([0-9a-f]{{64}})\b", message))}


def commands(selection, final, data, out_dir, boot=10000, seed=0):
    """{split: command} for the paired reports of the chosen RL arm against the chosen comparator."""
    rl, comparator = selection["rl"], selection["comparator"]
    a, b = Path(final) / rl / "scored.v3.jsonl", Path(final) / comparator / "scored.v3.jsonl"
    return {split: [sys.executable, "-m", "rlexp.metrics", "report", "--scored", str(a), "--vs", str(b), "--split", split, "--data", str(data),
                    "--boot", str(boot), "--seed", str(seed), "--json", str(Path(out_dir) / f"confirmatory_paired_{split}.json")]
            for split in (CONFIRMATORY, EXPLORATORY)}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_scored(final, arm, wanted):
    """<final>/<arm>/scored.v3.jsonl was scored from this arm's own samples.jsonl: the same (id, sample_idx) pairs, each once, and every scored row carries the settings in `wanted`
    ({field: value}, e.g. the manifest entry's model, mode and temperature). ValueError otherwise; the message holds counts only (row ids carry drug names)."""
    scored, samples = read_rows(Path(final) / arm / "scored.v3.jsonl"), Path(final) / arm / "samples.jsonl"
    if not samples.is_file():
        raise ValueError(f"{samples} is missing: the scored file of {arm} cannot be tied to its samples")
    pairs = lambda rows: Counter((r.get("id"), r.get("sample_idx")) for r in rows)
    have, want = pairs(scored), pairs(read_rows(samples))
    if have != want:
        raise ValueError(f"arm {arm}: scored.v3.jsonl and samples.jsonl do not hold the same (id, sample_idx) pairs: {sum((have - want).values())} only in the scored file, "
                         f"{sum((want - have).values())} only in the samples")
    for field, value in wanted.items():
        seen = {r.get(field) for r in scored}
        if seen != {value}:
            raise ValueError(f"arm {arm}: its scored rows carry {field} {sorted(seen, key=str)}, not {value!r}: that scored file is not this arm's")


def check_control(final, arm):
    """The CPU control `arm` was made by scripts/final_controls.py for both Test splits (its meta.json), and its scored file is its own; ValueError if not."""
    path = Path(final) / arm / "meta.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing: the CPU control {arm} is made by scripts/final_controls.py")
    meta = json.loads(path.read_text(encoding="utf-8"))
    frozen = {"control": arm, "k": 8, "splits": [CONFIRMATORY, EXPLORATORY], "rows": CONTROL_ROWS}
    differs = sorted(key for key, value in frozen.items() if meta.get(key) != value)
    if differs:
        raise ValueError(f"control {arm}: {path} differs from {frozen} in {differs}")
    missing = [key for key in ("source_sha256", *(["pool_sha256"] if arm == "C3" else [])) if not meta.get(key)]
    if missing:
        raise ValueError(f"control {arm}: {path} records no {', '.join(missing)}")
    if not (isinstance(meta.get("model"), str) and meta["model"].startswith("control_")):
        raise ValueError(f"control {arm}: {path} records no control_* model name for its rows")
    check_scored(final, arm, {"model": meta["model"]})


def check_sanity(final, selection, arms):
    """The Val reproduction of each adapter arm of the pair (a selection arm with adapter_sha256) in <final>/sanity.json, as gpu/run_final.py appends it: the file's tolerance is the
    gate's (final_eval.SANITY_TOLERANCE), the arm has an attempt, and its LATEST attempt passed, is consistent with its own numbers and carries the selection's score.
    Returns {arm: {"attempts": n, "latest": the latest attempt}}; ValueError otherwise."""
    adapter_arms = [a for a in arms if "adapter_sha256" in selection["arms"][a]]
    if not adapter_arms:
        return {}
    path = Path(final) / "sanity.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing: gpu/run_final.py writes the Val reproduction of every adapter arm there")
    data, tolerance = json.loads(path.read_text(encoding="utf-8")), final_eval().SANITY_TOLERANCE
    if data.get("tolerance") != tolerance:
        raise ValueError(f"{path} was made with tolerance {data.get('tolerance')}, the gate's (SANITY_TOLERANCE of gpu/final_eval.py) is {tolerance}")
    attempts, out = data.get("attempts"), {}
    if not isinstance(attempts, list):
        raise ValueError(f"{path} has no list of attempts")
    for arm in adapter_arms:
        mine = [a for a in attempts if isinstance(a, dict) and a.get("arm") == arm]
        if not mine:
            raise ValueError(f"{path} has no Val attempt for the adapter arm {arm}")
        last = mine[-1]
        try:
            consistent = abs(last["value"] - last["selection_score"]) <= tolerance and last["selection_score"] == selection["arms"][arm]["score"]
        except (KeyError, TypeError):
            consistent = False
        if last.get("passed") is not True or not consistent:
            raise ValueError(f"the latest Val reproduction of {arm} (attempt {len(mine)} of {len(mine)}) did not pass: {path} says {{'value': {last.get('value')}, "
                             f"'selection_score': {last.get('selection_score')}, 'passed': {last.get('passed')}}}, the selection's score is {selection['arms'][arm].get('score')}, tolerance {tolerance}")
        out[arm] = {"attempts": len(mine), "latest": last}
    return out


def check_code(manifest, tag):
    """The five gpu/ files whose sha256 gpu/final_eval.py recorded in the run manifest are the files of the tag: `git show <tag>:<path>` hashes the same. ValueError lists the differences."""
    wanted = {f"gpu/{name}" for name in final_eval().CODE_FILES}
    code = manifest.get("code_sha256")
    if not isinstance(code, dict) or set(code) != wanted:
        raise ValueError(f"the run manifest does not record the sha256 of exactly {sorted(wanted)} under code_sha256")
    differs = sorted(path for path, digest in code.items() if tag_file_sha256(tag, path) != digest)
    if differs:
        raise ValueError(f"the code that sampled is not the code of git tag {tag}: {', '.join(differs)} differ")


def check_runs(selection, selection_path, final, spec_sha256, tag="prereg-v1"):
    """Check that the pair to be reported was sampled under the frozen spec by the tagged code and that its scored files are its own (see the module docstring); ValueError if not.
    Returns (the run manifest's path, {arm: {"attempts", "latest"}} of the Val reproductions)."""
    path = Path(final) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing: the samples were not made by gpu/final_eval.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("arms")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} lists no arms")
    if manifest.get("spec_sha256") != spec_sha256 or any(e.get("spec_sha256") != spec_sha256 for e in entries.values()):
        raise ValueError("the run manifest is not (entirely) from the spec given by --spec-sha256")
    if manifest.get("selection_sha256") != sha256(selection_path):
        raise ValueError("the selection file is not the one the spec was frozen against (the run manifest's selection_sha256)")
    check_code(manifest, tag)
    arms = (selection["rl"], selection["comparator"])
    for arm in arms:
        try:
            sampled = selection["arms"][arm]["temperature"] is not None
        except KeyError as e:
            raise ValueError(f"the selection lacks {e}") from None
        if not sampled:
            check_control(final, arm)                                                 # a CPU control has no sampler and no manifest entry
            continue
        entry = entries.get(arm)
        if entry is None:
            raise ValueError(f"arm {arm} has no entry in the run manifest: it was not sampled by gpu/final_eval.py under this spec")
        meta = Path(final) / arm / "meta.json"
        if not meta.is_file():
            raise ValueError(f"{meta} is missing")
        run = json.loads(meta.read_text(encoding="utf-8")).get("configuration") or {}
        if any(key not in entry for key in SAMPLER):
            raise ValueError(f"the run manifest entry of arm {arm} lacks some of {list(SAMPLER)}")
        frozen = {key: entry[key] for key in (*SAMPLER, *(["data_sha256"] if entry.get("data_sha256") else []))}
        differs = sorted(key for key, value in frozen.items() if run.get(key) != value)
        if differs:
            raise ValueError(f"arm {arm}: the run's meta.json differs from the run manifest in {differs}")
        check_scored(final, arm, {key: entry[key] for key in ("model", "mode", "temperature")})
    return path, check_sanity(final, selection, arms)


def check_cohort(split, report):
    """The paired block of a `rlexp.metrics report` JSON for `split` covers its whole cohort, with fv_tanimoto defined on every drug; else ValueError."""
    drugs, rows = COHORT[split]
    paired = (report.get("splits") or {}).get(split, {}).get("paired") or {}
    fv = (paired.get("metrics") or {}).get("fv_tanimoto") or {}
    got = (paired.get("n_drugs"), paired.get("n_rows"), fv.get("n_drugs"))
    if got != (drugs, rows, drugs) or fv.get("p_value") is None:
        raise ValueError(f"{split}: the pair covers {got[0]} drugs and {got[1]} prompt rows, fv_tanimoto {got[2]} drugs, p-value {fv.get('p_value')}; "
                         f"the cohort is {drugs} drugs and {rows} rows, all with fv_tanimoto and a p-value")


def expected_spec_sha256(argument, tag):
    """(the spec sha256 the run manifest is held to, the tag's hashes, where it came from). The annotated git tag is the anchor and must exist: its message carries spec=, selection= and
    manifest= (all three), and a typed --spec-sha256 must agree with it; a typed hash alone, a lightweight tag (whose "message" is a commit's) or a tag without the three keys is refused."""
    message = read_tag(tag)
    if message is None:
        raise ValueError(f"git tag {tag} does not exist: the locked result is checked against it, and a typed --spec-sha256 alone is no anchor")
    kind = tag_object_type(tag)
    if kind != "tag":
        raise ValueError(f"git tag {tag} is not an annotated tag (git cat-file -t says {kind!r}): a lightweight tag has no message of its own")
    hashes = tag_hashes(message)
    missing = [f"{key}=<sha256>" for key in ("spec", "selection", "manifest") if key not in hashes]
    if missing:
        raise ValueError(f"git tag {tag} lacks {', '.join(missing)} in its message")
    if argument is not None and hashes["spec"] != argument:
        raise ValueError(f"--spec-sha256 differs from the spec= of git tag {tag}")
    return hashes["spec"], hashes, f"git tag {tag}" + (" (agrees with --spec-sha256)" if argument is not None else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--spec-sha256", help="sha256 of the frozen spec; the annotated git tag's spec= is the anchor (it must exist), a typed hash must agree with it")
    ap.add_argument("--tag", default="prereg-v1", help="the annotated git tag whose message names spec=, selection= and manifest= hashes")
    ap.add_argument("--manifest", type=Path, help="data/MANIFEST.sha256: the Test files read through --data must be the ones it lists (required unless --dry-run)")
    ap.add_argument("--final", type=Path, default=Path("runs/final"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.manifest:
        ap.error("--manifest is required")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    cmds = commands(selection, args.final, args.data, args.out.parent, args.boot)
    if args.dry_run:
        print(json.dumps(cmds, indent=1))
        return
    try:
        spec_sha256, tag, source = expected_spec_sha256(args.spec_sha256, args.tag)
        for key, path in (("selection", args.selection), ("manifest", args.manifest)):
            if tag[key] != sha256(path):
                raise ValueError(f"git tag {args.tag} names another {key}= than the sha256 of {path}")
        test_files = check_test_files(args.data, args.manifest)
        manifest_path, sanity = check_runs(selection, args.selection, args.final, spec_sha256, args.tag)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(f"spec sha256 from {source}")
    for arm, record in sanity.items():
        print(f"Val reproduction of {arm}: {record['attempts']} attempt(s), the latest passed (value {record['latest']['value']:.4f}, selection {record['latest']['selection_score']:.4f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inputs = {}
    result = {"rl": selection["rl"], "comparator": selection["comparator"], "confirmatory_split": CONFIRMATORY, "commands": cmds, "paired": {},
              "spec_sha256": spec_sha256, "spec_sha256_source": source, "tag": {"name": args.tag, **tag}, "selection_sha256": sha256(args.selection),
              "data_manifest_sha256": sha256(args.manifest), "test_files_sha256": test_files, "run_manifest_sha256": sha256(manifest_path),
              "sanity": sanity}
    for split, cmd in cmds.items():
        for path in (cmd[cmd.index("--scored") + 1], cmd[cmd.index("--vs") + 1]):
            if not Path(path).is_file():
                sys.exit(f"error: missing scored file {path}: run rlexp.metrics score on the final samples first")
            inputs[path] = sha256(path)
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode:
            sys.exit(f"error: report failed for {split}: {done.stderr.strip()[-300:]}")
        paired = json.loads(Path(cmd[cmd.index("--json") + 1]).read_text(encoding="utf-8"))
        try:
            check_cohort(split, paired)
        except ValueError as e:
            sys.exit(f"error: {e}")
        result["paired"][split] = paired
        (args.out.parent / f"confirmatory_paired_{split}.txt").write_text(done.stdout, encoding="utf-8")
    result["input_sha256"] = inputs
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {selection['rl']} against {selection['comparator']} on {CONFIRMATORY} (confirmatory) and {EXPLORATORY} (exploratory)")


if __name__ == "__main__":
    main()
