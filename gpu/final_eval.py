#!/usr/bin/env python3
"""Locked evaluation driver (docs/prereg-v1.md): sample every frozen arm on the Test splits, nothing else.

  python gpu/final_eval.py --spec results/frozen_arms.json --spec-sha256 <hex> --selection results/selection.json \
      --manifest data/MANIFEST.sha256 --data data --out runs/final [--only a,b] [--dry-run]

The spec lists, per arm, the exact sampler settings frozen at the tag, and carries "selection_sha256": the sha256 of the --selection file (the Val
selection of scripts/select_arms.py). The driver refuses a spec whose sha256 differs from the one given (record the hash in the tag message), an unfrozen
spec, a setting outside the frozen grid, any split other than test_t and test_r, a spec that is not what the selection chose (check_against_selection:
the selected arms exactly, every other arm a "reference", a "diagnostic" of a selected arm or a "probe" of an untrained model), and any data file it would read that is missing from --manifest (data/MANIFEST.sha256 of scripts/data_manifest.py) or differs from the hash there. The spec carries
"data_manifest_sha256", the sha256 of that manifest file: another manifest is refused. --data may point anywhere: the manifest, not the path, says whether the files
are the frozen ones. Before it samples an adapter arm (not in --dry-run) <model>/PREPARED.json must name the spec's adapter hashes and the merged files must still hash
as it recorded (gpu/prepare_final.py).
It calls gpu/sample_vllm.py once per arm over both splits; that script is resumable and prompt-atomic, so a killed run can be started again with
the same command. <out>/manifest.json collects, per finished arm, the spec hash, the settings and the command; it is never overwritten by another
spec (use a new --out) and --dry-run does not write it. Scoring happens later, on the CPU, with rlexp.metrics. Needs the GPU environment of
gpu/sample_vllm.py (Colab A100).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("test_t", "test_r")
TEMPERATURES = (0.3, 0.6, 1.0)          # the selectable grid, chosen on Val by fv_tanimoto at k=8
NEAR_GREEDY = 0.01                       # reported at k=1 only, never selected
MODES = ("think", "nothink")
DATA_VARIANTS = {"direct": "processed", "icl": "processed/arms/a4", "name": "processed/arms/name", "ident": "processed/arms/ident",
                 "shuffle": "processed/arms/shuffle", "blank": "processed/arms/blank"}   # directories under --data holding <split>.jsonl
SYSTEM_TEXTS = {"direct": "Reply with only the SMILES string of the drug described. No explanation.", "dataset": None}
REQUIRED = ("id", "model", "mode", "data", "system", "temperature", "k", "max_tokens")
SELECTED_DATA = ("direct", "icl")        # the real profile (with retrieved examples for A4); the other variants are diagnostics or probes
SELECTED_K = 8                           # the endpoint's k
SPEC_KEYS = ("mode", "max_tokens", "data", "system", "model")     # what a selected arm's "spec" block of the selection repeats (scripts/select_arms.py)
SPEC_OPTIONAL = ("quantization", "gpu_memory_utilization")         # sampler options a selected arm may carry only if its block names them
CONTROL_IDS = ("C0", "C1", "C2", "C3")                            # runs/final/<id> of the CPU controls (scripts/final_controls.py): no spec arm may use them
DIAGNOSTIC_DATA = ("shuffle", "blank")   # a diagnostic of a selected arm: the profile of another drug, or none
PROBE_DATA = ("name", "ident")           # a probe of an untrained model: the name given (oracle) or asked for (identification)
ROLES = ("reference", "diagnostic", "probe")
DIAGNOSTIC_FREE = {"id", "role", "of", "data", "system", "splits", "label"}      # all a diagnostic may change from the selected arm it belongs to (label: a table name only)
HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")      # org/name, never a path
MODELS_DIR = "final_models"              # where gpu/prepare_final.py puts merged models (its --models default)


class SpecError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path, expected_sha256, selection_path=None):
    """The validated spec. Raises SpecError for a hash mismatch, an unfrozen spec, or a setting outside the frozen grid. With selection_path (required for any
    real run, which the CLIs do) the spec must also carry that file's sha256 and be what it selected (check_against_selection)."""
    if file_sha256(path) != expected_sha256:
        raise SpecError("The spec's sha256 differs from the frozen one: this is not the registered protocol")
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("frozen") is not True:
        raise SpecError("The spec is not marked frozen")
    if tuple(spec.get("splits", ())) != SPLITS:
        raise SpecError(f"splits must be exactly {list(SPLITS)}")
    seen = set()
    for arm in spec.get("arms", []):
        missing = [key for key in REQUIRED if key not in arm]
        if missing:
            raise SpecError(f"arm {arm.get('id')!r} lacks {missing}")
        if arm["id"] in seen or not str(arm["id"]).replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise SpecError(f"arm id {arm['id']!r} is repeated or not a plain name")
        seen.add(arm["id"])
        if "splits" in arm and (not arm["splits"] or not set(arm["splits"]) <= set(SPLITS) or len(set(arm["splits"])) != len(arm["splits"])):
            raise SpecError(f"arm {arm['id']}: splits must be a non-empty subset of {list(SPLITS)}")
        if arm["mode"] not in MODES or arm["data"] not in DATA_VARIANTS or arm["system"] not in SYSTEM_TEXTS:
            raise SpecError(f"arm {arm['id']}: mode, data or system outside the frozen vocabulary")
        if isinstance(arm["temperature"], bool):
            raise SpecError(f"arm {arm['id']}: temperature must be a number")
        if arm["temperature"] == NEAR_GREEDY:
            if arm.get("k") != 1:
                raise SpecError(f"arm {arm['id']}: near-greedy decoding is a k=1 report, not a selectable temperature")
        elif arm["temperature"] not in TEMPERATURES:
            raise SpecError(f"arm {arm['id']}: temperature {arm['temperature']} is outside the frozen grid {TEMPERATURES}")
        if not (type(arm["k"]) is int and 1 <= arm["k"] <= 8) or not (type(arm["max_tokens"]) is int and arm["max_tokens"] >= 1):      # a bool is not an integer here
            raise SpecError(f"arm {arm['id']}: k must be an integer in 1..8 and max_tokens a positive integer")
    if not seen:
        raise SpecError("the spec has no arms")
    if selection_path is not None:
        if spec.get("selection_sha256") != file_sha256(selection_path):
            raise SpecError("the spec's selection_sha256 is missing or differs from the sha256 of the selection file: the spec was not frozen against it")
        check_against_selection(spec, json.loads(Path(selection_path).read_text(encoding="utf-8")))
    return spec


def is_hub_id(model, trained):
    """A plain Hugging Face hub id (org/name): no path, nothing under the models directory, not the merged model of an adapter arm (`trained`)."""
    return isinstance(model, str) and bool(HUB_ID.fullmatch(model)) and ".." not in model and model.split("/")[0] != MODELS_DIR and model not in trained


def check_against_selection(spec, selection):
    """Refuse a spec that is not what Val selected (selection = results/selection.json of scripts/select_arms.py). The selected arms are the RL arm
    selection["rl"] and the arms of selection["comparator_set"] that have a decoding (temperature not null). Each is a spec arm with the selected temperature,
    k=8, both Test splits (no "splits" key), the real-profile data and exactly the selection's "spec" block (model, mode, max_tokens, data, system; a sampled arm
    without one is an error; quantization and gpu_memory_utilization only if the block names them). An arm the selection gives no adapter_sha256 (and it must
    give one for an arm with a checkpoint) is the untrained hub model of its block with no adapter block; an arm with one samples final_models/<id> and carries
    exactly that adapter. A CPU control (temperature null: C0 to C3, scored from its own samples) is not a spec arm. The other arms of selection["rl_set"] are
    Val-only candidates: a spec arm that carries their id is refused (only the Val-selected RL arm is scored on Test). Every other spec arm has a role, does not
    reuse an id of the selection and is not named C0 to C3:
      "reference": an untrained model (a plain hub id, no adapter) on the direct-answer text (data and system "direct") at a grid temperature;
      "diagnostic": "of" a selected arm, data "shuffle" or "blank", and differs from that arm only in data, system and splits;
      "probe": a name-given or identification run ("data" name or ident) of an untrained model (a plain hub id, no adapter) at a grid temperature.
    quantization and gpu_memory_utilization stay optional for references, diagnostics and probes (a 27B FP8 arm sets gpu_memory_utilization, not quantization). Raises SpecError."""
    arms = {a["id"]: a for a in spec["arms"]}
    try:
        known, rl = selection["arms"], selection["rl"]
        if rl not in selection["rl_set"] or selection["comparator"] not in selection["comparator_set"]:
            raise SpecError("the selection's rl or comparator is not in its own set")
        ids = list(dict.fromkeys([rl, *selection["comparator_set"]]))
        val_only = [a for a in selection["rl_set"] if a not in ids]
        chosen = {i: known[i] for i in ids}
        sampled = [i for i in ids if chosen[i]["temperature"] is not None]
    except KeyError as e:
        raise SpecError(f"the selection lacks {e}") from None
    if rl not in sampled:
        raise SpecError(f"the selected RL arm {rl} has no decoding in the selection")
    for i in ids:
        if i not in sampled and i in arms:
            raise SpecError(f"{i} is a CPU control in the selection (no decoding): it is scored from its own samples, not sampled by this driver")
    trained = {a["model"] for a in spec["arms"] if a.get("adapter") and a.get("role") != "diagnostic"}     # merged models: never a reference, a probe or an untrained selected arm
    for i in sampled:
        sel, arm = chosen[i], arms.get(i)
        if arm is None:
            raise SpecError(f"the selected arm {i} is not in the spec")
        if arm["temperature"] != sel["temperature"]:
            raise SpecError(f"arm {i}: temperature {arm['temperature']} is not the selected {sel['temperature']}")
        if arm["k"] != SELECTED_K or "splits" in arm:
            raise SpecError(f"arm {i}: a selected arm is sampled at k={SELECTED_K} on both Test splits (no splits key)")
        block = sel.get("spec")
        if not isinstance(block, dict) or not set(SPEC_KEYS) <= set(block) <= {*SPEC_KEYS, *SPEC_OPTIONAL}:
            raise SpecError(f"arm {i}: the selection has no spec block {list(SPEC_KEYS)} for it, so the spec cannot be tied to what Val ran: "
                            "add it to the candidates file and select again")
        differs = [k for k in SPEC_KEYS if arm[k] != block[k]] + [k for k in SPEC_OPTIONAL if arm.get(k) != block.get(k)]
        if differs:
            raise SpecError(f"arm {i}: {differs} differ from the selection's spec block {block}")
        if arm["data"] not in SELECTED_DATA:
            raise SpecError(f"arm {i}: a selected arm is sampled on the real profile (data {' or '.join(SELECTED_DATA)}), not on {arm['data']!r}")
        want = sel.get("adapter_sha256")
        have = ((arm.get("adapter") or {}).get("sha256") or {}).get("adapter_model.safetensors")
        if want is None and sel.get("checkpoint"):
            raise SpecError(f"arm {i}: the selection names checkpoint {sel['checkpoint']} but no adapter_sha256, so the spec cannot be tied to it: "
                            "add the hash to the candidates file and select again")
        if want is None and not is_hub_id(arm["model"], trained):               # an arm with an adapter block has its own model in `trained`
            raise SpecError(f"arm {i}: the selection gives it no adapter, so it samples the untrained hub model {block['model']!r} and carries no adapter block")
        if want is not None and arm["model"] != f"{MODELS_DIR}/{i}":
            raise SpecError(f"arm {i}: the selected checkpoint is merged into {MODELS_DIR}/{i}, not {arm['model']!r}")
        if want is not None and have != want:
            raise SpecError(f"arm {i}: the spec's adapter is not the selected checkpoint (adapter_model.safetensors sha256 differs)")
    for arm in spec["arms"]:
        if arm["id"] in val_only:
            raise SpecError(f"arm {arm['id']} is a Val-only RL candidate: only the Val-selected RL arm ({rl}) is scored on Test")
        if arm["id"] in sampled:
            continue
        role, name, model = arm.get("role"), arm["id"], arm["model"]
        if name in CONTROL_IDS:
            raise SpecError(f"arm {name}: {name} is the id of a CPU control; runs/final/{name} holds its samples, so no spec arm may be called {name}")
        if role not in ROLES:
            raise SpecError(f"arm {name} is not a selected arm and has no role: it must be a reference, a diagnostic (of a selected arm) or a probe")
        if name in known:
            raise SpecError(f"arm {name} reuses the id of an arm of the selection without being that arm")
        if role == "diagnostic":
            if arm.get("of") not in sampled:
                raise SpecError(f"arm {name}: a diagnostic names a selected arm in 'of'")
            if arm["data"] not in DIAGNOSTIC_DATA:
                raise SpecError(f"arm {name}: a diagnostic runs on the {' or '.join(DIAGNOSTIC_DATA)} prompts, not on {arm['data']!r}")
            base = arms[arm["of"]]
            differs = sorted(k for k in (set(arm) | set(base)) - DIAGNOSTIC_FREE if arm.get(k) != base.get(k))
            if differs:
                raise SpecError(f"arm {name} differs from {arm['of']} in {differs}: a diagnostic changes only data, system and splits")
            continue
        if not is_hub_id(model, trained):               # an arm with an adapter block has its own model in `trained`
            raise SpecError(f"arm {name}: a {role} is an untrained model: no adapter, and a plain hub id (org/name) that is not a merged model")
        if arm["temperature"] not in TEMPERATURES:
            raise SpecError(f"arm {name}: a {role} runs at a grid temperature {TEMPERATURES}")
        if role == "reference" and (arm["data"], arm["system"]) != ("direct", "direct"):
            raise SpecError(f"arm {name}: a reference runs on the direct-answer text (data and system 'direct')")
        if role == "probe" and arm["data"] not in PROBE_DATA:
            raise SpecError(f"arm {name}: a probe is a name-given or identification run (data {' or '.join(PROBE_DATA)}), not {arm['data']!r}")


def merged_hashes(directory):
    """{file: sha256} of a merged export's weights (*.safetensors) and config.json; empty when there are none."""
    directory = Path(directory)
    names = sorted(p.name for p in directory.glob("*.safetensors")) + (["config.json"] if (directory / "config.json").is_file() else [])
    return {name: file_sha256(directory / name) for name in names}


def check_prepared(arms):
    """Every adapter arm of the plan samples an export that gpu/prepare_final.py made from the frozen adapter: <model>/PREPARED.json records the spec's adapter hashes,
    and the merged files hash now as it recorded. SpecError otherwise (a stale or foreign model directory would be sampled under the arm's name)."""
    for arm in dict(((a["model"], a) for a in arms if a.get("adapter"))).values():
        marker = Path(arm["model"]) / "PREPARED.json"
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except OSError:
            raise SpecError(f"{marker} is missing: {arm['model']} is not an export made by gpu/prepare_final.py") from None
        if recorded.get("adapter_sha256") != arm["adapter"]["sha256"]:
            raise SpecError(f"{marker} records other adapter hashes than the spec's for arm {arm['id']}: {arm['model']} was made from another adapter")
        if not recorded.get("merged_sha256") or recorded["merged_sha256"] != merged_hashes(arm["model"]):
            raise SpecError(f"the files of {arm['model']} no longer hash as {marker} records: prepare it again")


def check_manifest_pin(spec, manifest_path):
    """The --manifest file is the one the spec was frozen with: the spec's data_manifest_sha256 equals its sha256, else SpecError."""
    if spec.get("data_manifest_sha256") != file_sha256(manifest_path):
        raise SpecError("the spec's data_manifest_sha256 is missing or differs from the sha256 of --manifest: that manifest is not the frozen one")


def read_data_manifest(path):
    """{relative path: sha256} of a data/MANIFEST.sha256 (lines 'sha256  path', paths relative to the data directory)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, sep, name = line.strip().partition("  ")
            if not sep:
                raise SpecError(f"{path} is not a sha256 manifest")
            out[name.strip()] = digest
    return out


def check_data(arms, data_root, manifest_path):
    """Every <data>/<variant dir>/<split>.jsonl that the arms will read is in the manifest with its sha256, else SpecError."""
    manifest, checked = read_data_manifest(manifest_path), set()
    for arm in arms:
        for split in arm.get("splits", SPLITS):
            rel = f"{DATA_VARIANTS[arm['data']]}/{split}.jsonl"
            if rel in checked:
                continue
            if rel not in manifest:
                raise SpecError(f"{rel} has no entry in {manifest_path}")
            path = Path(data_root) / rel
            if not path.is_file():
                raise SpecError(f"{path} is missing")
            if file_sha256(path) != manifest[rel]:
                raise SpecError(f"{path} differs from its sha256 in {manifest_path}")
            checked.add(rel)


def command(arm, data_root, out_root):
    """The sample_vllm.py command line of one arm over both Test splits (or the subset its spec names, e.g. only test_t for a 27B reference)."""
    out = [sys.executable, str(HERE / "sample_vllm.py"), "--model", arm["model"], "--data", str(Path(data_root) / DATA_VARIANTS[arm["data"]]),
           "--splits", ",".join(arm.get("splits", SPLITS)), "--mode", arm["mode"], "--k", str(arm["k"]), "--max-tokens", str(arm["max_tokens"]),
           "--temperature", str(arm["temperature"]), "--prompts-per-batch", "auto", "--out", str(Path(out_root) / arm["id"])]
    if SYSTEM_TEXTS[arm["system"]] is not None:
        out += ["--system-text", SYSTEM_TEXTS[arm["system"]]]
    if arm.get("quantization"):
        out += ["--quantization", arm["quantization"]]
    if arm.get("gpu_memory_utilization"):
        out += ["--gpu-memory-utilization", str(arm["gpu_memory_utilization"])]
    return out


def read_run_manifest(path, spec_sha256, selection_sha256):
    """The run manifest of an --out directory, or a new one. One directory holds one spec: a manifest of another spec is an error, never overwritten."""
    if not path.exists():
        return {"spec_sha256": spec_sha256, "selection_sha256": selection_sha256, "arms": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("spec_sha256") != spec_sha256:
        raise SpecError(f"{path} was written under another spec: use a new --out directory")
    if not isinstance(manifest.get("arms"), dict):
        raise SpecError(f"{path} is not a manifest of this driver")
    return manifest


def manifest_entry(arm, cmd, out_root, spec_sha256):
    """What scripts/confirmatory.py checks a finished arm's run directory against: the frozen settings, plus the sampler's own data hash once it exists."""
    entry = {"spec_sha256": spec_sha256, "command": cmd, "splits": list(arm.get("splits", SPLITS)),
             **{key: arm[key] for key in ("model", "mode", "temperature", "max_tokens", "k")}}
    meta = Path(out_root) / arm["id"] / "meta.json"
    data_sha256 = json.loads(meta.read_text(encoding="utf-8")).get("configuration", {}).get("data_sha256") if meta.is_file() else None
    return {**entry, "data_sha256": data_sha256} if data_sha256 else entry


def write_run_manifest(path, manifest):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--selection", type=Path, required=True, help="results/selection.json: the Val selection the spec must realise")
    ap.add_argument("--manifest", type=Path, required=True, help="data/MANIFEST.sha256: the sha256 of every data file the run reads")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/final"))
    ap.add_argument("--only", help="comma-separated arm ids")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them (nothing is written)")
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec, args.spec_sha256, args.selection)
        wanted = set(args.only.split(",")) if args.only else None
        arms = [a for a in spec["arms"] if wanted is None or a["id"] in wanted]
        if wanted and {a["id"] for a in arms} != wanted:
            raise SpecError("--only names an arm that is not in the spec")
        check_manifest_pin(spec, args.manifest)
        check_data(arms, args.data, args.manifest)
        if not args.dry_run:
            check_prepared(arms)
        manifest = read_run_manifest(args.out / "manifest.json", args.spec_sha256, spec["selection_sha256"])
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        cmd = command(arm, args.data, args.out)
        print(" ".join(cmd[:2] + ["..."] + cmd[-4:]) if not args.dry_run else json.dumps(cmd), flush=True)
        if args.dry_run:
            continue
        if subprocess.run(cmd).returncode:
            sys.exit(f"error: arm {arm['id']} failed; rerun the same command to resume")
        manifest["arms"][arm["id"]] = manifest_entry(arm, cmd, args.out, args.spec_sha256)
        write_run_manifest(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
