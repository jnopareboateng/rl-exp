import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu"
ADAPTER = "a" * 64


@pytest.fixture(scope="module")
def fe():
    spec = importlib.util.spec_from_file_location("final_eval", GPU / "final_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm(**kw):
    return {"id": "A2-4b", "model": "merged/a2", "mode": "think", "data": "direct", "system": "direct", "temperature": 0.6, "k": 8, "max_tokens": 256, **kw}


def write(tmp_path, **kw):
    spec = {"frozen": True, "splits": ["test_t", "test_r"], "arms": [arm()], **kw}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ vocabulary and grid (no selection)
def test_a_frozen_spec_with_the_right_hash_loads(fe, tmp_path):
    path, digest = write(tmp_path)
    assert fe.load_spec(path, digest)["arms"][0]["id"] == "A2-4b"


@pytest.mark.parametrize("change,message", [
    ({"frozen": False}, "not marked frozen"), ({"splits": ["val"]}, "splits must be"), ({"splits": ["test_t", "test_r", "val"]}, "splits must be"),
    ({"arms": []}, "no arms"), ({"arms": [arm(temperature=0.5)]}, "frozen grid"), ({"arms": [arm(mode="auto")]}, "frozen vocabulary"),
    ({"arms": [arm(data="train")]}, "frozen vocabulary"), ({"arms": [arm(k=9)]}, "k must be"), ({"arms": [arm(max_tokens=0)]}, "max_tokens"),
    ({"arms": [arm(temperature=0.01)]}, "k=1 report"), ({"arms": [arm(), arm()]}, "repeated"), ({"arms": [arm(id="../x")]}, "plain name"), ({"arms": [{"id": "x"}]}, "lacks")])
def test_invalid_specs_are_refused(fe, tmp_path, change, message):
    path, digest = write(tmp_path, **change)
    with pytest.raises(fe.SpecError, match=message):
        fe.load_spec(path, digest)


def test_near_greedy_is_allowed_only_as_a_k1_report(fe, tmp_path):
    path, digest = write(tmp_path, arms=[arm(temperature=0.01, k=1)])
    assert fe.load_spec(path, digest)["arms"][0]["temperature"] == 0.01


def test_an_arm_may_restrict_itself_to_a_subset_of_the_test_splits(fe, tmp_path):
    path, digest = write(tmp_path, arms=[arm(splits=["test_t"])])
    spec = fe.load_spec(path, digest)
    cmd = fe.command(spec["arms"][0], "data", "runs/final")
    assert cmd[cmd.index("--splits") + 1] == "test_t"
    for bad in ([], ["val"], ["test_t", "test_t"], ["test_t", "test_r", "test_x"]):
        p2, d2 = write(tmp_path, arms=[arm(splits=bad)])
        with pytest.raises(fe.SpecError, match="subset"):
            fe.load_spec(p2, d2)


def test_a_changed_spec_is_refused_even_if_valid(fe, tmp_path):
    path, digest = write(tmp_path)
    path.write_text(path.read_text().replace("0.6", "1.0"))
    with pytest.raises(fe.SpecError, match="sha256"):
        fe.load_spec(path, digest)


def test_command_covers_both_test_splits_with_the_frozen_settings(fe):
    cmd = fe.command(arm(), "data", "runs/final")
    assert cmd[cmd.index("--splits") + 1] == "test_t,test_r" and cmd[cmd.index("--data") + 1] == "data/processed"
    assert cmd[cmd.index("--temperature") + 1] == "0.6" and cmd[cmd.index("--k") + 1] == "8" and cmd[cmd.index("--out") + 1] == "runs/final/A2-4b"
    assert cmd[cmd.index("--system-text") + 1].startswith("Reply with only the SMILES")
    plain = fe.command(arm(system="dataset", data="icl", quantization="fp8", gpu_memory_utilization=0.93), "data", "runs/final")
    assert "--system-text" not in plain and plain[plain.index("--data") + 1] == "data/processed/arms/a4"
    assert plain[plain.index("--quantization") + 1] == "fp8" and plain[plain.index("--gpu-memory-utilization") + 1] == "0.93"


def test_the_blank_profile_diagnostic_is_in_the_vocabulary(fe, tmp_path):
    path, digest = write(tmp_path, arms=[arm(data="blank")])
    assert fe.load_spec(path, digest)["arms"][0]["data"] == "blank"
    cmd = fe.command(arm(data="blank"), "data", "runs/final")
    assert cmd[cmd.index("--data") + 1] == "data/processed/arms/blank"


# ------------------------------------------------------------------------------------------------ the spec against the Val selection
def rl_adapter(digest=ADAPTER):
    return {"path": "adapters/rl-full/step-40", "base": "Qwen/Qwen3.5-4B", "sha256": {"adapter_model.safetensors": digest, "adapter_config.json": "b" * 64}}


def selected_arms():
    """The spec arms that the selection below requires: an untrained arm and an RL checkpoint (C3, a CPU control, is not sampled here)."""
    return {"A0n": arm(id="A0n", model="Qwen/Qwen3.5-4B", mode="nothink"), "RL-full": arm(id="RL-full", model="final_models/RL-full", adapter=rl_adapter())}


def block(mode="think", **kw):
    """The selection's spec block of an arm (what scripts/select_arms.py copies from the candidates file)."""
    return {"mode": mode, "max_tokens": 256, "data": "direct", "system": "direct", **kw}


def selection():
    """C3 a CPU control, A0n an untrained arm, RL-full the selected RL arm, RL-contrastive a Val-only RL candidate (in rl_set, not selected)."""
    return {"arms": {"C3": {"temperature": None, "checkpoint": None},
                     "A0n": {"temperature": 0.6, "checkpoint": None, "spec": block("nothink")},
                     "RL-full": {"temperature": 0.6, "checkpoint": "step-40", "adapter_sha256": ADAPTER, "spec": block()},
                     "RL-contrastive": {"temperature": 1.0, "checkpoint": "step-10", "adapter_sha256": "c" * 64, "spec": block()}},
            "comparator": "C3", "rl": "RL-full", "comparator_set": ["C3", "A0n"], "rl_set": ["RL-full", "RL-contrastive"]}


def diagnostic(of="A0n", **kw):
    return {**selected_arms()[of], "id": f"{of}-shuffle", "role": "diagnostic", "of": of, "data": "shuffle", **kw}


def reference(**kw):
    return {**arm(id="ref-9b", model="Qwen/Qwen3.5-9B", mode="nothink", role="reference"), **kw}


def probe(**kw):
    """A name-given run of an untrained model, as the Val probes ran (T=1.0, no think)."""
    return {**arm(id="name-9b", model="Qwen/Qwen3.5-9B", mode="nothink", role="probe", data="name", temperature=1.0), **kw}


def frozen(tmp_path, arms=None, extra=(), sel=None, **spec_kw):
    """(spec path, spec sha256, selection path): a frozen spec written against a selection file."""
    sel_path = tmp_path / "selection.json"
    sel_path.write_text(json.dumps(sel if sel is not None else selection()))
    spec = {"frozen": True, "splits": ["test_t", "test_r"], "selection_sha256": sha(sel_path), **spec_kw,
            "arms": [*(selected_arms() if arms is None else arms).values(), *extra]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, sha(path), sel_path


def test_a_spec_that_realises_the_selection_loads_with_references_diagnostics_and_probes(fe, tmp_path):
    fp8 = dict(model="Qwen/Qwen3.5-27B-FP8", splits=["test_t"], temperature=1.0, gpu_memory_utilization=0.93)      # 27B FP8: a memory fraction, no quantization flag
    extra = [reference(), reference(id="ref-27b", k=2, **fp8), diagnostic(), diagnostic("RL-full", data="blank"),
             probe(), probe(id="ident-4b", model="Qwen/Qwen3.5-4B", data="ident", max_tokens=128), probe(id="ident-27b", data="ident", k=2, max_tokens=128, **fp8)]
    path, digest, sel_path = frozen(tmp_path, extra=extra)
    arms = fe.load_spec(path, digest, sel_path)["arms"]
    assert [a["id"] for a in arms] == ["A0n", "RL-full", "ref-9b", "ref-27b", "A0n-shuffle", "RL-full-shuffle", "name-9b", "ident-4b", "ident-27b"]
    cmd = fe.command(arms[3], "data", "runs/final")
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.93" and "--quantization" not in cmd and cmd[cmd.index("--splits") + 1] == "test_t"


CASES = {
    "temperature differs from the selected one": (lambda a, e, s: a["RL-full"].update(temperature=1.0), "is not the selected 0.6"),
    "a sampled selected arm is missing": (lambda a, e, s: a.pop("A0n"), "selected arm A0n is not in the spec"),
    "an extra arm has no role": (lambda a, e, s: e.append(arm(id="surprise")), "has no role"),
    "an extra arm has an unknown role": (lambda a, e, s: e.append(arm(id="surprise", role="bonus")), "has no role"),
    "a selected arm is restricted to one split": (lambda a, e, s: a["RL-full"].update(splits=["test_t"]), "both Test splits"),
    "a selected arm is sampled at k below 8": (lambda a, e, s: a["A0n"].update(k=4), "k=8"),
    "a CPU control is a spec arm": (lambda a, e, s: e.append(arm(id="C3")), "CPU control"),
    "the adapter is not the selected checkpoint": (lambda a, e, s: a["RL-full"].update(adapter=rl_adapter("c" * 64)), "not the selected checkpoint"),
    "the spec arm has no adapter": (lambda a, e, s: a["RL-full"].pop("adapter"), "not the selected checkpoint"),
    "a checkpoint without a hash in the selection": (lambda a, e, s: s["arms"]["RL-full"].pop("adapter_sha256"), "no adapter_sha256"),
    "a selected arm on a diagnostic prompt set": (lambda a, e, s: a["A0n"].update(data="shuffle"), r"\['data'\] differ"),
    "the selected RL arm is missing": (lambda a, e, s: a.pop("RL-full"), "selected arm RL-full is not in the spec"),
    "the selected RL arm has no decoding": (lambda a, e, s: s["arms"]["RL-full"].update(temperature=None), "RL-full has no decoding"),
    "a Val-only RL candidate is a spec arm": (lambda a, e, s: e.append(arm(id="RL-contrastive", model="final_models/RL-contrastive", adapter=rl_adapter("c" * 64), temperature=1.0)), "Val-only RL candidate"),
    "a Val-only RL candidate as a reference": (lambda a, e, s: e.append(reference(id="RL-contrastive")), "Val-only RL candidate"),
    "a Val-only RL candidate as a probe": (lambda a, e, s: e.append(probe(id="RL-contrastive")), "Val-only RL candidate"),
    "a diagnostic of a Val-only RL candidate": (lambda a, e, s: e.append({**diagnostic("RL-full"), "of": "RL-contrastive"}), "names a selected arm"),
    "rl is not in rl_set": (lambda a, e, s: s.update(rl="A0n"), "own set"),
    "the selection has no spec block for a comparator": (lambda a, e, s: s["arms"]["A0n"].pop("spec"), "no spec block"),
    "the selection has no spec block for the RL arm": (lambda a, e, s: s["arms"]["RL-full"].pop("spec"), "no spec block"),
    "the spec block lacks a key": (lambda a, e, s: s["arms"]["A0n"].update(spec={"mode": "nothink", "max_tokens": 256, "data": "direct"}), "no spec block"),
    "the spec block has an extra key": (lambda a, e, s: s["arms"]["A0n"].update(spec={**block("nothink"), "temperature": 0.6}), "no spec block"),
    "the spec block is not a block": (lambda a, e, s: s["arms"]["A0n"].update(spec="nothink"), "no spec block"),
    "the arm's mode differs from the spec block": (lambda a, e, s: a["A0n"].update(mode="think"), r"\['mode'\] differ"),
    "the arm's max_tokens differ from the spec block": (lambda a, e, s: a["RL-full"].update(max_tokens=512), r"\['max_tokens'\] differ"),
    "the arm's data differs from the spec block": (lambda a, e, s: a["A0n"].update(data="icl"), r"\['data'\] differ"),
    "the arm's system differs from the spec block": (lambda a, e, s: a["A0n"].update(system="dataset"), r"\['system'\] differ"),
    "the spec block itself names a diagnostic prompt set": (lambda a, e, s: (s["arms"]["A0n"]["spec"].update(data="shuffle"), a["A0n"].update(data="shuffle")), "real profile"),
    "comparator is not in comparator_set": (lambda a, e, s: s.update(comparator="RL-full"), "own set"),
    "the selection lacks an arm entry": (lambda a, e, s: s["arms"].pop("A0n"), "selection lacks"),
    "a diagnostic changes the model": (lambda a, e, s: e.append(diagnostic(model="Qwen/Qwen3.5-9B")), "changes only data"),
    "a diagnostic changes the temperature": (lambda a, e, s: e.append(diagnostic(temperature=1.0)), "changes only data"),
    "a diagnostic changes the mode": (lambda a, e, s: e.append(diagnostic(mode="think")), "changes only data"),
    "a diagnostic changes max_tokens": (lambda a, e, s: e.append(diagnostic(max_tokens=512)), "changes only data"),
    "a diagnostic changes k": (lambda a, e, s: e.append(diagnostic(k=2)), "changes only data"),
    "a diagnostic changes the adapter": (lambda a, e, s: e.append(diagnostic("RL-full", adapter=rl_adapter("c" * 64))), "changes only data"),
    "a diagnostic adds an adapter": (lambda a, e, s: e.append(diagnostic("A0n", adapter=rl_adapter())), "changes only data"),
    "a diagnostic of an unknown arm": (lambda a, e, s: e.append({**diagnostic(), "of": "nope"}), "names a selected arm"),
    "a diagnostic of a CPU control": (lambda a, e, s: e.append({**diagnostic(), "of": "C3"}), "names a selected arm"),
    "a diagnostic of a diagnostic": (lambda a, e, s: e.extend([diagnostic(), {**diagnostic(), "id": "again", "of": "A0n-shuffle"}]), "names a selected arm"),
    "a diagnostic on name prompts": (lambda a, e, s: e.append(diagnostic(data="name")), "shuffle or blank"),
    "a diagnostic on the real prompts": (lambda a, e, s: e.append(diagnostic(data="direct", system="dataset")), "shuffle or blank"),
    "a reference with an adapter": (lambda a, e, s: e.append(reference(adapter=rl_adapter())), "untrained model"),
    "a reference at near-greedy": (lambda a, e, s: e.append(reference(temperature=0.01, k=1)), "grid temperature"),
    "a reference off the grid": (lambda a, e, s: e.append(reference(temperature=0.5)), "frozen grid"),
    "a reference on the blank prompts": (lambda a, e, s: e.append(reference(data="blank")), "direct-answer text"),
    "a reference on name prompts": (lambda a, e, s: e.append(reference(data="name")), "direct-answer text"),
    "a reference with the dataset system text": (lambda a, e, s: e.append(reference(system="dataset")), "direct-answer text"),
    "a reference that is a merged model": (lambda a, e, s: e.append(reference(model="final_models/RL-full")), "plain hub id"),
    "a reference with an absolute path": (lambda a, e, s: e.append(reference(model="/content/Qwen/Qwen3.5-9B")), "plain hub id"),
    "a reference with a relative path": (lambda a, e, s: e.append(reference(model="./Qwen/Qwen3.5-9B")), "plain hub id"),
    "a reference with a parent path": (lambda a, e, s: e.append(reference(model="../x/Qwen3.5-9B")), "plain hub id"),
    "a reference with a deep path": (lambda a, e, s: e.append(reference(model="a/b/Qwen3.5-9B")), "plain hub id"),
    "a reference with a bare name": (lambda a, e, s: e.append(reference(model="Qwen3.5-9B")), "plain hub id"),
    "a reference with a dotted segment": (lambda a, e, s: e.append(reference(model="Qwen/..")), "plain hub id"),
    "a reference with a double dot inside the name": (lambda a, e, s: e.append(reference(model="Qwen/Qwen3..5-9B")), "plain hub id"),
    "a reference that reuses a selection id": (lambda a, e, s: (s["arms"].update(A1={"temperature": 0.6, "checkpoint": None}), e.append(reference(id="A1"))), "reuses the id"),
    "a probe with an adapter": (lambda a, e, s: e.append(probe(adapter=rl_adapter())), "untrained model"),
    "a probe of a merged model": (lambda a, e, s: e.append(probe(model="final_models/RL-full")), "plain hub id"),
    "a probe of a model under the models directory that no arm of the spec uses": (lambda a, e, s: e.append(probe(model="final_models/old-run")), "plain hub id"),
    "a reference under the models directory that no arm of the spec uses": (lambda a, e, s: e.append(reference(model="final_models/old-run")), "plain hub id"),
    "a probe of the merged model of an adapter arm": (lambda a, e, s: (a["RL-full"].update(model="stage/RL-full"), e.append(probe(model="stage/RL-full"))), "plain hub id"),
    "a probe of a local path": (lambda a, e, s: e.append(probe(model="/content/Qwen3.5-9B")), "plain hub id"),
    "a probe at near-greedy": (lambda a, e, s: e.append(probe(temperature=0.01, k=1)), "grid temperature"),
    "a probe off the grid": (lambda a, e, s: e.append(probe(temperature=0.5)), "frozen grid"),
    "a probe on the real profile": (lambda a, e, s: e.append(probe(data="direct")), "name-given or identification"),
    "a probe on shuffled prompts": (lambda a, e, s: e.append(probe(data="shuffle")), "name-given or identification"),
    "a probe on blank prompts": (lambda a, e, s: e.append(probe(data="blank")), "name-given or identification"),
    "a probe that reuses a selection id": (lambda a, e, s: (s["arms"].update(A1={"temperature": 0.6, "checkpoint": None}), e.append(probe(id="A1"))), "reuses the id"),
    "a diagnostic that reuses a selection id": (lambda a, e, s: (s["arms"].update(A1={"temperature": 0.6, "checkpoint": None}), e.append(diagnostic(id="A1"))), "reuses the id"),
}


@pytest.mark.parametrize("case", CASES)
def test_a_spec_that_departs_from_the_selection_is_refused(fe, tmp_path, case):
    edit, message = CASES[case]
    arms, extra, sel = selected_arms(), [], selection()
    edit(arms, extra, sel)
    path, digest, sel_path = frozen(tmp_path, arms, extra, sel)
    with pytest.raises(fe.SpecError, match=message):
        fe.load_spec(path, digest, sel_path)


def test_the_val_only_rl_candidate_is_neither_required_nor_looked_up(fe, tmp_path):
    sel = selection()
    del sel["arms"]["RL-contrastive"]                                    # its entry is not consulted: only selection["rl"] and the comparators are required
    path, digest, sel_path = frozen(tmp_path, sel=sel)
    assert [a["id"] for a in fe.load_spec(path, digest, sel_path)["arms"]] == ["A0n", "RL-full"]


def test_a_comparator_that_is_sampled_is_required_but_the_other_rl_arms_are_not(fe, tmp_path):
    sel = selection()
    sel["comparator_set"] = ["C3", "A0n", "A2"]
    sel["arms"]["A2"] = {"temperature": 1.0, "checkpoint": None, "spec": block(), "adapter_sha256": "d" * 64}
    path, digest, sel_path = frozen(tmp_path, sel=sel)
    with pytest.raises(fe.SpecError, match="selected arm A2 is not in the spec"):
        fe.load_spec(path, digest, sel_path)
    a2 = arm(id="A2", model="final_models/A2", adapter={**rl_adapter("d" * 64)}, temperature=1.0)
    path, digest, sel_path = frozen(tmp_path, extra=[a2], sel=sel)
    assert [a["id"] for a in fe.load_spec(path, digest, sel_path)["arms"]] == ["A0n", "RL-full", "A2"]


def test_the_spec_must_carry_the_hash_of_the_selection_it_was_frozen_against(fe, tmp_path):
    path, digest, sel_path = frozen(tmp_path)
    assert fe.load_spec(path, digest, sel_path)
    sel_path.write_text(sel_path.read_text() + " ")                     # the selection changed after the freeze
    with pytest.raises(fe.SpecError, match="selection_sha256"):
        fe.load_spec(path, digest, sel_path)
    spec = json.loads(path.read_text())
    del spec["selection_sha256"]
    path.write_text(json.dumps(spec))
    with pytest.raises(fe.SpecError, match="selection_sha256"):
        fe.load_spec(path, sha(path), sel_path)


# ------------------------------------------------------------------------------------------------ the CLI
def make_data(tmp_path, variants=("processed",), splits=("test_t", "test_r")):
    """A synthetic data directory and its manifest (sha256sum format, paths relative to the data directory)."""
    root, lines = tmp_path / "data", []
    for variant in variants:
        for split in splits:
            f = root / variant / f"{split}.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"synthetic {variant} {split}\n")
            lines.append(f"{sha(f)}  {variant}/{split}.jsonl")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return root, root / "MANIFEST.sha256"


def cli(fe, tmp_path, spec, selection_path, root, manifest, *extra, out="out"):
    path, digest = spec
    fe.main(["--spec", str(path), "--spec-sha256", digest, "--selection", str(selection_path), "--manifest", str(manifest), "--data", str(root),
             "--out", str(tmp_path / out), *extra])


@pytest.fixture
def world(fe, tmp_path, monkeypatch):
    """A spec with three arms (A0n and RL-full selected, A0n-shuffle a diagnostic), its data, and a fake sampler that records meta.json like sample_vllm.py."""
    path, digest, sel_path = frozen(tmp_path, extra=[diagnostic()])
    root, manifest = make_data(tmp_path, variants=("processed", "processed/arms/shuffle"))
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps({"configuration": {"data_sha256": "d" * 64, "k": 8}}))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(fe.subprocess, "run", fake_run)
    return type("World", (), {"spec": (path, digest), "sel": sel_path, "root": root, "manifest": manifest, "calls": calls, "tmp": tmp_path})


def test_cli_dry_run_prints_commands_and_writes_nothing(fe, world, capsys):
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--dry-run")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and json.loads(lines[0])[1].endswith("sample_vllm.py") and not world.calls
    assert not (world.tmp / "out").exists()                              # neither the directory nor a manifest
    with pytest.raises(SystemExit):
        cli(fe, world.tmp, (world.spec[0], "0" * 64), world.sel, world.root, world.manifest, "--dry-run")
    with pytest.raises(SystemExit, match="--only names"):
        cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--dry-run", "--only", "nope")


def test_the_selection_and_the_data_manifest_are_required_arguments(fe, world):
    path, digest = world.spec
    base = ["--spec", str(path), "--spec-sha256", digest, "--out", str(world.tmp / "out"), "--dry-run"]
    for missing in ("--selection", "--manifest"):
        given = {"--selection": str(world.sel), "--manifest": str(world.manifest)}
        argv = base + [x for k, v in given.items() if k != missing for x in (k, v)]
        with pytest.raises(SystemExit) as e:
            fe.main(argv)
        assert e.value.code == 2


def test_a_spec_that_is_not_the_selection_stops_the_cli_before_any_sampling(fe, world):
    path, digest, sel_path = frozen(world.tmp, arms={**selected_arms(), "RL-full": arm(id="RL-full", model="final_models/RL-full", adapter=rl_adapter(), temperature=1.0)})
    with pytest.raises(SystemExit, match="is not the selected 0.6"):
        cli(fe, world.tmp, (path, digest), sel_path, world.root, world.manifest)
    assert not world.calls


def test_the_run_manifest_accumulates_per_arm_and_records_the_sampler_data_hash(fe, world):
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--only", "A0n")
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--only", "RL-full")
    manifest = json.loads((world.tmp / "out" / "manifest.json").read_text())
    assert manifest["spec_sha256"] == world.spec[1] and manifest["selection_sha256"] == sha(world.sel)
    assert sorted(manifest["arms"]) == ["A0n", "RL-full"]                # the second invocation kept the first one's entry
    entry = manifest["arms"]["RL-full"]
    assert (entry["spec_sha256"], entry["model"], entry["mode"], entry["temperature"], entry["k"], entry["max_tokens"]) == \
        (world.spec[1], "final_models/RL-full", "think", 0.6, 8, 256)
    assert entry["splits"] == ["test_t", "test_r"] and entry["data_sha256"] == "d" * 64 and entry["command"][1].endswith("sample_vllm.py")
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--only", "A0n")        # a rerun replaces its own entry only
    assert sorted(json.loads((world.tmp / "out" / "manifest.json").read_text())["arms"]) == ["A0n", "RL-full"]


def test_a_manifest_written_under_another_spec_is_never_overwritten(fe, world):
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--only", "A0n")
    before = (world.tmp / "out" / "manifest.json").read_text()
    other = frozen(world.tmp, extra=[diagnostic(), reference()])         # another spec (one more arm) in the same --out
    assert other[1] != world.spec[1]
    world.calls.clear()
    for flags in ((), ("--dry-run",)):
        with pytest.raises(SystemExit, match="another spec"):
            cli(fe, world.tmp, other[:2], other[2], world.root, world.manifest, *flags)
    assert (world.tmp / "out" / "manifest.json").read_text() == before and not world.calls


def test_a_failing_arm_stops_the_run_with_a_resume_hint_and_leaves_no_entry(fe, world, monkeypatch):
    results = iter([0, 3])

    def run(cmd):
        world.calls.append(cmd)
        return type("R", (), {"returncode": next(results)})()

    monkeypatch.setattr(fe.subprocess, "run", run)
    with pytest.raises(SystemExit, match="rerun the same command"):
        cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest)
    arms = json.loads((world.tmp / "out" / "manifest.json").read_text())["arms"]
    assert sorted(arms) == ["A0n"] and "data_sha256" not in arms["A0n"]       # the finished arm only; no meta.json, so no sampler hash


def test_the_data_files_must_match_the_manifest_before_anything_is_sampled(fe, world):
    victim = world.root / "processed" / "test_t.jsonl"
    victim.write_text("a different file\n")
    for flags in ((), ("--dry-run",)):
        with pytest.raises(SystemExit, match="differs from its sha256"):
            cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, *flags)
    assert not world.calls
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--only", "A0n-shuffle")      # that arm reads other files
    assert len(world.calls) == 1
    victim.unlink()
    with pytest.raises(SystemExit, match="is missing"):
        cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--only", "A0n")


def test_a_data_file_without_a_manifest_entry_is_refused(fe, world):
    manifest_lines = world.manifest.read_text().splitlines()
    world.manifest.write_text("\n".join(l for l in manifest_lines if "shuffle" not in l) + "\n")
    cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--dry-run", "--only", "A0n")
    with pytest.raises(SystemExit, match="processed/arms/shuffle/test_t.jsonl has no entry"):
        cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--dry-run")
    world.manifest.write_text("not a manifest line\n")
    with pytest.raises(SystemExit, match="not a sha256 manifest"):
        cli(fe, world.tmp, world.spec, world.sel, world.root, world.manifest, "--dry-run")


def test_an_arm_restricted_to_one_split_reads_and_checks_only_that_file(fe, tmp_path, monkeypatch):
    big = reference(id="ref-27b", splits=["test_t"], k=2)
    path, digest, sel_path = frozen(tmp_path, extra=[big])
    root, manifest = make_data(tmp_path)
    (root / "processed" / "test_r.jsonl").write_text("changed\n")        # the other file is bad: only the arms that read it care
    with pytest.raises(SystemExit, match="test_r.jsonl differs"):
        cli(fe, tmp_path, (path, digest), sel_path, root, manifest, "--dry-run")
    cli(fe, tmp_path, (path, digest), sel_path, root, manifest, "--dry-run", "--only", "ref-27b")


def test_the_data_directory_is_free_when_the_manifest_matches(fe, world):
    moved = world.tmp / "elsewhere"
    (world.root).rename(moved)
    cli(fe, world.tmp, world.spec, world.sel, moved, moved / "MANIFEST.sha256", "--dry-run")
    cmd = fe.command(selected_arms()["A0n"], moved, "out")
    assert cmd[cmd.index("--data") + 1] == f"{moved}/processed"


def test_the_data_manifest_check_applies_to_the_blank_profile_prompts_too(fe, tmp_path, monkeypatch):
    path, digest, sel_path = frozen(tmp_path, extra=[diagnostic("RL-full", data="blank")])
    root, manifest = make_data(tmp_path, variants=("processed", "processed/arms/blank"))
    cli(fe, tmp_path, (path, digest), sel_path, root, manifest, "--dry-run")
    blank = root / "processed" / "arms" / "blank" / "test_r.jsonl"
    blank.write_text("every row has the same prompt, but not this file\n")
    with pytest.raises(SystemExit, match="blank/test_r.jsonl differs from its sha256"):
        cli(fe, tmp_path, (path, digest), sel_path, root, manifest, "--dry-run")
    manifest.write_text("".join(line + "\n" for line in manifest.read_text().splitlines() if "blank" not in line))
    with pytest.raises(SystemExit, match="processed/arms/blank/test_t.jsonl has no entry"):
        cli(fe, tmp_path, (path, digest), sel_path, root, manifest, "--dry-run")
    cli(fe, tmp_path, (path, digest), sel_path, root, manifest, "--dry-run", "--only", "RL-full")           # the blank files are read only by the arm that names them
