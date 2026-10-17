import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "confirmatory.py"
SPEC = "5" * 64
DATA_HASH = "d" * 64


@pytest.fixture(scope="module")
def cf():
    spec = importlib.util.spec_from_file_location("confirmatory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEL = {"rl": "RL-full", "comparator": "C3", "arms": {"RL-full": {"temperature": 0.6}, "C3": {"temperature": None}}}
SETTINGS = {"model": "final_models/RL-full", "mode": "nothink", "temperature": 0.6, "max_tokens": 256, "k": 8}


def control_meta(control="C3", **kw):
    """The meta.json that scripts/final_controls.py writes for a CPU control (C3 also records the pool it came from)."""
    meta = {"control": control, "k": 8, "splits": ["test_t", "test_r"], "rows": 180, "source": "runs/x/samples.jsonl", "source_sha256": "e" * 64}
    return {**meta, **({"pool_sha256": "f" * 64, "train_score": 0.19, "source_kind": "pool"} if control == "C3" else {}), **kw}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_commands_pair_the_chosen_arms_on_both_test_splits(cf):
    cmds = cf.commands(SEL, "runs/final", "data/processed", "results", boot=100, seed=3)
    assert set(cmds) == {"test_t", "test_r"}
    for split, cmd in cmds.items():
        assert cmd[cmd.index("--scored") + 1] == "runs/final/RL-full/scored.v3.jsonl" and cmd[cmd.index("--vs") + 1] == "runs/final/C3/scored.v3.jsonl"
        assert cmd[cmd.index("--split") + 1] == split and cmd[cmd.index("--boot") + 1] == "100" and cmd[cmd.index("--seed") + 1] == "3"
        assert cmd[cmd.index("--json") + 1] == f"results/paired_{split}.json"


def test_dry_run_prints_and_runs_nothing(cf, tmp_path, capsys):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SEL))
    cf.main(["--selection", str(sel), "--out", str(tmp_path / "r.json"), "--dry-run"])
    assert "paired_test_t.json" in capsys.readouterr().out and not (tmp_path / "r.json").exists()


def make_report(split, drugs=None, rows=None, fv_drugs=None, p=0.0312):
    """What `rlexp.metrics report --json` writes for a paired run on one split: the cohort the pair covers and the fv_tanimoto block."""
    cohort = {"test_t": (41, 100), "test_r": (30, 80)}[split]
    drugs, rows = cohort[0] if drugs is None else drugs, cohort[1] if rows is None else rows
    fv = {"n_drugs": drugs if fv_drugs is None else fv_drugs, "p_value": p, "wins": 7}
    return {"splits": {split: {"n_drugs": drugs, "n_rows": rows, "paired": {"n_drugs": drugs, "n_rows": rows, "metrics": {"fv_tanimoto": fv}}}}}


@pytest.fixture
def world(cf, tmp_path, monkeypatch):
    """A selection (RL-full sampled, C3 a CPU control), a final directory with the run manifest, meta.json and scored files, and a fake `rlexp.metrics report`."""
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SEL))
    final = tmp_path / "final"
    for arm in ("RL-full", "C3"):
        (final / arm).mkdir(parents=True)
        (final / arm / "scored.v3.jsonl").write_text("{}\n")
    (final / "RL-full" / "meta.json").write_text(json.dumps({"configuration": {**SETTINGS, "data_sha256": DATA_HASH, "top_p": 1.0, "seed": 0}}))
    (final / "C3" / "meta.json").write_text(json.dumps(control_meta()))
    entry = {"spec_sha256": SPEC, **SETTINGS, "k": 8, "splits": ["test_t", "test_r"], "data_sha256": DATA_HASH, "command": ["x"]}
    manifest = {"spec_sha256": SPEC, "selection_sha256": sha(sel), "arms": {"RL-full": entry}}
    (final / "manifest.json").write_text(json.dumps(manifest))
    reports = {"overrides": {}}

    def fake_run(cmd, capture_output, text):
        split = cmd[cmd.index("--split") + 1]
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(reports["overrides"][split] if split in reports["overrides"] else make_report(split)))
        return type("R", (), {"returncode": 0, "stdout": "table", "stderr": ""})()

    monkeypatch.setattr(cf.subprocess, "run", fake_run)

    def run(*extra, spec=SPEC, out=None):
        cf.main(["--selection", str(sel), "--spec-sha256", spec, "--final", str(final), "--data", str(tmp_path), "--out", str(out or tmp_path / "res" / "confirmatory.json"), *extra])

    def edit(mutate):
        data = json.loads((final / "manifest.json").read_text())
        mutate(data)
        (final / "manifest.json").write_text(json.dumps(data))

    return type("World", (), {"run": staticmethod(run), "final": final, "sel": sel, "tmp": tmp_path, "edit": staticmethod(edit), "reports": reports, "out": tmp_path / "res" / "confirmatory.json"})


def test_missing_scored_files_stop_with_a_hint(cf, world):
    (world.final / "C3" / "scored.v3.jsonl").unlink()
    with pytest.raises(SystemExit, match="rlexp.metrics score"):
        world.run()
    assert not world.out.exists()


def test_a_run_collects_both_pairs_the_input_hashes_and_the_provenance(world):
    world.run()
    res = json.loads(world.out.read_text())
    assert res["paired"]["test_t"]["splits"]["test_t"]["paired"]["n_drugs"] == 41 and res["paired"]["test_r"]["splits"]["test_r"]["paired"]["n_drugs"] == 30
    assert len(res["input_sha256"]) == 2 and (world.out.parent / "paired_test_t.txt").read_text() == "table"
    assert res["spec_sha256"] == SPEC and res["selection_sha256"] == sha(world.sel) and res["run_manifest_sha256"] == sha(world.final / "manifest.json")
    assert res["input_sha256"][str(world.final / "RL-full" / "scored.v3.jsonl")] == sha(world.final / "RL-full" / "scored.v3.jsonl")


def test_the_output_directory_is_created_before_anything_is_written_into_it(world):
    out = world.tmp / "deep" / "er" / "confirmatory.json"
    world.run(out=out)
    assert out.is_file() and (out.parent / "paired_test_t.json").is_file()


def test_the_spec_hash_is_required_for_a_real_run(cf, world):
    with pytest.raises(SystemExit) as e:
        cf.main(["--selection", str(world.sel), "--final", str(world.final), "--out", str(world.out)])
    assert e.value.code == 2


@pytest.mark.parametrize("split,override,message", [
    ("test_t", make_report("test_t", drugs=40), "test_t: the pair covers 40 drugs"),            # a drug missing from either run
    ("test_t", make_report("test_t", rows=99), "99 prompt rows"),
    ("test_t", make_report("test_t", drugs=0, rows=0, fv_drugs=0, p=None), "0 drugs"),            # val rows reported as test_t: the empty pair of the review
    ("test_t", make_report("test_t", fv_drugs=30), "fv_tanimoto 30 drugs"),                       # k<8 for some drugs: the endpoint is undefined for them
    ("test_t", make_report("test_t", p=None), "p-value None"),                                    # k<8 for all: no test
    ("test_r", make_report("test_r", drugs=29), "test_r: the pair covers 29 drugs"),
    ("test_r", make_report("test_r", rows=100), "100 prompt rows"),                               # the Test-T row count on Test-R
    ("test_t", {"splits": {"test_t": {"n_drugs": 0, "n_rows": 0, "n_samples": 0}}}, "None drugs"),    # no paired block at all
    ("test_t", {}, "None drugs"),
])
def test_a_pair_that_does_not_cover_the_cohort_is_an_error_and_writes_no_result(world, split, override, message):
    world.reports["overrides"][split] = override
    with pytest.raises(SystemExit, match=message):
        world.run()
    assert not world.out.exists()


# ------------------------------------------------------------------------------------------------ the pair comes from the frozen spec
def test_the_manifest_must_exist_and_name_the_given_spec_in_itself_and_in_every_arm(world):
    with pytest.raises(SystemExit, match="not .*from the spec given"):
        world.run(spec="0" * 64)
    world.edit(lambda m: m["arms"].__setitem__("stale", {**m["arms"]["RL-full"], "spec_sha256": "0" * 64}))      # an arm left over from another spec
    with pytest.raises(SystemExit, match="not .*from the spec given"):
        world.run()
    world.edit(lambda m: m["arms"].pop("stale"))
    world.edit(lambda m: m.update(spec_sha256="0" * 64))
    with pytest.raises(SystemExit, match="not .*from the spec given"):
        world.run()
    (world.final / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest.json is missing"):
        world.run()
    assert not world.out.exists()


def test_the_selection_file_must_be_the_one_the_spec_was_frozen_against(world):
    world.sel.write_text(json.dumps({**SEL, "comparator": "C0"}))
    with pytest.raises(SystemExit, match="selection file is not the one"):
        world.run()
    assert not world.out.exists()


def test_a_sampled_arm_needs_a_manifest_entry_but_a_cpu_control_does_not(world):
    world.edit(lambda m: m["arms"].clear())
    with pytest.raises(SystemExit, match="lists no arms"):
        world.run()
    world.edit(lambda m: m["arms"].__setitem__("A0n", {"spec_sha256": SPEC}))                  # other arms are there, the pair's RL arm is not
    with pytest.raises(SystemExit, match="arm RL-full has no entry"):
        world.run()
    world.edit(lambda m: m["arms"].pop("A0n"))
    world.edit(lambda m: m["arms"].__setitem__("RL-full", {"spec_sha256": SPEC, **SETTINGS}))
    world.run()                                                                                  # C3 (a control) has none and is accepted
    assert world.out.is_file()


@pytest.mark.parametrize("field,value", [("model", "final_models/other"), ("mode", "think"), ("temperature", 1.0), ("max_tokens", 512), ("k", 4), ("data_sha256", "e" * 64)])
def test_the_runs_own_meta_json_must_repeat_the_frozen_sampler_settings(world, field, value):
    meta = world.final / "RL-full" / "meta.json"
    data = json.loads(meta.read_text())
    data["configuration"][field] = value
    meta.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match=rf"differs from the run manifest in \['{field}'\]"):
        world.run()
    assert not world.out.exists()


def test_the_data_hash_is_compared_only_when_the_manifest_recorded_one_and_meta_json_must_exist(world):
    world.edit(lambda m: m["arms"]["RL-full"].pop("data_sha256"))
    meta = world.final / "RL-full" / "meta.json"
    data = json.loads(meta.read_text())
    data["configuration"]["data_sha256"] = "e" * 64
    meta.write_text(json.dumps(data))
    world.run()
    meta.unlink()
    with pytest.raises(SystemExit, match="meta.json is missing"):
        world.run()


def test_a_manifest_entry_without_sampler_settings_is_refused(world):
    world.edit(lambda m: m["arms"]["RL-full"].pop("k"))
    with pytest.raises(SystemExit, match="lacks some of"):
        world.run()


def test_a_selection_that_does_not_say_whether_an_arm_is_sampled_is_refused(world):
    world.sel.write_text(json.dumps({"rl": "RL-full", "comparator": "C3"}))
    world.edit(lambda m: m.update(selection_sha256=sha(world.sel)))
    with pytest.raises(SystemExit, match="selection lacks"):
        world.run()


# ------------------------------------------------------------------------------------------------ a CPU control comes from scripts/final_controls.py
def edit_control(world, **kw):
    path = world.final / "C3" / "meta.json"
    meta = json.loads(path.read_text())
    meta.update(kw)
    path.write_text(json.dumps({k: v for k, v in meta.items() if v != "<drop>"}))


@pytest.mark.parametrize("change,message", [
    ({"control": "C0"}, r"differs from .* in \['control'\]"), ({"k": 4}, r"in \['k'\]"), ({"splits": ["test_t"]}, r"in \['splits'\]"),
    ({"splits": ["test_r", "test_t"]}, r"in \['splits'\]"), ({"rows": 100}, r"in \['rows'\]"), ({"rows": "<drop>"}, r"in \['rows'\]"),
    ({"source_sha256": "<drop>"}, "records no source_sha256"), ({"source_sha256": ""}, "records no source_sha256"),
    ({"pool_sha256": None}, "records no pool_sha256"), ({"pool_sha256": "<drop>"}, "records no pool_sha256"),
    ({"source_sha256": None, "pool_sha256": None}, "records no source_sha256, pool_sha256")])
def test_a_cpu_control_must_carry_the_meta_json_of_final_controls(world, change, message):
    edit_control(world, **change)
    with pytest.raises(SystemExit, match=message):
        world.run()
    assert not world.out.exists()


def test_a_cpu_control_without_a_meta_json_is_refused(world):
    (world.final / "C3" / "meta.json").unlink()
    with pytest.raises(SystemExit, match=r"C3/meta.json is missing: the CPU control C3 is made by scripts/final_controls.py"):
        world.run()
    assert not world.out.exists()


def test_only_c3_needs_a_pool_hash(world):
    (world.final / "C0").mkdir()
    (world.final / "C0" / "scored.v3.jsonl").write_text("{}\n")
    (world.final / "C0" / "meta.json").write_text(json.dumps(control_meta("C0")))                  # no pool_sha256: C0 is a Train reference
    world.sel.write_text(json.dumps({**SEL, "comparator": "C0", "arms": {**SEL["arms"], "C0": {"temperature": None}}}))
    world.edit(lambda m: m.update(selection_sha256=sha(world.sel)))
    world.run()
    assert json.loads(world.out.read_text())["comparator"] == "C0"
    (world.final / "C0" / "meta.json").write_text(json.dumps(control_meta("C0", source_sha256="")))
    with pytest.raises(SystemExit, match="records no source_sha256"):
        world.run(out=world.tmp / "again" / "confirmatory.json")
