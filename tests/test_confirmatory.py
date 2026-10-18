import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "confirmatory.py"
SPEC = "5" * 64
DATA_HASH = "d" * 64
ADAPTER = "a" * 64


@pytest.fixture(scope="module")
def cf():
    spec = importlib.util.spec_from_file_location("confirmatory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEL = {"rl": "RL-full", "comparator": "C3", "arms": {"RL-full": {"temperature": 0.6}, "C3": {"temperature": None}}}
SEL_ADAPTER = {"rl": "RL-full", "comparator": "C3", "arms": {"RL-full": {"temperature": 0.6, "adapter_sha256": ADAPTER, "score": 0.1887}, "C3": {"temperature": None}}}
SETTINGS = {"model": "final_models/RL-full", "mode": "nothink", "temperature": 0.6, "max_tokens": 256, "k": 8}
IDS = [f"drug{i}/{j}" for i in range(3) for j in range(2)]


def control_meta(control="C3", **kw):
    """The meta.json that scripts/final_controls.py writes for a CPU control (C3 also records the pool it came from)."""
    meta = {"control": control, "model": {"C3": "control_constant_pool", "C0": "control_constant"}[control], "k": 8, "splits": ["test_t", "test_r"], "rows": 180,
            "source": "runs/x/samples.jsonl", "source_sha256": "e" * 64}
    return {**meta, **({"pool_sha256": "f" * 64, "train_score": 0.19, "source_kind": "pool"} if control == "C3" else {}), **kw}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def sample_rows(model, mode, temperature, ids=IDS):
    return [{"id": i, "group": i.split("/")[0], "split": "test_t", "sample_idx": j, "text": "x", "model": model, "mode": mode, "temperature": temperature} for i in ids for j in range(8)]


def scored_rows(model, mode, temperature, ids=IDS):
    return [{**r, "reward": 0.1, "valid": True} for r in sample_rows(model, mode, temperature, ids)]


def test_commands_pair_the_chosen_arms_on_both_test_splits(cf):
    cmds = cf.commands(SEL, "runs/final", "data/processed", "results", boot=100, seed=3)
    assert set(cmds) == {"test_t", "test_r"}
    for split, cmd in cmds.items():
        assert cmd[cmd.index("--scored") + 1] == "runs/final/RL-full/scored.v3.jsonl" and cmd[cmd.index("--vs") + 1] == "runs/final/C3/scored.v3.jsonl"
        assert cmd[cmd.index("--split") + 1] == split and cmd[cmd.index("--boot") + 1] == "100" and cmd[cmd.index("--seed") + 1] == "3"
        assert cmd[cmd.index("--json") + 1] == f"results/paired_{split}.json"


def test_dry_run_prints_and_runs_nothing_and_needs_no_hash_manifest_or_tag(cf, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cf, "read_tag", lambda tag, cwd=None: pytest.fail("a dry run reads no tag"))
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
    """A selection (RL-full sampled, C3 a CPU control), the Test data files and their manifest, a final directory with the run manifest, meta.json, samples and scored files of
    both arms, a git tag that names the hashes, and a fake `rlexp.metrics report`."""
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SEL))
    data = tmp_path / "data"
    data.mkdir()
    for split in ("test_t", "test_r"):
        (data / f"{split}.jsonl").write_text(f"synthetic rows of {split}\n")
    manifest_file = tmp_path / "MANIFEST.sha256"
    manifest_file.write_text("".join(f"{sha(data / f'{s}.jsonl')}  processed/{s}.jsonl\n" for s in ("test_t", "test_r")))
    final = tmp_path / "final"
    for arm in ("RL-full", "C3"):
        (final / arm).mkdir(parents=True)
    write_rows(final / "RL-full" / "samples.jsonl", sample_rows("final_models/RL-full", "nothink", 0.6))
    write_rows(final / "RL-full" / "scored.v3.jsonl", scored_rows("final_models/RL-full", "nothink", 0.6))
    write_rows(final / "C3" / "samples.jsonl", sample_rows("control_constant_pool", "control", None))
    write_rows(final / "C3" / "scored.v3.jsonl", scored_rows("control_constant_pool", "control", None))
    (final / "RL-full" / "meta.json").write_text(json.dumps({"configuration": {**SETTINGS, "data_sha256": DATA_HASH, "top_p": 1.0, "seed": 0}}))
    (final / "C3" / "meta.json").write_text(json.dumps(control_meta()))
    entry = {"spec_sha256": SPEC, **SETTINGS, "k": 8, "splits": ["test_t", "test_r"], "data_sha256": DATA_HASH, "command": ["x"]}
    (final / "manifest.json").write_text(json.dumps({"spec_sha256": SPEC, "selection_sha256": sha(sel), "arms": {"RL-full": entry}}))
    reports = {"overrides": {}}
    tag = {"message": f"prereg-v1        prereg-v1 spec={SPEC} selection={sha(sel)} manifest={sha(manifest_file)}\n"}
    monkeypatch.setattr(cf, "read_tag", lambda name, cwd=None: tag["message"])

    def fake_run(cmd, capture_output, text):
        split = cmd[cmd.index("--split") + 1]
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps(reports["overrides"][split] if split in reports["overrides"] else make_report(split)))
        return type("R", (), {"returncode": 0, "stdout": "table", "stderr": ""})()

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    out = tmp_path / "res" / "confirmatory.json"

    def run(*extra, spec=None, out=out):
        argv = ["--selection", str(sel), "--manifest", str(manifest_file), "--final", str(final), "--data", str(data), "--out", str(out)]
        cf.main(argv + (["--spec-sha256", spec] if spec else []) + list(extra))

    def edit(mutate):
        data_ = json.loads((final / "manifest.json").read_text())
        mutate(data_)
        (final / "manifest.json").write_text(json.dumps(data_))

    def use_selection(selection):
        """Write another selection, with the manifest and the tag following it (as a freeze would)."""
        sel.write_text(json.dumps(selection))
        edit(lambda m: m.update(selection_sha256=sha(sel)))
        tag["message"] = f"prereg-v1  prereg-v1 spec={SPEC} selection={sha(sel)} manifest={sha(manifest_file)}\n"

    return type("World", (), {"run": staticmethod(run), "final": final, "sel": sel, "tmp": tmp_path, "edit": staticmethod(edit), "reports": reports, "out": out, "data": data,
                              "manifest": manifest_file, "tag": tag, "use_selection": staticmethod(use_selection)})


def test_missing_scored_files_stop_with_a_hint(cf, world):
    (world.final / "C3" / "scored.v3.jsonl").unlink()
    with pytest.raises(SystemExit):
        world.run()
    assert not world.out.exists()


def test_a_run_collects_both_pairs_the_input_hashes_and_the_provenance(world):
    world.run()
    res = json.loads(world.out.read_text())
    assert res["paired"]["test_t"]["splits"]["test_t"]["paired"]["n_drugs"] == 41 and res["paired"]["test_r"]["splits"]["test_r"]["paired"]["n_drugs"] == 30
    assert len(res["input_sha256"]) == 2 and (world.out.parent / "paired_test_t.txt").read_text() == "table"
    assert res["spec_sha256"] == SPEC and res["selection_sha256"] == sha(world.sel) and res["run_manifest_sha256"] == sha(world.final / "manifest.json")
    assert res["input_sha256"][str(world.final / "RL-full" / "scored.v3.jsonl")] == sha(world.final / "RL-full" / "scored.v3.jsonl")
    assert res["data_manifest_sha256"] == sha(world.manifest) and res["test_files_sha256"] == {s: sha(world.data / f"{s}.jsonl") for s in ("test_t", "test_r")}
    assert res["tag"] == {"name": "prereg-v1", "spec": SPEC, "selection": sha(world.sel), "manifest": sha(world.manifest)} and "git tag prereg-v1" in res["spec_sha256_source"]
    assert res["sanity"] == {} and res["sanity_skipped_and_allowed"] == []


def test_the_output_directory_is_created_before_anything_is_written_into_it(world):
    out = world.tmp / "deep" / "er" / "confirmatory.json"
    world.run(out=out)
    assert out.is_file() and (out.parent / "paired_test_t.json").is_file()


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
        world.tag["message"] = None
        world.run(spec="0" * 64)
    world.tag["message"] = f"prereg-v1 spec={SPEC} selection={sha(world.sel)} manifest={sha(world.manifest)}"
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
    world.tag["message"] = f"prereg-v1 spec={SPEC}"
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
    world.use_selection({"rl": "RL-full", "comparator": "C3"})
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
    ({"source_sha256": None, "pool_sha256": None}, "records no source_sha256, pool_sha256"),
    ({"model": "<drop>"}, "records no control_"), ({"model": "final_models/RL-full"}, "records no control_"), ({"model": 7}, "records no control_")])
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
    write_rows(world.final / "C0" / "samples.jsonl", sample_rows("control_constant", "control", None))
    write_rows(world.final / "C0" / "scored.v3.jsonl", scored_rows("control_constant", "control", None))
    (world.final / "C0" / "meta.json").write_text(json.dumps(control_meta("C0")))                  # no pool_sha256: C0 is a Train reference
    world.use_selection({**SEL, "comparator": "C0", "arms": {**SEL["arms"], "C0": {"temperature": None}}})
    world.run()
    assert json.loads(world.out.read_text())["comparator"] == "C0"
    (world.final / "C0" / "meta.json").write_text(json.dumps(control_meta("C0", source_sha256="")))
    with pytest.raises(SystemExit, match="records no source_sha256"):
        world.run(out=world.tmp / "again" / "confirmatory.json")


# ------------------------------------------------------------------------------------------------ a scored file is its arm's own
def test_a_scored_file_of_another_model_in_an_arms_directory_is_refused(world):
    write_rows(world.final / "RL-full" / "scored.v3.jsonl", scored_rows("final_models/A2", "think", 1.0))         # A2's scored file copied into RL-full/
    with pytest.raises(SystemExit, match=r"arm RL-full: its scored rows carry model \['final_models/A2'\], not 'final_models/RL-full'"):
        world.run()
    assert not world.out.exists()


@pytest.mark.parametrize("field,value,shown", [("mode", "think", r"mode \['nothink', 'think'\], not 'nothink'"), ("temperature", 1.0, r"temperature \[0.6, 1.0\], not 0.6"),
                                               ("model", "final_models/RL-full-v2", "model")])
def test_every_scored_row_must_carry_the_manifest_entrys_model_mode_and_temperature(world, field, value, shown):
    rows = scored_rows("final_models/RL-full", "nothink", 0.6)
    rows[3][field] = value                                                  # one row only
    write_rows(world.final / "RL-full" / "scored.v3.jsonl", rows)
    with pytest.raises(SystemExit, match=shown):
        world.run()


def test_a_scored_file_of_a_model_run_in_the_control_directory_is_refused(world):
    write_rows(world.final / "C3" / "scored.v3.jsonl", scored_rows("final_models/A2", "think", 1.0))
    with pytest.raises(SystemExit, match=r"arm C3: its scored rows carry model \['final_models/A2'\], not 'control_constant_pool'"):
        world.run()
    write_rows(world.final / "C3" / "scored.v3.jsonl", scored_rows("control_constant", "control", None))        # another control's rows
    with pytest.raises(SystemExit, match=r"model \['control_constant'\], not 'control_constant_pool'"):
        world.run()


def test_the_scored_pairs_must_be_the_pairs_of_the_samples_each_once(world):
    rows = scored_rows("final_models/RL-full", "nothink", 0.6)
    for name, changed, message in (("a row missing", rows[:-1], "0 only in the scored file, 1 only in the samples"),
                                   ("a row of another prompt", [*rows[:-1], {**rows[-1], "id": "other/0"}], "1 only in the scored file, 1 only in the samples"),
                                   ("a row twice", [*rows[:-1], rows[0]], "1 only in the scored file, 1 only in the samples"),
                                   ("a row added a second time", [*rows, rows[0]], "1 only in the scored file, 0 only in the samples"),
                                   ("an extra sample", [*rows, {**rows[0], "sample_idx": 9}], "1 only in the scored file, 0 only in the samples")):
        write_rows(world.final / "RL-full" / "scored.v3.jsonl", changed)
        with pytest.raises(SystemExit, match=message):
            world.run()
    write_rows(world.final / "RL-full" / "scored.v3.jsonl", rows)
    (world.final / "RL-full" / "samples.jsonl").unlink()
    with pytest.raises(SystemExit, match="samples.jsonl is missing"):
        world.run()
    assert not world.out.exists()


def test_a_control_scored_file_is_tied_to_its_own_samples_too(world):
    write_rows(world.final / "C3" / "samples.jsonl", sample_rows("control_constant_pool", "control", None, ids=IDS[:-1]))
    with pytest.raises(SystemExit, match=r"arm C3: scored.v3.jsonl and samples.jsonl do not hold the same"):
        world.run()


def test_an_error_about_a_scored_file_names_counts_not_ids(world):
    write_rows(world.final / "RL-full" / "scored.v3.jsonl", scored_rows("final_models/RL-full", "nothink", 0.6)[:-1])
    with pytest.raises(SystemExit) as e:
        world.run()
    assert "drug" not in str(e.value)                                                       # row ids carry the drug name


# ------------------------------------------------------------------------------------------------ the Test files read through --data are the frozen ones
def test_the_test_files_must_be_the_ones_the_manifest_lists(world):
    (world.data / "test_r.jsonl").write_text("another file\n")
    with pytest.raises(SystemExit, match=r"test_r.jsonl differs from its sha256 in"):
        world.run()
    (world.data / "test_r.jsonl").unlink()
    with pytest.raises(SystemExit, match="test_r.jsonl is missing"):
        world.run()
    (world.data / "test_r.jsonl").write_text("synthetic rows of test_r\n")
    world.manifest.write_text("".join(line + "\n" for line in world.manifest.read_text().splitlines() if "test_t" not in line))
    world.tag["message"] = f"prereg-v1 spec={SPEC}"
    with pytest.raises(SystemExit, match="processed/test_t.jsonl has no entry"):
        world.run()
    world.manifest.write_text("garbage\n")
    with pytest.raises(SystemExit, match="not a sha256 manifest"):
        world.run()
    assert not world.out.exists()


def test_the_manifest_is_required_for_a_real_run(cf, world):
    with pytest.raises(SystemExit) as e:
        cf.main(["--selection", str(world.sel), "--final", str(world.final), "--data", str(world.data), "--out", str(world.out)])
    assert e.value.code == 2


# ------------------------------------------------------------------------------------------------ the spec hash comes from the tag
def test_without_an_argument_the_spec_hash_is_the_tags_and_a_missing_tag_is_refused(world):
    world.run()
    assert json.loads(world.out.read_text())["spec_sha256"] == SPEC
    world.out.unlink()
    world.tag["message"] = None
    with pytest.raises(SystemExit, match="no --spec-sha256 and no git tag prereg-v1"):
        world.run()
    assert not world.out.exists()


def test_an_argument_must_agree_with_the_tag_and_stands_alone_only_without_one(world):
    world.run(spec=SPEC)
    assert "agrees with --spec-sha256" in json.loads(world.out.read_text())["spec_sha256_source"]
    with pytest.raises(SystemExit, match="--spec-sha256 differs from the spec= of git tag prereg-v1"):
        world.run(spec="0" * 64)
    world.tag["message"] = None
    world.run(spec=SPEC)                                                    # a rehearsal before the tag exists
    assert json.loads(world.out.read_text())["spec_sha256_source"] == "--spec-sha256 (no git tag)"


@pytest.mark.parametrize("message", ["prereg-v1  Freeze 2026-10-18", f"prereg-v1  spec={SPEC[:63]}", f"prereg-v1  spec={('ab' * 32).upper()}", f"prereg-v1  myspec={SPEC}", f"prereg-v1  spec = {SPEC}"])
def test_a_tag_without_a_usable_spec_hash_is_refused_even_with_an_argument(world, message):
    world.tag["message"] = message
    for argument in (None, SPEC):
        with pytest.raises(SystemExit, match="has no spec=<sha256> in its message"):
            world.run(spec=argument)


def test_the_tags_selection_and_manifest_hashes_must_be_those_of_the_files_given(world):
    world.tag["message"] = f"prereg-v1  spec={SPEC} selection={'0' * 64} manifest={sha(world.manifest)}"
    with pytest.raises(SystemExit, match="names another selection="):
        world.run()
    world.tag["message"] = f"prereg-v1  spec={SPEC} selection={sha(world.sel)} manifest={'0' * 64}"
    with pytest.raises(SystemExit, match="names another manifest="):
        world.run()
    world.tag["message"] = f"prereg-v1  spec={SPEC}"                         # a tag that names only the spec checks only the spec
    world.run()


def test_tag_hashes_reads_the_first_token_of_each_key(cf):
    a, b, c = "1" * 64, "ab" * 32, "3" * 64
    assert cf.tag_hashes(f"prereg-v1        prereg-v1 spec={a} selection={b} manifest={c}\n    second line spec={c}") == {"spec": a, "selection": b, "manifest": c}
    assert cf.tag_hashes(f"x spec={a}") == {"spec": a} and cf.tag_hashes("nothing here") == {}
    assert cf.tag_hashes(f"spec={a[:-1]} selection={b.upper()} nospec={c}") == {}


def test_read_tag_asks_git_for_the_message_of_a_real_tag(cf, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *a], capture_output=True, text=True, check=True)
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "x")
    assert cf.read_tag("prereg-v1", cwd=repo) is None                          # no such tag
    git("tag", "-a", "prereg-v1", "-m", f"prereg-v1 spec={SPEC} selection={'1' * 64} manifest={'2' * 64}")
    assert cf.tag_hashes(cf.read_tag("prereg-v1", cwd=repo)) == {"spec": SPEC, "selection": "1" * 64, "manifest": "2" * 64}
    assert cf.read_tag("prereg-v1", cwd=tmp_path / "not-a-repo") is None


# ------------------------------------------------------------------------------------------------ the Val reproduction of an adapter arm
def sanity_record(**kw):
    return {"value": 0.1884, "selection_score": 0.1887, "tolerance": 0.005, "passed": True, "seed": 0, **kw}


def with_adapter_arm(world, record):
    world.use_selection(SEL_ADAPTER)
    if record is not None:
        (world.final / "sanity.json").write_text(json.dumps({"RL-full": record}))


def test_an_adapter_arm_needs_a_passed_val_reproduction(world):
    with_adapter_arm(world, None)
    with pytest.raises(SystemExit, match="sanity.json is missing: gpu/run_final.py writes"):
        world.run()
    (world.final / "sanity.json").write_text(json.dumps({"A2": sanity_record()}))
    with pytest.raises(SystemExit, match="has no Val reproduction for the adapter arm RL-full"):
        world.run()
    (world.final / "sanity.json").write_text(json.dumps({"RL-full": sanity_record()}))
    world.run()
    res = json.loads(world.out.read_text())
    assert res["sanity"] == {"RL-full": sanity_record()} and res["sanity_skipped_and_allowed"] == []


@pytest.mark.parametrize("change", [{"passed": False}, {"passed": "true"}, {"passed": None}, {"value": 0.17}, {"tolerance": 0.0001}, {"selection_score": 0.19}, {"value": None},
                                    {"tolerance": "<drop>"}, {"selection_score": "<drop>"}, {"value": "<drop>"}])
def test_a_failed_or_inconsistent_val_reproduction_is_refused(world, change):
    record = {k: v for k, v in sanity_record(**change).items() if v != "<drop>"}
    with_adapter_arm(world, record)
    with pytest.raises(SystemExit, match="the Val reproduction of RL-full did not pass"):
        world.run()
    assert not world.out.exists()


def test_a_skipped_val_reproduction_is_refused_unless_allowed_and_then_it_is_printed_and_recorded(world, capsys):
    with_adapter_arm(world, {"skipped": True})
    with pytest.raises(SystemExit, match="was skipped .*--allow-skipped-sanity"):
        world.run()
    world.run("--allow-skipped-sanity")
    assert "WARNING: the Val reproduction of RL-full was skipped; accepted by --allow-skipped-sanity" in capsys.readouterr().out
    res = json.loads(world.out.read_text())
    assert res["sanity_skipped_and_allowed"] == ["RL-full"] and res["sanity"] == {"RL-full": {"skipped": True}}
    (world.final / "sanity.json").write_text(json.dumps({"RL-full": sanity_record(passed=False)}))        # allowing skips does not allow failures
    with pytest.raises(SystemExit, match="did not pass"):
        world.run("--allow-skipped-sanity")


def test_an_untrained_pair_needs_no_sanity_file(world):
    world.run()
    assert not (world.final / "sanity.json").exists() and world.out.is_file()
