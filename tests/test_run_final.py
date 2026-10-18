import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_final", ROOT / "gpu" / "run_final.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["run_final"] = rf
spec.loader.exec_module(rf)

TOL = rf.SANITY_TOLERANCE


def arms(*ids, **extra):
    return [{"id": i, **extra.get(i, {})} for i in ids]


def files(tmp_path, spec_arms, selection=None):
    (tmp_path / "spec.json").write_text(json.dumps({"arms": spec_arms}))
    (tmp_path / "spec.sha256").write_text("ab" * 32 + "  results/frozen_arms.json\n")
    (tmp_path / "sel.json").write_text(json.dumps(selection or {"arms": {}}))
    return ["--spec", str(tmp_path / "spec.json"), "--spec-sha256-file", str(tmp_path / "spec.sha256"), "--selection", str(tmp_path / "sel.json"), "--dry-run"]


def test_every_arm_of_the_spec_is_in_a_stage_and_the_order_is_fixed():
    ids = ["RL-full", "A2", "A0n", "A4", "9B", "27B-3.5", "27B-3.8", "name-9B", "ident-27B"]
    stages = rf.plan({"arms": arms(*reversed(ids))})
    assert [name for name, _, _ in stages] == ["4B arms", "9B", "Qwen3.5-27B", "Qwen3.8-27B"]
    assert stages[0][1] == ["RL-full", "A2", "A0n", "A4"]                  # the confirmatory pair first, whatever the spec's order
    assert [clear for _, _, clear in stages] == [False, True, True, True]


def test_an_arm_in_no_stage_stops_before_anything_runs():
    with pytest.raises(ValueError, match="surprise"):
        rf.plan({"arms": arms("A2", "surprise")})


def test_a_stage_without_present_arms_is_dropped():
    assert [n for n, _, _ in rf.plan({"arms": arms("A2", "9B")})] == ["4B arms", "9B"]


ADAPTER = {"adapter": {"path": "x"}, "model": "final_models/A2", "mode": "think", "temperature": 1.0, "max_tokens": 256}
RL_ADAPTER = {**ADAPTER, "model": "final_models/RL-full"}
SELECTION = {"arms": {"A2": {"score": 0.1557}, "RL-full": {"score": 0.1949}}}


def test_dry_run_prints_prepare_sanity_and_stage_commands_without_running(tmp_path, capsys):
    rf.main(files(tmp_path, arms("A2", "9B", "27B-3.5", **{"A2": ADAPTER}), SELECTION))
    out = capsys.readouterr().out
    assert "gpu/prepare_final.py" in out and "--only A2" in out
    assert "sanity" in out and "--splits val" in out and "gpu/final_eval.py" in out
    assert out.index("== stage 4B arms") < out.index("== stage 9B") < out.index("== stage Qwen3.5-27B")
    assert out.rstrip().endswith("locked evaluation sampling finished")


def test_from_stage_skips_earlier_stages_and_rejects_an_unknown_one(tmp_path, capsys):
    argv = files(tmp_path, arms("A2", "9B", "27B-3.5"))
    rf.main([*argv, "--from-stage", "Qwen3.5-27B"])
    out = capsys.readouterr().out
    assert "== stage Qwen3.5-27B" in out and "== stage 9B" not in out and "== stage 4B arms" not in out
    with pytest.raises(SystemExit, match="not a stage"):
        rf.main([*argv, "--from-stage", "nope"])


# ------------------------------------------------------------------------------------------------ the Val sanity gate
def fake_environment(tmp_path, monkeypatch, value=0.1975):
    """A Val sanity run that 'samples' into tmp_path/sanity and reports `value`; rf.run records every command."""
    stale = tmp_path / "sanity" / "A2"
    monkeypatch.setattr(rf, "SANITY", tmp_path / "sanity")
    monkeypatch.setattr(rf, "now", lambda: "2026-10-18T01:00:00+00:00")
    seen = {"commands": [], "stale_present_at_sampling": []}

    def fake_run(cmd, dry, **kw):
        if dry:
            return 0                                                  # like rf.run: a dry run executes nothing
        seen["commands"].append([str(c) for c in cmd])
        text = " ".join(map(str, cmd))
        if "sample_vllm.py" in text:
            seen["stale_present_at_sampling"].append(stale.exists())
            stale.mkdir(parents=True, exist_ok=True)
            (stale / "samples.jsonl").write_text("")
        elif "report" in cmd:
            (stale / "report.v3.json").write_text(json.dumps({"splits": {"val": {"metrics": {"fv_tanimoto": {"value": value}}}}}))
        return 0

    monkeypatch.setattr(rf, "run", fake_run)
    return stale, seen


ARM = {"id": "A2", "model": "m", "mode": "think", "temperature": 1.0, "max_tokens": 256}


def test_the_tolerance_is_final_evals_one_constant_and_there_is_no_flag_for_it():
    spec_ = importlib.util.spec_from_file_location("final_eval_for_run_final_test", ROOT / "gpu" / "final_eval.py")
    fe = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(fe)
    assert TOL == fe.SANITY_TOLERANCE == 0.005
    assert "SANITY_TOLERANCE =" not in (ROOT / "gpu" / "run_final.py").read_text() and "--tolerance" not in (ROOT / "gpu" / "run_final.py").read_text().split('ap.add_argument("--drive")')[1]


def test_a_tolerance_flag_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        rf.main([*files(tmp_path, arms("9B")), "--tolerance", "0.05"])
    assert e.value.code == 2


def test_the_attempt_holds_the_arm_the_value_the_score_the_verdict_and_the_time_and_judges_by_the_one_tolerance(tmp_path, monkeypatch):
    fake_environment(tmp_path, monkeypatch)                                # the Val sample reproduces 0.1975
    sel = lambda score: {"arms": {"A2": {"score": score}}}
    assert rf.sanity(ARM, sel(0.195), False) == {"arm": "A2", "value": 0.1975, "selection_score": 0.195, "passed": True, "time": "2026-10-18T01:00:00+00:00"}
    assert rf.sanity(ARM, sel(0.15), False)["passed"] is False                                                           # 0.0475 apart
    assert rf.sanity(ARM, sel(0.1925), False)["passed"] is False                                                         # 0.0050000000000000044 apart: over the line, with float noise
    assert rf.sanity(ARM, sel(0.1950), False)["passed"] is True and rf.sanity(ARM, sel(0.1999), False)["passed"] is True      # 0.0025 and 0.0024 apart
    assert rf.sanity(ARM, sel(0.190), False)["passed"] is False                                                          # 0.0075 apart: the old 0.01 would have let it through


def test_a_stale_sanity_directory_is_deleted_before_the_sample_but_not_in_a_dry_run(tmp_path, monkeypatch):
    stale, seen = fake_environment(tmp_path, monkeypatch)
    stale.mkdir(parents=True)
    (stale / "samples.jsonl").write_text("samples of an older merge")
    assert rf.sanity(ARM, {"arms": {"A2": {"score": 0.2}}}, True) is None
    assert (stale / "samples.jsonl").read_text() == "samples of an older merge" and seen["stale_present_at_sampling"] == []      # a dry run deletes and samples nothing real
    rf.sanity(ARM, {"arms": {"A2": {"score": 0.2}}}, False)
    assert seen["stale_present_at_sampling"] == [False]                                                                         # gone when the sampler starts


def main_argv(tmp_path, spec_arms, selection, *extra):
    argv = files(tmp_path, spec_arms, selection)
    return [a for a in argv if a != "--dry-run"] + ["--out", str(tmp_path / "out"), *extra]


def passing(score, value):
    return lambda arm, selection, dry: {"arm": arm["id"], "value": value, "selection_score": score, "passed": abs(value - score) <= TOL, "time": "t"}


def test_every_attempt_is_appended_a_failure_stays_on_record_and_the_stop_comes_after_it(tmp_path, monkeypatch):
    results = {"A2": passing(0.1557, 0.1560), "RL-full": passing(0.1949, 0.2100)}
    monkeypatch.setattr(rf, "sanity", lambda arm, selection, dry: results[arm["id"]](arm, selection, dry))
    commands = []
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    spec_arms = arms("A2", "RL-full", **{"A2": ADAPTER, "RL-full": RL_ADAPTER})
    with pytest.raises(SystemExit, match="RL-full reproduces 0.2100, the selection says 0.1949"):
        rf.main(main_argv(tmp_path, spec_arms, SELECTION))
    written = json.loads((tmp_path / "out" / "sanity.json").read_text())
    assert written["tolerance"] == TOL and [(a["arm"], a["passed"]) for a in written["attempts"]] == [("A2", True), ("RL-full", False)]
    assert not any("final_eval.py" in " ".join(map(str, c)) for c in commands)                  # no Test sampling after a failed gate
    results["RL-full"] = passing(0.1949, 0.1951)                                                  # the same command again: a second attempt that passes
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    with pytest.raises(SystemExit, match="not finished"):                                         # (the fake sampler wrote no manifest)
        rf.main(main_argv(tmp_path, spec_arms, SELECTION))
    again = json.loads((tmp_path / "out" / "sanity.json").read_text())
    arms_in_order = [(a["arm"], a["passed"]) for a in again["attempts"]]
    assert arms_in_order[-2:] == [("A2", True), ("RL-full", True)] and ("RL-full", False) in arms_in_order                      # the failed attempt is still there
    assert again["attempts"][:len(written["attempts"])] == written["attempts"]                    # append-only: the earlier attempts are untouched
    assert any("final_eval.py" in " ".join(map(str, c)) for c in commands)


def test_the_gate_refuses_a_sanity_file_made_with_another_tolerance_and_leaves_it_alone(tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    old = json.dumps({"tolerance": 0.05, "attempts": [{"arm": "A2", "value": 0.0, "selection_score": 0.1, "passed": True, "time": "t"}]})
    (tmp_path / "out" / "sanity.json").write_text(old)
    monkeypatch.setattr(rf, "sanity", lambda arm, selection, dry: passing(0.15, 0.15)(arm, selection, dry))
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)
    with pytest.raises(SystemExit, match="made with tolerance 0.05, the gate's is 0.005"):
        rf.main(main_argv(tmp_path, arms("A2", **{"A2": ADAPTER}), SELECTION))
    assert (tmp_path / "out" / "sanity.json").read_text() == old


def test_skip_sanity_keeps_the_record_and_needs_a_passed_latest_attempt_of_every_arm_still_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "sanity", lambda *a: pytest.fail("the gate must not run"))
    commands = []
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    spec_arms = arms("A2", "RL-full", "9B", **{"A2": ADAPTER, "RL-full": RL_ADAPTER})
    argv = main_argv(tmp_path, spec_arms, SELECTION, "--skip-sanity")
    with pytest.raises(SystemExit, match=r"needs a passed Val check of \['A2', 'RL-full'\] in .*sanity.json already"):                  # no file at all
        rf.main(argv)
    assert not (tmp_path / "out" / "sanity.json").exists() and not commands[1:]                  # nothing written; at most the prepare step was planned
    (tmp_path / "out").mkdir()
    good = {"arm": "A2", "value": 0.1557, "selection_score": 0.1557, "passed": True, "time": "t"}
    only_a2 = json.dumps({"tolerance": TOL, "attempts": [good]})
    (tmp_path / "out" / "sanity.json").write_text(only_a2)
    with pytest.raises(SystemExit, match=r"Val check of \['RL-full'\]"):                          # one arm has no attempt
        rf.main(argv)
    failed_last = json.dumps({"tolerance": TOL, "attempts": [good, {**good, "arm": "RL-full", "value": 0.3, "passed": True}, {**good, "arm": "RL-full", "value": 0.3, "passed": False}]})
    (tmp_path / "out" / "sanity.json").write_text(failed_last)
    with pytest.raises(SystemExit, match=r"Val check of \['RL-full'\]"):                          # the latest attempt decides, and it failed
        rf.main(argv)
    complete = json.dumps({"tolerance": TOL, "attempts": [{**good, "arm": "RL-full", "passed": False}, good, {**good, "arm": "RL-full"}]}, indent=1)
    (tmp_path / "out" / "sanity.json").write_text(complete)
    with pytest.raises(SystemExit, match="not finished"):                                         # accepted: the gate was skipped, sampling went on, the fake wrote no manifest
        rf.main(argv)
    assert (tmp_path / "out" / "sanity.json").read_text() == complete                            # kept byte for byte, never {"skipped": true}
    assert any("final_eval.py" in " ".join(map(str, c)) for c in commands)


def test_skip_sanity_in_a_dry_run_only_plans_and_nothing_ever_writes_a_skipped_record(tmp_path):
    rf.main([*files(tmp_path, arms("A2", "9B", **{"A2": ADAPTER}), SELECTION), "--out", str(tmp_path / "out"), "--skip-sanity"])
    assert not (tmp_path / "out").exists()
    assert '"skipped"' not in (ROOT / "gpu" / "run_final.py").read_text()                          # the old {"skipped": true} record is gone for good


def test_a_dry_run_writes_no_sanity_file(tmp_path):
    rf.main([*files(tmp_path, arms("A2", "9B", **{"A2": ADAPTER}), SELECTION), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_a_spec_without_adapter_arms_has_no_gate_and_no_sanity_file(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    with pytest.raises(SystemExit, match="not finished"):
        rf.main(main_argv(tmp_path, arms("9B", "27B-3.5"), {"arms": {}}))
    assert not (tmp_path / "out" / "sanity.json").exists() and not any("prepare_final" in " ".join(map(str, c)) for c in commands)


def test_append_sanity_keeps_every_attempt_and_the_tolerance(tmp_path):
    for value in (1, 2, 3):
        rf.append_sanity(tmp_path / "o", {"arm": "A2", "value": value, "selection_score": 0.1, "passed": value == 3, "time": "t"})
    rf.append_sanity(tmp_path / "o", {"arm": "RL-full", "value": 0, "selection_score": 0.1, "passed": False, "time": "t"})
    data = json.loads((tmp_path / "o" / "sanity.json").read_text())
    assert data["tolerance"] == TOL and [(a["arm"], a["value"]) for a in data["attempts"]] == [("A2", 1), ("A2", 2), ("A2", 3), ("RL-full", 0)]
    assert not list((tmp_path / "o").glob("*.tmp")) and rf.last_attempt(data, "A2")["value"] == 3 and rf.last_attempt(data, "none") is None


# ------------------------------------------------------------------------------------------------ which stages need the export and the gate
def test_prepare_and_the_gate_run_only_when_a_stage_still_to_run_holds_an_adapter_arm(tmp_path, capsys):
    argv = files(tmp_path, arms("A2", "A2-shuffle", "9B", "27B-3.5", **{"A2": ADAPTER, "A2-shuffle": {**ADAPTER, "role": "diagnostic", "of": "A2"}}), SELECTION)
    rf.main(argv)
    assert "gpu/prepare_final.py" in capsys.readouterr().out
    for stage in ("9B", "Qwen3.5-27B"):
        rf.main([*argv, "--from-stage", stage])
        out = capsys.readouterr().out
        assert "prepare_final" not in out and "--splits val" not in out and f"== stage {stage}" in out and "== stage 4B arms" not in out
    rf.main([*argv, "--from-stage", "4B arms"])
    assert "gpu/prepare_final.py" in capsys.readouterr().out


def test_a_diagnostic_in_a_pending_stage_makes_its_base_arms_export_and_gate_necessary(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rf, "STAGES", [("first", ["A2"], False), ("second", ["A2-shuffle", "9B"], True)])        # the base arm is done in an earlier stage, its diagnostic is still to run
    spec_arms = arms("A2", "A2-shuffle", "9B", **{"A2": ADAPTER, "A2-shuffle": {**ADAPTER, "role": "diagnostic", "of": "A2"}})
    rf.main([*files(tmp_path, spec_arms, SELECTION), "--from-stage", "second"])
    out = capsys.readouterr().out
    assert "--only A2" in out and "--splits val" in out and "== stage first" not in out                          # the diagnostic samples A2's export, so A2 is prepared and gated
    monkeypatch.setattr(rf, "STAGES", [("first", ["A2", "A2-shuffle"], False), ("second", ["9B"], True)])
    rf.main([*files(tmp_path, spec_arms, SELECTION), "--from-stage", "second"])
    assert "prepare_final" not in capsys.readouterr().out                                                          # nothing adapter-bound is left to run


# ------------------------------------------------------------------------------------------------ Drive
def drive_world(tmp_path, monkeypatch):
    """A fake rsync (rf.run records it and can be told to fail), a Drive folder and the arm-free spec of a 9B-only plan."""
    drive = tmp_path / "drive"
    drive.mkdir()
    commands, fail = [], {"rsync": 0}

    def fake_run(cmd, dry, **kw):
        commands.append([str(c) for c in cmd])
        text = " ".join(map(str, cmd))
        if dry:
            return 0
        if cmd[0] == "rsync":
            return fail["rsync"]
        if "final_eval.py" in text:                                   # a stage "samples": its arms reach the manifest
            out = tmp_path / "out"
            out.mkdir(exist_ok=True)
            (out / "manifest.json").write_text(json.dumps({"arms": {i: {} for i in "9B,name-9B,ident-9B".split(",")}}))
        return 0

    monkeypatch.setattr(rf, "run", fake_run)
    argv = [a for a in files(tmp_path, arms("9B", "name-9B", "ident-9B")) if a != "--dry-run"] + ["--out", str(tmp_path / "out"), "--drive", str(drive)]
    return drive, commands, fail, argv


def test_a_drive_path_that_does_not_exist_is_refused_before_anything_runs_even_in_a_dry_run(tmp_path, monkeypatch):
    drive, commands, fail, argv = drive_world(tmp_path, monkeypatch)
    argv[argv.index("--drive") + 1] = str(tmp_path / "no-such-drive")
    for flags in ((), ("--dry-run",)):
        with pytest.raises(SystemExit, match="--drive .* does not exist: mount Drive first"):
            rf.main([*argv, *flags])
    assert not commands


def test_a_fresh_runtime_gets_the_earlier_run_back_from_drive_before_anything_is_sampled(tmp_path, monkeypatch, capsys):
    drive, commands, fail, argv = drive_world(tmp_path, monkeypatch)
    (drive / "final" / "9B").mkdir(parents=True)
    (drive / "final" / "9B" / "samples.jsonl").write_text("the longer partial file")
    rf.main(argv)
    restore = [c for c in commands if c[0] == "rsync"][0]
    assert restore == ["rsync", "-a", "--ignore-existing", f"{drive}/final/", f"{tmp_path / 'out'}/"] and commands.index(restore) < next(i for i, c in enumerate(commands) if any("final_eval.py" in x for x in c))
    assert "restoring it from Drive before anything else" in capsys.readouterr().out
    # the same state once <out> has an arm directory: no restore, only the copies to Drive
    (tmp_path / "out" / "9B").mkdir()
    commands.clear()
    rf.main(argv)
    assert all(c[:3] == ["rsync", "-a", f"{tmp_path / 'out'}/"] for c in commands if c[0] == "rsync")


def test_nothing_is_restored_when_drive_holds_nothing_or_out_already_holds_arms(tmp_path, monkeypatch):
    drive, commands, fail, argv = drive_world(tmp_path, monkeypatch)
    rf.main(argv)                                                                                    # empty Drive: no restore
    assert not any(c[0] == "rsync" and c[-2].startswith(str(drive)) for c in commands)
    (drive / "final").mkdir(exist_ok=True)
    (drive / "final" / "stale.txt").write_text("x")
    (tmp_path / "out" / "9B").mkdir(parents=True, exist_ok=True)                                     # out holds an arm
    commands.clear()
    rf.main(argv)
    assert not any(c[0] == "rsync" and c[-2].startswith(str(drive)) for c in commands)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_the_real_restore_copies_the_arms_back_but_never_overwrites_a_file_that_is_already_in_out(tmp_path):
    drive, out = tmp_path / "drive", tmp_path / "out"
    (drive / "final" / "9B").mkdir(parents=True)
    (drive / "final" / "9B" / "samples.jsonl").write_text("partial samples")
    (drive / "final" / "sanity.json").write_text("the attempts Drive knew about")
    out.mkdir()
    (out / "sanity.json").write_text("one attempt more, written since the last copy to Drive")
    assert rf.restore_from_drive(out, drive, False) is True
    assert (out / "9B" / "samples.jsonl").read_text() == "partial samples"
    assert (out / "sanity.json").read_text() == "one attempt more, written since the last copy to Drive"


def test_a_restore_that_fails_stops_everything(tmp_path, monkeypatch):
    drive, commands, fail, argv = drive_world(tmp_path, monkeypatch)
    (drive / "final").mkdir()
    (drive / "final" / "9B").mkdir()
    fail["rsync"] = 23
    with pytest.raises(SystemExit, match=r"restoring .* failed \(rsync code 23\); nothing was sampled"):
        rf.main(argv)
    assert not any("final_eval.py" in " ".join(c) for c in commands)


def test_a_failed_copy_to_drive_is_reported_and_the_last_one_aborts(tmp_path, monkeypatch, capsys):
    drive, commands, fail, argv = drive_world(tmp_path, monkeypatch)
    fail["rsync"] = 12
    with pytest.raises(SystemExit, match=r"the copy of .* to .*final failed \(rsync code 12\): copy it by hand before the runtime ends"):
        rf.main(argv)
    err = capsys.readouterr().err
    assert "WARNING: the copy of" in err and "(rsync code 12)" in err                              # the per-stage copy warned and went on
    assert sum(c[0] == "rsync" for c in commands) == 2                                                # one after the stage, one at the end


def test_a_missing_rsync_is_a_failed_copy_not_a_crash(monkeypatch, capsys):
    def no_rsync(cmd, dry, **kw):
        raise FileNotFoundError("rsync")

    monkeypatch.setattr(rf, "run", no_rsync)
    rf.sync("out", "/drive", False)
    assert "WARNING: the copy of out to /drive/final failed (rsync code FileNotFoundError)" in capsys.readouterr().err
    with pytest.raises(SystemExit, match=r"failed \(rsync code FileNotFoundError\): copy it by hand before the runtime ends"):
        rf.sync("out", "/drive", False, last=True)


def test_the_periodic_copy_reports_a_failure_and_survives_it(tmp_path, monkeypatch, capsys):
    results = iter([type("R", (), {"returncode": 0})(), type("R", (), {"returncode": 11})(), FileNotFoundError("rsync")])
    stop = rf.threading.Event()

    def fake_subprocess_run(cmd, **kw):
        result = next(results)
        if isinstance(result, Exception):
            stop.set()
            raise result
        return result

    monkeypatch.setattr(rf.subprocess, "run", fake_subprocess_run)
    rf.sync_forever("runs/final", "/drive", 0.001, stop)
    err = capsys.readouterr().err
    assert "periodic copy of runs/final to /drive/final failed (rsync code 11)" in err and "(rsync code FileNotFoundError)" in err and err.count("WARNING") == 2


# ------------------------------------------------------------------------------------------------ the run is finished only when every arm is in the manifest
def test_finished_is_printed_only_when_every_arm_of_the_plan_is_in_the_manifest(tmp_path, monkeypatch, capsys):
    drive, commands, fail, argv = drive_world(tmp_path, monkeypatch)
    rf.main(argv)
    assert "locked evaluation sampling finished" in capsys.readouterr().out
    (tmp_path / "out" / "manifest.json").write_text(json.dumps({"arms": {"9B": {}}}))
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)                                          # a stage that returns 0 but whose arms are not all recorded
    with pytest.raises(SystemExit, match=r"not finished: 2 arm\(s\) of the plan are not in .*manifest.json: \['name-9B', 'ident-9B'\]"):
        rf.main(argv)
    assert "locked evaluation sampling finished" not in capsys.readouterr().out


def test_a_from_stage_over_an_unfinished_stage_cannot_print_finished(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "manifest.json").write_text(json.dumps({"arms": {"27B-3.5": {}, "name-27B": {}, "ident-27B": {}}}))
    argv = [a for a in files(tmp_path, arms("A0n", "27B-3.5", "name-27B", "ident-27B")) if a != "--dry-run"] + ["--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match=r"not finished: 1 arm\(s\) .*\['A0n'\]"):
        rf.main([*argv, "--from-stage", "Qwen3.5-27B"])
    (tmp_path / "out" / "manifest.json").write_text(json.dumps({"arms": {"A0n": {}, "27B-3.5": {}, "name-27B": {}, "ident-27B": {}}}))
    rf.main([*argv, "--from-stage", "Qwen3.5-27B"])
    assert "locked evaluation sampling finished" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the Hugging Face cache
def test_the_cache_to_clear_is_the_hub_libraries_and_falls_back_to_the_default(monkeypatch, tmp_path):
    constants = type(sys)("huggingface_hub.constants")
    constants.HF_HUB_CACHE = str(tmp_path / "hf" / "hub")
    package = type(sys)("huggingface_hub")
    package.constants = constants
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    assert rf.hub_cache() == tmp_path / "hf" / "hub"
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert rf.hub_cache() == tmp_path / "home" / ".cache" / "huggingface" / "hub"


def test_a_stage_that_clears_removes_the_hub_cache_and_the_merged_models_but_a_resume_with_keep_cache_does_not(tmp_path, monkeypatch):
    removed = []
    monkeypatch.setattr(rf, "hub_cache", lambda: tmp_path / "the-hub-cache")
    monkeypatch.setattr(rf.shutil, "rmtree", lambda path, ignore_errors=False: removed.append(Path(path)))
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)
    argv = [a for a in files(tmp_path, arms("9B")) if a != "--dry-run"] + ["--out", str(tmp_path / "out"), "--models", str(tmp_path / "models")]
    with pytest.raises(SystemExit, match="not finished"):
        rf.main(argv)
    assert removed == [tmp_path / "the-hub-cache", tmp_path / "models"]
    removed.clear()
    with pytest.raises(SystemExit, match="not finished"):
        rf.main([*argv, "--keep-cache"])
    assert removed == []
    rf.main([*argv, "--dry-run"])                                                                       # a dry run deletes nothing
    assert removed == []


def test_keep_cache_protects_only_the_stage_being_resumed(tmp_path, monkeypatch):
    removed = []
    monkeypatch.setattr(rf.shutil, "rmtree", lambda path, **kw: removed.append(str(path)))
    monkeypatch.setattr(rf, "hub_cache", lambda: tmp_path / "hub")
    monkeypatch.setattr(rf, "STAGES", [("first", ["9B"], True), ("second", ["27B-3.5"], True)])
    commands = []
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    for extra, expected in (([], 4), (["--keep-cache"], 2)):                    # a clear is two rmtree calls: the hub cache and the merged models
        removed.clear()
        with pytest.raises(SystemExit, match="not finished"):
            rf.main([*main_argv(tmp_path, arms("9B", "27B-3.5"), {"arms": {}}), *extra])
        assert len(removed) == expected, (extra, removed)
    stage_failure = lambda cmd, dry, **kw: 1 if "final_eval.py" in " ".join(map(str, cmd)) else 0
    monkeypatch.setattr(rf, "run", stage_failure)
    with pytest.raises(SystemExit, match=r"--skip-sanity --from-stage 'first' --keep-cache"):
        rf.main(main_argv(tmp_path, arms("9B", "27B-3.5"), {"arms": {}}))
