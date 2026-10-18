import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_final", ROOT / "gpu" / "run_final.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["run_final"] = rf
spec.loader.exec_module(rf)


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


def test_dry_run_prints_prepare_sanity_and_stage_commands_without_running(tmp_path, capsys):
    ad = {"adapter": {"path": "x"}, "model": "final_models/A2", "mode": "think", "temperature": 1.0, "max_tokens": 256}
    sel = {"arms": {"A2": {"score": 0.15}}}
    rf.main(files(tmp_path, arms("A2", "9B", "27B-3.5", **{"A2": ad}), sel))
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


ADAPTER = {"adapter": {"path": "x"}, "model": "final_models/A2", "mode": "think", "temperature": 1.0, "max_tokens": 256}


def fake_environment(tmp_path, monkeypatch, value=0.1975):
    """A Val sanity run that 'samples' into tmp_path/sanity and reports `value`; rf.run records every command."""
    stale = tmp_path / "sanity" / "A2"
    monkeypatch.setattr(rf, "SANITY", tmp_path / "sanity")
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


def test_the_sanity_record_holds_the_value_the_score_the_tolerance_and_the_verdict(tmp_path, monkeypatch):
    fake_environment(tmp_path, monkeypatch)                                # the Val sample reproduces 0.1975
    arm = {"id": "A2", "model": "m", "mode": "think", "temperature": 1.0, "max_tokens": 256}
    assert rf.sanity(arm, {"arms": {"A2": {"score": 0.195}}}, 0.005, False) == {"value": 0.1975, "selection_score": 0.195, "tolerance": 0.005, "passed": True, "seed": 0}
    assert rf.sanity(arm, {"arms": {"A2": {"score": 0.15}}}, 0.005, False)["passed"] is False                            # 0.0475 apart
    assert rf.sanity(arm, {"arms": {"A2": {"score": 0.195}}}, 0.002, False)["passed"] is False                            # 0.0025 apart, tolerance 0.002
    assert rf.sanity(arm, {"arms": {"A2": {"score": 0.190}}}, 0.005, False)["passed"] is False                            # 0.0075 apart: refused at 0.005 ...
    assert rf.sanity(arm, {"arms": {"A2": {"score": 0.190}}}, 0.01, False)["passed"] is True                              # ... and accepted at the old 0.01


def test_a_stale_sanity_directory_is_deleted_before_the_sample_but_not_in_a_dry_run(tmp_path, monkeypatch):
    stale, seen = fake_environment(tmp_path, monkeypatch)
    stale.mkdir(parents=True)
    (stale / "samples.jsonl").write_text("samples of an older merge")
    arm = {"id": "A2", "model": "m", "mode": "think", "temperature": 1.0, "max_tokens": 256}
    assert rf.sanity(arm, {"arms": {"A2": {"score": 0.2}}}, 0.005, True) is None
    assert (stale / "samples.jsonl").read_text() == "samples of an older merge" and seen["stale_present_at_sampling"] == []      # a dry run deletes and samples nothing real
    rf.sanity(arm, {"arms": {"A2": {"score": 0.2}}}, 0.005, False)
    assert seen["stale_present_at_sampling"] == [False]                                                                         # gone when the sampler starts


def main_argv(tmp_path, spec_arms, selection, *extra):
    argv = files(tmp_path, spec_arms, selection)
    return [a for a in argv if a != "--dry-run"] + ["--out", str(tmp_path / "out"), *extra]


def test_every_result_is_written_to_sanity_json_and_a_failure_is_on_record_before_the_stop(tmp_path, monkeypatch):
    records = {"A2": {"value": 0.1560, "selection_score": 0.1557, "tolerance": 0.005, "passed": True, "seed": 0},
               "RL-full": {"value": 0.20, "selection_score": 0.1887, "tolerance": 0.005, "passed": False, "seed": 0}}
    monkeypatch.setattr(rf, "sanity", lambda arm, selection, tolerance, dry: records[arm["id"]])
    commands = []
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    spec_arms = arms("A2", "RL-full", **{"A2": ADAPTER, "RL-full": {**ADAPTER, "model": "final_models/RL-full"}})
    with pytest.raises(SystemExit, match="RL-full reproduces 0.2000, the selection says 0.1887"):
        rf.main(main_argv(tmp_path, spec_arms, {"arms": {}}))
    assert json.loads((tmp_path / "out" / "sanity.json").read_text()) == records
    assert not any("final_eval.py" in " ".join(map(str, c)) for c in commands)                  # no Test sampling after a failed gate


def test_a_passing_gate_leaves_all_records_and_the_default_tolerance_is_half_a_percent(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rf, "sanity", lambda arm, selection, tolerance, dry: seen.append(tolerance) or
                        {"value": 0.15, "selection_score": 0.15, "tolerance": tolerance, "passed": True, "seed": 0})
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)
    rf.main(main_argv(tmp_path, arms("A2", "9B", **{"A2": ADAPTER}), {"arms": {}}))
    assert seen == [0.005] and json.loads((tmp_path / "out" / "sanity.json").read_text())["A2"]["passed"] is True
    rf.main(main_argv(tmp_path, arms("A2", **{"A2": ADAPTER}), {"arms": {}}, "--tolerance", "0.002"))
    assert seen == [0.005, 0.002]


def test_skip_sanity_leaves_a_skipped_record_for_every_adapter_arm_and_no_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "sanity", lambda *a: pytest.fail("the gate must not run"))
    commands = []
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: commands.append(cmd) or 0)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "sanity.json").write_text(json.dumps({"A2": {"value": 0.15, "selection_score": 0.15, "tolerance": 0.005, "passed": True, "seed": 0}}))      # an earlier pass
    spec_arms = arms("A2", "RL-full", "9B", **{"A2": ADAPTER, "RL-full": {**ADAPTER, "model": "final_models/RL-full"}})
    rf.main(main_argv(tmp_path, spec_arms, {"arms": {}}, "--skip-sanity"))
    assert json.loads((tmp_path / "out" / "sanity.json").read_text()) == {"A2": {"skipped": True}, "RL-full": {"skipped": True}}      # the old pass is not carried over
    assert any("final_eval.py" in " ".join(map(str, c)) for c in commands)                                                     # sampling goes on


def test_a_dry_run_writes_no_sanity_file_and_skip_sanity_in_a_dry_run_neither(tmp_path, monkeypatch, capsys):
    argv = files(tmp_path, arms("A2", "9B", **{"A2": ADAPTER}), {"arms": {"A2": {"score": 0.15}}})
    rf.main([*argv, "--out", str(tmp_path / "out")])
    rf.main([*argv, "--out", str(tmp_path / "out"), "--skip-sanity"])
    assert not (tmp_path / "out").exists()


def test_a_spec_without_adapter_arms_writes_no_sanity_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)
    rf.main(main_argv(tmp_path, arms("9B", "27B-3.5"), {"arms": {}}))
    assert not (tmp_path / "out" / "sanity.json").exists()


def test_write_sanity_keeps_the_other_arms_and_is_atomic(tmp_path):
    rf.write_sanity(tmp_path / "o", "A2", {"skipped": True})
    rf.write_sanity(tmp_path / "o", "RL-full", {"value": 0.1, "selection_score": 0.1, "tolerance": 0.005, "passed": True, "seed": 0})
    rf.write_sanity(tmp_path / "o", "A2", {"value": 0.2, "selection_score": 0.2, "tolerance": 0.005, "passed": True, "seed": 0})
    assert sorted(json.loads((tmp_path / "o" / "sanity.json").read_text())) == ["A2", "RL-full"] and not list((tmp_path / "o").glob("*.tmp"))
