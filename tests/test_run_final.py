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


def test_the_sanity_gate_stops_on_a_mismatch(tmp_path, monkeypatch):
    out = tmp_path / "sanity" / "A2"
    out.mkdir(parents=True)
    (out / "samples.jsonl").write_text("")
    (out / "report.v3.json").write_text(json.dumps({"splits": {"val": {"metrics": {"fv_tanimoto": {"value": 0.20}}}}}))
    monkeypatch.setattr(rf, "SANITY", tmp_path / "sanity")
    monkeypatch.setattr(rf, "run", lambda cmd, dry, **kw: 0)
    arm = {"id": "A2", "model": "m", "mode": "think", "temperature": 1.0, "max_tokens": 256}
    rf.sanity(arm, {"arms": {"A2": {"score": 0.195}}}, 0.01, False)           # within tolerance: passes
    with pytest.raises(SystemExit, match="reproduces"):
        rf.sanity(arm, {"arms": {"A2": {"score": 0.15}}}, 0.01, False)        # 0.05 apart: stops
