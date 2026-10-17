import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu"


@pytest.fixture(scope="module")
def fe():
    spec = importlib.util.spec_from_file_location("final_eval", GPU / "final_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm(**kw):
    return {"id": "A2-4b", "model": "merged/a2", "mode": "think", "data": "direct", "system": "direct", "temperature": 0.6, "k": 8, "max_tokens": 256, **kw}


def write(tmp_path, **kw):
    spec = {"frozen": True, "splits": ["test_t", "test_r"], "arms": [arm()], **kw}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_cli_dry_run_prints_commands_and_writes_a_manifest(fe, tmp_path, capsys):
    path, digest = write(tmp_path)
    fe.main(["--spec", str(path), "--spec-sha256", digest, "--out", str(tmp_path / "out"), "--dry-run"])
    assert json.loads(capsys.readouterr().out)[1].endswith("sample_vllm.py")
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["spec_sha256"] == digest
    with pytest.raises(SystemExit):
        fe.main(["--spec", str(path), "--spec-sha256", "0" * 64, "--out", str(tmp_path / "out"), "--dry-run"])
    with pytest.raises(SystemExit):
        fe.main(["--spec", str(path), "--spec-sha256", digest, "--out", str(tmp_path / "out"), "--dry-run", "--only", "nope"])


def test_a_failing_arm_stops_the_run_with_a_resume_hint(fe, tmp_path, monkeypatch):
    path, digest = write(tmp_path)
    monkeypatch.setattr(fe.subprocess, "run", lambda cmd: type("R", (), {"returncode": 3})())
    with pytest.raises(SystemExit, match="rerun the same command"):
        fe.main(["--spec", str(path), "--spec-sha256", digest, "--out", str(tmp_path / "out")])
