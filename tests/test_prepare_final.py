import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu"


@pytest.fixture(scope="module")
def pf():
    spec = importlib.util.spec_from_file_location("prepare_final", GPU / "prepare_final.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapter(tmp_path):
    d = tmp_path / "ad"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "adapter_config.json").write_text("{}")
    return d, {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("adapter_model.safetensors", "adapter_config.json")}


def write_spec(tmp_path, block):
    spec = {"frozen": True, "arms": [{"id": "RL-full", "adapter": block}, {"id": "C0"}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_matching_adapter_is_merged_once(pf, tmp_path, monkeypatch, capsys):
    d, hashes = adapter(tmp_path)
    path, digest = write_spec(tmp_path, {"path": str(d), "base": "Qwen/Qwen3.5-4B", "sha256": hashes})
    calls = []
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd: calls.append(cmd) or type("R", (), {"returncode": 0})())
    pf.main(["--spec", str(path), "--spec-sha256", digest, "--models", str(tmp_path / "models")])
    assert len(calls) == 1 and calls[0][calls[0].index("--out") + 1] == str(tmp_path / "models" / "RL-full") and calls[0][calls[0].index("--adapter") + 1] == str(d)
    assert json.loads((tmp_path / "models" / "RL-full" / "PREPARED.json").read_text())["adapter_sha256"] == hashes


def test_a_changed_adapter_file_is_refused(pf, tmp_path):
    d, hashes = adapter(tmp_path)
    path, digest = write_spec(tmp_path, {"path": str(d), "sha256": hashes})
    (d / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="differs from the frozen hash"):
        pf.main(["--spec", str(path), "--spec-sha256", digest, "--models", str(tmp_path / "m"), "--dry-run"])


def test_every_adapter_file_must_be_frozen_and_the_spec_hash_checked(pf, tmp_path):
    d, hashes = adapter(tmp_path)
    path, digest = write_spec(tmp_path, {"path": str(d), "sha256": {"adapter_model.safetensors": hashes["adapter_model.safetensors"]}})
    with pytest.raises(SystemExit, match="every adapter file"):
        pf.main(["--spec", str(path), "--spec-sha256", digest, "--models", str(tmp_path / "m"), "--dry-run"])
    with pytest.raises(SystemExit, match="sha256 differs"):
        pf.main(["--spec", str(path), "--spec-sha256", "0" * 64, "--models", str(tmp_path / "m"), "--dry-run"])


def test_a_prepared_arm_is_skipped_and_dry_run_merges_nothing(pf, tmp_path, monkeypatch, capsys):
    d, hashes = adapter(tmp_path)
    path, digest = write_spec(tmp_path, {"path": str(d), "sha256": hashes})
    out = tmp_path / "m" / "RL-full"
    out.mkdir(parents=True)
    (out / "PREPARED.json").write_text(json.dumps({"adapter_sha256": hashes}))
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd: pytest.fail("must not merge"))
    pf.main(["--spec", str(path), "--spec-sha256", digest, "--models", str(tmp_path / "m")])
    assert "already prepared" in capsys.readouterr().out
    pf.main(["--spec", str(path), "--spec-sha256", digest, "--models", str(tmp_path / "m2"), "--dry-run"])
    assert not (tmp_path / "m2").exists()
