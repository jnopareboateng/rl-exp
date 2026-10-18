import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu"
MODELS = "final_models"          # the models directory, relative to the working directory (a selected adapter arm samples final_models/<id>)


@pytest.fixture(scope="module")
def pf():
    spec = importlib.util.spec_from_file_location("prepare_final", GPU / "prepare_final.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter(tmp_path):
    d = tmp_path / "ad"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"weights")
    (d / "adapter_config.json").write_text("{}")
    return d, {f: sha(d / f) for f in ("adapter_model.safetensors", "adapter_config.json")}


def write_spec(tmp_path, block, temperature=0.6, model=None, extra=()):
    """A frozen spec with one adapter arm (RL-full, the selected RL arm, sampled from final_models/RL-full) and an untrained one (A0n), plus the selection it realises."""
    spec_block = {"mode": "nothink", "max_tokens": 256, "data": "direct", "system": "direct"}
    selection = {"arms": {"C0": {"temperature": None, "checkpoint": None},
                          "A0n": {"temperature": 0.6, "checkpoint": None, "spec": {**spec_block, "model": "Qwen/Qwen3.5-4B"}},
                          "RL-full": {"temperature": 0.6, "checkpoint": "step-40", "adapter_sha256": block["sha256"].get("adapter_model.safetensors"),
                                      "spec": {**spec_block, "model": "final_models/RL-full"}}},
                 "comparator": "C0", "rl": "RL-full", "comparator_set": ["C0", "A0n"], "rl_set": ["RL-full"]}
    sel_path = tmp_path / "selection.json"
    sel_path.write_text(json.dumps(selection))
    base = {"mode": "nothink", "data": "direct", "system": "direct", "k": 8, "max_tokens": 256}
    spec = {"frozen": True, "splits": ["test_t", "test_r"], "selection_sha256": sha(sel_path),
            "arms": [{"id": "RL-full", "model": model or f"{MODELS}/RL-full", "temperature": temperature, "adapter": block, **base},
                     {"id": "A0n", "model": "Qwen/Qwen3.5-4B", "temperature": 0.6, **base}, *extra]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path, sha(path), sel_path


def run_main(pf, tmp_path, spec, models=MODELS, *extra):
    path, digest, sel_path = spec
    pf.main(["--spec", str(path), "--spec-sha256", digest, "--selection", str(sel_path), "--models", models, *extra])


def fake_merge(calls, weights=b"merged weights"):
    """A merge_lora.py that writes a one-shard export."""
    def run(cmd):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if weights is not None:
            (out / "model-00001-of-00001.safetensors").write_bytes(weights)
        (out / "config.json").write_text("{}")
        return type("R", (), {"returncode": 0})()
    return run


def prepared_marker(models, spec_block):
    out = Path(models) / "RL-full"
    out.mkdir(parents=True)
    (out / "model-00001-of-00001.safetensors").write_bytes(b"merged weights")
    (out / "config.json").write_text("{}")
    marker = {"adapter_sha256": spec_block["sha256"], "merged_sha256": {"config.json": sha(out / "config.json"),
              "model-00001-of-00001.safetensors": sha(out / "model-00001-of-00001.safetensors")}}
    (out / "PREPARED.json").write_text(json.dumps(marker))
    return out


def test_a_matching_adapter_is_merged_once_and_the_merged_files_are_hashed(pf, tmp_path, monkeypatch):
    d, hashes = adapter(tmp_path)
    block = {"path": str(d), "base": "Qwen/Qwen3.5-4B", "sha256": hashes}
    calls = []
    monkeypatch.setattr(pf.subprocess, "run", fake_merge(calls))
    run_main(pf, tmp_path, write_spec(tmp_path, block))
    assert len(calls) == 1 and calls[0][calls[0].index("--out") + 1] == f"{MODELS}/RL-full" and calls[0][calls[0].index("--adapter") + 1] == str(d)
    out = tmp_path / MODELS / "RL-full"
    marker = json.loads((out / "PREPARED.json").read_text())
    assert marker["adapter_sha256"] == hashes
    assert marker["merged_sha256"] == {"config.json": sha(out / "config.json"), "model-00001-of-00001.safetensors": sha(out / "model-00001-of-00001.safetensors")}


def test_what_prepare_final_records_is_what_final_eval_checks_before_sampling(pf, tmp_path, monkeypatch):
    """The two scripts share one definition of the merged files (final_eval.merged_hashes): a freshly prepared export passes final_eval's check, a changed one does not."""
    d, hashes = adapter(tmp_path)
    block = {"path": str(d), "base": "Qwen/Qwen3.5-4B", "sha256": hashes}
    monkeypatch.setattr(pf.subprocess, "run", fake_merge([]))
    run_main(pf, tmp_path, write_spec(tmp_path, block))
    arm = {"id": "RL-full", "model": f"{MODELS}/RL-full", "adapter": block}
    sys_fe = pf.sys.modules["final_eval"]
    sys_fe.check_prepared([arm])
    (tmp_path / MODELS / "RL-full" / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    with pytest.raises(sys_fe.SpecError, match="no longer hash"):
        sys_fe.check_prepared([arm])


def test_a_merge_that_wrote_no_weights_is_an_error(pf, tmp_path, monkeypatch):
    d, hashes = adapter(tmp_path)
    monkeypatch.setattr(pf.subprocess, "run", fake_merge([], weights=None))
    with pytest.raises(SystemExit, match="wrote no"):
        run_main(pf, tmp_path, write_spec(tmp_path, {"path": str(d), "sha256": hashes}))
    assert not (tmp_path / MODELS / "RL-full" / "PREPARED.json").exists()


def test_a_changed_adapter_file_is_refused(pf, tmp_path):
    d, hashes = adapter(tmp_path)
    spec = write_spec(tmp_path, {"path": str(d), "sha256": hashes})
    (d / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="differs from the frozen hash"):
        run_main(pf, tmp_path, spec, MODELS, "--dry-run")


def test_every_adapter_file_must_be_frozen_and_the_spec_hash_checked(pf, tmp_path):
    d, hashes = adapter(tmp_path)
    path, digest, sel_path = write_spec(tmp_path, {"path": str(d), "sha256": {"adapter_model.safetensors": hashes["adapter_model.safetensors"]}})
    with pytest.raises(SystemExit, match="every adapter file"):
        run_main(pf, tmp_path, (path, digest, sel_path), MODELS, "--dry-run")
    with pytest.raises(SystemExit, match="sha256 differs"):
        run_main(pf, tmp_path, (path, "0" * 64, sel_path), MODELS, "--dry-run")


def test_an_adapter_file_that_is_missing_is_a_plain_error_not_a_traceback(pf, tmp_path):
    d, hashes = adapter(tmp_path)
    with pytest.raises(SystemExit, match="lacks.*notes.txt"):
        run_main(pf, tmp_path, write_spec(tmp_path, {"path": str(d), "sha256": {**hashes, "notes.txt": "0" * 64}}), MODELS, "--dry-run")
    (d / "adapter_config.json").unlink()
    with pytest.raises(SystemExit, match="lacks.*adapter_config.json"):
        run_main(pf, tmp_path, write_spec(tmp_path, {"path": str(d), "sha256": hashes}), MODELS, "--dry-run")


def test_a_prepared_arm_is_skipped_and_dry_run_merges_nothing(pf, tmp_path, monkeypatch, capsys):
    d, hashes = adapter(tmp_path)
    block = {"path": str(d), "sha256": hashes}
    spec = write_spec(tmp_path, block)
    prepared_marker(MODELS, block)
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd: pytest.fail("must not merge"))
    run_main(pf, tmp_path, spec)
    assert "already prepared" in capsys.readouterr().out
    shutil.rmtree(tmp_path / MODELS)
    run_main(pf, tmp_path, spec, MODELS, "--dry-run")
    assert not (tmp_path / MODELS).exists()


@pytest.mark.parametrize("stale", ["weights changed", "weights deleted", "old marker without merged hashes", "other adapter", "marker is not json"])
def test_a_marker_is_trusted_only_when_the_merged_files_still_hash_as_recorded(pf, tmp_path, monkeypatch, stale):
    d, hashes = adapter(tmp_path)
    block = {"path": str(d), "sha256": hashes}
    spec = write_spec(tmp_path, block)
    out = prepared_marker(MODELS, block)
    if stale == "weights changed":
        (out / "model-00001-of-00001.safetensors").write_bytes(b"wrong weights")
    elif stale == "weights deleted":
        (out / "model-00001-of-00001.safetensors").unlink()
    elif stale == "old marker without merged hashes":
        (out / "PREPARED.json").write_text(json.dumps({"adapter_sha256": hashes}))
    elif stale == "other adapter":
        (out / "PREPARED.json").write_text(json.dumps({**json.loads((out / "PREPARED.json").read_text()), "adapter_sha256": {**hashes, "adapter_config.json": "0" * 64}}))
    else:
        (out / "PREPARED.json").write_text("{")
    calls = []
    monkeypatch.setattr(pf.subprocess, "run", fake_merge(calls))
    run_main(pf, tmp_path, spec)
    assert len(calls) == 1                                                # merge_lora.py runs again (it checks an existing export itself)
    assert json.loads((out / "PREPARED.json").read_text())["merged_sha256"]["model-00001-of-00001.safetensors"] == sha(out / "model-00001-of-00001.safetensors")


def test_an_only_typo_is_refused_before_anything_is_merged(pf, tmp_path, monkeypatch):
    d, hashes = adapter(tmp_path)
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd: pytest.fail("must not merge"))
    with pytest.raises(SystemExit, match="--only names"):
        run_main(pf, tmp_path, write_spec(tmp_path, {"path": str(d), "sha256": hashes}), MODELS, "--only", "RL-full,RL-fulll")


def test_the_spec_is_validated_like_final_eval_does(pf, tmp_path, monkeypatch):
    d, hashes = adapter(tmp_path)
    block = {"path": str(d), "sha256": hashes}
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd: pytest.fail("must not merge"))
    with pytest.raises(SystemExit, match="frozen grid"):                  # a setting final_eval refuses
        run_main(pf, tmp_path, write_spec(tmp_path, block, temperature=0.5))
    path, digest, sel_path = write_spec(tmp_path, block)
    sel = json.loads(sel_path.read_text())
    sel["arms"]["RL-full"]["adapter_sha256"] = "f" * 64                   # a spec that is not what Val selected
    sel_path.write_text(json.dumps(sel))
    with pytest.raises(SystemExit, match="selection_sha256"):
        run_main(pf, tmp_path, (path, digest, sel_path))
    spec = json.loads(path.read_text())
    spec["selection_sha256"] = sha(sel_path)
    path.write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match="not the selected checkpoint"):
        run_main(pf, tmp_path, (path, sha(path), sel_path))


def test_the_selection_is_a_required_argument(pf, tmp_path):
    d, hashes = adapter(tmp_path)
    path, digest, _ = write_spec(tmp_path, {"path": str(d), "sha256": hashes})
    with pytest.raises(SystemExit) as e:
        pf.main(["--spec", str(path), "--spec-sha256", digest, "--models", MODELS, "--dry-run"])
    assert e.value.code == 2


def test_a_diagnostic_of_an_adapter_arm_is_not_merged_again(pf, tmp_path, monkeypatch):
    d, hashes = adapter(tmp_path)
    block = {"path": str(d), "sha256": hashes}
    base = json.loads(write_spec(tmp_path, block)[0].read_text())["arms"][0]
    shuffle = {**base, "id": "RL-full-shuffle", "role": "diagnostic", "of": "RL-full", "data": "shuffle"}       # samples the merged model of RL-full
    calls = []
    monkeypatch.setattr(pf.subprocess, "run", fake_merge(calls))
    run_main(pf, tmp_path, write_spec(tmp_path, block, extra=[shuffle]))
    assert [c[c.index("--out") + 1] for c in calls] == [f"{MODELS}/RL-full"] and not (tmp_path / MODELS / "RL-full-shuffle").exists()


def test_the_merge_target_must_be_the_model_the_spec_samples(pf, tmp_path, monkeypatch):
    d, hashes = adapter(tmp_path)
    monkeypatch.setattr(pf.subprocess, "run", lambda cmd: pytest.fail("must not merge"))
    for models in ("elsewhere", str(tmp_path / "elsewhere"), "final_models/sub"):           # the spec samples final_models/RL-full
        with pytest.raises(SystemExit, match="samples from .* but the merge is written to"):
            run_main(pf, tmp_path, write_spec(tmp_path, {"path": str(d), "sha256": hashes}), models, "--dry-run")
    run_main(pf, tmp_path, write_spec(tmp_path, {"path": str(d), "sha256": hashes}), str(tmp_path / MODELS), "--dry-run")      # the same directory, spelled absolutely
