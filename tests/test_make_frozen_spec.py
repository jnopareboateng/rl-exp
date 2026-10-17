import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_frozen_spec", ROOT / "scripts" / "make_frozen_spec.py")
mf = importlib.util.module_from_spec(spec)
sys.modules["make_frozen_spec"] = mf
spec.loader.exec_module(mf)

W, C = "adapter_model.safetensors", "adapter_config.json"
BLOCK = {"mode": "think", "max_tokens": 256, "data": "direct", "system": "direct"}


def selection(rl_run="rlfull-s40-t0.6", rl_ckpt="step-40"):
    return {"rl": "RL-full", "comparator": "C3", "arms": {
        "C3": {"temperature": None},
        "A0n": {"temperature": 0.6, "spec": {**BLOCK, "mode": "nothink"}},
        "A4": {"temperature": 0.6, "spec": {**BLOCK, "mode": "nothink", "data": "icl", "system": "dataset"}},
        "A2": {"temperature": 1.0, "spec": BLOCK},
        "RL-full": {"temperature": 0.6, "run": rl_run, "checkpoint": rl_ckpt, "spec": BLOCK}}}


HASHES = {k: {W: k[:2] * 32, C: "c" * 64} for k in ("A2-epoch1", "R1-step-40", "R5-step-60")}


def test_the_spec_copies_what_the_selection_decided():
    out = mf.build(selection(), HASHES, "m" * 64)
    ids = [a["id"] for a in out["arms"]]
    assert ids[:4] == ["A0n", "A4", "A2", "RL-full"] and len(ids) == len(set(ids)) == 19
    by = {a["id"]: a for a in out["arms"]}
    assert by["A0n"]["temperature"] == 0.6 and by["A2"]["temperature"] == 1.0 and by["A4"]["data"] == "icl" and by["A4"]["system"] == "dataset"
    assert by["RL-full"]["adapter"]["path"] == ".agent-runs/rl_R1/adapters/step-40" and by["RL-full"]["model"] == "final_models/RL-full"
    assert by["RL-full-shuffle"]["of"] == "RL-full" and by["RL-full-shuffle"]["adapter"] == by["RL-full"]["adapter"] and by["RL-full-blank"]["data"] == "blank"
    assert by["27B-3.5"]["splits"] == ["test_t"] and by["27B-3.5"]["gpu_memory_utilization"] == 0.93 and by["ident-4B"]["max_tokens"] == 128 and by["ident-27B"]["k"] == 2
    assert out["data_manifest_sha256"] == "m" * 64 and all(a["k"] == 8 for a in out["arms"] if a["id"] in ("A0n", "A4", "A2", "RL-full"))


def test_an_r5_checkpoint_is_read_from_its_own_run_and_hash():
    out = mf.build(selection("rlfull5-s60-t0.6", "step-60"), HASHES, "m" * 64)
    rl = {a["id"]: a for a in out["arms"]}["RL-full"]
    assert rl["adapter"]["path"] == ".agent-runs/rl_R5/adapters/step-60" and rl["adapter"]["sha256"][W] == HASHES["R5-step-60"][W]


def test_main_writes_the_spec_and_its_hash_file(tmp_path):
    (tmp_path / "sel.json").write_text(json.dumps(selection()))
    (tmp_path / "h.json").write_text(json.dumps(HASHES))
    (tmp_path / "man").write_text("x")
    out = tmp_path / "frozen_arms.json"
    mf.main(["--selection", str(tmp_path / "sel.json"), "--hashes", str(tmp_path / "h.json"), "--manifest", str(tmp_path / "man"), "--out", str(out)])
    written = json.loads(out.read_text())
    assert written["selection_sha256"] == mf.sha256(tmp_path / "sel.json") and written["frozen"] is True
    assert (tmp_path / "frozen_arms.sha256").read_text().split()[0] == mf.sha256(out)


def test_a_missing_hash_fails_loudly():
    with pytest.raises(KeyError):
        mf.build(selection("rlfull5-s70-t0.6", "step-70"), HASHES, "m" * 64)
