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


REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8c3d0"
HASHES = {k: {W: k[:2] * 32, C: "c" * 64, "revision": REVISION} for k in ("A2-epoch1", "R1-step-40", "R5-step-60")}


def test_the_spec_copies_what_the_selection_decided():
    out = mf.build(selection(), HASHES, "m" * 64)
    ids = [a["id"] for a in out["arms"]]
    assert ids[:4] == ["A0n", "A4", "A2", "RL-full"] and len(ids) == len(set(ids)) == 10
    by = {a["id"]: a for a in out["arms"]}
    assert by["A0n"]["temperature"] == 0.6 and by["A2"]["temperature"] == 1.0 and by["A4"]["data"] == "icl" and by["A4"]["system"] == "dataset"
    assert by["RL-full"]["adapter"]["path"] == ".agent-runs/adapters_plain/R1-step-40" and by["RL-full"]["model"] == "final_models/RL-full"
    assert by["RL-full-shuffle"]["of"] == "RL-full" and by["RL-full-shuffle"]["adapter"] == by["RL-full"]["adapter"] and by["RL-full-blank"]["data"] == "blank"
    assert by["ident-4B"]["max_tokens"] == 128 and by["name-4B"]["role"] == "probe" and not any(i.startswith(("9B", "27B", "name-9B", "ident-9B", "name-27B", "ident-27B", "name-3.8", "ident-3.8")) for i in ids)
    assert out["data_manifest_sha256"] == "m" * 64 and all(a["k"] == 8 for a in out["arms"] if a["id"] in ("A0n", "A4", "A2", "RL-full"))


def test_an_r5_checkpoint_is_read_from_its_own_run_and_hash():
    out = mf.build(selection("rlfull5-s60-t0.6", "step-60"), HASHES, "m" * 64)
    rl = {a["id"]: a for a in out["arms"]}["RL-full"]
    assert rl["adapter"]["path"] == ".agent-runs/adapters_plain/R5-step-60" and rl["adapter"]["sha256"][W] == HASHES["R5-step-60"][W]


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


def test_every_adapter_block_names_the_base_revision_of_its_hash_entry():
    hashes = {**HASHES, "A2-epoch1": {**HASHES["A2-epoch1"], "revision": "a" * 40}}
    out = mf.build(selection(), hashes, "m" * 64)
    blocks = {a["id"]: a["adapter"] for a in out["arms"] if "adapter" in a}
    assert sorted(blocks) == ["A2", "A2-blank", "A2-shuffle", "RL-full", "RL-full-blank", "RL-full-shuffle"]
    assert blocks["A2"]["revision"] == "a" * 40 == blocks["A2-shuffle"]["revision"] == blocks["A2-blank"]["revision"]
    assert blocks["RL-full"]["revision"] == REVISION == blocks["RL-full-blank"]["revision"] and blocks["RL-full-shuffle"] == blocks["RL-full"]


def test_a_hash_entry_without_a_revision_fails_loudly_instead_of_freezing_an_unpinned_adapter():
    for key in ("A2-epoch1", "R1-step-40"):
        hashes = {**HASHES, key: {W: "1" * 64, C: "c" * 64}}
        with pytest.raises(KeyError, match="revision"):
            mf.build(selection(), hashes, "m" * 64)


def test_the_written_spec_passes_the_drivers_revision_rule(tmp_path):
    import importlib.util as iu
    fe_spec = iu.spec_from_file_location("final_eval_for_frozen_spec", ROOT / "gpu" / "final_eval.py")
    fe = iu.module_from_spec(fe_spec)
    fe_spec.loader.exec_module(fe)
    assert all(fe.REVISION.fullmatch(a["adapter"]["revision"]) for a in mf.build(selection(), HASHES, "m" * 64)["arms"] if "adapter" in a)
