import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_controls", ROOT / "scripts" / "final_controls.py")
fc = importlib.util.module_from_spec(spec)
sys.modules["final_controls"] = fc
spec.loader.exec_module(fc)


def rows(splits=("val", "test_t", "test_r"), model="control_x"):
    return [{"id": f"{s}-{i}", "group": f"d{i}", "split": s, "sample_idx": 0, "text": "x", "answer": "CCO", "model": model} for s in splits for i in range(2)]


def write(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")


def test_only_test_rows_are_kept_and_both_splits_are_required(tmp_path):
    write(tmp_path / "a.jsonl", rows())
    kept = fc.test_rows(tmp_path / "a.jsonl")
    assert {r["split"] for r in kept} == {"test_t", "test_r"} and len(kept) == 4
    write(tmp_path / "b.jsonl", rows(("val", "test_t")))
    with pytest.raises(ValueError, match="test_r"):
        fc.test_rows(tmp_path / "b.jsonl")


def test_write_control_replicates_eight_times_and_records_the_model_and_the_source_hash(tmp_path):
    src = tmp_path / "src.jsonl"
    write(src, rows(("test_t", "test_r")))
    out = fc.write_control(tmp_path / "final", "C0", fc.test_rows(src), src, {"extra": 1})
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 4 * 8 and sorted({r["sample_idx"] for r in samples}) == list(range(8))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["source_sha256"] == fc.sha256(src) and meta["k"] == 8 and meta["extra"] == 1 and meta["rows"] == 4
    assert meta["control"] == "C0" and meta["model"] == "control_x" and meta["splits"] == ["test_t", "test_r"]


@pytest.mark.parametrize("model", [None, "", "final_models/RL-full", "Qwen/Qwen3.5-4B"])
def test_rows_that_are_not_a_controls_are_refused_before_anything_is_written(tmp_path, model):
    src = tmp_path / "src.jsonl"
    write(src, [{**r, "model": model} for r in rows(("test_t", "test_r"))])
    with pytest.raises(ValueError, match="one control_\\* model name"):
        fc.write_control(tmp_path / "final", "C0", fc.test_rows(src), src)
    assert not (tmp_path / "final").exists()
    mixed = rows(("test_t", "test_r"))
    mixed[0]["model"] = "control_y"
    write(src, mixed)
    with pytest.raises(ValueError, match="2 names"):
        fc.write_control(tmp_path / "final", "C0", fc.test_rows(src), src)


def test_there_is_no_k_flag_the_controls_always_have_eight_samples(tmp_path):
    with pytest.raises(SystemExit) as e:
        fc.main(["--final", str(tmp_path / "final"), "--manifest", str(tmp_path / "m"), "--k", "4"])
    assert e.value.code == 2 and fc.K == 8


def world(tmp_path, monkeypatch):
    """Control sources, synthetic Test files with their manifest, a fake scorer and a fake best_constant."""
    for name, source in {"C0": "c0", "C1": "c1", "C2": "c2"}.items():
        write(tmp_path / source / "samples.jsonl", rows(model=f"control_{source}"))
    monkeypatch.setattr(fc, "SOURCES", {"C0": str(tmp_path / "c0"), "C1": str(tmp_path / "c1"), "C2": str(tmp_path / "c2")})
    data = tmp_path / "data"
    data.mkdir()
    for split in ("test_t", "test_r"):
        (data / f"{split}.jsonl").write_text(f"synthetic {split}\n")
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text("".join(f"{hashlib.sha256((data / f'{s}.jsonl').read_bytes()).hexdigest()}  processed/{s}.jsonl\n" for s in ("test_t", "test_r")))
    scored = []
    monkeypatch.setattr(fc, "score", lambda out, data_: scored.append(out.name))

    class FakeBest:
        calls = []

        @staticmethod
        def main(argv):
            FakeBest.calls.append(argv)
            out = Path(argv[argv.index("--out") + 1])
            assert argv[argv.index("--splits") + 1] == "test_t,test_r"          # the C3 answer is made for the Test rows only
            write(out / "samples.jsonl", rows(("test_t", "test_r"), model="control_constant_pool"))
            (out / "meta.json").write_text(json.dumps({"pool_sha256": "abc", "train_score": 0.19, "source": "pool", "pool_winner": {"chembl_id": "CHEMBL1", "conn": "KEY"}}))

    real = fc.load_script
    monkeypatch.setattr(fc, "load_script", lambda name: FakeBest if name == "best_constant" else real(name))
    argv = ["--final", str(tmp_path / "final"), "--data", str(data), "--manifest", str(manifest)]
    return argv, data, manifest, scored, FakeBest


def test_main_builds_all_four_controls_with_their_model_and_c3s_pool(tmp_path, monkeypatch):
    argv, data, manifest, scored, best = world(tmp_path, monkeypatch)
    fc.main(argv)
    assert scored == ["C0", "C1", "C2", "C3"]
    metas = {c: json.loads((tmp_path / "final" / c / "meta.json").read_text()) for c in ("C0", "C1", "C2", "C3")}
    assert metas["C3"]["pool_sha256"] == "abc" and metas["C3"]["pool_winner"] == {"chembl_id": "CHEMBL1", "conn": "KEY"}
    assert [metas[c]["model"] for c in ("C0", "C1", "C2", "C3")] == ["control_c0", "control_c1", "control_c2", "control_constant_pool"]
    assert all(m["k"] == 8 and m["rows"] == 4 and m["splits"] == ["test_t", "test_r"] and m["source_sha256"] for m in metas.values())
    assert len((tmp_path / "final" / "C1" / "samples.jsonl").read_text().splitlines()) == 4 * 8


def test_the_test_files_must_be_the_manifests_before_anything_is_read_or_written(tmp_path, monkeypatch):
    argv, data, manifest, scored, best = world(tmp_path, monkeypatch)
    (data / "test_t.jsonl").write_text("another file\n")
    with pytest.raises(SystemExit, match="test_t.jsonl differs from its sha256"):
        fc.main(argv)
    (data / "test_t.jsonl").unlink()
    with pytest.raises(SystemExit, match="test_t.jsonl is missing"):
        fc.main(argv)
    (data / "test_t.jsonl").write_text("synthetic test_t\n")
    manifest.write_text(manifest.read_text().replace("processed/test_r.jsonl", "processed/other.jsonl"))
    with pytest.raises(SystemExit, match="processed/test_r.jsonl has no entry"):
        fc.main(argv)
    assert not scored and not best.calls and not (tmp_path / "final").exists()


def test_the_manifest_is_a_required_argument(tmp_path, monkeypatch):
    argv, *_ = world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        fc.main([a for i, a in enumerate(argv) if a != "--manifest" and argv[i - 1] != "--manifest"])
    assert e.value.code == 2
