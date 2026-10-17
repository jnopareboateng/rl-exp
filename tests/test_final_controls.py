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


def rows(splits=("val", "test_t", "test_r")):
    return [{"id": f"{s}-{i}", "group": f"d{i}", "split": s, "sample_idx": 0, "text": "x", "answer": "CCO"} for s in splits for i in range(2)]


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


def test_write_control_replicates_k_times_and_records_the_source_hash(tmp_path):
    src = tmp_path / "src.jsonl"
    write(src, rows(("test_t", "test_r")))
    out = fc.write_control(tmp_path / "final", "C0", fc.test_rows(src), 8, src, {"extra": 1})
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 4 * 8 and sorted({r["sample_idx"] for r in samples}) == list(range(8))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["source_sha256"] == fc.sha256(src) and meta["k"] == 8 and meta["extra"] == 1 and meta["rows"] == 4


def test_main_builds_all_four_controls(tmp_path, monkeypatch):
    for name, source in {"C0": "c0", "C1": "c1", "C2": "c2"}.items():
        write(tmp_path / source / "samples.jsonl", rows())
    monkeypatch.setattr(fc, "SOURCES", {"C0": str(tmp_path / "c0"), "C1": str(tmp_path / "c1"), "C2": str(tmp_path / "c2")})
    scored = []
    monkeypatch.setattr(fc, "score", lambda out, data: scored.append(out.name))

    class FakeBest:
        @staticmethod
        def main(argv):
            out = Path(argv[argv.index("--out") + 1])
            assert argv[argv.index("--splits") + 1] == "test_t,test_r"          # the C3 answer is made for the Test rows only
            write(out / "samples.jsonl", rows(("test_t", "test_r")))
            (out / "meta.json").write_text(json.dumps({"pool_sha256": "abc", "train_score": 0.19, "source": "pool"}))

    real = fc.load_script
    monkeypatch.setattr(fc, "load_script", lambda name: FakeBest if name == "best_constant" else real(name))
    fc.main(["--final", str(tmp_path / "final"), "--data", str(tmp_path / "data")])
    assert scored == ["C0", "C1", "C2", "C3"]
    assert json.loads((tmp_path / "final" / "C3" / "meta.json").read_text())["pool_sha256"] == "abc"
    assert len((tmp_path / "final" / "C1" / "samples.jsonl").read_text().splitlines()) == 4 * 8
