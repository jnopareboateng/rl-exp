import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replicate_samples.py"


@pytest.fixture(scope="module")
def rs():
    spec = importlib.util.spec_from_file_location("replicate_samples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_row_is_repeated_k_times_with_indices_and_nothing_else_changes(rs):
    rows = [{"id": "a", "sample_idx": 0, "text": "t1"}, {"id": "b", "sample_idx": 0, "text": "t2"}]
    out = rs.replicate(rows, 3)
    assert [(r["id"], r["sample_idx"]) for r in out] == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]
    assert all(r["text"] == {"a": "t1", "b": "t2"}[r["id"]] for r in out)


def test_input_with_several_samples_per_row_is_refused(rs):
    with pytest.raises(ValueError, match="several samples"):
        rs.replicate([{"id": "a", "sample_idx": 1}], 2)
    with pytest.raises(ValueError, match="several samples"):
        rs.replicate([{"id": "a", "sample_idx": 0}, {"id": "a", "sample_idx": 1}], 2)


def test_a_repeated_id_at_index_zero_is_a_duplicate_pair_not_several_samples(rs):
    rows = [{"id": "a", "sample_idx": 0, "text": "t1"}, {"id": "b", "sample_idx": 0, "text": "t2"}, {"id": "a", "sample_idx": 0, "text": "t3"}]
    with pytest.raises(ValueError, match=r"row 3 repeats the \(id, sample_idx\) pair"):
        rs.replicate(rows, 8)


@pytest.mark.parametrize("row", [{"id": "a", "text": "t"}, {"id": "a", "sample_idx": None}, {"id": "a", "sample_idx": "0"}, {"id": "a", "sample_idx": 0.0},
                                 {"id": "a", "sample_idx": False}])
def test_a_row_without_an_integer_sample_idx_is_refused(rs, row):
    with pytest.raises(ValueError, match="row 2 has no integer sample_idx"):
        rs.replicate([{"id": "z", "sample_idx": 0}, row], 8)


def test_cli_roundtrip(rs, tmp_path, capsys):
    src = tmp_path / "s.jsonl"
    src.write_text(json.dumps({"id": "a", "sample_idx": 0, "text": "x"}) + "\n")
    rs.main(["--samples", str(src), "--k", "8", "--out", str(tmp_path / "o" / "s.jsonl")])
    assert len((tmp_path / "o" / "s.jsonl").read_text().splitlines()) == 8 and "8 per row" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rs.main(["--samples", str(src), "--k", "0", "--out", str(tmp_path / "x.jsonl")])
    src.write_text(json.dumps({"id": "a", "sample_idx": 0}) + "\n" + json.dumps({"id": "a", "sample_idx": 0}) + "\n")
    with pytest.raises(SystemExit, match="repeats the"):
        rs.main(["--samples", str(src), "--k", "8", "--out", str(tmp_path / "dup.jsonl")])
    assert not (tmp_path / "dup.jsonl").exists()
