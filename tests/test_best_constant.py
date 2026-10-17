import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "best_constant.py"


@pytest.fixture(scope="module")
def bc():
    spec = importlib.util.spec_from_file_location("best_constant", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = {"d1": "CCO", "d2": "CCCO", "d3": "CCCCO", "d4": "c1ccccc1O"}      # three small alcohols and a phenol


def test_choose_prefers_the_candidate_most_similar_to_the_train_references_and_never_scores_a_train_entry_against_itself(bc):
    smiles, score, source = bc.choose(TRAIN)
    assert source == "train" and smiles in {"CCCO", "CCCCO"}                # the middle alcohols are closest to the others
    assert 0 < score < 1                                                    # a Train candidate is not compared with its own entry (that would add a 1.0)
    pooled, pscore, psource = bc.choose(TRAIN, ["CCCCCO"])
    assert psource in {"train", "pool"} and pscore >= score


def test_a_pool_candidate_wins_only_when_it_is_better_and_ties_go_to_the_smaller_smiles(bc):
    smiles, score, source = bc.choose({"a": "CCO", "b": "CCCO"}, ["CCCO", "CCO"])
    assert smiles == "CCCO" or smiles == "CCO"
    again = bc.choose({"a": "CCO", "b": "CCCO"}, ["CCO", "CCCO"])
    assert again[0] == smiles                                              # candidate order does not matter


def test_invalid_pool_entries_are_skipped(bc):
    assert bc.choose(TRAIN, ["not a smiles", "C1CC"])[1] > 0


def make_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    row = lambda i, g, split, s: {"id": f"{g}/{i}", "group": g, "split": split, "ref_smiles": s, "ref_conn": "x", "ref_scaffold": "", "prompt": "p"}
    (data / "train.jsonl").write_text("".join(json.dumps(row(0, g, "train", s)) + "\n" for g, s in TRAIN.items()))
    (data / "val.jsonl").write_text("".join(json.dumps(row(0, g, "val", s)) + "\n" for g, s in {"v1": "CCCCCO", "v2": "CCN"}.items()))
    return data


def make_pool(tmp_path, rows, record=True, digest=None):
    """A pool.jsonl and the manifest.json beside it as scripts/build_pool.py writes it (derived.files["pool.jsonl"].sha256)."""
    pool = tmp_path / "chembl_pool"
    pool.mkdir()
    (pool / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if record:
        recorded = digest or hashlib.sha256((pool / "pool.jsonl").read_bytes()).hexdigest()
        (pool / "manifest.json").write_text(json.dumps({"derived": {"files": {"pool.jsonl": {"records": len(rows), "sha256": recorded}}}}))
    return pool / "pool.jsonl"


POOL_ROWS = [{"smiles": "CCCO", "chembl_id": "CHEMBL42", "conn": "ABCDEFGHIJKLMN"}, {"smiles": "not a smiles", "chembl_id": "CHEMBL43", "conn": "NOPQRSTUVWXYZA"}]


def run(bc, tmp_path, pool=None, out="out"):
    args = ["--data", str(make_data(tmp_path) if not (tmp_path / "data").exists() else tmp_path / "data"), "--splits", "val", "--out", str(tmp_path / out)]
    bc.main(args + (["--pool", str(pool)] if pool else ["--pool", str(tmp_path / "no-pool" / "pool.jsonl")]))
    return json.loads((tmp_path / out / "meta.json").read_text())


def test_cli_writes_standard_samples_and_meta(bc, tmp_path, capsys):
    meta = run(bc, tmp_path)
    samples = [json.loads(l) for l in (tmp_path / "out" / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 2 and len({s["text"] for s in samples}) == 1 and samples[0]["text"].startswith("</think>")
    assert meta["control"] == "constant_pool" and meta["splits"] == ["val"] and meta["source"] == "train"
    assert meta["pool_sha256"] is None and "pool_winner" not in meta                       # no pool file: Train only
    assert "CC" not in capsys.readouterr().out.replace("constant_pool", "")                    # the chosen molecule string is not printed


def test_a_verified_pool_is_used_and_its_hash_and_the_winners_id_are_recorded(bc, tmp_path, capsys):
    pool = make_pool(tmp_path, POOL_ROWS)
    meta = run(bc, tmp_path, pool)
    assert meta["source"] == "pool" and meta["candidates"] == len(TRAIN) + 2               # every pool row with a SMILES counts; the unparseable one is skipped when scored
    assert meta["pool_sha256"] == hashlib.sha256(pool.read_bytes()).hexdigest()
    assert meta["pool_winner"] == {"chembl_id": "CHEMBL42", "conn": "ABCDEFGHIJKLMN"}
    printed = capsys.readouterr().out
    assert "CHEMBL42" in printed and "CCCO" not in printed                                  # the id may be shown, the structure never


def test_a_pool_that_differs_from_the_recorded_hash_is_refused(bc, tmp_path):
    pool = make_pool(tmp_path, POOL_ROWS)
    pool.write_text(pool.read_text() + json.dumps({"smiles": "CCCCCCCO", "chembl_id": "CHEMBL44", "conn": "X"}) + "\n")       # a row added after build_pool.py
    with pytest.raises(ValueError, match="differs from the sha256 recorded"):
        run(bc, tmp_path, pool)
    assert not (tmp_path / "out" / "samples.jsonl").exists()


def test_a_pool_with_no_recorded_hash_is_refused(bc, tmp_path):
    with pytest.raises(ValueError, match="no recorded sha256"):
        run(bc, tmp_path, make_pool(tmp_path, POOL_ROWS, record=False))
    other = tmp_path / "other"
    other.mkdir()
    pool = make_pool(other, POOL_ROWS, record=False)
    (pool.parent / "manifest.json").write_text(json.dumps({"files": {}}))                    # a manifest without the derived record
    with pytest.raises(ValueError, match="no recorded sha256"):
        run(bc, other, pool)
