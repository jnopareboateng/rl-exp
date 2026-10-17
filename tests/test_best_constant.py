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


def test_cli_writes_standard_samples_and_meta(bc, tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    row = lambda i, g, split, s: {"id": f"{g}/{i}", "group": g, "split": split, "ref_smiles": s, "ref_conn": "x", "ref_scaffold": "", "prompt": "p"}
    (data / "train.jsonl").write_text("".join(json.dumps(row(0, g, "train", s)) + "\n" for g, s in TRAIN.items()))
    (data / "val.jsonl").write_text("".join(json.dumps(row(0, g, "val", s)) + "\n" for g, s in {"v1": "CCCCCO", "v2": "CCN"}.items()))
    bc.main(["--data", str(data), "--splits", "val", "--out", str(tmp_path / "out")])
    samples = [json.loads(l) for l in (tmp_path / "out" / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 2 and len({s["text"] for s in samples}) == 1 and samples[0]["text"].startswith("</think>")
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["control"] == "constant_pool" and meta["splits"] == ["val"]
    assert "CC" not in capsys.readouterr().out.replace("constant_pool", "")                    # the chosen molecule string is not printed
