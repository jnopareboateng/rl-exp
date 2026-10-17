import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("answer_diversity", ROOT / "scripts" / "answer_diversity.py")
ad = importlib.util.module_from_spec(spec)
sys.modules["answer_diversity"] = ad
spec.loader.exec_module(ad)

ASPIRIN, CAFFEINE = "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(drug, answer, valid=True, finish="stop", split="val"):
    return {"split": split, "group": drug, "answer": answer, "valid": valid, "closed": True, "finish_reason": finish}


def test_one_answer_for_everything_is_fully_concentrated():
    agg = ad.diversity([("a", ASPIRIN), ("b", ASPIRIN), ("b", ASPIRIN)], ASPIRIN)
    assert agg["distinct_structures"] == 1 and agg["top_structure_share"] == 1.0
    assert agg["drugs_with_top_structure_as_mode"] == 1.0
    assert agg["mean_tanimoto_to_constant"] == pytest.approx(1.0) and agg["mean_tanimoto_between_drugs"] == pytest.approx(1.0)


def test_distinct_answers_per_drug_lower_the_concentration():
    agg = ad.diversity([("a", ASPIRIN), ("a", ASPIRIN), ("b", CAFFEINE), ("b", CAFFEINE)], ASPIRIN)
    assert agg["distinct_structures"] == 2 and agg["top_structure_share"] == 0.5
    assert agg["drugs_with_top_structure_as_mode"] == 0.5
    assert agg["mean_tanimoto_between_drugs"] < 0.5 and agg["mean_tanimoto_to_constant"] < 1.0


def test_only_usable_answers_count(tmp_path):
    path = tmp_path / "scored.v3.jsonl"
    write(path, [row("a", ASPIRIN), row("a", ASPIRIN, valid=False), row("b", ASPIRIN, finish="length"), row("c", ASPIRIN, split="test_t")])
    assert ad.usable_answers(path, "val") == [("a", ASPIRIN)]


def test_constant_answer_must_be_one_structure(tmp_path):
    path = tmp_path / "scored.v3.jsonl"
    write(path, [row("a", ASPIRIN), row("b", CAFFEINE)])
    with pytest.raises(ValueError):
        ad.constant_answer(path, "val")


def test_nothing_usable_is_reported_not_crashed():
    assert ad.diversity([], ASPIRIN) == {"n_usable": 0}


def test_cli_prints_aggregates_without_answer_text(tmp_path, capsys):
    runs = tmp_path / "runs"
    write(runs / "c" / "scored.v3.jsonl", [row("a", ASPIRIN), row("b", ASPIRIN)])
    write(runs / "m" / "scored.v3.jsonl", [row("a", ASPIRIN), row("b", CAFFEINE)])
    out = tmp_path / "out.json"
    ad.main(["--runs", str(runs), "--constant", "c", "--names", "m", "--json", str(out)])
    text = capsys.readouterr().out + out.read_text()
    assert "Cn1cnc2" not in text and "CC(=O)O" not in text
    assert json.loads(out.read_text())["runs"]["m"]["distinct_structures"] == 2
