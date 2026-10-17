import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def sr():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("strict_rescore", SCRIPTS / "strict_rescore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_text_keeps_a_smiles_only_answer_and_drops_a_sentence_embedded_one(sr):
    assert sr.strict_text("</think>\n\nCC(=O)Oc1ccccc1C(=O)O") == "</think>\nCC(=O)Oc1ccccc1C(=O)O"
    assert sr.strict_text("</think>\n\nThe answer is CC(=O)Oc1ccccc1C(=O)O.") == "</think>\n"
    assert sr.strict_text("no closing tag CCO") == "</think>\n"
    assert sr.strict_text("</think>\n\n**CC(=O)Oc1ccccc1C(=O)O**") == "</think>\n" + "CC(=O)Oc1ccccc1C(=O)O"


def test_rewrite_changes_only_the_text(sr):
    row = {"id": "a", "sample_idx": 0, "text": "</think>\n\nsome words CCO", "finish_reason": "stop", "n_tokens": 7}
    out = sr.rewrite([row])
    assert out == [{**row, "text": "</think>\n"}]


def test_cli_writes_the_rewritten_file(sr, tmp_path, capsys):
    src = tmp_path / "s.jsonl"
    src.write_text(json.dumps({"id": "a", "sample_idx": 0, "text": "</think>\n\nCCOCC"}) + "\n" + json.dumps({"id": "b", "sample_idx": 0, "text": "</think>\n\nwords only"}) + "\n")
    sr.main(["--samples", str(src), "--out", str(tmp_path / "o" / "s.jsonl")])
    rows = [json.loads(l) for l in (tmp_path / "o" / "s.jsonl").read_text().splitlines()]
    assert [r["text"] for r in rows] == ["</think>\nCCOCC", "</think>\n"] and "1 have no strict answer" in capsys.readouterr().out
