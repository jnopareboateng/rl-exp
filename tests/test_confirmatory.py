import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "confirmatory.py"


@pytest.fixture(scope="module")
def cf():
    spec = importlib.util.spec_from_file_location("confirmatory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEL = {"rl": "RL-full", "comparator": "C3"}


def test_commands_pair_the_chosen_arms_on_both_test_splits(cf):
    cmds = cf.commands(SEL, "runs/final", "data/processed", "results", boot=100, seed=3)
    assert set(cmds) == {"test_t", "test_r"}
    for split, cmd in cmds.items():
        assert cmd[cmd.index("--scored") + 1] == "runs/final/RL-full/scored.v3.jsonl" and cmd[cmd.index("--vs") + 1] == "runs/final/C3/scored.v3.jsonl"
        assert cmd[cmd.index("--split") + 1] == split and cmd[cmd.index("--boot") + 1] == "100" and cmd[cmd.index("--seed") + 1] == "3"
        assert cmd[cmd.index("--json") + 1] == f"results/paired_{split}.json"


def test_dry_run_prints_and_runs_nothing(cf, tmp_path, capsys):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SEL))
    cf.main(["--selection", str(sel), "--out", str(tmp_path / "r.json"), "--dry-run"])
    assert "paired_test_t.json" in capsys.readouterr().out and not (tmp_path / "r.json").exists()


def test_missing_scored_files_stop_with_a_hint(cf, tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SEL))
    with pytest.raises(SystemExit, match="rlexp.metrics score"):
        cf.main(["--selection", str(sel), "--final", str(tmp_path / "none"), "--out", str(tmp_path / "r.json")])


def test_a_run_collects_both_pairs_and_input_hashes(cf, tmp_path, monkeypatch):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SEL))
    for arm in ("RL-full", "C3"):
        (tmp_path / "final" / arm).mkdir(parents=True)
        (tmp_path / "final" / arm / "scored.v3.jsonl").write_text("{}\n")

    def fake_run(cmd, capture_output, text):
        Path(cmd[cmd.index("--json") + 1]).write_text(json.dumps({"split": cmd[cmd.index("--split") + 1]}))
        return type("R", (), {"returncode": 0, "stdout": "table", "stderr": ""})()

    monkeypatch.setattr(cf.subprocess, "run", fake_run)
    out = tmp_path / "res" / "confirmatory.json"
    out.parent.mkdir()
    cf.main(["--selection", str(sel), "--final", str(tmp_path / "final"), "--data", str(tmp_path), "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["paired"]["test_t"] == {"split": "test_t"} and res["paired"]["test_r"] == {"split": "test_r"}
    assert len(res["input_sha256"]) == 2 and (tmp_path / "res" / "paired_test_t.txt").read_text() == "table"
