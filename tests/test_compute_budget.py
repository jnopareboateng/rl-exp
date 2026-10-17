import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compute_budget", ROOT / "scripts" / "compute_budget.py")
cb = importlib.util.module_from_spec(spec)
sys.modules["compute_budget"] = cb
spec.loader.exec_module(cb)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


def jl(rows):
    return "".join(json.dumps(r) + "\n" for r in rows)


@pytest.fixture
def runs(tmp_path):
    r = tmp_path / "runs"
    write(r / "sft" / "A2" / "summary.json", {"status": "complete", "global_step": 129, "training_seconds": 540.0})
    write(r / "rl" / "R1" / "metrics.jsonl", jl([{"seconds_step": 60.0}, {"seconds_step": 70.0}]))
    write(r / "rl" / "R1" / "evals.jsonl", jl([{"seconds": 30.0}, {"seconds": 20.0}]))
    write(r / "rl" / "R1" / "summary.json", {"status": "complete"})
    write(r / "colab-s1" / "a2-val" / "meta.json", {"wall_time_s": 100.0, "calls": [{"wall_time_s": 40.0}, {"wall_time_s": 20.0}]})
    write(r / "colab-s1" / "name-4b" / "meta.json", {"wall_time_s": 50.0, "calls": [{"wall_time_s": 10.0}]})
    write(r / "colab-s1" / "val8-control_constant" / "samples.jsonl", "")
    write(r / "final" / "A2" / "meta.json", {"wall_time_s": 200.0, "calls": [{"wall_time_s": 150.0}]})
    return r


def test_training_rows_use_steps_and_evals(runs):
    result = cb.build(runs, runs / "none.json")
    by = {t["label"]: t for t in result["training"]}
    assert by["SFT A2 (answer-only, 3 epochs)"]["seconds"] == 540.0
    rl = by["RL-full pilot R1 (40 steps)"]
    assert rl["steps"] == 2 and rl["step_seconds"] == 130.0 and rl["eval_seconds"] == 50.0 and rl["seconds"] == 180.0 and rl["status"] == "complete"
    assert len(result["training"]) == 2                               # runs without a record are skipped, never invented


def test_sampling_is_grouped_and_missing_meta_is_listed(runs):
    result = cb.build(runs, runs / "none.json")
    s = result["sampling"]
    assert s["A2 Val sampling"] == {"runs": 1, "wall_seconds": 100.0, "generation_seconds": 60.0}
    assert s["Diagnostics: identification and name-given"]["runs"] == 1
    assert s["Locked Test evaluation"]["wall_seconds"] == 200.0
    assert result["runs_without_meta"] == ["colab-s1/val8-control_constant"]
    assert result["totals_hours"]["sampling_wall"] == pytest.approx(350.0 / 3600)


def test_sessions_are_deterministic_and_open_ones_end_at_as_of(runs, tmp_path):
    sess = {"as_of": "2026-10-17T12:00:00Z", "runtimes": [{"start": "2026-10-17T10:00:00Z", "end": "2026-10-17T11:00:00Z", "gpu": "A100", "note": "lost"},
                                                          {"start": "2026-10-17T11:30:00Z", "end": None, "gpu": "A100"}]}
    write(tmp_path / "s.json", sess)
    a = cb.build(runs, tmp_path / "s.json")
    b = cb.build(runs, tmp_path / "s.json")
    assert a == b
    assert [round(x["hours"], 3) for x in a["runtime_sessions"]["runtimes"]] == [1.0, 0.5]


def test_markdown_and_cli_write_files(runs, tmp_path, capsys):
    out, md = tmp_path / "o.json", tmp_path / "o.md"
    cb.main(["--runs", str(runs), "--sessions", str(tmp_path / "none.json"), "--out", str(out), "--md", str(md)])
    text = md.read_text()
    assert "SFT A2" in text and "Total logged" in text and "without a meta.json" in text
    assert json.loads(out.read_text())["totals_hours"]["training"] == pytest.approx(720.0 / 3600)
