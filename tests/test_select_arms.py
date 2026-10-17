import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_arms.py"


@pytest.fixture(scope="module")
def sa():
    spec = importlib.util.spec_from_file_location("select_arms", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(fv=None, mean=None):
    metrics = {}
    if fv is not None:
        metrics["fv_tanimoto"] = {"value": fv}
    if mean is not None:
        metrics["mean_tanimoto"] = {"value": mean}
    return {"splits": {"val": {"metrics": metrics}}}


REPORTS = {"a-t03": report(0.150, 0.1), "a-t06": report(0.160, 0.1), "a-t10": report(0.160, 0.1), "c0": report(None, 0.176),
           "r-s20": report(0.170), "r-s40": report(0.189), "r-s40b": report(0.189), "nan": {"splits": {"val": {"metrics": {"fv_tanimoto": {"value": float("nan")}, "mean_tanimoto": {"value": 0.05}}}}}}


def test_val_score_prefers_fv_then_mean_and_ignores_nan(sa):
    assert sa.val_score(REPORTS["a-t03"]) == (0.150, "fv_tanimoto")
    assert sa.val_score(REPORTS["c0"]) == (0.176, "mean_tanimoto")
    assert sa.val_score(REPORTS["nan"]) == (0.05, "mean_tanimoto")
    assert sa.val_score(report()) == (None, None)


def test_pick_takes_the_best_score_and_breaks_ties_by_lower_temperature_then_checkpoint(sa):
    arm = {"candidates": [{"run": "a-t10", "temperature": 1.0}, {"run": "a-t06", "temperature": 0.6}, {"run": "a-t03", "temperature": 0.3}]}
    best = sa.pick(arm, REPORTS.__getitem__)
    assert (best["run"], best["temperature"], best["score"]) == ("a-t06", 0.6, 0.160)                    # the tie with T=1.0 goes to the lower temperature
    assert [c["run"] for c in best["candidates"]] == ["a-t06", "a-t10", "a-t03"]
    rl = sa.pick({"candidates": [{"run": "r-s40b", "temperature": 0.6, "checkpoint": "step-40"}, {"run": "r-s40", "temperature": 0.6, "checkpoint": "step-30"},
                                 {"run": "r-s20", "temperature": 0.6, "checkpoint": "step-20"}]}, REPORTS.__getitem__)
    assert rl["run"] == "r-s40" and rl["checkpoint"] == "step-30"                                         # equal score and temperature: the earlier checkpoint
    with pytest.raises(ValueError):
        sa.pick({"candidates": []}, REPORTS.__getitem__)


def test_select_picks_comparator_and_rl_arm_and_validates_the_sets(sa):
    spec = {"arms": {"A": {"candidates": [{"run": "a-t06", "temperature": 0.6}]}, "C0": {"candidates": [{"run": "c0"}]},
                     "RL": {"candidates": [{"run": "r-s40", "temperature": 0.6, "checkpoint": "step-40"}, {"run": "r-s20", "temperature": 0.6, "checkpoint": "step-20"}]}},
            "comparator_set": ["A", "C0"], "rl_set": ["RL"]}
    out = sa.select(spec, REPORTS.__getitem__)
    assert out["comparator"] == "C0" and out["rl"] == "RL" and out["arms"]["RL"]["run"] == "r-s40"
    with pytest.raises(ValueError, match="without candidates"):
        sa.select({**spec, "rl_set": ["nope"]}, REPORTS.__getitem__)


def test_cli_reads_reports_from_run_directories(sa, tmp_path, capsys):
    runs = tmp_path / "runs"
    for name, rep in (("x", report(0.2)), ("y", report(0.1))):
        (runs / name).mkdir(parents=True)
        (runs / name / "report.v3.json").write_text(json.dumps(rep))
    cand = tmp_path / "c.json"
    cand.write_text(json.dumps({"arms": {"X": {"candidates": [{"run": "x"}]}, "Y": {"candidates": [{"run": "y"}]}}, "comparator_set": ["X"], "rl_set": ["Y"]}))
    sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(tmp_path / "out" / "sel.json")])
    assert json.loads((tmp_path / "out" / "sel.json").read_text())["comparator"] == "X"
    assert json.loads(capsys.readouterr().out)["scores"] == {"X": 0.2, "Y": 0.1}
    with pytest.raises(SystemExit):
        sa.main(["--candidates", str(cand), "--runs", str(tmp_path / "missing"), "--out", str(tmp_path / "o.json")])
