import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_arms.py"
SHA = "a1" * 32


@pytest.fixture(scope="module")
def sa():
    spec = importlib.util.spec_from_file_location("select_arms", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(fv=None, mean=None, rows=58, drugs=20, cover=(58, 20)):
    """The Val block of a report.v3.json: `rows`/`drugs` are what fv_tanimoto covers, `cover` the panel (rows, drugs) of the data files (None: no --data)."""
    metrics = {}
    if fv is not None:
        metrics["fv_tanimoto"] = {"value": fv, "n_rows": rows, "n_drugs": drugs, "skipped_rows": 58 - rows}
    if mean is not None:
        metrics["mean_tanimoto"] = {"value": mean, "n_rows": 58, "n_drugs": 20, "skipped_rows": 0}
    val = {"n_rows": 58, "n_drugs": 20, "n_samples": 464, "metrics": metrics}
    if cover:
        val["coverage"] = {"rows": cover[0], "drugs": cover[1]}
    return {"splits": {"val": val}}


REPORTS = {"a-t03": report(0.150, 0.1), "a-t06": report(0.160, 0.1), "a-t10": report(0.160, 0.1), "c0": report(0.176, 0.176),
           "r-s20": report(0.170), "r-s40": report(0.189), "r-s40b": report(0.189),
           "mean-only": report(None, 0.18), "nan": report(float("nan"), 0.05), "k4": report(None, 0.048),
           "partial": report(0.2, rows=40, drugs=14), "gaps": report(0.2, rows=50, drugs=20), "no-cover": report(0.2, cover=None),
           "other-panel": report(0.2, cover=(60, 20)), "test-only": {"splits": {"test_t": report(0.9)["splits"]["val"]}}}


def get(name):
    return REPORTS[name]


def test_val_score_is_fv_tanimoto_and_nothing_else(sa):
    assert sa.val_score(REPORTS["a-t03"]) == 0.150 and sa.val_score(REPORTS["c0"]) == 0.176


@pytest.mark.parametrize("name,message", [
    ("mean-only", "no fv_tanimoto"),                        # a mean_tanimoto is never a stand-in
    ("k4", "no fv_tanimoto"),                               # a k<8 run has no fv_tanimoto
    ("nan", "no fv_tanimoto"),
    ("partial", "40 of 58 Val rows and 14 of 20 drugs"),    # a partial run
    ("gaps", "50 of 58 Val rows"),                          # some rows have fewer than 8 samples
    ("other-panel", "of 60 Val rows"),                      # the panel is larger than the run
    ("no-cover", "no coverage block"),                      # a report that cannot show completeness
])
def test_val_score_refuses_a_report_that_is_not_k8_complete(sa, name, message):
    with pytest.raises(ValueError, match=message):
        sa.val_score(REPORTS[name])


def test_no_test_number_can_reach_the_selection(sa):
    with pytest.raises(KeyError):
        sa.val_score(REPORTS["test-only"])


def test_pick_takes_the_best_score_and_breaks_ties_by_lower_temperature_then_checkpoint(sa):
    arm = {"candidates": [{"run": "a-t10", "temperature": 1.0}, {"run": "a-t06", "temperature": 0.6}, {"run": "a-t03", "temperature": 0.3}]}
    best = sa.pick(arm, get)
    assert (best["run"], best["temperature"], best["score"], best["metric"]) == ("a-t06", 0.6, 0.160, "fv_tanimoto")        # the tie with T=1.0 goes to the lower temperature
    assert [c["run"] for c in best["candidates"]] == ["a-t06", "a-t10", "a-t03"]
    rl = sa.pick({"candidates": [{"run": "r-s40b", "temperature": 0.6, "checkpoint": "step-40"}, {"run": "r-s40", "temperature": 0.6, "checkpoint": "step-30"},
                                 {"run": "r-s20", "temperature": 0.6, "checkpoint": "s20"}]}, get)
    assert rl["run"] == "r-s40" and rl["checkpoint"] == "step-30"                                         # equal score and temperature: the earlier checkpoint
    with pytest.raises(ValueError):
        sa.pick({"candidates": []}, get)


def test_a_candidate_without_a_usable_val_score_is_an_error_not_a_dropout(sa):
    arm = {"candidates": [{"run": "a-t06", "temperature": 0.6}, {"run": "k4", "temperature": 0.3}, {"run": "a-t03", "temperature": 0.3}]}
    with pytest.raises(ValueError, match="candidate k4: .*no fv_tanimoto"):
        sa.pick(arm, get)


def test_a_candidate_on_another_metric_never_competes_with_fv_candidates(sa):
    spec = {"arms": {"M": {"candidates": [{"run": "mean-only"}]}, "F": {"candidates": [{"run": "r-s20"}]}}, "comparator_set": ["M", "F"], "rl_set": ["F"]}
    with pytest.raises(ValueError, match="no fv_tanimoto"):          # 0.18 on mean_tanimoto would beat 0.170 on fv_tanimoto if it were let in
        sa.select(spec, get)


def test_select_picks_comparator_and_rl_arm_and_validates_the_sets(sa):
    spec = {"arms": {"A": {"candidates": [{"run": "a-t06", "temperature": 0.6}]}, "C0": {"candidates": [{"run": "c0"}]},
                     "RL": {"candidates": [{"run": "r-s40", "temperature": 0.6, "checkpoint": "step-40"}, {"run": "r-s20", "temperature": 0.6, "checkpoint": "step-20"}]}},
            "comparator_set": ["A", "C0"], "rl_set": ["RL"]}
    out = sa.select(spec, get)
    assert out["comparator"] == "C0" and out["rl"] == "RL" and out["arms"]["RL"]["run"] == "r-s40"
    with pytest.raises(ValueError, match="without candidates"):
        sa.select({**spec, "rl_set": ["nope"]}, get)
    with pytest.raises(ValueError, match="rl_set is empty"):
        sa.select({**spec, "rl_set": []}, get)


def test_a_tie_between_arms_goes_to_the_smaller_arm_id_whatever_the_order(sa):
    arms = {n: {"candidates": [{"run": "r-s40"}]} for n in ("B", "A", "C")}
    for order in (["B", "A", "C"], ["C", "B", "A"], ["A", "B", "C"]):
        out = sa.select({"arms": arms, "comparator_set": order, "rl_set": order}, get)
        assert out["comparator"] == "A" and out["rl"] == "A"
    better = {"A": {"candidates": [{"run": "a-t06"}]}, "B": {"candidates": [{"run": "r-s40"}]}, "C": {"candidates": [{"run": "c0"}]}}      # 0.160, 0.189, 0.176
    assert sa.select({"arms": better, "comparator_set": ["A", "B", "C"], "rl_set": ["A"]}, get)["comparator"] == "B"      # a strictly better score wins, whatever the id


@pytest.mark.parametrize("checkpoint,step", [("step-40", 40), ("s7", 7), ("step-0", 0), (None, 0), ("", 0)])
def test_step_of_reads_step_n_and_s_n(sa, checkpoint, step):
    assert sa.step_of({"checkpoint": checkpoint}) == step and sa.step_of({}) == 0


@pytest.mark.parametrize("checkpoint", ["rl2/step-40", "checkpoint-7-of-40", "step40", "step-4-0", "epoch-1", "step-", "s", "step-40.5", " step-40", 40])
def test_step_of_refuses_anything_it_would_have_to_guess(sa, checkpoint):
    with pytest.raises(ValueError, match="neither step-<n> nor s<n>"):
        sa.step_of({"checkpoint": checkpoint})


@pytest.mark.parametrize("run", ["../x", "a/b", "..", ".", "a..b", "", "a b", "a\\b", "a:b", None, 7])
def test_run_names_must_be_plain_directory_names(sa, run):
    def never(_):
        pytest.fail("the report of a bad run name must not be opened")

    with pytest.raises(ValueError, match="not a plain run directory name"):
        sa.pick({"candidates": [{"run": run}]}, never)
    assert sa.pick({"candidates": [{"run": "r-s40"}]}, get)["run"] == "r-s40"


@pytest.mark.parametrize("run", ["a2-t0.6", "val8-control_constant", "x_y.z-1", "A.B", ".x", "_x", "-x"])
def test_the_names_of_the_real_runs_pass_the_name_check(sa, run):
    assert sa.pick({"candidates": [{"run": run}]}, lambda _: REPORTS["r-s40"])["run"] == run


def test_the_winning_candidates_adapter_hash_is_carried_into_the_arm_record(sa):
    arm = {"candidates": [{"run": "r-s40", "checkpoint": "step-40", "adapter_sha256": SHA}, {"run": "r-s20", "checkpoint": "step-20", "adapter_sha256": "b2" * 32}]}
    best = sa.pick(arm, get)
    assert best["run"] == "r-s40" and best["adapter_sha256"] == SHA
    assert "adapter_sha256" not in sa.pick({"candidates": [{"run": "r-s40"}]}, get)
    for bad in ("abc", SHA.upper(), SHA + "0", 7):
        with pytest.raises(ValueError, match="adapter_sha256"):
            sa.pick({"candidates": [{"run": "r-s40", "adapter_sha256": bad}]}, get)


SPEC = {"mode": "nothink", "max_tokens": 256, "data": "direct", "system": "direct"}


def test_an_arms_spec_block_is_copied_into_the_selection_and_an_arm_without_one_gets_none(sa):
    spec = {"arms": {"A": {"spec": SPEC, "candidates": [{"run": "a-t06", "temperature": 0.6}]}, "C0": {"candidates": [{"run": "c0"}]},
                     "RL": {"spec": {**SPEC, "mode": "think"}, "candidates": [{"run": "r-s40", "temperature": 0.6, "checkpoint": "step-40"}]}},
            "comparator_set": ["A", "C0"], "rl_set": ["RL"]}
    out = sa.select(spec, get)
    assert out["arms"]["A"]["spec"] == SPEC and out["arms"]["RL"]["spec"] == {**SPEC, "mode": "think"} and "spec" not in out["arms"]["C0"]
    assert out["arms"]["A"]["run"] == "a-t06"                                             # the rest of the record is unchanged


@pytest.mark.parametrize("block", [{}, {**SPEC, "extra": 1}, {k: v for k, v in SPEC.items() if k != "system"}, {**SPEC, "mode": "auto"}, {**SPEC, "max_tokens": 0},
                                   {**SPEC, "max_tokens": "256"}, {**SPEC, "max_tokens": True}, {**SPEC, "max_tokens": 256.0}, {**SPEC, "data": ""}, {**SPEC, "system": None}, "nothink", [SPEC]])
def test_a_malformed_spec_block_is_refused(sa, block):
    spec = {"arms": {"A": {"spec": block, "candidates": [{"run": "a-t06", "temperature": 0.6}]}}, "comparator_set": ["A"], "rl_set": ["A"]}
    with pytest.raises(ValueError, match="arm A: spec must be exactly"):
        sa.select(spec, get)


def test_cli_reads_reports_from_run_directories(sa, tmp_path, capsys):
    runs = tmp_path / "runs"
    for name, rep in (("x", report(0.2)), ("y", report(0.1)), ("short", report(None, 0.3))):
        (runs / name).mkdir(parents=True)
        (runs / name / "report.v3.json").write_text(json.dumps(rep))
    cand = tmp_path / "c.json"
    cand.write_text(json.dumps({"arms": {"X": {"candidates": [{"run": "x"}]}, "Y": {"candidates": [{"run": "y"}]}}, "comparator_set": ["X"], "rl_set": ["Y"]}))
    sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(tmp_path / "out" / "sel.json")])
    assert json.loads((tmp_path / "out" / "sel.json").read_text())["comparator"] == "X"
    assert json.loads(capsys.readouterr().out)["scores"] == {"X": 0.2, "Y": 0.1}
    with pytest.raises(SystemExit):
        sa.main(["--candidates", str(cand), "--runs", str(tmp_path / "missing"), "--out", str(tmp_path / "o.json")])
    cand.write_text(json.dumps({"arms": {"X": {"spec": SPEC, "candidates": [{"run": "x"}]}, "Y": {"candidates": [{"run": "y"}]}}, "comparator_set": ["X"], "rl_set": ["Y"]}))
    sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(tmp_path / "o3.json")])
    assert json.loads((tmp_path / "o3.json").read_text())["arms"]["X"]["spec"] == SPEC                  # through the CLI as well
    cand.write_text(json.dumps({"arms": {"X": {"candidates": [{"run": "x"}, {"run": "short"}]}, "Y": {"candidates": [{"run": "y"}]}}, "comparator_set": ["X"], "rl_set": ["Y"]}))
    with pytest.raises(SystemExit, match="candidate short"):
        sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(tmp_path / "o2.json")])
    assert not (tmp_path / "o2.json").exists()                                      # nothing is written for an incomplete selection
