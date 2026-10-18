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


SPEC = {"model": "Qwen/Qwen3.5-4B", "mode": "nothink", "max_tokens": 256, "data": "direct", "system": "direct"}


def test_an_arms_spec_block_is_copied_into_the_selection_and_an_arm_without_one_gets_none(sa):
    spec = {"arms": {"A": {"spec": SPEC, "candidates": [{"run": "a-t06", "temperature": 0.6}]}, "C0": {"candidates": [{"run": "c0"}]},
                     "RL": {"spec": {**SPEC, "mode": "think"}, "candidates": [{"run": "r-s40", "temperature": 0.6, "checkpoint": "step-40"}]}},
            "comparator_set": ["A", "C0"], "rl_set": ["RL"]}
    out = sa.select(spec, get)
    assert out["arms"]["A"]["spec"] == SPEC and out["arms"]["RL"]["spec"] == {**SPEC, "mode": "think"} and "spec" not in out["arms"]["C0"]
    assert out["arms"]["A"]["run"] == "a-t06"                                             # the rest of the record is unchanged


@pytest.mark.parametrize("block", [{}, {**SPEC, "extra": 1}, {k: v for k, v in SPEC.items() if k != "system"}, {k: v for k, v in SPEC.items() if k != "model"}, {**SPEC, "mode": "auto"},
                                   {**SPEC, "max_tokens": 0}, {**SPEC, "max_tokens": "256"}, {**SPEC, "max_tokens": True}, {**SPEC, "max_tokens": 256.0}, {**SPEC, "data": ""},
                                   {**SPEC, "system": None}, {**SPEC, "model": ""}, {**SPEC, "model": 7}, "nothink", [SPEC],
                                   {**SPEC, "quantization": ""}, {**SPEC, "quantization": 8}, {**SPEC, "gpu_memory_utilization": 0}, {**SPEC, "gpu_memory_utilization": 1.5},
                                   {**SPEC, "gpu_memory_utilization": True}, {**SPEC, "gpu_memory_utilization": "0.9"}])
def test_a_malformed_spec_block_is_refused(sa, block):
    spec = {"arms": {"A": {"spec": block, "candidates": [{"run": "a-t06", "temperature": 0.6}]}}, "comparator_set": ["A"], "rl_set": ["A"]}
    with pytest.raises(ValueError, match="arm A: spec must be exactly"):
        sa.select(spec, get)


def test_the_block_may_name_a_quantization_and_a_memory_fraction_and_they_are_copied(sa):
    block = {**SPEC, "quantization": "fp8", "gpu_memory_utilization": 0.93}
    spec = {"arms": {"A": {"spec": block, "candidates": [{"run": "a-t06", "temperature": 0.6}]}}, "comparator_set": ["A"], "rl_set": ["A"]}
    assert sa.select(spec, get)["arms"]["A"]["spec"] == block
    spec["arms"]["A"]["spec"] = {**SPEC, "gpu_memory_utilization": 1}
    assert sa.select(spec, get)["arms"]["A"]["spec"]["gpu_memory_utilization"] == 1


# ------------------------------------------------------------------------------------------------ the winning run was sampled the way the block says
def settings(model="Qwen/Qwen3.5-4B", mode="nothink", temperature=0.6, max_tokens=256):
    return {"model": {model}, "mode": {mode}, "temperature": {temperature}, "max_tokens": {max_tokens}}


def arms_with_spec(**block):
    """A (untrained, T=0.6) and RL (adapter arm, T=0.6, merged path in its Val run) with their blocks; the winners are a-t06 and r-s40."""
    return {"arms": {"A": {"spec": {**SPEC, **block}, "candidates": [{"run": "a-t06", "temperature": 0.6}, {"run": "a-t03", "temperature": 0.3}]},
                     "RL": {"spec": {**SPEC, "mode": "think", "model": "final_models/RL"},
                            "candidates": [{"run": "r-s40", "temperature": 0.6, "checkpoint": "step-40", "adapter_sha256": SHA}]}},
            "comparator_set": ["A"], "rl_set": ["RL"]}


RUNS = {"a-t06": settings(), "a-t03": settings(temperature=0.3), "r-s40": settings("/content/merged_R1_s40", "think", 0.6)}


def test_a_winning_run_that_was_sampled_as_declared_passes_and_an_adapter_arms_merged_path_is_not_compared(sa):
    assert sa.select(arms_with_spec(), get, RUNS.__getitem__)["arms"]["A"]["run"] == "a-t06"


@pytest.mark.parametrize("field,seen,message", [
    ("mode", {"mode": {"think"}}, r"run a-t06 was sampled with mode \['think'\] but the candidates file says 'nothink'"),
    ("max_tokens", {"max_tokens": {512}}, r"max_tokens \[512\] but the candidates file says 256"),
    ("temperature", {"temperature": {1.0}}, r"temperature \[1.0\] but the candidates file says 0.6"),
    ("model", {"model": {"Qwen/Qwen3.5-9B"}}, r"model \['Qwen/Qwen3.5-9B'\] but the candidates file says 'Qwen/Qwen3.5-4B'"),
    ("model path", {"model": {"final_models/RL-full"}}, "but the candidates file says 'Qwen/Qwen3.5-4B'"),
    ("two temperatures", {"temperature": {0.6, 1.0}}, r"temperature \[0.6, 1.0\]"),
    ("two models", {"model": {"Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}}, "model"),
])
def test_a_winning_run_that_was_sampled_otherwise_is_refused_by_name(sa, field, seen, message):
    with pytest.raises(ValueError, match=message):
        sa.select(arms_with_spec(), get, lambda run: {**RUNS[run], **seen})


def test_the_samples_of_the_winning_run_only_are_compared_and_missing_fields_or_files_are_not(sa):
    only_winner = lambda run: {**RUNS[run], "mode": {"think"}} if run == "a-t03" else RUNS[run]          # a losing candidate may differ
    assert sa.select(arms_with_spec(), get, only_winner)["arms"]["A"]["run"] == "a-t06"
    assert sa.select(arms_with_spec(), get, lambda run: {})["arms"]["A"]["run"] == "a-t06"              # a run without a samples file
    assert sa.select(arms_with_spec(), get, lambda run: {"mode": {"think" if run == "r-s40" else "nothink"}})["arms"]["A"]["run"] == "a-t06"      # rows without the other fields
    assert sa.select(arms_with_spec(), get)["arms"]["A"]["run"] == "a-t06"                              # no loader: nothing is compared
    a_model_of_the_adapter_arm = lambda run: {**RUNS[run], "model": {"/content/some/other/merge"}} if run == "r-s40" else RUNS[run]
    assert sa.select(arms_with_spec(), get, a_model_of_the_adapter_arm)["arms"]["RL"]["run"] == "r-s40"
    with pytest.raises(ValueError, match="run r-s40 was sampled with mode"):                           # but its mode is
        sa.select(arms_with_spec(), get, lambda run: {**RUNS[run], "mode": {"nothink"}} if run == "r-s40" else RUNS[run])


def test_an_arm_without_a_block_is_not_compared(sa):
    spec = arms_with_spec()
    del spec["arms"]["A"]["spec"]
    assert sa.select(spec, get, lambda run: RUNS[run] if run == "r-s40" else {"mode": {"think"}, "temperature": {9.0}})["arms"]["A"]["run"] == "a-t06"


def test_run_settings_reads_the_distinct_values_of_a_samples_file(sa, tmp_path):
    (tmp_path / "r").mkdir()
    rows = [{"id": "x", "model": "m", "mode": "think", "temperature": 0.6, "max_tokens": 256, "text": "a"}, {"id": "y", "model": "m", "mode": "think", "temperature": 1.0, "text": "b"}, {"id": "z"}]
    (tmp_path / "r" / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    assert sa.run_settings(tmp_path, "r") == {"model": {"m"}, "mode": {"think"}, "temperature": {0.6, 1.0}, "max_tokens": {256}}
    assert sa.run_settings(tmp_path, "none") == {}


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
    cand.write_text(json.dumps({"arms": {"X": {"spec": SPEC, "candidates": [{"run": "x", "temperature": 0.6}]}, "Y": {"candidates": [{"run": "y"}]}}, "comparator_set": ["X"], "rl_set": ["Y"]}))
    row = {"id": "a", "model": "Qwen/Qwen3.5-4B", "mode": "nothink", "temperature": 0.6, "max_tokens": 256}
    (runs / "x" / "samples.jsonl").write_text(json.dumps(row) + "\n")
    sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(tmp_path / "o4.json")])              # as declared
    (runs / "x" / "samples.jsonl").write_text(json.dumps({**row, "model": "Qwen/Qwen3.5-9B"}) + "\n")
    with pytest.raises(SystemExit, match="run x was sampled with model"):
        sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(tmp_path / "o5.json")])
    assert not (tmp_path / "o5.json").exists()


# ------------------------------------------------------------------------------------------------ the committed selection is derived, not typed
ROOT = Path(__file__).resolve().parents[1]


def assert_selection_is_derived(sa, candidates_path, selection_path, runs):
    """selection_path holds the candidates' sets (equal, not a subset) and is the output of select_arms on candidates_path and the Val reports and samples under runs, byte for byte."""
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    committed = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    assert committed["comparator_set"] == candidates["comparator_set"] and committed["rl_set"] == candidates["rl_set"], "the selection's sets are not the candidates' sets"
    result = sa.select(candidates, lambda run: json.loads((Path(runs) / run / "report.v3.json").read_text(encoding="utf-8")), lambda run: sa.run_settings(runs, run))
    assert (json.dumps(result, indent=1) + "\n").encode("utf-8") == Path(selection_path).read_bytes(), "the selection is not what select_arms derives from the candidates"


def test_the_committed_selection_is_what_select_arms_derives_from_the_committed_candidates(sa):
    """A hand edit, a candidates file that moved on, or a new Val report that changes a pick fails here until select_arms is run again."""
    candidates, selection, runs = ROOT / "results" / "arm_candidates.json", ROOT / "results" / "selection.json", ROOT / "runs" / "colab-s1"
    if not (runs.is_dir() and candidates.is_file() and selection.is_file()):
        pytest.skip("needs results/arm_candidates.json, results/selection.json and the (untracked) runs/colab-s1")
    assert_selection_is_derived(sa, candidates, selection, runs)


def test_a_hand_edited_selection_is_noticed(sa, tmp_path):
    runs = tmp_path / "runs"
    for name, rep in (("a", report(0.16)), ("b", report(0.17)), ("c", report(0.15))):
        (runs / name).mkdir(parents=True)
        (runs / name / "report.v3.json").write_text(json.dumps(rep))
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"arms": {"A": {"candidates": [{"run": "a"}]}, "B": {"candidates": [{"run": "b"}]}, "C": {"candidates": [{"run": "c"}]}},
                                "comparator_set": ["A", "B"], "rl_set": ["C"]}))
    sel = tmp_path / "selection.json"
    sa.main(["--candidates", str(cand), "--runs", str(runs), "--out", str(sel)])
    assert_selection_is_derived(sa, cand, sel, runs)                                       # as written by the CLI: derived
    derived = json.loads(sel.read_text())
    for edit, message in ((lambda d: d.update(comparator="A"), "not what select_arms derives"), (lambda d: d["arms"]["A"].update(score=0.9), "not what select_arms derives"),
                          (lambda d: d.update(comparator_set=["A"]), "sets are not the candidates' sets"), (lambda d: d.update(rl_set=["C", "A"]), "sets are not the candidates' sets"),
                          (lambda d: d.update(comparator_set=["B", "A"]), "sets are not the candidates' sets")):
        edited = json.loads(json.dumps(derived))
        edit(edited)
        sel.write_text(json.dumps(edited, indent=1) + "\n")
        with pytest.raises(AssertionError, match=message):
            assert_selection_is_derived(sa, cand, sel, runs)
    cand.write_text(json.dumps({"arms": {"A": {"candidates": [{"run": "a"}]}, "B": {"candidates": [{"run": "b"}]}, "C": {"candidates": [{"run": "c"}]}},
                                "comparator_set": ["B", "A"], "rl_set": ["C"]}))              # the candidates' sets moved on (order counts: the selection copies them)
    sel.write_text(json.dumps(derived, indent=1) + "\n")
    with pytest.raises(AssertionError, match="sets are not the candidates' sets"):
        assert_selection_is_derived(sa, cand, sel, runs)
