"""scripts/make_figures.py on small synthetic inputs: the data frames (lists of rows, built before anything is drawn), the refusal of Test inputs,
missing optional inputs, the table writers and a full run that must be byte-for-byte repeatable. No real data, no molecule strings."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"


@pytest.fixture(scope="module")
def mf():
    spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ref(split, year):
    return {"split": split, "approval_year": year}


REFS = [ref("train", 2019)] * 3 + [ref("train", 2012), ref("val", 2019), ref("test_r", 2021), ref("buffer", 2024), ref("test_t", 2025),
                                   ref("test_t", 2025), ref("test_t", 2026), ref("excluded", 2020), ref("excluded", None)]
REPORT = {"splits": {"train": {"drugs_by_year": {"<2015": 1, "2015-2021": 3, "2022": 0, "2024": 0}},
                     "val": {"drugs_by_year": {"<2015": 0, "2015-2021": 1}},
                     "test_r": {"drugs_by_year": {"2015-2021": 1}},
                     "buffer": {"drugs_by_year": {"2024": 1}},
                     "test_t": {"drugs_by_year": {"2025": 2, "2026": 1}},
                     "excluded": {"drugs_by_year": {"2015-2021": 1}}}}      # the excluded drug without a year is not in any bin


def trow(run, tani=None, tokens=None, fv_lift=None, **extra):
    return {"run": run, "label": f"label of {run}", "decoding": "T=1.0, k=8", "drugs": 20, "samples": 464, "prompt_tokens": 500.0, "valid_closed": 0.9,
            "fv_tanimoto": tani, "fv_tokens": tokens, "fv_lift": fv_lift, "fv_samples": 1.5, "lift": 0.01, "mean_tanimoto": 0.1, "core@1": 0.0, "exact@1": 0.0,
            **extra}


TABLE_ROWS = [trow("val8-control_constant_pool", 0.1865, 0.0, 0.0), trow("val8-control_copy_nearest", 0.174, 0.0, 0.0218),
              trow("val-control_constant", None, None, None), trow("rlfull-s20-t0.6", 0.168, 31.0, 0.006), trow("rlfull-s40-t0.6", 0.1887, 31.3, 0.008),
              trow("rlfull-s40-t1.0", 0.1869, 31.3, 0.0096), trow("a2-val", 0.1557, 61.6, 0.0196), trow("a1-val", None, None, None),
              trow("name-4b", 0.141, 120.0, 0.009), trow("mystery-run", 0.1, 10.0, 0.0), trow("only-lift", 0.12, None, 0.02)]


def table_file(path, split="val", rows=TABLE_ROWS):
    path.write_text(json.dumps({"split": split, "columns": [], "rows": rows, "constant": {}}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------------- F1
def test_f1_counts_each_year_and_split_and_leaves_excluded_drugs_out(mf):
    rows = mf.f1_rows(REFS, REPORT)
    cells = {(r["year"], r["split"]): r["drugs"] for r in rows}
    assert cells == {(2012, "train"): 1, (2019, "train"): 3, (2019, "val"): 1, (2021, "test_r"): 1, (2024, "buffer"): 1, (2025, "test_t"): 2,
                     (2026, "test_t"): 1}
    assert [r["year"] for r in rows] == sorted(r["year"] for r in rows)
    assert mf.split_totals(rows) == {"train": 4, "val": 1, "test_r": 1, "buffer": 1, "test_t": 3}


def test_f1_stops_when_refs_and_report_disagree_or_a_drawn_drug_has_no_year(mf):
    wrong = json.loads(json.dumps(REPORT))
    wrong["splits"]["train"]["drugs_by_year"]["2015-2021"] = 4
    with pytest.raises(ValueError, match="disagree for train 2015-2021"):
        mf.f1_rows(REFS, wrong)
    with pytest.raises(ValueError, match="no approval year"):
        mf.f1_rows(REFS + [ref("val", None)], REPORT)
    assert mf.f1_rows(REFS)                                                    # without a report there is nothing to cross-check


def test_year_bin_reads_the_report_keys(mf):
    assert mf.year_bin(2014, "<2015") and not mf.year_bin(2015, "<2015")
    assert mf.year_bin(2015, "2015-2021") and mf.year_bin(2021, "2015-2021") and not mf.year_bin(2022, "2015-2021")
    assert mf.year_bin(2022, "2022") and not mf.year_bin(2023, "2022")


# ---------------------------------------------------------------------------------------------------------------- F2 and F4
def test_f2_rows_keep_runs_with_both_values_and_label_the_best_run_of_each_main_arm(mf):
    rows = mf.f2_rows(TABLE_ROWS)
    runs = [r["run"] for r in rows]
    assert "val-control_constant" not in runs and "a1-val" not in runs and "only-lift" not in runs      # no first-valid value or no tokens
    assert "mystery-run" in runs
    by = {r["run"]: r for r in rows}
    assert by["val8-control_constant_pool"]["family"] == "control" and by["val8-control_constant_pool"]["arm"] == "Best constant answer (C3)"
    assert by["val8-control_constant_pool"]["fv_tokens"] == 0.0                                   # a CPU control generates nothing and is kept
    assert by["mystery-run"]["family"] == "other" and not by["mystery-run"]["labelled"]             # an unknown run is drawn, never labelled
    assert not by["name-4b"]["labelled"]                                                         # a diagnostic, not a main arm
    assert [r["run"] for r in rows if r["family"] == "rl" and r["labelled"]] == ["rlfull-s40-t0.6"]  # the best of the RL runs only
    assert by["a2-val"]["labelled"] and by["val8-control_copy_nearest"]["labelled"]
    assert list(rows[0]) == list(mf.F2_FIELDS)


def test_on_a_test_split_the_labels_use_no_outcome(mf):
    rows = mf.f2_rows(TABLE_ROWS, "test_t")
    assert [r["run"] for r in rows if r["family"] == "rl" and r["labelled"]] == ["rlfull-s20-t0.6", "rlfull-s40-t0.6", "rlfull-s40-t1.0"]


def test_f4_needs_lift_not_tokens_and_both_figures_say_which_runs_are_missing(mf):
    assert "only-lift" in [r["run"] for r in mf.f4_rows(TABLE_ROWS)] and "only-lift" not in [r["run"] for r in mf.f2_rows(TABLE_ROWS)]
    kept, omitted = mf.usable(TABLE_ROWS, "fv_tanimoto", "fv_tokens")
    assert omitted == ["val-control_constant", "a1-val", "only-lift"] and len(kept) == len(TABLE_ROWS) - 3
    assert [r["fv_lift"] for r in mf.f4_rows(TABLE_ROWS) if r["run"] == "val8-control_copy_nearest"] == [0.0218]
    assert mf.usable([{"run": "x", "fv_tanimoto": float("nan"), "fv_tokens": 1.0}], "fv_tanimoto", "fv_tokens")[1] == ["x"]    # NaN is missing


def test_the_constant_level_is_c3_first_valid_else_its_mean(mf):
    assert mf.constant_level(TABLE_ROWS) == 0.1865
    assert mf.constant_level([trow("val-control_constant_pool", None, None, None, mean_tanimoto=0.18)]) == 0.18
    assert mf.constant_level([trow("a2-val", 0.15, 1.0, 0.0)]) is None


# ---------------------------------------------------------------------------------------------------------------- F3
def test_moving_average_uses_up_to_five_values_and_ignores_none(mf):
    assert mf.moving_average([1, 2, 3, 4, 5, 6, 7]) == [1, 1.5, 2, 2.5, 3, 4, 5]
    assert mf.moving_average([None, 2, None, 4]) == [None, 2, 2, 3]
    assert mf.moving_average([None, None]) == [None, None]


def rl_run(fv, objective_differs, steps=7):
    metrics = [{"step": i, "reward_mean": 0.01 * i, "lift_batch": 0.001 * i, "collapse_score": None if i == 2 else 1.0, "top_answer_share": 0.01,
                **({"reward_full_mean": 0.1 + 0.01 * i} if objective_differs else {})} for i in range(1, steps + 1)]
    evals = [{"step": s, "metrics": {"validity": 0.5 + 0.1 * s / 10, "tanimoto": 0.08 + 0.01 * s / 10, **({"fv_tanimoto": 0.15} if fv else {})}} for s in (0, 10)]
    return {"metrics": metrics, "evals": evals, "label": "run"}


def test_f3_rows_carry_both_val_similarities_the_standard_training_reward_and_the_averages(mf):
    rows = mf.f3_rows({"R1": rl_run(False, False), "R3": rl_run(True, True)})
    series = {(r["run"], r["series"]) for r in rows}
    assert ("R1", "val_similarity_first_valid") not in series and ("R3", "val_similarity_first_valid") in series     # only a run with 8 samples has it
    assert {("R1", "val_similarity_all"), ("R3", "val_similarity_all"), ("R1", "val_validity")} <= series
    def reward(run, name):
        return [(r["step"], r["value"]) for r in rows if r["run"] == run and r["series"] == name]

    assert reward("R1", "train_reward")[0] == (1, 0.01)                                           # no standard reward logged: the objective is used
    assert reward("R3", "train_reward")[0] == (1, pytest.approx(0.11))                            # the standard reward, not the objective 0.01
    assert [r["source"] for r in rows if r["run"] == "R3" and r["series"] == "train_reward"] == ["reward_full_mean"] * 7
    assert reward("R1", "train_reward_avg5")[6] == (7, pytest.approx(0.05))                       # mean of steps 3 to 7
    assert 2 not in [step for step, _ in reward("R1", "collapse_score")]                           # a None is left out, not drawn as 0
    assert set(rows[0]) == set(mf.F3_FIELDS)


def write_run(folder, metrics, evals=None, summary=None, tail=""):
    folder.mkdir(parents=True)
    (folder / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics) + tail, encoding="utf-8")
    if evals is not None:
        (folder / "evals.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evals), encoding="utf-8")
    if summary is not None:
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_load_rl_runs_skips_absent_runs_drops_a_truncated_last_line_and_names_the_objective(mf, tmp_path_factory):
    base = tmp_path_factory.mktemp("figs") / "rl"
    summary = {"identity": {"options": {"reward_mode": "contrastive", "profile_mode": "none"}}, "command": ["x.py", "--val", "data/processed/val.jsonl"]}
    write_run(base / "R1", [{"step": 1}, {"step": 2}], [{"step": 0, "metrics": {}}], summary, tail='{"step": 3, "rewa')
    write_run(base / "R2", [{"step": 1}])
    runs, skipped = mf.load_rl_runs(base, ["R1", "R2", "R9"])
    assert list(runs) == ["R1", "R2"] and skipped == ["R9"]
    assert [m["step"] for m in runs["R1"]["metrics"]] == [1, 2]                                    # the half-written last line is dropped
    assert runs["R1"]["label"] == "R1: contrastive reward, profile removed" and runs["R2"]["label"] == "R2"
    assert runs["R2"]["evals"] == []                                                              # no evals.jsonl yet: still drawn
    bad = tmp_path_factory.mktemp("figs") / "rl"
    write_run(bad / "R1", [{"step": 1}], tail="not json\n" + json.dumps({"step": 2}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        mf.load_rl_runs(bad, ["R1"])                                                              # only the LAST line may be truncated


# ---------------------------------------------------------------------------------------------------------------- Test inputs
def test_test_named_paths_and_the_locked_evaluation_are_refused_unless_allowed(mf):
    for path in ("results/test_t_baselines.json", "data/processed/test_r.jsonl", "runs/final/C3/scored.v3.jsonl", "results/confirmatory.json",
                 "runs/test-t/x.jsonl"):
        with pytest.raises(mf.Refused):
            mf.refuse_test_path(path, False)
        mf.refuse_test_path(path, True)
    for path in ("results/val_baselines.json", "runs/rl/R1", "data/processed/refs.jsonl", "/tmp/latest_run/x.json", "results/test_results.json"):
        mf.refuse_test_path(path, False)


def test_a_test_split_is_refused_before_anything_is_read_or_written(mf, tmp_path_factory):
    folder = tmp_path_factory.mktemp("figs")
    out = folder / "out"
    for split in ("test_t", "test_r"):
        with pytest.raises(SystemExit) as stop:
            mf.main(["--split", split, "--out", str(out)])
        assert "refused" in str(stop.value) and "--allow-test" in str(stop.value)
        table = table_file(folder / f"tbl_{split[-1]}.json", split)           # a table of that very split under a neutral name: only --split stops it
        with pytest.raises(SystemExit) as stop:
            mf.main(["--split", split, "--table", str(table), "--out", str(out), "--tables"])
        assert "is a Test split" in str(stop.value)
    assert not out.exists()


def test_a_table_of_another_split_is_refused(mf, tmp_path_factory):
    folder = tmp_path_factory.mktemp("figs")
    table = table_file(folder / "tbl.json", split="test_t")
    with pytest.raises(mf.Refused, match="holds split 'test_t', not 'val'"):
        mf.load_table(table, "val")
    assert mf.load_table(table, "test_t")["split"] == "test_t"                                    # the path itself is neutral
    with pytest.raises(mf.Refused):
        mf.load_table(folder / "test_t_baselines.json", "test_t")                                 # a Test-named path needs the flag
    with pytest.raises(SystemExit):
        mf.main(["--split", "val", "--table", str(table), "--out", str(folder / "out"), "--tables"])
    assert not (folder / "out" / "table_val.md").exists()


def test_an_rl_run_evaluated_on_a_test_file_is_refused(mf, tmp_path_factory):
    base = tmp_path_factory.mktemp("figs") / "rl"
    write_run(base / "R1", [{"step": 1}], summary={"command": ["x.py", "--val", "data/processed/test_t.jsonl"]})
    with pytest.raises(mf.Refused):
        mf.load_rl_runs(base, ["R1"])
    assert list(mf.load_rl_runs(base, ["R1"], allow_test=True)[0]) == ["R1"]


# ---------------------------------------------------------------------------------------------------------------- tables
def test_the_table_writers_format_signs_missing_values_and_pipes(mf):
    rows = mf.table_rows({"rows": [trow("a", 0.1234, 12.34, -1e-18, lift=-0.0004, label="A | B"), trow("b", None, None, None)]})
    text = mf.table_md(rows, "Val", 20)
    lines = text.splitlines()
    assert lines[0].startswith("| Arm | Decoding | Valid answers | Delivered similarity | Samples per answer | Tokens per answer |")
    assert lines[1].startswith("|---|---|---:|") and len(lines[0].split("|")) == len(lines[2].split("|")) == len(lines[3].split("|"))
    assert "| A / B | T=1.0, k=8 | 0.900 | 0.123 | 1.50 | 12.3 | +0.000 | +0.000 | 0.100 | 0.000 | 0.000 |" in lines[2]   # a tiny negative rounds to +0.000
    assert "n/a" in lines[3] and "Val split, 20 drugs" in text


def test_write_csv_leaves_missing_values_empty_and_uses_plain_newlines(mf, tmp_path_factory):
    path = tmp_path_factory.mktemp("figs") / "t.csv"
    mf.write_csv(path, [{"a": 1, "b": None, "c": 0.123456789, "d": True, "e": 8.3e-18, "f": -3e-9}], ("a", "b", "c", "d", "e", "f"))
    assert path.read_bytes() == b"a,b,c,d,e,f\n1,,0.123456789,True,0,-3e-09\n"       # float noise reads 0, a real small value stays


# ---------------------------------------------------------------------------------------------------------------- whole runs
def project(folder, split="val", with_rl=True):
    """A small project tree under `folder`: refs and split report, a results table and (optionally) one RL run."""
    data = folder / "data"
    data.mkdir(parents=True)
    (data / "refs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in REFS), encoding="utf-8")
    (data / "split_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    table = table_file(folder / f"tbl_{split}.json", split)
    if with_rl:
        write_run(folder / "rl" / "R1", [{"step": i, "reward_mean": 0.1, "lift_batch": 0.01, "collapse_score": 1.0, "top_answer_share": 0.01}
                                         for i in range(1, 13)], [{"step": s, "metrics": {"validity": 0.6, "tanimoto": 0.08 + s / 500}} for s in (0, 10)],
                  {"identity": {"options": {"reward_mode": "full"}}, "command": ["x.py", "--val", "data/processed/val.jsonl"]})
    return ["--table", str(table), "--data", str(data), "--rl-dir", str(folder / "rl"), "--rl-runs", "R1,R3"]


def snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_a_val_run_writes_every_figure_csv_caption_and_table_and_is_repeatable(mf, tmp_path_factory, capsys):
    pytest.importorskip("matplotlib")
    folder = tmp_path_factory.mktemp("figs")
    args = project(folder)
    mf.main(["--split", "val", "--out", str(folder / "out1"), *args])
    said = capsys.readouterr().out
    assert "F3: runs not on disk, skipped: R3" in said and "F2: 3 runs without fv_tanimoto or fv_tokens are not drawn" in said
    assert "runs drawn as Other (not in ARMS): mystery-run" in said
    files = snapshot(folder / "out1")
    assert set(files) == {"f1_approval_years.png", "f1_approval_years.csv", "f2_similarity_vs_tokens_val.png", "f2_similarity_vs_tokens_val.csv",
                          "f3_rl_training.png", "f3_rl_training.csv", "f4_lift_vs_similarity_val.png", "f4_lift_vs_similarity_val.csv",
                          "table_val.md", "table_val.csv", "captions.json"}
    for name, content in files.items():
        if name.endswith(".png"):
            assert content[:8] == b"\x89PNG\r\n\x1a\n" and b"Software" not in content and b"tIME" not in content    # no version tag, no timestamp
    captions = json.loads(files["captions.json"])
    assert set(captions) == {"f1_approval_years", "f2_similarity_vs_tokens_val", "f3_rl_training", "f4_lift_vs_similarity_val", "table_val"}
    assert all(0 < len(text) < 1200 for text in captions.values()) and "20 Val drugs" in captions["f2_similarity_vs_tokens_val"]
    with open(folder / "out1" / "f2_similarity_vs_tokens_val.csv", newline="", encoding="utf-8") as f:                 # the csv is the plotted data
        assert [r["run"] for r in csv.DictReader(f)] == [r["run"] for r in mf.f2_rows(TABLE_ROWS)]
    mf.main(["--split", "val", "--out", str(folder / "out2"), *args])
    assert snapshot(folder / "out2") == files                                                                          # same inputs, same bytes


def test_without_any_rl_run_the_other_outputs_are_still_written(mf, tmp_path_factory, capsys):
    pytest.importorskip("matplotlib")
    folder = tmp_path_factory.mktemp("figs")
    args = project(folder, with_rl=False)
    mf.main(["--split", "val", "--out", str(folder / "out"), *args])
    names = set(snapshot(folder / "out"))
    assert "f3_rl_training.png" not in names and "f3_rl_training.csv" not in names
    assert {"f1_approval_years.png", "f2_similarity_vs_tokens_val.png", "f4_lift_vs_similarity_val.png", "table_val.md", "captions.json"} <= names
    assert "f3_rl_training" not in json.loads((folder / "out" / "captions.json").read_text())
    assert "no RL run with logged steps" in capsys.readouterr().out


def test_a_run_that_has_logged_no_step_yet_is_skipped_like_an_absent_one(mf, tmp_path_factory, capsys):
    pytest.importorskip("matplotlib")
    folder = tmp_path_factory.mktemp("figs")
    args = project(folder, with_rl=False)
    write_run(folder / "rl" / "R1", [])                                                           # metrics.jsonl exists but is empty
    mf.main(["--split", "val", "--out", str(folder / "out"), *args])
    assert "f3_rl_training.png" not in snapshot(folder / "out") and "no RL run with logged steps" in capsys.readouterr().out


def test_tables_only_needs_neither_data_nor_matplotlib_runs_and_writes_no_figure(mf, tmp_path_factory):
    folder = tmp_path_factory.mktemp("figs")
    table = table_file(folder / "tbl.json")
    mf.main(["--split", "val", "--table", str(table), "--out", str(folder / "out"), "--tables", "--data", str(folder / "nowhere")])
    assert set(snapshot(folder / "out")) == {"table_val.md", "table_val.csv", "captions.json"}
    with open(folder / "out" / "table_val.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run"] for r in rows] == [r["run"] for r in TABLE_ROWS] and rows[2]["fv_tanimoto"] == ""             # a missing value stays empty


def test_a_test_table_is_drawn_only_with_the_flag_and_is_named_for_its_split(mf, tmp_path_factory):
    pytest.importorskip("matplotlib")
    folder = tmp_path_factory.mktemp("figs")
    args = project(folder, split="test_t", with_rl=False)
    with pytest.raises(SystemExit):
        mf.main(["--split", "test_t", "--out", str(folder / "out"), *args])
    mf.main(["--split", "test_t", "--out", str(folder / "out"), "--allow-test", *args])
    names = set(snapshot(folder / "out"))
    assert {"f2_similarity_vs_tokens_test_t.png", "f4_lift_vs_similarity_test_t.png", "table_test_t.md", "table_test_t.csv"} <= names
    assert "Test-T" in json.loads((folder / "out" / "captions.json").read_text())["f2_similarity_vs_tokens_test_t"]


def test_a_missing_required_input_is_an_error_message_not_a_traceback(mf, tmp_path_factory):
    folder = tmp_path_factory.mktemp("figs")
    with pytest.raises(SystemExit) as stop:
        mf.main(["--split", "val", "--table", str(folder / "absent.json"), "--out", str(folder / "out")])
    assert "missing input" in str(stop.value)
