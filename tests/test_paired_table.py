"""scripts/paired_table.py: the pair set (explicit, --auto), the confirmatory label, the schema and sign convention of the collected paired block, the restriction to common
drugs (the REAL `rlexp.metrics report --vs` on a tiny synthetic data directory), refusals, outputs without answers or drug names, determinism. A fake report for the many-pair
cases. No real data."""
import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
          "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "CN1CCCC1c1cccnc1", "OC(=O)CC(O)(CC(O)=O)C(O)=O",
          "NCCc1ccc(O)c(O)c1", "CC(C)CC(N)C(O)=O", "OC(=O)c1ccccc1O"]
SELECTION = {"rl": "RL-full", "comparator": "C3", "comparator_set": ["C0", "C1", "C3", "A0n", "A2", "A4"], "rl_set": ["RL-full"], "arms": {}}
TESTED = ("fv_tanimoto", "mean_tanimoto", "fv_lift", "lift")


@pytest.fixture(scope="module")
def pt():
    spec = importlib.util.spec_from_file_location("paired_table", SCRIPTS / "paired_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------------------------------------- a fake `rlexp.metrics report --vs`
def entry(a, b, tested, n=41, rows=100):
    e = {"a": a, "b": b, "diff": a - b, "lo": a - b - 0.03, "hi": a - b + 0.03, "n_drugs": n, "n_rows": rows, "skipped_rows": 0, "degenerate": False, "boot_se": 0.01, "excludes_zero": False,
         "SECRET_FIELD": "SECRET"}
    if tested:
        e.update(wins=20, losses=15, ties=6, n_clusters=n, n_nontied=n - 6, p_unit="cluster", p_value=0.1234, p_method="monte_carlo", studentized=1.5)
    return e


def paired_report(split, n=41, rows=100, cover=(41, 100), a=0.30, b=0.20):
    """What `rlexp.metrics report --vs --json` writes for one split: A, the coverage of the split and the paired block of 12 metrics (A = a, B = b on every one)."""
    names = ("fv_tanimoto", "mean_tanimoto", "tanimoto_valid", "fv_lift", "lift", "valid_closed", "core@1", "exact@1", "fv_tokens")
    paired = {"n_drugs": n, "n_rows": rows, "metrics": {m: entry(a, b, m in ("fv_tanimoto", "mean_tanimoto", "fv_lift", "lift", "core@1", "exact@1"), n, rows) for m in names}}
    block = {"n_drugs": n, "n_rows": rows, "n_samples": rows * 8, "metrics": {}, "paired": paired, "constant": {"smiles": SMILES[0], "metrics": {}}}
    if cover:
        block["coverage"] = {"drugs": cover[0], "rows": cover[1]}
    return {"meta": {"a": "x", "b": "y", "boot": 10, "seed": 0, "ci": 95}, "notes": [], "splits": {split: block}}


class Fake:
    def __init__(self, pt, monkeypatch, report=None):
        self.calls, self.report = [], report or (lambda a, b, split: paired_report(split))
        monkeypatch.setattr(pt, "run_cli", self)

    def __call__(self, module, *args):
        self.calls.append((module, args))
        arg = lambda name: args[args.index(name) + 1]
        a, b = Path(arg("--scored")).parent.name, Path(arg("--vs")).parent.name
        Path(arg("--json")).write_text(json.dumps(self.report(a, b, arg("--split"))))
        return "a text report with SECRET"

    def pairs(self):
        return [(Path(a[a.index("--scored") + 1]).parent.name, Path(a[a.index("--vs") + 1]).parent.name) for _, a in self.calls]


def make_final(tmp_path, *arms):
    for arm in arms:
        (tmp_path / "final" / arm).mkdir(parents=True, exist_ok=True)
        (tmp_path / "final" / arm / "scored.v3.jsonl").write_text("{}\n")
    return tmp_path / "final"


def go(pt, tmp_path, *extra, split="test_t", allow=True, md=True):
    argv = ["--final", str(tmp_path / "final"), "--split", split, "--out", str(tmp_path / "res" / "paired.json"), "--data", str(tmp_path / "data"), "--boot", "55", *extra]
    argv += ["--md", str(tmp_path / "res" / "paired.md")] if md else []
    argv += ["--allow-test"] if allow else []
    pt.main(argv)
    return json.loads((tmp_path / "res" / "paired.json").read_text()), ((tmp_path / "res" / "paired.md").read_text() if md else None)


# ------------------------------------------------------------------------------------------------- the pair set
def test_pairs_are_a_colon_b_with_two_different_plain_arm_ids(pt):
    assert pt.parse_pairs("RL-full:C3,RL-full:A2") == [("RL-full", "C3"), ("RL-full", "A2")] and pt.parse_pairs("") == [] and pt.parse_pairs("27B-3.5:9B,") == [("27B-3.5", "9B")]
    for bad in ("RL-full", "RL-full:", ":C3", "A2:A2", "../x:C3", "a/b:C3", "A:B:C", "A:..", "A:.hidden", "A: B", "a..b:C3", "A:b..c"):
        with pytest.raises(SystemExit, match="error: pair .* is not A:B"):
            pt.parse_pairs(bad)


def test_auto_adds_the_standard_exploratory_set_and_nothing_else(pt):
    present = ["27B-3.5", "9B", "A0n", "A2", "A2-blank", "A2-shuffle", "A4", "A4-shuffle", "C0", "C1", "C2", "C3", "RL-full", "RL-full-blank", "RL-full-shuffle", "RL-contrastive",
               "name-4B", "ident-4B", "27B-3.8"]
    assert pt.auto_pairs(SELECTION, present) == [
        ("RL-full", "C0"), ("RL-full", "C1"), ("RL-full", "C3"), ("RL-full", "A0n"), ("RL-full", "A2"), ("RL-full", "A4"),      # the comparator set (C2 is not in it)
        ("RL-full", "27B-3.5"), ("RL-full", "9B"), ("RL-full", "27B-3.8"),                                                      # the references present
        ("RL-full", "RL-full-shuffle"), ("A2", "A2-shuffle")]                                                                   # the profile tests: the RL arm and A2, nothing else
    other = {**SELECTION, "rl": "RL-contrastive", "comparator_set": ["C3", "A2"], "comparator": "C3"}
    assert pt.auto_pairs(other, ["9B"]) == [("RL-contrastive", "C3"), ("RL-contrastive", "A2"), ("RL-contrastive", "9B"), ("RL-contrastive", "RL-contrastive-shuffle"), ("A2", "A2-shuffle")]
    assert pt.auto_pairs({**SELECTION, "comparator_set": ["C3", "RL-full"]}, []) == [("RL-full", "C3"), ("RL-full", "RL-full-shuffle"), ("A2", "A2-shuffle")]     # never the RL arm against itself


def test_auto_needs_a_selection_with_an_rl_arm_and_a_comparator_set(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "A2")
    Fake(pt, monkeypatch)
    with pytest.raises(SystemExit, match="--auto needs --selection"):
        go(pt, tmp_path, "--auto")
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"comparator_set": ["A2"]}))
    with pytest.raises(SystemExit, match="the selection lacks 'rl'"):
        go(pt, tmp_path, "--auto", "--selection", str(sel))
    with pytest.raises(SystemExit, match="cannot read the selection"):
        go(pt, tmp_path, "--auto", "--selection", str(tmp_path / "missing.json"))
    sel.write_text("[1, 2]")
    with pytest.raises(SystemExit, match="is not a JSON object"):
        go(pt, tmp_path, "--pairs", "A2:A2x", "--selection", str(sel))


def test_a_run_pairs_explicit_pairs_first_then_the_auto_set_without_repeats(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "RL-full", "RL-full-shuffle", "A2", "A2-shuffle", "C0", "C1", "C3", "A0n", "A4", "9B", "extra")
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SELECTION))
    fake = Fake(pt, monkeypatch)
    res, _ = go(pt, tmp_path, "--pairs", "A4:C3,RL-full:C3,A4:C3", "--auto", "--selection", str(sel))
    want = [("A4", "C3"), ("RL-full", "C3"), ("RL-full", "C0"), ("RL-full", "C1"), ("RL-full", "A0n"), ("RL-full", "A2"), ("RL-full", "A4"), ("RL-full", "9B"),
            ("RL-full", "RL-full-shuffle"), ("A2", "A2-shuffle")]
    assert fake.pairs() == want and [(r["a"], r["b"]) for r in res["pairs"]] == want


def test_no_pairs_is_an_error(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "A2")
    Fake(pt, monkeypatch)
    with pytest.raises(SystemExit, match="no pairs"):
        go(pt, tmp_path)


# ------------------------------------------------------------------------------------------------- confirmatory or exploratory
def test_only_the_selected_pair_on_test_t_is_confirmatory(pt):
    sel = SELECTION
    assert pt.label(("RL-full", "C3"), "test_t", sel) == "confirmatory"
    assert pt.label(("RL-full", "C3"), "test_r", sel) == "exploratory" and pt.label(("RL-full", "C3"), "val", sel) == "exploratory"
    assert pt.label(("C3", "RL-full"), "test_t", sel) == "exploratory"                 # the registered direction only
    assert pt.label(("RL-full", "A2"), "test_t", sel) == "exploratory" and pt.label(("A2", "C3"), "test_t", sel) == "exploratory"
    assert pt.label(("RL-full", "C3"), "test_t", None) == "exploratory"                # without a selection nothing is registered


def freeze(tmp_path, selection_path, **over):
    """<final>/manifest.json as gpu/final_eval.py writes it: the selection the run was frozen against."""
    digest = hashlib.sha256(Path(selection_path).read_bytes()).hexdigest()
    (tmp_path / "final" / "manifest.json").write_text(json.dumps({"spec_sha256": "5" * 64, "selection_sha256": digest, "arms": {}, **over}))


def test_labels_reach_the_json_and_the_markdown(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "RL-full", "C3", "A2")
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SELECTION))
    freeze(tmp_path, sel)
    Fake(pt, monkeypatch)
    res, md = go(pt, tmp_path, "--pairs", "RL-full:A2,RL-full:C3", "--selection", str(sel))
    assert [(r["a"], r["b"], r["label"]) for r in res["pairs"]] == [("RL-full", "A2", "exploratory"), ("RL-full", "C3", "confirmatory")]
    assert res["selection"] == {"rl": "RL-full", "comparator": "C3"}
    assert "### RL-full minus A2 (exploratory, Test-T)" in md and "### RL-full minus C3 (confirmatory, Test-T)" in md
    res, md = go(pt, tmp_path, "--pairs", "RL-full:C3", "--selection", str(sel), split="test_r")
    assert res["pairs"][0]["label"] == "exploratory" and "(exploratory, Test-R)" in md and "confirmatory, Test-R" not in md


@pytest.mark.parametrize("how", ["no manifest", "another selection", "manifest without the hash", "manifest is not json"])
def test_a_selection_that_is_not_the_runs_labels_nothing_confirmatory_and_says_so(pt, tmp_path, monkeypatch, capsys, how):
    make_final(tmp_path, "RL-full", "C3")
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SELECTION))
    if how == "another selection":
        freeze(tmp_path, sel, selection_sha256=hashlib.sha256(b"another").hexdigest())
    elif how == "manifest without the hash":
        (tmp_path / "final" / "manifest.json").write_text(json.dumps({"arms": {}}))
    elif how == "manifest is not json":
        (tmp_path / "final" / "manifest.json").write_text("{")
    Fake(pt, monkeypatch)
    res, md = go(pt, tmp_path, "--pairs", "RL-full:C3", "--selection", str(sel))
    assert [(r["a"], r["b"], r["label"]) for r in res["pairs"]] == [("RL-full", "C3", "exploratory")] and res["selection_is_the_runs"] is False
    assert "no pair is labelled confirmatory" in capsys.readouterr().err
    assert "(exploratory, Test-T)" in md and "confirmatory, Test-T" not in md and "no pair above is registered" in md


def test_a_selection_that_is_the_runs_prints_no_warning_and_keeps_the_label(pt, tmp_path, monkeypatch, capsys):
    make_final(tmp_path, "RL-full", "C3", "A2")
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SELECTION))
    freeze(tmp_path, sel)
    Fake(pt, monkeypatch)
    res, md = go(pt, tmp_path, "--pairs", "RL-full:C3", "--selection", str(sel))
    assert res["pairs"][0]["label"] == "confirmatory" and res["selection_is_the_runs"] is True and "warning" not in capsys.readouterr().err and "not registered" not in md
    freeze(tmp_path, sel)
    sel.write_text(json.dumps({**SELECTION, "comparator": "A2"}))                     # edited after the run: the same file name, other bytes
    res, md = go(pt, tmp_path, "--pairs", "RL-full:A2", "--selection", str(sel))
    assert res["pairs"][0]["label"] == "exploratory" and res["selection_is_the_runs"] is False
    capsys.readouterr()
    res, _ = go(pt, tmp_path, "--pairs", "RL-full:C3")                                 # no selection at all: nothing registered, nothing to warn about
    assert res["selection_is_the_runs"] is None and "warning" not in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------- what is collected
def test_the_record_holds_the_paired_block_of_six_metrics_with_the_endpoint_first(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "A2", "C3")
    Fake(pt, monkeypatch, lambda a, b, split: paired_report(split, a=0.31, b=0.2))
    res, _ = go(pt, tmp_path, "--pairs", "A2:C3")
    rec, = res["pairs"]
    assert list(rec["metrics"]) == ["fv_tanimoto", "mean_tanimoto", "tanimoto_valid", "fv_lift", "lift", "valid_closed"]       # six, the confirmatory endpoint first
    e = rec["metrics"]["fv_tanimoto"]
    assert (e["a"], e["b"], e["diff"], e["ci"]) == (0.31, 0.2, pytest.approx(0.11), [pytest.approx(0.08), pytest.approx(0.14)])
    assert (e["wins"], e["losses"], e["ties"], e["p_value"], e["p_method"], e["studentized"], e["n_drugs"], e["n_rows"]) == (20, 15, 6, 0.1234, "monte_carlo", 1.5, 41, 100)
    assert (e["p_unit"], e["n_clusters"]) == ("cluster", 41)
    for name in ("tanimoto_valid", "valid_closed"):                                       # not tested by the report: no wins, losses, ties or p-value
        u = rec["metrics"][name]
        assert (u["wins"], u["losses"], u["ties"], u["p_value"], u["studentized"]) == (None, None, None, None, None) and u["diff"] == pytest.approx(0.11) and u["ci"] is not None
    assert "core@1" not in rec["metrics"] and "fv_tokens" not in rec["metrics"]
    assert res["split"] == "test_t" and rec["split"] == "test_t" and "A minus B" in res["sign"] and (res["boot"], res["seed"]) == (55, 0)
    assert rec["metrics"]["lift"]["wins"] == 20


def test_the_report_is_run_for_a_minus_b_on_the_split_with_the_data_boot_and_seed(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "A2", "C3")
    fake = Fake(pt, monkeypatch)
    go(pt, tmp_path, "--pairs", "A2:C3", split="test_r")
    (module, args), = fake.calls
    arg = lambda name: args[args.index(name) + 1]
    assert module == "rlexp.metrics" and args[0] == "report"
    assert arg("--scored") == str(tmp_path / "final" / "A2" / "scored.v3.jsonl") and arg("--vs") == str(tmp_path / "final" / "C3" / "scored.v3.jsonl")
    assert (arg("--split"), arg("--data"), arg("--boot"), arg("--seed")) == ("test_r", str(tmp_path / "data"), "55", "0")


def test_the_pair_covers_the_cohort_or_is_flagged_partial(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "A2", "C3", "RL-full")
    reports = {("A2", "C3"): paired_report("test_t"), ("RL-full", "C3"): paired_report("test_t", n=39, rows=95, cover=(41, 100)),
               ("RL-full", "A2"): paired_report("test_t", cover=None)}
    Fake(pt, monkeypatch, lambda a, b, split: reports[a, b])
    res, md = go(pt, tmp_path, "--pairs", "A2:C3,RL-full:C3,RL-full:A2")
    full, part, unknown = res["pairs"]
    reports.update({("A2", "A4"): paired_report("test_t", n=41, rows=99), ("C3", "A4"): paired_report("test_t", n=40, rows=100)})     # fewer rows only, fewer drugs only
    make_final(tmp_path, "A4")
    res2, _ = go(pt, tmp_path, "--pairs", "A2:A4,C3:A4")
    assert [r["partial"] for r in res2["pairs"]] == [True, True]
    assert (full["n_drugs"], full["n_rows"], full["cohort"], full["partial"]) == (41, 100, {"drugs": 41, "rows": 100}, False)
    assert (part["n_drugs"], part["n_rows"], part["cohort"], part["partial"]) == (39, 95, {"drugs": 41, "rows": 100}, True)
    assert unknown["cohort"] is None and unknown["partial"] is None
    assert "39 common drugs, 95 prompt rows; Test-T has 41 drugs and 100 rows (PARTIAL" in md and md.count("PARTIAL") == 1


def test_a_pair_whose_endpoint_is_undefined_shows_n_a_not_zero_wins(pt, tmp_path, monkeypatch, capsys):
    make_final(tmp_path, "RL-full", "27B-3.8")

    def short_k(a, b, split):
        r = paired_report(split)
        e = r["splits"][split]["paired"]["metrics"]["fv_tanimoto"]
        e.update(a=None, b=None, diff=None, lo=None, hi=None, n_drugs=0, wins=0, losses=0, ties=0, p_value=None, p_method=None, studentized=None, n_clusters=0)       # as the CLI writes it
        return r
    Fake(pt, monkeypatch, short_k)
    res, md = go(pt, tmp_path, "--pairs", "RL-full:27B-3.8")
    e = res["pairs"][0]["metrics"]["fv_tanimoto"]
    assert e["n_drugs"] == 0 and e["p_value"] is None and e["diff"] is None and e["ci"] is None
    row = next(x for x in md.splitlines() if x.startswith("| Delivered similarity"))
    assert [c.strip() for c in row.strip("|").split("|")][1:] == ["n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "0"]
    assert "fv_tanimoto undefined (fewer than 8 samples per row in a run)" in capsys.readouterr().out


def test_a_pair_with_no_rows_in_common_is_listed_with_n_zero(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "RL-full", "27B-3.5")

    def none_in_common(a, b, split):
        r = paired_report(split, cover=(30, 80))
        r["splits"][split]["paired"] = {"n_drugs": 0, "n_rows": 0, "metrics": None}
        return r
    Fake(pt, monkeypatch, none_in_common)
    res, md = go(pt, tmp_path, "--pairs", "RL-full:27B-3.5", split="test_r")
    rec, = res["pairs"]
    assert (rec["n_drugs"], rec["n_rows"], rec["metrics"], rec["partial"]) == (0, 0, None, True)
    assert "share no prompt rows on Test-R" in md
    both = lambda a, b, split: {"splits": {split: {"n_drugs": 0, "n_rows": 0, "n_samples": 0}}}                 # A itself has no rows on the split: no paired block at all
    Fake(pt, monkeypatch, both)
    res, _ = go(pt, tmp_path, "--pairs", "RL-full:27B-3.5", split="test_r")
    assert res["pairs"][0]["metrics"] is None and res["pairs"][0]["cohort"] is None


# ------------------------------------------------------------------------------------------------- refusals
def test_a_pair_is_refused_when_a_scored_file_is_missing_and_every_missing_file_is_named(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "RL-full", "A2")
    fake = Fake(pt, monkeypatch)
    with pytest.raises(SystemExit) as e:
        go(pt, tmp_path, "--pairs", "RL-full:C3,RL-full:A2,C0:C3,A2:A4")
    message = str(e.value)
    assert message.startswith("error: missing scored file(s)") and message.count("scored.v3.jsonl") == 3
    for arm in ("C3", "C0", "A4"):
        assert str(tmp_path / "final" / arm / "scored.v3.jsonl") in message
    assert fake.calls == [] and not (tmp_path / "res" / "paired.json").exists()                  # nothing ran, nothing was written
    (tmp_path / "final" / "A2" / "scored.v3.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing scored file"):
        go(pt, tmp_path, "--pairs", "RL-full:A2")


def test_an_auto_pair_whose_arm_is_missing_is_refused_too(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "RL-full", "C0", "C1", "C3", "A0n", "A2", "A4", "A2-shuffle")             # RL-full-shuffle is missing
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SELECTION))
    fake = Fake(pt, monkeypatch)
    with pytest.raises(SystemExit, match="RL-full-shuffle"):
        go(pt, tmp_path, "--auto", "--selection", str(sel))
    assert fake.calls == []


@pytest.mark.parametrize("split", ["test_t", "test_r"])
def test_a_test_split_needs_allow_test_and_val_does_not(pt, tmp_path, monkeypatch, split):
    make_final(tmp_path, "A2", "C3")
    fake = Fake(pt, monkeypatch)
    with pytest.raises(SystemExit, match="refused: .* is a Test split"):
        go(pt, tmp_path, "--pairs", "A2:C3", split=split, allow=False)
    assert fake.calls == []
    res, md = go(pt, tmp_path, "--pairs", "A2:C3", split="val", allow=False)
    assert res["split"] == "val" and res["pairs"][0]["split"] == "val" and "Paired comparisons on Val" in md and "(exploratory, Val)" in md
    for bad in ("train", "test", "val,test_t", ""):
        with pytest.raises(SystemExit, match="error: splits must come from"):
            go(pt, tmp_path, "--pairs", "A2:C3", split=bad, allow=False)


def test_p_values_print_like_the_metrics_report(pt):
    assert [pt.p_text(p) for p in (None, 1.0, 0.1234, 0.00031, 0.0001, 0.00009, 1e-5)] == ["n/a", "1.0000", "0.1234", "0.0003", "0.0001", "9.0e-05", "1.0e-05"]


def test_the_split_is_named_in_every_progress_line_and_block(pt, tmp_path, monkeypatch, capsys):
    make_final(tmp_path, "A2", "C3", "A4")
    Fake(pt, monkeypatch)
    res, md = go(pt, tmp_path, "--pairs", "A2:C3,A4:C3", split="val", allow=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("[val] A2 minus C3 (exploratory): fv_tanimoto +0.1000") and lines[1].startswith("[val] A4 minus C3") and "wins 20 losses 15 ties 6" in lines[0]
    assert all(r["split"] == "val" for r in res["pairs"]) and md.count(", Val)") == 2 and "Paired comparisons on Val" in md and "Test-R" not in md


def test_a_missing_final_directory_and_a_failing_report_are_plain_errors(pt, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="error: .*final is not a directory"):
        go(pt, tmp_path, "--pairs", "A2:C3")
    make_final(tmp_path, "A2", "C3")
    stderr = "error: " + "x" * 400 + " unknown id(s), e.g. ['secretdrug/0']"
    monkeypatch.setattr(pt.run_cli.__globals__["subprocess"], "run", lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr=stderr))
    with pytest.raises(SystemExit) as e:
        go(pt, tmp_path, "--pairs", "A2:C3")
    assert "rlexp.metrics report failed" in str(e.value) and "secretdrug" not in str(e.value) and not (tmp_path / "res" / "paired.json").exists()


# ------------------------------------------------------------------------------------------------- output
def test_markdown_states_the_sign_convention_and_lists_the_six_measures_in_plain_english(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "RL-full", "RL-full-shuffle", "C3")
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(SELECTION))
    Fake(pt, monkeypatch)
    _, md = go(pt, tmp_path, "--pairs", "RL-full:C3,RL-full:RL-full-shuffle", "--selection", str(sel))
    assert "Differences are A minus B: a positive difference means the first arm of the pair is higher" in md
    assert "every other number is exploratory, with no adjustment for multiplicity" in md and "profile test of the protocol" in md
    section = md.split("### RL-full minus C3")[1].split("### ")[0]
    rows = [x for x in section.splitlines() if x.startswith("| ") and "Measure" not in x]
    assert [x.split("|")[1].strip() for x in rows] == ["Delivered similarity (the confirmatory endpoint)", "Mean similarity, all samples", "Similarity of valid answers", "Delivered lift",
                                                     "Lift, all samples", "Valid answers"]
    assert rows[0].split("|")[2:9] == [" 0.300 ", " 0.200 ", " +0.100 ", " [+0.070, +0.130] ", " 20 / 15 / 6 ", " 0.1234 ", " 41 "]
    assert rows[2].split("|")[6:8] == [" n/a ", " n/a "]                                   # similarity of valid answers: not a tested metric
    assert "fv_" not in md and "tanimoto_valid" not in md
    _, md = go(pt, tmp_path, "--pairs", "RL-full:C3", "--selection", str(sel))
    assert "profile test of the protocol" not in md                                       # the sentence appears only with a shuffle pair


def test_the_output_is_deterministic_and_holds_no_report_text_molecule_or_machine_path(pt, tmp_path, monkeypatch):
    make_final(tmp_path, "A2", "C3")
    Fake(pt, monkeypatch)
    res, md = go(pt, tmp_path, "--pairs", "A2:C3")
    first = (tmp_path / "res" / "paired.json").read_bytes()
    go(pt, tmp_path, "--pairs", "A2:C3")
    assert (tmp_path / "res" / "paired.json").read_bytes() == first and (tmp_path / "res" / "paired.md").read_text() == md
    seen = first.decode() + md
    for marker in ("SECRET", SMILES[0], "a text report", str(tmp_path), "smiles"):
        assert marker not in seen
    assert list(res["pairs"][0]) == ["a", "b", "split", "label", "n_drugs", "n_rows", "cohort", "partial", "metrics"]


# ------------------------------------------------------------------------------------------------- the real CLI on a tiny data directory
@pytest.fixture(scope="module")
def real(pt, tmp_path_factory):
    """data/<split>.jsonl with real references, A = the exact reference five of eight times (eight copies of an unparseable answer otherwise), B = a constant, and B covers only the
    first three of the four test_t drugs: scored with the real reward, paired by the real `rlexp.metrics report --vs` (A:B and B:A, one run)."""
    from rlexp import metrics
    tmp_path = tmp_path_factory.mktemp("pair")
    data, rows, k = tmp_path / "data", {}, 0
    data.mkdir()
    for split, groups in {"train": ["trainone", "traintwo", "trainthree"], "test_t": ["testtone", "testttwo", "testtthree", "testtfour"]}.items():
        rows[split] = []
        for g in groups:
            rows[split] += [{"id": f"{g}/{i}", "group": g, "task_type": "A", "split": split, "cluster_id": k, "ref_smiles": SMILES[k], "ref_conn": f"conn-{g}",
                             "ref_scaffold": f"scaf-{g}", "ref_alt_smiles": [], "n_prompt_tokens": 100} for i in range(1 + k % 2)]
            k += 1
        write_jsonl(data / f"{split}.jsonl", rows[split])
    test_t = rows["test_t"]                                                              # drugs with 1+k%2 rows: k = 3..6 -> 2, 1, 2, 1 rows
    sample = lambda r, j, answer: {"id": r["id"], "group": r["group"], "split": r["split"], "sample_idx": j, "text": f"t</think>\n{answer}", "finish_reason": "stop", "n_tokens": 7}
    a = [sample(r, j, r["ref_smiles"] if j >= 3 else "no molecule here") for r in test_t for j in range(8)]
    b = [sample(r, j, SMILES[11]) for r in test_t if r["group"] != "testtfour" for j in range(8)]
    for arm, samples in (("A", a), ("B", b)):
        (tmp_path / "final" / arm).mkdir(parents=True)
        metrics.write_jsonl(tmp_path / "final" / arm / "scored.v3.jsonl", metrics.score_samples(samples, test_t))
    pt.main(["--final", str(tmp_path / "final"), "--split", "test_t", "--pairs", "A:B,B:A", "--out", str(tmp_path / "paired.json"), "--md", str(tmp_path / "paired.md"),
             "--data", str(data), "--boot", "100", "--allow-test"])
    return tmp_path, json.loads((tmp_path / "paired.json").read_text()), (tmp_path / "paired.md").read_text(), test_t


def test_a_pair_over_different_drugs_is_restricted_to_the_common_drugs_and_reports_n(real):
    _, res, md, _ = real
    ab = res["pairs"][0]
    assert (ab["n_drugs"], ab["n_rows"]) == (3, 5) and ab["cohort"] == {"drugs": 4, "rows": 6} and ab["partial"] is True      # testtfour is not in B's run: left out, and said
    assert all(e["n_drugs"] == 3 and e["n_rows"] == 5 for e in ab["metrics"].values())
    assert "3 common drugs, 5 prompt rows; Test-T has 4 drugs and 6 rows (PARTIAL" in md


def test_the_real_paired_block_has_hand_checked_values_and_the_sign_convention(real):
    from rlexp import chem
    _, res, _, test_t = real
    ab, ba = res["pairs"]
    constant = chem.standardize(SMILES[11])
    refs = {r["group"]: r["ref_smiles"] for r in test_t}
    b_mean = sum(chem.tanimoto(constant, chem.standardize(refs[g])) for g in ("testtone", "testttwo", "testtthree")) / 3
    e = ab["metrics"]["fv_tanimoto"]
    assert e["a"] == pytest.approx(1.0) and e["b"] == pytest.approx(b_mean) and e["diff"] == pytest.approx(1.0 - b_mean)      # A minus B, positive: A is better
    assert (e["wins"], e["losses"], e["ties"]) == (3, 0, 0) and e["p_value"] == 0.25 and e["p_method"] == "exact" and (e["p_unit"], e["n_clusters"]) == ("cluster", 3)
    r = ba["metrics"]["fv_tanimoto"]                                                     # the reverse pair: swapped arms, negated difference, mirrored counts, the same p
    assert (r["a"], r["b"], r["diff"]) == (e["b"], e["a"], pytest.approx(-e["diff"])) and (r["wins"], r["losses"], r["ties"]) == (0, 3, 0) and r["p_value"] == e["p_value"]
    assert ab["metrics"]["valid_closed"]["a"] == pytest.approx(5 / 8) and ab["metrics"]["valid_closed"]["b"] == 1.0 and ab["metrics"]["valid_closed"]["wins"] is None
    assert ab["label"] == ba["label"] == "exploratory"


def test_the_real_paired_block_equals_an_in_process_report_of_the_same_rows(real):
    from rlexp import metrics
    tmp_path, res, *_ = real
    load = lambda arm: metrics.read_jsonl(tmp_path / "final" / arm / "scored.v3.jsonl", drop=("text",))
    want = metrics.report(load("A"), load("B"), ["test_t"], boot=100, seed=0, data=metrics.load_data(tmp_path / "data"))["splits"]["test_t"]["paired"]
    got = res["pairs"][0]
    assert (got["n_drugs"], got["n_rows"]) == (want["n_drugs"], want["n_rows"])
    for name, e in got["metrics"].items():
        for key in ("a", "b", "diff", "wins", "losses", "ties", "p_value", "p_method", "studentized", "n_drugs", "n_rows"):
            assert e[key] == want["metrics"][name].get(key)
        assert e["ci"] == [want["metrics"][name]["lo"], want["metrics"][name]["hi"]]


def test_the_real_outputs_hold_no_answer_drug_name_or_molecule(real):
    _, res, md, _ = real
    seen = json.dumps(res) + md
    for marker in (*SMILES, "testtone", "testttwo", "testtthree", "testtfour", "trainone", "no molecule here", "</think>", "smiles"):
        assert marker not in seen


# ------------------------------------------------------------------------------------------------- the three scripts on the same runs
def test_final_table_paired_table_and_exposure_terciles_agree_on_the_same_runs(pt, tmp_path):
    """final_table scores two runs of the same four drugs (real CLIs), paired_table pairs the scored files it leaves, exposure_terciles splits the same contrast: one set of numbers."""
    import sys

    def load(name):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    ft, ex = sys.modules["final_table"], load("exposure_terciles")
    data, k, rows = tmp_path / "data", 0, {}
    data.mkdir()
    for split, groups in {"train": ["trainone", "traintwo", "trainthree"], "test_t": ["testtone", "testttwo", "testtthree", "testtfour"]}.items():
        rows[split] = []
        for g in groups:
            rows[split] += [{"id": f"{g}/{i}", "group": g, "task_type": "A", "split": split, "cluster_id": k, "ref_smiles": SMILES[k], "ref_conn": f"conn-{g}",
                             "ref_scaffold": f"scaf-{g}", "ref_alt_smiles": [], "n_prompt_tokens": 100} for i in range(1 + k % 2)]
            k += 1
        write_jsonl(data / f"{split}.jsonl", rows[split])
    (tmp_path / "curated.json").write_text("[]")
    answers = {"A": lambda r, j: r["ref_smiles"] if j >= 2 + (r["group"] == "testttwo") else "no molecule here", "B": lambda r, j: SMILES[11]}
    for arm, answer in answers.items():
        (tmp_path / "final" / arm).mkdir(parents=True)
        write_jsonl(tmp_path / "final" / arm / "samples.jsonl", [{"id": r["id"], "group": r["group"], "split": "test_t", "sample_idx": j, "text": f"t</think>\n{answer(r, j)}",
                                                                   "finish_reason": "stop", "n_tokens": 5} for r in rows["test_t"] for j in range(8)])
    ft.main(["--final", str(tmp_path / "final"), "--splits", "test_t", "--out", str(tmp_path / "table.json"), "--data", str(data), "--curated", str(tmp_path / "curated.json"),
             "--boot", "50", "--allow-test"])
    pt.main(["--final", str(tmp_path / "final"), "--split", "test_t", "--pairs", "A:B", "--out", str(tmp_path / "paired.json"), "--data", str(data), "--boot", "50", "--allow-test"])
    refs = [{"group": g, "split": "test_t", "approval_year": 2025, "pubchem_create_year": 2025 - n, "pubmed_count": 10 * n, "ref_smiles": SMILES[0]}
            for n, g in enumerate(["testtone", "testttwo", "testtthree", "testtfour"], 1)]
    write_jsonl(tmp_path / "refs.jsonl", refs)
    (tmp_path / "exposure.json").write_text("{}")
    ex.main(["--final", str(tmp_path / "final"), "--split", "test_t", "--a", "A", "--b", "B", "--refs", str(tmp_path / "refs.jsonl"), "--exposure", str(tmp_path / "exposure.json"),
             "--out", str(tmp_path / "exposed.json"), "--allow-test"])
    table = {r["run"]: r for r in json.loads((tmp_path / "table.json").read_text())["rows"]}
    pair = json.loads((tmp_path / "paired.json").read_text())["pairs"][0]
    expo = json.loads((tmp_path / "exposed.json").read_text())
    e = pair["metrics"]["fv_tanimoto"]
    assert (e["a"], e["b"]) == (table["A"]["fv_tanimoto"], table["B"]["fv_tanimoto"]) and e["diff"] == pytest.approx(table["A"]["fv_tanimoto"] - table["B"]["fv_tanimoto"], abs=1e-12)
    assert pair["n_drugs"] == 4 and pair["partial"] is False and pair["cohort"] == {"drugs": 4, "rows": 6}
    o = expo["overall"]
    assert (o["mean_a"], o["mean_b"], o["mean_diff"]) == (e["a"], e["b"], e["diff"]) and (o["wins"], o["losses"], o["ties"], o["p_value"]) == (e["wins"], e["losses"], e["ties"], e["p_value"])
    assert sum(t["n_drugs"] for t in expo["covariates"]["pubmed_count"]["terciles"]) == 4
