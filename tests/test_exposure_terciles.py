"""scripts/exposure_terciles.py on synthetic scored runs and refs: the cut (rank terciles, ties by drug id), the covariates (PubMed count, record age = approval year minus PubChem
record year, refs first then exposure.json), the per-tercile numbers against hand computations and an independent exact sign-flip, equality of the overall block with the REAL
`rlexp.metrics report --vs` CLI, missing covariates counted, the sign convention, common-drug restriction, refusals, no drug names in the outputs, determinism. No real data."""
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SMILES_MARKER = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("exposure_terciles", SCRIPTS / "exposure_terciles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# nine drugs; B scores 0.2 on all of them and A scores 0.2 + diff. PubMed counts and ages (years public before approval) are shuffled against each other, so the two cuts differ.
DRUGS = [f"zqdrug{i}" for i in range(1, 10)]
DIFF = dict(zip(DRUGS, [-0.05, 0.10, -0.05, 0.10, -0.20, 0.10, -0.10, 0.05, 0.0]))
PUBMED = dict(zip(DRUGS, [90, 10, 50, 20, 70, 30, 80, 40, 60]))
AGE = dict(zip(DRUGS, [11, 3, 7, 9, 4, 10, 5, 8, 6]))
APPROVAL = {g: 2025 if i < 5 else 2026 for i, g in enumerate(DRUGS)}
CREATE = {g: APPROVAL[g] - AGE[g] for g in DRUGS}                         # the PubChem record year
LOW_PUBMED, MID_PUBMED, HIGH_PUBMED = ["zqdrug2", "zqdrug4", "zqdrug6"], ["zqdrug8", "zqdrug3", "zqdrug9"], ["zqdrug5", "zqdrug7", "zqdrug1"]
LOW_AGE, MID_AGE, HIGH_AGE = ["zqdrug2", "zqdrug5", "zqdrug7"], ["zqdrug9", "zqdrug3", "zqdrug8"], ["zqdrug4", "zqdrug6", "zqdrug1"]


def exact_p(units):
    """The two-sided exact sign-flip p-value of per-unit totals, by enumerating every sign pattern (tiny n only): an independent check of metrics.sign_flip_p."""
    units = [u for u in units if abs(u) >= 1e-9]
    if not units:
        return 1.0
    t, eps = abs(sum(units)), 1e-9 * sum(abs(u) for u in units)
    return sum(abs(sum(s * u for s, u in zip(signs, units))) >= t - eps for signs in itertools.product((1, -1), repeat=len(units))) / 2 ** len(units)


def scored(split, values, cluster=None, k=8, rows=1):
    """Scored rows: values = {drug: first-valid Tanimoto}; k identical valid samples per prompt row (so fv_tanimoto = that value); cluster = {drug: cluster id}."""
    return [{"id": f"{g}/{i}", "group": g, "split": split, "sample_idx": j, "finish_reason": "stop", "n_tokens": 10, "closed": True, "answer": None, "valid": True,
             "tanimoto": v, "scaffold_match": False, "conn_match": False, "stereo_match": False, "reward": v, "lift": v - 0.1, "text": SMILES_MARKER,
             "cluster_id": (cluster or {}).get(g, n)} for n, (g, v) in enumerate(values.items()) for i in range(rows) for j in range(k)]


def refs_rows(split="test_t", pubmed=PUBMED, create=CREATE, approval=APPROVAL, drugs=DRUGS):
    return [{"group": g, "split": split, "approval_year": approval.get(g), "pubchem_create_year": create.get(g), "pubmed_count": pubmed.get(g), "ref_smiles": SMILES_MARKER,
             "cluster_id": 1} for g in drugs]


def world(tmp_path, a=None, b=None, refs=None, exposure=None, split="test_t", **kw):
    """final/A and final/B scored files, refs.jsonl and exposure.json under tmp_path."""
    a = a if a is not None else {g: 0.2 + DIFF[g] for g in DRUGS}
    b = b if b is not None else {g: 0.2 for g in DRUGS}
    for arm, values in (("A", a), ("B", b)):
        (tmp_path / "final" / arm).mkdir(parents=True, exist_ok=True)
        (tmp_path / "final" / arm / "scored.v3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored(split, values, **kw)))
    (tmp_path / "refs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (refs if refs is not None else refs_rows(split))))
    (tmp_path / "exposure.json").write_text(json.dumps(exposure if exposure is not None else {}))
    return tmp_path


def run(ex, tmp_path, *extra, split="test_t", a="A", b="B", allow=True, md=True):
    argv = ["--final", str(tmp_path / "final"), "--split", split, "--a", a, "--b", b, "--refs", str(tmp_path / "refs.jsonl"), "--exposure", str(tmp_path / "exposure.json"),
            "--out", str(tmp_path / "res" / "exposure.json"), *extra]
    argv += ["--md", str(tmp_path / "res" / "exposure.md")] if md else []
    argv += ["--allow-test"] if allow else []
    ex.main(argv)
    return json.loads((tmp_path / "res" / "exposure.json").read_text()), ((tmp_path / "res" / "exposure.md").read_text() if md else None)


def tercile_of(res, covariate):
    return {t["tercile"]: t for t in res["covariates"][covariate]["terciles"]}


def mean(values):
    return sum(values) / len(values)


# ------------------------------------------------------------------------------------------------- the cut
def test_terciles_are_rank_thirds_lowest_first_with_sizes_rank_times_three_over_n(ex):
    bins = ex.terciles({f"d{i:02d}": i for i in range(41)})
    assert [len(b) for b in bins] == [14, 14, 13] and bins[0][0] == "d00" and bins[0][-1] == "d13" and bins[1][0] == "d14" and bins[2][0] == "d28" and bins[2][-1] == "d40"
    assert [len(b) for b in ex.terciles({f"d{i}": i for i in range(10)})] == [4, 3, 3] and [len(b) for b in ex.terciles({"a": 1, "b": 2, "c": 3})] == [1, 1, 1]
    assert [len(b) for b in ex.terciles({f"d{i}": i for i in range(7)})] == [3, 2, 2]


def test_ties_in_the_covariate_break_by_drug_id_so_the_cut_is_deterministic(ex):
    same = {name: 5 for name in ("delta", "alpha", "charlie", "bravo", "echo", "foxtrot")}
    assert ex.terciles(same) == [["alpha", "bravo"], ["charlie", "delta"], ["echo", "foxtrot"]]
    assert ex.terciles(dict(reversed(list(same.items())))) == ex.terciles(same)                  # not the order the drugs arrive in
    mixed = {"a": 2, "b": 1, "c": 2, "d": 2, "e": 3, "f": 2}
    assert ex.terciles(mixed) == [["b", "a"], ["c", "d"], ["f", "e"]]                               # by (covariate, id): 1 | 2, 2 | 2, 2 | 3 ... lowest first
    low, mid, high = ex.terciles({g: PUBMED[g] for g in DRUGS})
    assert (low, mid, high) == (LOW_PUBMED, MID_PUBMED, HIGH_PUBMED) and ex.terciles({g: AGE[g] for g in DRUGS}) == [LOW_AGE, MID_AGE, HIGH_AGE]


# ------------------------------------------------------------------------------------------------- numbers against hand computations
def test_each_tercile_has_the_hand_computed_mean_difference_wins_losses_ties_and_exact_p(ex, tmp_path):
    res, _ = run(ex, world(tmp_path))
    for covariate, groups in (("pubmed_count", (LOW_PUBMED, MID_PUBMED, HIGH_PUBMED)), ("record_age", (LOW_AGE, MID_AGE, HIGH_AGE))):
        got = tercile_of(res, covariate)
        assert list(got) == ["low", "middle", "high"]
        for name, members in zip(("low", "middle", "high"), groups):
            t, diffs = got[name], [DIFF[g] for g in members]
            assert t["n_drugs"] == 3 and t["mean_diff"] == pytest.approx(mean(diffs), abs=1e-12) and t["mean_a"] == pytest.approx(mean([0.2 + d for d in diffs]))
            assert t["mean_b"] == pytest.approx(0.2) and (t["wins"], t["losses"], t["ties"]) == (sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), sum(d == 0 for d in diffs))
            assert t["p_value"] == pytest.approx(exact_p(diffs)) and t["p_method"] == "exact" and (t["p_unit"], t["n_clusters"]) == ("cluster", 3)
    low = tercile_of(res, "pubmed_count")["low"]
    assert (low["wins"], low["p_value"], low["range"]) == (3, 0.25, [10, 30])                          # three wins: 2 of 8 sign patterns reach the observed sum
    assert tercile_of(res, "pubmed_count")["middle"]["p_value"] == 1.0 and tercile_of(res, "pubmed_count")["middle"]["ties"] == 1
    assert tercile_of(res, "pubmed_count")["high"]["range"] == [70, 90]
    assert tercile_of(res, "record_age")["low"]["range"] == [3, 5] and tercile_of(res, "record_age")["high"]["range"] == [9, 11]


def test_the_terciles_partition_the_drugs_and_the_overall_block_is_the_report_of_all_of_them(ex, tmp_path):
    res, _ = run(ex, world(tmp_path))
    o = res["overall"]
    assert res["n_drugs"] == 9 and o["n_drugs"] == 9 and (o["wins"], o["losses"], o["ties"]) == (4, 4, 1) and o["mean_diff"] == pytest.approx(mean(list(DIFF.values())), abs=1e-12)
    assert o["p_value"] == pytest.approx(exact_p(list(DIFF.values())))
    for covariate in ("pubmed_count", "record_age"):
        ts = res["covariates"][covariate]["terciles"]
        assert sum(t["n_drugs"] for t in ts) == 9 and sum(t["n_drugs"] * t["mean_diff"] for t in ts) / 9 == pytest.approx(o["mean_diff"], abs=1e-12)
        assert [sum(t[k] for t in ts) for k in ("wins", "losses", "ties")] == [o["wins"], o["losses"], o["ties"]]


def test_the_overall_block_equals_the_real_metrics_cli_paired_block(ex, tmp_path):
    root = world(tmp_path)
    res, _ = run(ex, root)
    cli = tmp_path / "cli.json"
    subprocess.run([sys.executable, "-m", "rlexp.metrics", "report", "--scored", str(root / "final/A/scored.v3.jsonl"), "--vs", str(root / "final/B/scored.v3.jsonl"), "--split", "test_t",
                    "--data", str(root / "no-data"), "--boot", "20", "--seed", "0", "--json", str(cli)], check=True, capture_output=True)
    want = json.loads(cli.read_text())["splits"]["test_t"]["paired"]["metrics"]["fv_tanimoto"]
    got = res["overall"]
    assert (got["mean_a"], got["mean_b"], got["mean_diff"]) == (want["a"], want["b"], want["diff"])                                   # the very same drug-level means
    assert [got[k] for k in ("wins", "losses", "ties", "p_value", "p_method", "n_drugs", "n_clusters", "p_unit")] == [want[k] for k in
                                                                                                                       ("wins", "losses", "ties", "p_value", "p_method", "n_drugs", "n_clusters", "p_unit")]


def test_p_values_print_like_the_metrics_report(ex):
    assert [ex.p_text(p) for p in (None, 1.0, 0.1234, 0.00031, 0.0001, 0.00009, 1e-5)] == ["n/a", "1.0000", "0.1234", "0.0003", "0.0001", "9.0e-05", "1.0e-05"]


def test_the_difference_is_a_minus_b_and_swapping_the_runs_negates_it(ex, tmp_path):
    res, _ = run(ex, world(tmp_path))
    back, _ = run(ex, tmp_path, a="B", b="A")
    assert res["sign"].startswith("A minus B") and (res["a"], res["b"]) == ("A", "B") and (back["a"], back["b"]) == ("B", "A")
    for covariate in ("pubmed_count", "record_age"):
        for t, u in zip(res["covariates"][covariate]["terciles"], back["covariates"][covariate]["terciles"]):
            assert u["mean_diff"] == pytest.approx(-t["mean_diff"], abs=1e-12) and (u["wins"], u["losses"], u["ties"]) == (t["losses"], t["wins"], t["ties"]) and u["p_value"] == t["p_value"]
            assert (u["mean_a"], u["mean_b"]) == (t["mean_b"], t["mean_a"])
    assert res["overall"]["mean_diff"] > -1 and back["overall"]["mean_diff"] == pytest.approx(-res["overall"]["mean_diff"], abs=1e-12)


def test_the_sign_flip_is_on_clusters_so_drugs_of_one_component_are_one_unit(ex, tmp_path):
    # zqdrug2 and zqdrug4 (both +0.10, both in the low PubMed tercile) share a component: one unit of total +0.20 and one of +0.10, not three units
    shared = {g: n for n, g in enumerate(DRUGS)} | {"zqdrug4": 1}
    res, _ = run(ex, world(tmp_path, cluster=shared))
    low = tercile_of(res, "pubmed_count")["low"]
    assert (low["wins"], low["n_clusters"], low["p_unit"]) == (3, 2, "cluster") and low["p_value"] == 0.5 == pytest.approx(exact_p([0.2, 0.1]))       # 0.25 if each drug were a unit
    assert res["overall"]["n_clusters"] == 8


def test_the_p_value_is_monte_carlo_above_twenty_untied_units_and_the_method_is_said(ex, tmp_path):
    drugs = [f"zqmany{i:02d}" for i in range(24)]
    a = {g: 0.3 + (0.01 if i % 3 else -0.005) for i, g in enumerate(drugs)}
    b = {g: 0.3 for g in drugs}
    root = world(tmp_path, a, b, refs=refs_rows(pubmed={g: i for i, g in enumerate(drugs)}, create={g: 1990 + i for i, g in enumerate(drugs)}, approval={g: 2025 for g in drugs}, drugs=drugs))
    res, _ = run(ex, root)
    assert res["overall"]["p_method"] == "monte_carlo" and res["overall"]["n_drugs"] == 24 and [t["n_drugs"] for t in res["covariates"]["pubmed_count"]["terciles"]] == [8, 8, 8]
    assert all(t["p_method"] == "exact" for t in res["covariates"]["pubmed_count"]["terciles"])
    cli = tmp_path / "cli.json"                                                                      # the seeded Monte Carlo p-value is the CLI's (seed 0), to the last digit
    subprocess.run([sys.executable, "-m", "rlexp.metrics", "report", "--scored", str(root / "final/A/scored.v3.jsonl"), "--vs", str(root / "final/B/scored.v3.jsonl"), "--split", "test_t",
                    "--data", str(root / "no-data"), "--boot", "20", "--seed", "0", "--json", str(cli)], check=True, capture_output=True)
    want = json.loads(cli.read_text())["splits"]["test_t"]["paired"]["metrics"]["fv_tanimoto"]
    assert res["overall"]["p_value"] == want["p_value"] and 0 < want["p_value"] < 1


# ------------------------------------------------------------------------------------------------- covariates and what is missing
def test_record_age_is_the_approval_year_minus_the_record_year_and_the_range_reports_it(ex, tmp_path):
    res, _ = run(ex, world(tmp_path))
    assert [t["range"] for t in res["covariates"]["record_age"]["terciles"]] == [[3, 5], [6, 8], [9, 11]]
    assert all(r["approval_year"] - r["pubchem_create_year"] == AGE[r["group"]] for r in refs_rows())       # the fixture itself: age = approval year minus record year
    shifted = {g: y - 1 for g, y in CREATE.items()}                                                       # records one year older: every age one year more
    root = world(tmp_path / "shifted", refs=refs_rows(create=shifted))
    res, _ = run(ex, root)
    assert [t["range"] for t in res["covariates"]["record_age"]["terciles"]] == [[4, 6], [7, 9], [10, 12]]


def test_drugs_without_a_covariate_are_counted_in_missing_and_left_out_of_that_covariate_only(ex, tmp_path):
    pubmed = {g: v for g, v in PUBMED.items() if g not in ("zqdrug1", "zqdrug5")}                      # two without a PubMed count
    approval = {g: y for g, y in APPROVAL.items() if g != "zqdrug3"}                                    # one without an approval year
    create = {g: v for g, v in CREATE.items() if g != "zqdrug8"}                                        # one without a record year
    res, _ = run(ex, world(tmp_path, refs=refs_rows(pubmed=pubmed, create=create, approval=approval)))
    assert res["missing"] == {"pubmed_count": 2, "pubchem_create_year": 1, "approval_year": 1, "record_age": 2}
    p, a = res["covariates"]["pubmed_count"], res["covariates"]["record_age"]
    assert (p["n_drugs"], p["missing"], a["n_drugs"], a["missing"]) == (7, 2, 7, 2)
    assert sum(t["n_drugs"] for t in p["terciles"]) == 7 and [t["n_drugs"] for t in p["terciles"]] == [3, 2, 2]      # 7 drugs: rank * 3 // 7
    members = {g for g in DRUGS if g in pubmed}
    assert sum(t["n_drugs"] * t["mean_diff"] for t in p["terciles"]) / 7 == pytest.approx(mean([DIFF[g] for g in members]), abs=1e-12)
    assert res["n_drugs"] == 9 and res["overall"]["n_drugs"] == 9                                      # the overall contrast keeps every common drug
    assert "2 do not and are left out" in (tmp_path / "res" / "exposure.md").read_text()


def test_a_covariate_that_is_not_a_number_counts_as_missing_not_as_a_value(ex, tmp_path):
    rows = refs_rows()
    for r in rows:
        if r["group"] == "zqdrug1":
            r["pubmed_count"] = "unknown"                                                                # text
        if r["group"] == "zqdrug2":
            r["pubchem_create_year"] = True                                                              # a bool is not a year
        if r["group"] == "zqdrug3":
            r["approval_year"] = None
    res, _ = run(ex, world(tmp_path, refs=rows))
    assert res["missing"] == {"pubmed_count": 1, "pubchem_create_year": 1, "approval_year": 1, "record_age": 2}
    assert res["covariates"]["pubmed_count"]["n_drugs"] == 8 and res["covariates"]["record_age"]["n_drugs"] == 7


def test_a_drug_with_no_refs_row_misses_every_covariate(ex, tmp_path):
    res, _ = run(ex, world(tmp_path, refs=[r for r in refs_rows() if r["group"] != "zqdrug7"]))
    assert res["missing"] == {"pubmed_count": 1, "pubchem_create_year": 1, "approval_year": 1, "record_age": 1}


def test_exposure_json_fills_a_value_the_refs_lack_but_never_overrides_the_refs(ex, tmp_path):
    rows = refs_rows()
    for r in rows:
        if r["group"] == "zqdrug1":
            r["pubmed_count"] = None
        if r["group"] == "zqdrug2":
            r["pubchem_create_year"] = None
    exposure = {"zqdrug1": {"pubchem_cid": 1, "pubchem_create_year": 1999, "pubmed_count": PUBMED["zqdrug1"]}, "zqdrug2": {"pubchem_create_year": APPROVAL["zqdrug2"] - AGE["zqdrug2"],
                "pubmed_count": 7777}, "zqdrug3": {"pubmed_count": 1}}                       # zqdrug3 has refs values, which win
    res, _ = run(ex, world(tmp_path, refs=rows, exposure=exposure))
    assert res["missing"] == {"pubmed_count": 0, "pubchem_create_year": 0, "approval_year": 0, "record_age": 0}
    assert tercile_of(res, "pubmed_count")["high"]["range"] == [70, 90] and tercile_of(res, "record_age")["low"]["range"] == [3, 5]


def test_refs_of_other_splits_are_not_used(ex, tmp_path):
    rows = refs_rows("test_r")                                                                       # the same group names, but refs of Test-R
    with pytest.raises(SystemExit, match="0 of 9 drugs have pubmed_count"):
        run(ex, world(tmp_path, refs=rows))
    rows = refs_rows()[:6] + [{**r, "pubmed_count": 99999} for r in refs_rows("test_r")[6:]]                # three drugs only have a Test-R row, with another count
    res, _ = run(ex, world(tmp_path, refs=rows))
    assert res["missing"]["pubmed_count"] == 3 and res["covariates"]["pubmed_count"]["n_drugs"] == 6 and max(t["range"][1] for t in res["covariates"]["pubmed_count"]["terciles"]) == 90


def test_fewer_than_three_drugs_with_a_covariate_is_an_error_not_an_empty_tercile(ex, tmp_path):
    pubmed = {"zqdrug1": 5, "zqdrug2": 6}
    world(tmp_path, refs=refs_rows(pubmed=pubmed))
    with pytest.raises(SystemExit, match="error: 2 of 9 drugs have pubmed_count: terciles need three"):
        run(ex, tmp_path)
    assert not (tmp_path / "res" / "exposure.json").exists()


# ------------------------------------------------------------------------------------------------- which drugs and rows
def test_only_the_drugs_both_runs_cover_are_analysed(ex, tmp_path):
    a = {g: 0.2 + DIFF[g] for g in DRUGS} | {"zqonlya": 0.9}
    b = {g: 0.2 for g in DRUGS} | {"zqonlyb": 0.0}
    res, _ = run(ex, world(tmp_path, a, b, refs=refs_rows(drugs=[*DRUGS, "zqonlya", "zqonlyb"], pubmed=PUBMED | {"zqonlya": 1, "zqonlyb": 2},
                                                          create=CREATE | {"zqonlya": 2024, "zqonlyb": 2024}, approval=APPROVAL | {"zqonlya": 2025, "zqonlyb": 2025})))
    assert res["n_drugs"] == 9 and res["overall"]["n_drugs"] == 9 and res["missing"]["pubmed_count"] == 0
    assert tercile_of(res, "pubmed_count")["low"]["range"] == [10, 30]                               # the covariate ranges are those of the nine
    assert res["overall"]["mean_diff"] == pytest.approx(mean(list(DIFF.values())), abs=1e-12)


def test_rows_of_other_splits_are_ignored(ex, tmp_path):
    root = world(tmp_path)
    other = scored("val", {"zqval": 0.9})
    for arm in ("A", "B"):
        path = root / "final" / arm / "scored.v3.jsonl"
        path.write_text(path.read_text() + "".join(json.dumps(r) + "\n" for r in other))
    res, _ = run(ex, root)
    assert res["n_drugs"] == 9 and res["split"] == "test_t"


def test_a_drugs_prompt_rows_are_averaged_inside_the_drug(ex, tmp_path):
    a = {g: 0.2 + DIFF[g] for g in DRUGS}
    root = world(tmp_path, a=a, rows=2)
    res, _ = run(ex, root)
    assert res["overall"]["mean_diff"] == pytest.approx(mean(list(DIFF.values())), abs=1e-12) and res["overall"]["n_drugs"] == 9


def test_fv_tanimoto_must_be_defined_for_every_common_drug(ex, tmp_path):
    root = world(tmp_path, k=8)
    path = root / "final" / "A" / "scored.v3.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows if not (r["group"] == "zqdrug9" and r["sample_idx"] >= 5)))          # one drug: 5 samples per row
    with pytest.raises(SystemExit, match="fv_tanimoto is defined for 8 of 9 common drugs"):
        run(ex, root)
    for arm in ("A", "B"):                                                                           # k = 4 everywhere: no fv_tanimoto at all
        (root / "final" / arm / "scored.v3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored("test_t", {g: 0.3 for g in DRUGS}, k=4)))
    with pytest.raises(SystemExit, match="defined for 0 of 9"):
        run(ex, root)


# ------------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("split", ["test_t", "test_r"])
def test_a_test_split_needs_allow_test_and_val_does_not(ex, tmp_path, split):
    world(tmp_path, split="val", refs=refs_rows("val"))
    with pytest.raises(SystemExit, match="refused: .* is a Test split"):
        run(ex, tmp_path, split=split, allow=False)
    res, md = run(ex, tmp_path, split="val", allow=False)
    assert res["split"] == "val" and "on Val (exploratory)" in md
    for bad in ("train", "test", ""):
        with pytest.raises(SystemExit, match="error: splits must come from"):
            run(ex, tmp_path, split=bad, allow=False)


def test_missing_files_are_listed_and_the_arm_ids_must_be_plain_and_different(ex, tmp_path):
    root = world(tmp_path)
    (root / "final" / "B" / "scored.v3.jsonl").unlink()
    (root / "refs.jsonl").unlink()
    with pytest.raises(SystemExit) as e:
        run(ex, root)
    assert str(e.value).startswith("error: missing input file(s)") and "B/scored.v3.jsonl" in str(e.value) and "refs.jsonl" in str(e.value) and "A/scored" not in str(e.value)
    for a, b in (("A", "A"), ("../A", "B"), ("A/x", "B"), ("A", ".."), ("", "B")):
        with pytest.raises(SystemExit, match="two different plain arm ids"):
            run(ex, root, a=a, b=b)


def test_a_run_with_no_rows_on_the_split_and_a_bad_exposure_file_are_plain_errors(ex, tmp_path):
    root = world(tmp_path)
    with pytest.raises(SystemExit, match="error: A has no rows on test_r"):
        run(ex, root, split="test_r")
    (root / "exposure.json").write_text("[1, 2]")
    with pytest.raises(SystemExit, match="is not a drug-to-covariates object"):
        run(ex, root)
    (root / "exposure.json").write_text("{not json")
    with pytest.raises(SystemExit, match="error: "):
        run(ex, root)
    assert not (root / "res" / "exposure.json").exists()


def test_a_scored_file_the_metrics_module_refuses_is_a_clean_error_without_ids(ex, tmp_path):
    root = world(tmp_path)
    path = root / "final" / "A" / "scored.v3.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    path.write_text("".join(json.dumps({**r, "cluster_id": 5} if r["group"] == "zqdrug1" else r) + "\n" for r in rows))        # run A puts a drug in another component than B does
    with pytest.raises(SystemExit) as e:
        run(ex, root)
    assert str(e.value).startswith("error: ") and "zqdrug" not in str(e.value)
    path.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "id"}) + "\n" for r in rows))                  # rows without an id: the metrics module's message, not a KeyError
    with pytest.raises(SystemExit, match="error: .*lacks"):
        run(ex, root)


# ------------------------------------------------------------------------------------------------- output
def test_every_number_is_labelled_exploratory_and_the_markdown_says_there_is_no_adjustment(ex, tmp_path):
    res, md = run(ex, world(tmp_path))
    assert res["label"] == "exploratory" and all(c["label"] == "exploratory" and all(t["label"] == "exploratory" for t in c["terciles"]) for c in res["covariates"].values())
    assert res["metric"] == "fv_tanimoto" and "A minus B" in res["sign"] and res["split"] == "test_t"
    assert "(exploratory)" in md.splitlines()[0] and "no adjustment for multiplicity" in md and "Test-T" in md
    assert "## PubMed articles (lowest to highest third)" in md and "## Years the structure was public before approval (lowest to highest third)" in md
    assert "| low | 10 to 30 | 3 |" in md and "| high | 9 to 11 | 3 |" in md and "p (sign-flip)" in md and "A positive difference means A is higher" in md and "holds only 3 drugs" in md
    assert "fv_" not in md and "pubmed_count" not in md and "record_age" not in md                       # plain English


def test_the_output_has_no_drug_name_id_sample_or_molecule_and_is_deterministic(ex, tmp_path, capsys):
    root = world(tmp_path)
    res, md = run(ex, root)
    first = (root / "res" / "exposure.json").read_bytes()
    out1 = capsys.readouterr().out
    run(ex, root)
    assert (root / "res" / "exposure.json").read_bytes() == first and (root / "res" / "exposure.md").read_text() == md and capsys.readouterr().out == out1
    seen = first.decode() + md + out1
    for marker in (*DRUGS, "zqdrug", SMILES_MARKER):
        assert marker not in seen
    assert set(res) == {"a", "b", "split", "metric", "sign", "label", "definitions", "n_drugs", "overall", "missing", "covariates"}
    assert "approval year minus the creation year" in res["definitions"]["record_age"] and "rank * 3 // n" in res["definitions"]["terciles"]
