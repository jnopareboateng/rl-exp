"""Tests for scripts/rank_readout.py, including a mutation pass (15 mutants, all killed).

The decisive property is that a policy whose answer does not depend on the drug scores EXACTLY chance: hit@1 = 1/N and
mean percentile rank = (N+1)/2N. Most mutations of the rank arithmetic break that equality, which is why the battery
below leads with it.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))                     # rank_readout imports final_table from scripts/
SOURCE = (ROOT / "scripts" / "rank_readout.py").read_text(encoding="utf-8")


def load(source=SOURCE, name="rank_readout"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "rank_readout.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    exec(compile(source, str(ROOT / "scripts" / "rank_readout.py"), "exec"), mod.__dict__)
    return mod


rr = load()

# Four well-known public structures, used only as test fixtures.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
REFS = {"a": ASPIRIN, "b": CAFFEINE, "c": IBUPROFEN, "d": PARACETAMOL}


def data_rows(split="val", clusters=None):
    clusters = clusters or {g: i for i, g in enumerate(REFS)}
    return [{"id": f"{g}-1", "group": g, "split": split, "cluster_id": clusters[g], "ref_smiles": s} for g, s in REFS.items()]


def sample(group, answer, idx=0, valid=True, closed=True, finish="stop", split="val", row=1):
    return {"id": f"{group}-{row}", "group": group, "split": split, "sample_idx": idx, "answer": answer,
            "valid": valid, "closed": closed, "finish_reason": finish}


def candidates(split="val"):
    return rr.candidates(data_rows(split), split)


def constant_rows(answer=ASPIRIN, split="val"):
    return [sample(g, answer, split=split) for g in REFS]


def perfect_rows(split="val"):
    return [sample(g, REFS[g], split=split) for g in REFS]


def eight(rows):
    """Every row repeated as eight samples (idx 0 to 7): the cohort the endpoint, and so the CLI, requires."""
    return [{**r, "sample_idx": i} for r in rows for i in range(8)]


# --- the battery: every mutant must break at least one of these ---------------------------------------------------

def battery(mod):
    groups, mols = mod.candidates(data_rows(), "val")
    n = len(groups)
    assert (n, groups) == (4, ["a", "b", "c", "d"])

    # 1. a drug-independent answer scores exactly chance, whatever it answers
    for answer in (ASPIRIN, CAFFEINE, PARACETAMOL):
        drugs = mod.per_drug([sample(g, answer) for g in REFS], "val", groups, mols)
        hit = sum(d["hit@1"] for d in drugs.values()) / n
        pct = sum(d["pct_rank"] for d in drugs.values()) / n
        assert hit == pytest.approx(1 / n), f"constant answer scored {hit} on hit@1, chance is {1/n}"
        assert pct == pytest.approx((n + 1) / (2 * n)), f"constant answer scored {pct} on percentile rank"
    assert mod.chance(n) == {"hit@1": pytest.approx(0.25), "pct_rank": pytest.approx(0.625)}

    # 2. answering its own reference is a top-1 hit for every drug
    drugs = mod.per_drug(perfect_rows(), "val", groups, mols)
    assert all(d["hit@1"] == 1.0 and d["pct_rank"] == pytest.approx(1 / n) for d in drugs.values())

    # 3. answering another drug's reference never hits, and ranks the own drug worse than chance on average
    swapped = [sample("a", REFS["b"]), sample("b", REFS["a"]), sample("c", REFS["d"]), sample("d", REFS["c"])]
    drugs = mod.per_drug(swapped, "val", groups, mols)
    assert all(d["hit@1"] == 0.0 for d in drugs.values())
    assert sum(d["pct_rank"] for d in drugs.values()) / n > (n + 1) / (2 * n)

    # 4. a row with no usable answer takes the worst rank and lowers the valid-answer share
    for broken in (dict(valid=False), dict(closed=False), dict(finish="length"), dict(finish="abort")):
        drugs = mod.per_drug([sample("a", REFS["a"], **broken)], "val", groups, mols)
        assert drugs["a"]["pct_rank"] == 1.0 and drugs["a"]["hit@1"] == 0.0 and drugs["a"]["fv_rate"] == 0.0, broken
    drugs = mod.per_drug([sample("a", "not a molecule")], "val", groups, mols)
    assert drugs["a"]["pct_rank"] == 1.0 and drugs["a"]["fv_rate"] == 0.0

    # 5. the FIRST valid answer is taken, not the best and not the last
    rows = [sample("a", REFS["b"], idx=0, valid=False), sample("a", REFS["b"], idx=1), sample("a", REFS["a"], idx=2)]
    assert mod.per_drug(rows, "val", groups, mols)["a"]["hit@1"] == 0.0, "the first valid answer must win over a better later one"
    assert mod.first_valid(rows, "val")[("a", "a-1")] == REFS["b"]

    # 6. only the first eight samples of a row count
    late = [sample("a", REFS["a"], idx=i, valid=False) for i in range(8)] + [sample("a", REFS["a"], idx=8)]
    assert mod.per_drug(late, "val", groups, mols)["a"]["fv_rate"] == 0.0, "a valid answer at sample 8 is past the endpoint's k"
    last = [sample("a", REFS["a"], idx=i, valid=False) for i in range(7)] + [sample("a", REFS["a"], idx=7)]
    assert mod.per_drug(last, "val", groups, mols)["a"]["fv_rate"] == 1.0, "the eighth sample (index 7) is inside the endpoint's k"

    # 7. a drug's rows are averaged, then drugs count equally
    two_rows = [sample("a", REFS["a"], row=1), sample("a", REFS["b"], row=2)] + [sample(g, REFS["b"]) for g in "bcd"]
    drugs = mod.per_drug(two_rows, "val", groups, mols)
    assert drugs["a"]["rows"] == 2 and drugs["a"]["hit@1"] == pytest.approx(0.5)

    # 8. ties share their mid-rank, so an answer equidistant from everything is not a top-1 hit
    ties = mod.rank_of(ASPIRIN, "a", ["a", "a2"], [mod.chem.standardize(ASPIRIN), mod.chem.standardize(ASPIRIN)])
    assert ties == pytest.approx(1.5), f"two identical references must share rank 1.5, got {ties}"

    # 9. the contrast sign and the wins follow each measure's direction (lower percentile rank is better, higher hit@1 is)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm, rows in (("good", perfect_rows()), ("flat", constant_rows())):
            (root / arm).mkdir()
            (root / arm / mod.SCORED).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        c = mod.contrast(root, ("good", "flat"), "val", groups, mols, mod.clusters(data_rows(), "val"))
        pct, hit = c["measures"]["pct_rank"], c["measures"]["hit@1"]
        assert pct["diff"] < 0 and pct["wins"] > pct["losses"], f"the better arm must have a negative percentile difference, got {pct['diff']}"
        assert hit["diff"] > 0 and hit["wins"] > hit["losses"], f"the better arm must have a positive hit@1 difference, got {hit['diff']}"
        assert pct["wins"] == hit["wins"] == 3 and pct["n_clusters"] == 4
    return True


def test_battery_passes_on_the_real_script():
    assert battery(rr)


# --- the contrast, through the files the script reads -------------------------------------------------------------

def write_run(root, arm, rows):
    (root / arm).mkdir(parents=True, exist_ok=True)
    (root / arm / rr.SCORED).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def contrast_fixture(tmp_path, clusters=None):
    groups, mols = candidates()
    write_run(tmp_path, "good", perfect_rows())
    write_run(tmp_path, "flat", constant_rows())
    cluster_of = rr.clusters(data_rows(clusters=clusters), "val")
    return groups, mols, cluster_of


def test_a_better_arm_gives_a_negative_percentile_difference_and_wins(tmp_path):
    groups, mols, cluster_of = contrast_fixture(tmp_path)
    c = rr.contrast(tmp_path, ("good", "flat"), "val", groups, mols, cluster_of)
    pct, hit = c["measures"]["pct_rank"], c["measures"]["hit@1"]
    # the constant answer IS the reference of one of the four drugs, so that drug ties and the other three are wins
    assert pct["diff"] < 0 and (pct["wins"], pct["losses"], pct["ties"]) == (3, 0, 1)
    assert hit["diff"] > 0 and (hit["wins"], hit["losses"], hit["ties"]) == (3, 0, 1)
    assert c["n_drugs"] == 4 and pct["n_clusters"] == 4


def test_the_p_value_is_the_cluster_sign_flip_of_the_component_totals(tmp_path):
    import numpy as np
    from rlexp import metrics
    groups, mols, cluster_of = contrast_fixture(tmp_path)
    c = rr.contrast(tmp_path, ("good", "flat"), "val", groups, mols, cluster_of)
    # three components differ (the fourth drug's own reference is the constant, so its difference is a tie and is dropped)
    expected, method, non_tied = metrics.sign_flip_p(np.array([0.25 - 0.625] * 3 + [0.0]), np.random.default_rng(0))
    assert c["measures"]["pct_rank"]["p_value"] == pytest.approx(expected) and c["measures"]["pct_rank"]["p_method"] == method
    assert (c["measures"]["pct_rank"]["non_tied"], non_tied) == (3, 3)
    assert c["measures"]["pct_rank"]["p_value"] == pytest.approx(0.25)      # 2^(1-3) with three units that agree


def test_related_drugs_share_one_exchangeable_unit(tmp_path):
    groups, mols, cluster_of = contrast_fixture(tmp_path, clusters={"a": 1, "b": 1, "c": 1, "d": 1})
    c = rr.contrast(tmp_path, ("good", "flat"), "val", groups, mols, cluster_of)
    assert c["measures"]["pct_rank"]["n_clusters"] == 1 and c["measures"]["pct_rank"]["p_value"] == pytest.approx(1.0)


def test_arms_with_no_drug_in_common_compare_to_nothing(tmp_path):
    groups, mols, cluster_of = contrast_fixture(tmp_path)
    write_run(tmp_path, "other", [sample("a", ASPIRIN, split="test_r")])
    c = rr.contrast(tmp_path, ("other", "flat"), "val", groups, mols, cluster_of)
    assert c["measures"] == {} and c["n_drugs"] == 0


def test_rows_of_another_split_are_ignored(tmp_path):
    groups, mols = candidates()
    rows = perfect_rows() + [sample(g, ASPIRIN, split="test_t") for g in REFS]
    drugs = rr.per_drug(rows, "val", groups, mols)
    assert all(d["rows"] == 1 for d in drugs.values()) and all(d["hit@1"] == 1.0 for d in drugs.values())


def test_a_reference_that_does_not_parse_is_an_error():
    from rlexp import metrics
    rows = data_rows()
    rows[0]["ref_smiles"] = "not a molecule"
    with pytest.raises(metrics.InputError):
        rr.candidates(rows, "val")
    with pytest.raises(metrics.InputError):
        rr.candidates([], "val")


# --- the command line -------------------------------------------------------------------------------------------

def test_a_test_split_needs_allow_test(tmp_path):
    with pytest.raises(SystemExit) as e:
        rr.main(["--final", str(tmp_path), "--split", "test_t", "--arms", "flat", "--out", str(tmp_path / "o.json")])
    assert "--allow-test" in str(e.value)


def test_bad_inputs_exit(tmp_path):
    for argv in (["--split", "val", "--arms", "../etc", "--out", "o.json"],            # not a plain id
                 ["--split", "val", "--a", "x", "--out", "o.json"],                     # --a without --b
                 ["--split", "val", "--out", "o.json"],                                # nothing to do
                 ["--split", "val", "--pairs", "x:x", "--out", "o.json"]):              # a pair of one arm
        with pytest.raises(SystemExit):
            rr.main(["--final", str(tmp_path)] + argv)


def test_the_output_file_records_the_chance_values_and_is_not_confirmatory(tmp_path):
    write_run(tmp_path, "flat", eight(constant_rows()))
    data = tmp_path / "data"
    data.mkdir()
    (data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data_rows()), encoding="utf-8")
    out, md = tmp_path / "o.json", tmp_path / "o.md"
    rr.main(["--final", str(tmp_path), "--split", "val", "--arms", "flat", "--data", str(data), "--out", str(out), "--md", str(md)])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["confirmatory"] is False and res["n_candidates"] == 4
    assert res["chance"] == {"hit@1": pytest.approx(0.25), "pct_rank": pytest.approx(0.625)}
    assert res["arms"][0]["hit@1"] == pytest.approx(0.25) and res["arms"][0]["pct_rank"] == pytest.approx(0.625)
    assert res["arms"][0]["n_distinct"] == 1, "one fixed answer is one distinct answer"
    text = md.read_text(encoding="utf-8")
    assert "Nothing in this table is confirmatory" in text and "0.625" in text and "Distinct answers" in text
    assert not any(s in text for s in (ASPIRIN, CAFFEINE, IBUPROFEN, PARACETAMOL)), "no structure may reach the report"


# --- guards and statistics added after the independent review ---------------------------------------------------

def fake_per_drug(table):
    return lambda rows, split, groups, mols: {g: {rr.PCT: v[0], rr.HIT: v[1], "fv_rate": 1.0, "rows": 1} for g, v in table.items()}


def test_the_p_value_tests_component_sums_and_wins_count_drugs(tmp_path, monkeypatch):
    # three components: A holds three drugs (-0.1 each, total -0.3), B one drug (-0.2), C one drug (+0.25). Sums give p = 6/8 = 0.75; means would give 1.0
    cluster_of = {"g1": "A", "g2": "A", "g3": "A", "g4": "B", "g5": "C"}
    a = {"g1": (0.4, 0), "g2": (0.4, 0), "g3": (0.4, 0), "g4": (0.3, 0), "g5": (0.75, 0)}
    b = {g: (0.5, 0) for g in a}
    write_run(tmp_path, "x", perfect_rows())
    write_run(tmp_path, "y", perfect_rows())
    tables = iter([a, b])
    monkeypatch.setattr(rr, "per_drug", lambda rows, split, groups, mols: fake_per_drug(next(tables))(rows, split, groups, mols))
    m = rr.contrast(tmp_path, ("x", "y"), "val", ["g"], [None], cluster_of)["measures"][rr.PCT]
    assert m["p_value"] == pytest.approx(0.75) and m["n_clusters"] == 3
    assert (m["wins"], m["losses"], m["ties"]) == (4, 1, 0), "wins count drugs, not components"
    assert m["diff"] == pytest.approx((-0.1 * 3 - 0.2 + 0.25) / 5), "the difference is the drug-balanced mean"


def test_a_contrast_of_more_than_twenty_components_is_deterministic(tmp_path, monkeypatch):
    import numpy as np
    rng = np.random.default_rng(7)
    diffs = rng.normal(-0.02, 0.05, 25)
    cluster_of = {f"g{i}": f"c{i}" for i in range(25)}
    a = {f"g{i}": (0.5 + d, 0) for i, d in enumerate(diffs)}
    b = {g: (0.5, 0) for g in a}
    write_run(tmp_path, "x", perfect_rows())
    write_run(tmp_path, "y", perfect_rows())
    ps = []
    for _ in range(2):
        tables = iter([a, b])
        monkeypatch.setattr(rr, "per_drug", lambda rows, split, groups, mols, t=tables: fake_per_drug(next(t))(rows, split, groups, mols))
        m = rr.contrast(tmp_path, ("x", "y"), "val", ["g"], [None], cluster_of)["measures"][rr.PCT]
        ps.append(m["p_value"])
    assert m["p_method"] == "monte_carlo" and ps[0] == ps[1], "the Monte Carlo test must be seeded: the same inputs give the same p"
    assert ps[0] == pytest.approx(0.00010999890001099989), "pinned: 25 components, 100,000 flips, seed 0"


def test_the_arm_mean_is_over_drugs_not_rows(tmp_path):
    groups, mols = candidates()
    rows = [sample("a", REFS["a"], row=r) for r in (1, 2, 3)] + [sample(g, ASPIRIN) for g in "bcd"]
    write_run(tmp_path, "x", rows)
    rec = rr.arm_record(tmp_path, "x", "val", groups, mols)
    drugs = rr.per_drug(rows, "val", groups, mols)
    assert rec["hit@1"] == pytest.approx(sum(d["hit@1"] for d in drugs.values()) / 4) and rec["n_rows"] == 6
    assert rec["hit@1"] != pytest.approx(sum(d["hit@1"] * d["rows"] for d in drugs.values()) / 6)


def test_a_tie_for_the_nearest_reference_is_not_a_hit():
    from rlexp import chem
    twin = chem.standardize(ASPIRIN)
    rec = rr.per_drug([sample("a", ASPIRIN)], "val", ["a", "a2"], [twin, chem.standardize(ASPIRIN)])
    assert rec["a"]["hit@1"] == 0.0 and rec["a"]["pct_rank"] == pytest.approx(1.5 / 2)


def check(rows, split="val"):
    from rlexp import metrics
    rr.check_arm(rows, split, data_rows(split), "x")


def test_a_complete_arm_passes_the_coverage_check_and_other_splits_are_ignored():
    check(eight(constant_rows()) + [sample("zz", ASPIRIN, split="test_t")])


@pytest.mark.parametrize("what", ["a row with seven samples", "a missing drug", "a row the data does not have", "another drug than the data", "a repeated sample"])
def test_a_partial_or_foreign_arm_is_refused(what):
    from rlexp import metrics
    rows = eight(constant_rows())
    if what == "a row with seven samples":
        rows = [r for r in rows if not (r["group"] == "a" and r["sample_idx"] == 7)]
    elif what == "a missing drug":
        rows = [r for r in rows if r["group"] != "a"]
    elif what == "a row the data does not have":
        rows += eight([sample("zz", ASPIRIN)])
    elif what == "another drug than the data":
        rows = [{**r, "group": "b"} if r["group"] == "a" else r for r in rows]
    else:
        rows.append(dict(rows[0]))
    with pytest.raises(metrics.InputError):
        check(rows)


def test_the_cli_refuses_a_partial_arm(tmp_path):
    write_run(tmp_path, "flat", constant_rows())              # one sample per row: not the endpoint's cohort
    data = tmp_path / "data"
    data.mkdir()
    (data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data_rows()), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        rr.main(["--final", str(tmp_path), "--split", "val", "--arms", "flat", "--data", str(data), "--out", str(tmp_path / "o.json")])
    assert "does not cover" in str(e.value)
    assert not (tmp_path / "o.json").exists()


def test_reference_rows_must_agree_and_exist():
    from rlexp import metrics
    rows = data_rows()
    rows.append({**rows[0], "id": "a-2", "ref_smiles": CAFFEINE})
    with pytest.raises(metrics.InputError):
        rr.candidates(rows, "val")
    rows = data_rows()
    rows[1]["ref_smiles"] = ""
    with pytest.raises(metrics.InputError):
        rr.candidates(rows, "val")


def test_a_null_cluster_id_makes_the_drug_its_own_unit():
    rows = data_rows()
    for r in rows:
        r["cluster_id"] = None
    assert rr.clusters(rows, "val") == {g: g for g in REFS}


def test_the_number_of_distinct_answers_is_reported(tmp_path):
    groups, mols = candidates()
    write_run(tmp_path, "mixed", [sample("a", ASPIRIN), sample("b", ASPIRIN), sample("c", CAFFEINE), sample("d", "not a molecule")])
    assert rr.arm_record(tmp_path, "mixed", "val", groups, mols)["n_distinct"] == 2


# --- the Val numbers this was pinned to (skipped when the run directory is absent) --------------------------------

VAL_EXPECTED = {"val8-control_constant": (0.0500, 0.5250), "val8-control_constant_pool": (0.0500, 0.5250),
                "val8-control_copy_nearest": (0.1292, 0.4385), "a4-t0.6": (0.1500, 0.4642), "a2-val": (0.1875, 0.3957),
                "a2-shuffle-t1.0": (0.0000, 0.5659), "rlfull5-s50-t0.3": (0.1000, 0.4729),
                "r5s50-shuffle-t0.3": (0.0708, 0.5281), "r5s50-blank-t0.3": (0.0500, 0.5250),
                "b9-t0.6": (0.1125, 0.4235), "b27-direct": (0.1958, 0.3785)}


@pytest.mark.parametrize("arm,expected", sorted(VAL_EXPECTED.items()))
def test_val_numbers_are_reproduced(arm, expected):
    runs = ROOT / "runs" / "colab-s1"
    if not (runs / arm / rr.SCORED).is_file() or not (ROOT / "data" / "processed" / "val.jsonl").is_file():
        pytest.skip("the Val runs or the Val data file are not in this tree")
    from rlexp import metrics
    groups, mols = rr.candidates(metrics.load_data(ROOT / "data" / "processed", splits=("val",)), "val")
    r = rr.arm_record(runs, arm, "val", groups, mols)
    assert (round(r["hit@1"], 4), round(r["pct_rank"], 4)) == expected


def test_val_contrasts_are_reproduced():
    runs = ROOT / "runs" / "colab-s1"
    if not (runs / "a2-val" / rr.SCORED).is_file() or not (ROOT / "data" / "processed" / "val.jsonl").is_file():
        pytest.skip("the Val runs or the Val data file are not in this tree")
    from rlexp import metrics
    rows = metrics.load_data(ROOT / "data" / "processed", splits=("val",))
    groups, mols = rr.candidates(rows, "val")
    cluster_of = rr.clusters(rows, "val")
    for pair, diff, wlt, p in ((("a2-val", "a2-shuffle-t1.0"), -0.1702, (16, 4, 0), 0.0045),
                               (("rlfull5-s50-t0.3", "r5s50-shuffle-t0.3"), -0.0552, (12, 6, 2), 0.0270)):
        m = rr.contrast(runs, pair, "val", groups, mols, cluster_of)["measures"]["pct_rank"]
        assert round(m["diff"], 4) == diff and (m["wins"], m["losses"], m["ties"]) == wlt and round(m["p_value"], 4) == p


# --- mutation pass ----------------------------------------------------------------------------------------------

MUTANTS = (
    ("rank off by one", "return better + 1 + tied / 2", "return better + tied / 2"),
    ("ties ignored", "return better + 1 + tied / 2", "return better + 1"),
    ("self counted as nearer", "if s > own + metrics.TIE", "if s >= own - metrics.TIE"),
    ("hit@1 becomes hit@2", "float(rank <= 1)", "float(rank <= 2)"),
    ("a failed row becomes the best rank", "1.0 if rank is None else rank / n", "0.0 if rank is None else rank / n"),
    ("a failed row stops counting against validity", "float(rank is not None)", "1.0"),
    ("percentile not normalised", "else rank / n", "else rank / (n + 1)"),
    ("chance percentile wrong", "return {HIT: 1 / n, PCT: (n + 1) / (2 * n)}", "return {HIT: 1 / n, PCT: 0.5}"),
    ("chance hit wrong", "return {HIT: 1 / n, PCT: (n + 1) / (2 * n)}", "return {HIT: 1 / (n + 1), PCT: (n + 1) / (2 * n)}"),
    ("sample cap removed", '[:k]', '[:]'),
    ("sample order ignored", 'sorted(samples, key=lambda s: s["sample_idx"])[:k]', 'list(reversed(samples))[:k]'),
    ("length-terminated answers accepted", 'and metrics.finished(s)', 'and True'),
    ("unclosed answers accepted", 'if s.get("valid") and s.get("closed")', 'if s.get("valid") and True'),
    ("last valid answer taken", "                out[key] = s.get(\"answer\")\n                break", "                out[key] = s.get(\"answer\")"),
    ("rows summed instead of averaged", 'float(np.mean(v)) for k, v in d.items()', 'float(np.sum(v)) for k, v in d.items()'),
    ("contrast sign reversed", "diffs = {g: da[g][key] - db[g][key] for g in common}", "diffs = {g: db[g][key] - da[g][key] for g in common}"),
    ("hit@1 wins not flipped back", "        if key == HIT:                                                     # higher hit@1 is better: a win is a positive difference\n            wins, losses = losses, wins", "        if False:\n            wins, losses = losses, wins"),
)


@pytest.mark.parametrize("name,before,after", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_is_killed(name, before, after):
    assert SOURCE.count(before) == 1, f"the mutation target {before!r} is not unique in the source"
    mod = load(SOURCE.replace(before, after), name=f"rank_readout_mutant_{abs(hash(name))}")
    with pytest.raises((AssertionError, ZeroDivisionError, TypeError, KeyError, IndexError)):
        battery(mod)
