"""scripts/make_release.py on a synthetic repository tree: the allow-list and length guard, determinism, blanks instead of inventions, no Test or prompt file opened."""
import copy
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from rlexp import chem

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_release", ROOT / "scripts" / "make_release.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["make_release"] = mr
spec.loader.exec_module(mr)

ASPIRIN, PARACETAMOL, CAFFEINE, IBUPROFEN = "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
NAPROXEN, PROCAINAMIDE, POOL_DRUG = "COc1ccc2cc(C(C)C(=O)O)ccc2c1", "CCN(CC)CCNC(=O)c1ccc(N)cc1", "CC(=O)Nc1ccc(OC(C)=O)cc1"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ"            # first block of the standard InChIKey of aspirin


def ref(group, smiles, split, year=2000, source="chembl", cid=None, cluster=0, exposed=True):
    mol = chem.standardize(smiles) if smiles else None
    return {"group": group, "components": [group], "ref_smiles": chem.canonical(mol) if mol else None, "ref_inchikey": chem.inchikey(mol) if mol else None,
            "ref_conn": chem.conn(mol) if mol else None, "ref_scaffold": chem.scaffold(mol) if mol else None, "ref_source": "chembl" if mol else None,
            "chembl_id": cid, "approval_year": year, "approval_source": source, "sft_agrees": True, "ref_alt_smiles": [], "cluster_id": cluster, "split": split,
            "exclude_reason": "manual_unresolvable" if split == "excluded" else None, "n_rows": 2, "n_heavy": mol.GetNumHeavyAtoms() if mol else None,
            "convention": None, "pubchem_create_year": 2005 if exposed else None, "pubmed_count": 7 if exposed else None}


def default_refs():
    return [ref("aspirin", ASPIRIN, "train", 1950, cid="CHEMBL25", cluster=0), ref("paracetamol", PARACETAMOL, "train", 1955, cid="CHEMBL112", cluster=1),
            ref("caffeine", CAFFEINE, "train", 1980, cid="CHEMBL113", cluster=2, exposed=False), ref("ibuprofen", IBUPROFEN, "train", 1974, cid="CHEMBL521", cluster=3),
            ref("naproxen", NAPROXEN, "val", 1976, cid="CHEMBL154", cluster=4),
            ref("procainamide", PROCAINAMIDE, "test_t", 2025, source="audit", cluster=5), ref("unresolved", None, "excluded", 1999, cluster=-1)]


def write(path, obj, text=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if text else json.dumps(obj), encoding="utf-8")


def val_mean(answer, val_smiles):
    mol = chem.standardize(answer)
    return sum(chem.tanimoto(mol, chem.standardize(s)) for s in val_smiles) / len(val_smiles)


def make_tree(root, refs=None, regulator="EMA/EC (EU)", calibration=None, meta=True, ceiling_sha=None):
    """A tiny repository: 4 Train drugs, 1 Val, 1 Test-T (dated by the audit), 1 excluded, a two-drug pool and the results files that go with them."""
    rows = refs if refs is not None else default_refs()
    write(root / "data/processed/refs.jsonl", "".join(json.dumps(r) + "\n" for r in rows), text=True)
    counts = {s: sum(r["split"] == s for r in rows) for s in mr.SPLIT_NAMES}
    write(root / "data/processed/split_report.json", {"test_t_min_year": 2025, "splits": {s: {"drugs": n} for s, n in counts.items()}})
    write(root / "data/curated/chembl_refs.json", [{"name": r["group"], "status": "ok", "chembl_id": r["chembl_id"], "parent_smiles": r["ref_smiles"],
                                                    "approval_year": r["approval_year"], "first_approval": r["approval_year"]} for r in rows if r["chembl_id"]])
    write(root / "data/curated/approval_audit.json", {"procainamide": {"fda_year": 2025, "earliest_global_year": 2025, "earliest_regulator": regulator,
                                                                    "evidence": [], "confidence": "high"}})
    write(root / "data/curated/exposure.json", {r["group"]: {"pubchem_cid": 100 + len(r["group"]), "pubchem_create_year": 2005, "pubmed_count": 7} for r in rows if r["pubchem_create_year"]})
    write(root / "data/curated/manual_refs.json", {})
    pool = [{"chembl_id": "CHEMBL9001", "smiles": chem.canonical(chem.standardize(POOL_DRUG))}, {"chembl_id": "CHEMBL9002", "smiles": "CCO"}]
    pool_text = "".join(json.dumps(p) + "\n" for p in pool)
    sha = hashlib.sha256(pool_text.encode()).hexdigest()
    write(root / mr.POOL, pool_text, text=True)
    write(root / mr.POOL.replace("pool.jsonl", "manifest.json"), {"chembl_release": "ChEMBL_99", "release_date": "2000-01-01",
                                                                   "derived": {"files": {"pool.jsonl": {"records": 2, "sha256": sha}}}})
    write(root / "results/constant_ceiling.json", {"pool_sha256": ceiling_sha or sha, "splits": {
        "val": {"oracle": 0.5, "c3": 0.3, "c3_rank": 2, "top5": [0.5, 0.4], "candidates": 6, "drugs": 1},
        "test_t": {"oracle": 0.9, "c3": 0.8, "c3_rank": 1, "top5": [0.9], "candidates": 6, "drugs": 1}}})
    if calibration is not None:
        write(root / "results/random_pair_calibration.json", calibration)
    train = {r["group"]: r["ref_smiles"] for r in rows if r["split"] == "train"}
    c3 = mr.c3_answer(root, train, {"pool_sha256": sha})
    _, answers = mr.controls_table(train, {r["group"]: r for r in rows}, c3)
    val = [r["ref_smiles"] for r in rows if r["split"] == "val"]
    base = [{"run": run, "mean_tanimoto": val_mean(answers[c], val) if c in answers else 0.2, "fv_tanimoto": 0.2, "fv_tanimoto_ci": [0.1, 0.3], "mean_reward": 0.1,
             "lift": 0.0, "lift_ci": [-0.1, 0.1], "exact@1": 0.0, "core@1": 0.0, "valid_closed": 1.0, "fv_samples": 1.0, "fv_tokens": 0.0, "drugs": 1}
            for c, run in mr.RUNS.items()]
    write(root / "results/val_baselines.json", {"split": "val", "rows": base})
    if meta:
        write(root / mr.META, {"control": "constant_pool", "train_score": round(c3["score"], 4), "source": "pool", "candidates": c3["candidates"], "answer_chars": len(c3["smiles"])})
    return root


def run(root, out=None):
    out = out or root / "release"
    mr.main(["--root", str(root), "--out", str(out)])
    return out


def table(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def test_the_tables_hold_the_allow_listed_columns_and_one_row_per_usable_drug(tmp_path, capsys):
    out = run(make_tree(tmp_path))
    rows = table(out / "splits.csv")
    assert list(rows[0]) == list(mr.FIELDS["splits.csv"]) and [r["drug_name"] for r in rows] == sorted(r["drug_name"] for r in rows)
    assert len(rows) == 6 and "unresolved" not in {r["drug_name"] for r in rows}          # the excluded group is not listed
    aspirin = next(r for r in rows if r["drug_name"] == "aspirin")
    assert aspirin["inchikey_first_block"] == ASPIRIN_KEY and aspirin["heavy_atoms"] == "13" and aspirin["split"] == "train" and aspirin["chembl_id"] == "CHEMBL25"
    assert aspirin["smiles"] == chem.canonical(chem.standardize(ASPIRIN)) and aspirin["approval_source"] == "chembl"
    assert next(r for r in rows if r["drug_name"] == "procainamide")["approval_source"] == "EMA/EC (EU)"           # the audit's regulator, not the word audit
    ctl = {r["control"]: r for r in table(out / "controls.csv")}
    assert list(ctl) == ["C0", "C1", "C2", "C3"] and ctl["C1"]["smiles"] == "" and ctl["C3"]["chembl_id"].startswith("CHEMBL900") and ctl["C0"]["smiles"]
    assert "ChEMBL_99" in ctl["C3"]["selection_rule"] and re.search(r"sha256 [0-9a-f]{64}", ctl["C3"]["selection_rule"])
    assert "\n" not in ctl["C3"]["selection_rule"]
    assert not any(r["split"] == "test_t" for r in table(out / "values.csv"))
    assert "C0, C2 and C3 reproduce their Val means" in capsys.readouterr().out


def test_the_output_is_deterministic_whatever_the_input_order(tmp_path):
    a = run(make_tree(tmp_path / "a"))
    b = run(make_tree(tmp_path / "b", refs=list(reversed(default_refs()))))
    again = run(tmp_path / "a", tmp_path / "again")
    for name in ("splits.csv", "controls.csv", "values.csv", "README.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes() == (again / name).read_bytes(), name


def test_a_leaked_long_string_is_refused_and_nothing_is_written(tmp_path, capsys):
    leak = default_refs()
    leak[0]["group"] = "x" * 2500                                  # a prompt-sized string in a field that reaches the output
    root = make_tree(tmp_path, refs=leak)
    with pytest.raises(SystemExit, match=r"drug_name holds 2500 characters"):
        run(root)
    assert not (root / "release").exists()
    assert "x" * 50 not in capsys.readouterr().out


def test_a_free_text_source_label_is_refused(tmp_path):
    root = make_tree(tmp_path, regulator="a regulator note " * 6)          # over 80 characters: free text, not a label
    with pytest.raises(SystemExit, match="approval_source holds"):
        run(root)
    assert not (root / "release").exists()


def test_guard_refuses_a_field_outside_the_allow_list_a_non_scalar_and_an_unknown_split():
    ok = {"control": "C0", "chembl_id": "CHEMBL1", "smiles": "CCO", "selection_rule": "rule"}
    mr.guard("controls.csv", [ok])
    with pytest.raises(mr.ReleaseError, match=r"outside the allow-list: \['prompt'\]"):
        mr.guard("controls.csv", [{**ok, "prompt": "short"}])
    with pytest.raises(mr.ReleaseError, match="not text or a number"):
        mr.guard("controls.csv", [{**ok, "smiles": ["CCO"]}])
    with pytest.raises(mr.ReleaseError, match="characters"):
        mr.guard("controls.csv", [{**ok, "selection_rule": "y" * 2001}])
    with pytest.raises(mr.ReleaseError, match="unknown split"):
        mr.guard("splits.csv", [{"drug_name": "a", "split": "secret"}])


def test_numbers_are_rounded_never_negative_zero_and_only_numeric_leaves_are_read():
    assert (mr.fmt(-1e-18), mr.fmt(0.1234567), mr.fmt(7), mr.fmt(1.0)) == ("0.0", "0.123457", "7", "1.0")
    got = dict(mr.leaves({"a": [1, 2], "b": True, "c": "text", "d": list(range(11)), "e": {"f": 1.5}}))
    assert got == {"a/1": 1, "a/2": 2, "e/f": 1.5}                    # a bool, a string and a list of 11 are not read


def test_unknown_values_stay_blank_and_are_reported(tmp_path, capsys):
    out = run(make_tree(tmp_path))
    rows = {r["drug_name"]: r for r in table(out / "splits.csv")}
    assert rows["caffeine"]["pubchem_cid"] == rows["caffeine"]["pubchem_create_year"] == rows["caffeine"]["pubmed_count"] == ""          # no exposure entry: nothing invented
    assert rows["procainamide"]["chembl_id"] == ""
    log = capsys.readouterr().out
    assert "blank fields: chembl_id 1, pubchem_cid 1, pubchem_create_year 1, pubmed_count 1" in log
    assert "results/random_pair_calibration.json absent" in log and not [r for r in table(out / "values.csv") if r["table"] == "random_pair"]
    assert "chembl_id 1, pubchem_cid 1" in (out / "README.md").read_text()


def test_the_random_pair_file_is_used_when_it_exists_and_test_named_entries_are_skipped(tmp_path, capsys):
    cal = {"refs": "data/processed/refs.jsonl", "splits": ["train", "val"], "n_pairs": 10, "mean": 0.12, "percentiles": {"50": 0.1, "99": 0.3},
           "at": {"0.1760": {"percentile": 80.0}}, "test_t": {"mean": 0.9}}
    out = run(make_tree(tmp_path, calibration=cal))
    rp = {r["statistic"]: r for r in table(out / "values.csv") if r["table"] == "random_pair"}
    assert set(rp) == {"n_pairs", "mean", "percentiles/50", "percentiles/99", "at/0.1760/percentile"} and rp["mean"]["value"] == "0.12"
    assert rp["mean"]["split"] == "train+val" and rp["mean"]["source_file"] == "results/random_pair_calibration.json"
    assert "7 Test-named entries skipped" in capsys.readouterr().out                    # test_t: 1 value in the calibration, 6 in the ceiling file
    assert {r["split"] for r in table(out / "values.csv") if r["table"] == "oracle_constant"} == {"val"}


def test_a_chembl_id_for_a_different_compound_is_withheld_and_one_with_no_structure_is_kept():
    refs = {g["group"]: g for g in default_refs()}
    refs["aspirin"]["chembl_id"], refs["caffeine"]["chembl_id"] = "CHEMBL999", "CHEMBL113"
    chembl = {"aspirin": {"parent_smiles": CAFFEINE}, "caffeine": {"parent_smiles": None}, "paracetamol": {"parent_smiles": PARACETAMOL}}
    rows, notes = mr.splits_table(refs, chembl, {}, {})
    ids = {r["drug_name"]: r["chembl_id"] for r in rows}
    assert ids["aspirin"] is None and notes["withheld"] == ["aspirin"]
    assert ids["caffeine"] == "CHEMBL113" and "caffeine" in notes["unchecked"] and "paracetamol" not in notes["unchecked"] and ids["paracetamol"] == "CHEMBL112"
    refs["aspirin"]["ref_alt_smiles"] = [CAFFEINE]                             # a recorded alternate of the reference counts as agreement
    assert {r["drug_name"]: r["chembl_id"] for r in mr.splits_table(refs, chembl, {}, {})[0]}["aspirin"] == "CHEMBL999"


def test_year_issues_flag_every_kind_of_inconsistency():
    refs = {g["group"]: g for g in default_refs()}
    chembl = {g: {"approval_year": r["approval_year"], "first_approval": r["approval_year"]} for g, r in refs.items() if r["approval_source"] == "chembl"}
    audit = {"procainamide": {"fda_year": 2025, "earliest_global_year": 2025, "earliest_regulator": "FDA"}}
    assert mr.year_issues(refs, chembl, audit, {}, 2025) == []
    bad = copy.deepcopy(refs)
    bad["aspirin"]["approval_year"] = 1949                                     # differs from its source file
    bad["paracetamol"]["approval_year"] = 2026                                 # a Test-T year outside Test-T
    bad["procainamide"]["approval_source"] = "chembl"                          # an audit year exists but is not the source
    issues = mr.year_issues(bad, chembl, audit, {}, 2025)
    assert any(i.startswith("aspirin: year 1949") for i in issues) and any(i.startswith("paracetamol") and "Test-T rule" in i for i in issues)
    assert any(i.startswith("procainamide") and "audit year exists" in i for i in issues)
    contradictory = {"x": {"fda_year": 2025, "earliest_global_year": 2026, "earliest_regulator": "EMA"}, "y": {"fda_year": 2025, "earliest_global_year": 2024, "earliest_regulator": "FDA"}}
    assert [i.split(":")[0] for i in mr.year_issues({}, {}, contradictory, {}, 2025)] == ["x", "y"]


def test_released_controls_must_reproduce_the_val_means_on_record(tmp_path):
    root = make_tree(tmp_path)
    path = root / "results/val_baselines.json"
    base = json.loads(path.read_text())
    next(r for r in base["rows"] if r["run"] == mr.RUNS["C3"])["mean_tanimoto"] += 0.01
    write(path, base)
    with pytest.raises(SystemExit, match="C3: the released answer scores"):
        run(root)


def test_a_pool_that_is_not_the_recorded_one_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="pool_sha256"):
        run(make_tree(tmp_path / "a", ceiling_sha="0" * 64))
    root = make_tree(tmp_path / "b")
    (root / mr.POOL).write_text((root / mr.POOL).read_text() + "\n", encoding="utf-8")        # the bytes no longer match the manifest
    with pytest.raises(SystemExit, match="differs from the sha256"):
        run(root)
    (root / mr.POOL).unlink()
    with pytest.raises(SystemExit, match="needs the ChEMBL pool"):
        run(root)


def test_the_run_meta_is_cross_checked_when_present_and_a_disagreement_is_refused(tmp_path, capsys):
    root = make_tree(tmp_path)
    run(root)
    assert "C3 agrees with the run meta.json" in capsys.readouterr().out
    meta = json.loads((root / mr.META).read_text())
    write(root / mr.META, {**meta, "pool_winner": {"chembl_id": "CHEMBL0"}})
    with pytest.raises(SystemExit, match="pool_winner"):
        run(root, tmp_path / "other")
    (root / mr.META).unlink()
    run(root, tmp_path / "third")
    assert "run meta.json absent" in capsys.readouterr().out


def test_a_missing_control_row_is_an_error_not_a_traceback(tmp_path):
    root = make_tree(tmp_path)
    path = root / "results/val_baselines.json"
    base = json.loads(path.read_text())
    base["rows"] = [r for r in base["rows"] if r["run"] != mr.RUNS["C2"]]
    write(path, base)
    with pytest.raises(SystemExit, match="no row for"):
        run(root)


def test_the_table_and_the_split_report_must_agree(tmp_path):
    root = make_tree(tmp_path)
    path = root / "data/processed/split_report.json"
    rep = json.loads(path.read_text())
    rep["splits"]["train"]["drugs"] = 5
    write(path, rep)
    with pytest.raises(SystemExit, match=r"train': \(4, 5\)"):
        run(root)


def test_no_prompt_file_no_test_file_and_nothing_under_runs_final_is_opened(tmp_path, monkeypatch):
    root = make_tree(tmp_path)
    for p in ("data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/test_r.jsonl", "data/processed/test_t.jsonl", "runs/final/c0/samples.jsonl"):
        write(root / p, "PROMPT TEXT", text=True)
    seen = []
    real_text, real_bytes = Path.read_text, Path.read_bytes
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (seen.append(str(self)), real_text(self, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: (seen.append(str(self)), real_bytes(self, *a, **k))[1])
    out = run(root)
    assert seen and not [p for p in seen if re.search(r"/(train|val|test_r|test_t)\.jsonl$|/runs/final/", p)]
    assert "PROMPT TEXT" not in "".join(p.read_text() for p in out.iterdir())


def test_the_readme_is_short_and_lists_the_checksum_of_every_other_file(tmp_path):
    out = run(make_tree(tmp_path))
    text = (out / "README.md").read_text()
    assert len(text.splitlines()) <= 40
    sums = {name: digest for digest, name in re.findall(r"^([0-9a-f]{64})  (\S+)$", text, re.M)}
    assert set(sums) == {"splits.csv", "controls.csv", "values.csv"}
    for name, digest in sums.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


RELEASE = ROOT / "release"
DOCS = {"docs/benchmark_card.md": 120, "docs/controls_checklist.md": 60, "release/README.md": 40}


@pytest.mark.skipif(not RELEASE.is_dir(), reason="release/ has not been built")
def test_the_published_release_matches_its_checksums_and_the_documents_keep_their_limits():
    text = (RELEASE / "README.md").read_text(encoding="utf-8")
    sums = {name: digest for digest, name in re.findall(r"^([0-9a-f]{64})  (\S+)$", text, re.M)}
    assert set(sums) == {"splits.csv", "controls.csv", "values.csv"}
    for name, digest in sums.items():
        assert hashlib.sha256((RELEASE / name).read_bytes()).hexdigest() == digest, f"{name} changed since the README was written: run scripts/make_release.py"
    for rel, limit in DOCS.items():
        body = (ROOT / rel).read_text(encoding="utf-8")
        assert len(body.splitlines()) <= limit, rel
        assert not re.search("[\U0001F300-\U0001FAFF\u2600-\u27BF]", body), f"{rel} holds an emoji"
        assert body.count("\u2014") <= 3, f"{rel} has more than a few em dashes"
    rows = table(RELEASE / "splits.csv")
    assert len(rows) == int(re.search(r"one row per usable drug \((\d+)\)", text).group(1))
    assert all(r["split"] in mr.SPLIT_NAMES[:5] for r in rows) and not any(len(v) > 2000 for r in rows for v in r.values())
