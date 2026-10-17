"""The ChEMBL pool pipeline (scripts/fetch_chembl_pool.py, build_pool.py, class_ceiling.py): pure functions and the
cache-to-pool flow on small fixtures. No network, no real dataset.

    .venv/bin/python -m pytest tests/test_pool.py
"""
import hashlib
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from rlexp import chem

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch, build, ceil = load("fetch_chembl_pool"), load("build_pool"), load("class_ceiling")
SIM = build.SIM

ASPIRIN, ASPIRIN_NA = "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
DIAZEPAM, NORDAZEPAM = "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc12", "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1"  # Tanimoto >= 0.4
MENTHOL, TRANEXAMIC = "CC(C)C1CCC(C)CC1O", "NCC1CCC(CC1)C(=O)O"  # one cyclohexane scaffold, low Tanimoto
MORPHINE, CAFFEINE, THEOPHYLLINE = "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"
METFORMIN, WARFARIN = "CN(C)C(=N)N=C(N)N", "CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O"
# acyclic pairs with a bit Tanimoto of exactly 0.4 and of 0.3846: the leakage threshold sits between them
AT_THRESHOLD, JUST_BELOW = ("CC(C)CCl", "CC(C)Cl"), ("CC(=O)N", "CC(=O)OC")


def rec(cid, parent=None, smiles="", **kw):
    """A cached molecule record (the trimmed form fetch_chembl_pool writes)."""
    return {"id": cid, "parent_id": parent or cid, "pref_name": kw.pop("name", None), "smiles": smiles or None,
            "first_approval": kw.pop("year", None), "withdrawn_flag": False, "therapeutic_flag": True, "oral": False,
            "parenteral": False, "topical": False, "atc": kw.pop("atc", []), "synonyms": kw.pop("syn", []),
            "max_phase": 4.0, "molecule_type": "Small molecule", **kw}


# ------------------------------------------------------------------------------------------------ fetch_chembl_pool
RAW = {  # what the API returns (a small fake ChEMBL); CHEMBL99 is a hierarchy parent outside the max_phase 4 set
    "CHEMBL10": {"molecule_chembl_id": "CHEMBL10", "pref_name": "ASPIRIN", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL10"},
                 "molecule_structures": {"canonical_smiles": ASPIRIN, "molfile": "x"}, "first_approval": 1950,
                 "withdrawn_flag": False, "therapeutic_flag": True, "oral": True, "parenteral": False, "topical": False,
                 "atc_classifications": ["N02BA01", "B01AC06"], "max_phase": "4.0", "molecule_type": "Small molecule",
                 "molecule_synonyms": [{"molecule_synonym": "Aspirin", "syn_type": "INN", "synonyms": "Aspirin"}]},
    "CHEMBL11": {"molecule_chembl_id": "CHEMBL11", "pref_name": "ASPIRIN SODIUM", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL10"},
                 "molecule_structures": {"canonical_smiles": ASPIRIN_NA}, "first_approval": None, "atc_classifications": None,
                 "max_phase": "4.0", "molecule_type": "Small molecule", "molecule_synonyms": []},
    "CHEMBL12": {"molecule_chembl_id": "CHEMBL12", "pref_name": "ETHYLAMINE HCL", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL99"},
                 "molecule_structures": {"canonical_smiles": "CCN.Cl"}, "max_phase": "4.0", "molecule_type": "Small molecule",
                 "atc_classifications": [], "molecule_synonyms": [{"molecule_synonym": None, "syn_type": "OTHER"}]},
    "CHEMBL13": {"molecule_chembl_id": "CHEMBL13", "pref_name": "NO STRUCTURE", "molecule_hierarchy": None,
                 "molecule_structures": None, "max_phase": "4.0", "molecule_type": "Small molecule", "atc_classifications": [],
                 "molecule_synonyms": []},
    "CHEMBL14": {"molecule_chembl_id": "CHEMBL14", "pref_name": "ETHANOL", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL14"},
                 "molecule_structures": {"canonical_smiles": "CCO"}, "max_phase": "4.0", "molecule_type": "Small molecule",
                 "atc_classifications": ["V03AB16"], "molecule_synonyms": []},
    "CHEMBL99": {"molecule_chembl_id": "CHEMBL99", "pref_name": "ETHYLAMINE", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL99"},
                 "molecule_structures": {"canonical_smiles": "CCN"}, "max_phase": "2.0", "molecule_type": "Small molecule",
                 "atc_classifications": [], "molecule_synonyms": []},
}
IN_SET = ["CHEMBL10", "CHEMBL11", "CHEMBL12", "CHEMBL13", "CHEMBL14"]
MECH = [{"mec_id": i, "molecule_chembl_id": m, "parent_molecule_chembl_id": p, "mechanism_of_action": f"moa {i}",
         "action_type": "INHIBITOR", "target_chembl_id": t} for i, m, p, t in
        [(1, "CHEMBL10", "CHEMBL10", "CHEMBL230"), (2, "CHEMBL99", "CHEMBL99", "CHEMBL231"), (3, "CHEMBL98", None, None)]]
IND = [{"drugind_id": i, "molecule_chembl_id": m, "parent_molecule_chembl_id": m, "mesh_heading": h, "efo_term": h,
        "max_phase_for_ind": "4.0"} for i, m, h in [(1, "CHEMBL10", "Pain"), (2, "CHEMBL10", "Fever"), (3, "CHEMBL14", "Anesthesia")]]
TARGETS = {t: {"target_chembl_id": t, "pref_name": f"target {t}", "target_type": "SINGLE PROTEIN"} for t in ("CHEMBL230", "CHEMBL231")}


def test_crawl_halves_the_page_when_the_server_gives_up(tmp_path, monkeypatch):
    table = [{"mec_id": i, "molecule_chembl_id": "CHEMBL1", "parent_molecule_chembl_id": None, "mechanism_of_action": "m",
              "action_type": None, "target_chembl_id": None} for i in range(5)]
    limits = []

    def api(path, params, pause):
        limits.append(params["limit"])
        if params["limit"] > 2:
            raise RuntimeError("gave up after 6 attempts")
        return {"mechanisms": table[params["offset"]:params["offset"] + params["limit"]], "page_meta": {"total_count": 5}}

    monkeypatch.setattr(fetch, "api", api)
    monkeypatch.setattr(fetch, "PAGE", 8)
    monkeypatch.setattr(fetch, "PAGE_MIN", 2)
    used = []
    got, total = fetch.crawl(tmp_path, used, "mechanism", "mechanism.json", "mechanisms", {}, "mec_id", "x", fetch.trim_mechanism, 0)
    assert [m["mec_id"] for m in got] == list(range(5)) and total == 5
    assert limits == [8, 4, 2, 2, 2] and [n for _, n in used] == [2, 2, 1]  # 8 -> 4 -> 2, then pages of 2
    monkeypatch.setattr(fetch, "api", lambda path, params, pause: (_ for _ in ()).throw(RuntimeError("still down")))
    with pytest.raises(RuntimeError, match="still down"):  # at the minimum size the failure is final
        fetch.crawl(tmp_path / "other", [], "mechanism", "mechanism.json", "mechanisms", {}, "mec_id", "x", fetch.trim_mechanism, 0)


class FakeApi:
    """Stands in for fetch.api: serves RAW, MECH, IND and TARGETS in pages and `__in` batches, and counts the calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, params, pause):
        self.calls.append(path)
        if path == "status.json":
            return {"chembl_db_version": "ChEMBL_TEST", "chembl_release_date": "2026-01-01", "other": 1}
        table, key = {"molecule.json": ([RAW[i] for i in IN_SET], "molecules"), "mechanism.json": (MECH, "mechanisms"),
                      "drug_indication.json": (IND, "drug_indications")}.get(path, (None, None))
        if "molecule_chembl_id__in" in params:
            return {"molecules": [RAW[i] for i in params["molecule_chembl_id__in"].split(",") if i in RAW], "page_meta": {}}
        if "target_chembl_id__in" in params:
            return {"targets": [TARGETS[i] for i in params["target_chembl_id__in"].split(",") if i in TARGETS], "page_meta": {}}
        o, n = params["offset"], params["limit"]
        return {key: table[o:o + n], "page_meta": {"total_count": len(table)}}


@pytest.fixture
def small_pages(monkeypatch):
    monkeypatch.setattr(fetch, "PAGE", 2)
    monkeypatch.setattr(fetch, "BATCH", 2)


def test_trim_molecule_keeps_the_pool_fields_and_survives_missing_ones():
    t = fetch.trim_molecule(RAW["CHEMBL10"])
    assert t == {"id": "CHEMBL10", "parent_id": "CHEMBL10", "pref_name": "ASPIRIN", "smiles": ASPIRIN, "first_approval": 1950,
                 "withdrawn_flag": False, "therapeutic_flag": True, "oral": True, "parenteral": False, "topical": False,
                 "atc": ["B01AC06", "N02BA01"], "synonyms": [{"name": "Aspirin", "type": "INN"}], "max_phase": 4.0,
                 "molecule_type": "Small molecule"}
    assert fetch.trim_molecule(RAW["CHEMBL13"])["smiles"] is None and fetch.trim_molecule(RAW["CHEMBL13"])["parent_id"] == "CHEMBL13"
    assert fetch.trim_molecule(RAW["CHEMBL11"])["atc"] == [] and fetch.trim_molecule(RAW["CHEMBL12"])["synonyms"] == []
    assert fetch.trim_indication(IND[0])["max_phase_for_ind"] == 4.0


def test_fetch_caches_every_page_and_a_rerun_needs_no_network(tmp_path, small_pages, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(fetch, "api", api)
    m = fetch.fetch(tmp_path)
    assert m["records"] == {"molecule": 5, "molecule_api_total": 5, "parent": 1, "mechanism": 3, "mechanism_api_total": 3,
                            "target": 2, "indication": 3, "indication_api_total": 3}
    assert m["chembl_release"] == "ChEMBL_TEST" and not (tmp_path / "manifest.json.tmp").exists()
    assert {f.name for f in tmp_path.iterdir()} == set(m["files"]) | {"manifest.json"}
    for name, meta in m["files"].items():
        assert meta["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert m["files"]["molecule_000004.json"]["records"] == 1 and m["files"]["parent_000.json"]["records"] == 1
    n_calls, first = len(api.calls), (tmp_path / "manifest.json").read_bytes()

    again = fetch.fetch(tmp_path)  # idempotent: same manifest, not one request
    assert len(api.calls) == n_calls and again == m and (tmp_path / "manifest.json").read_bytes() == first

    (tmp_path / "indication_000002.json").unlink()  # a lost page is the only thing fetched again
    assert fetch.fetch(tmp_path) == m
    assert api.calls[n_calls:] == ["drug_indication.json"]


def test_fetch_refetches_an_id_batch_whose_ids_changed(tmp_path, small_pages, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(fetch, "api", api)
    used = []
    fetch.lookup(tmp_path, used, "target", "target.json", "targets", "target_chembl_id__in", {"CHEMBL230"}, "x", fetch.trim_target, 0)
    out = fetch.lookup(tmp_path, used, "target", "target.json", "targets", "target_chembl_id__in", {"CHEMBL230", "CHEMBL231"}, "x",
                       fetch.trim_target, 0)
    assert len(api.calls) == 2 and [t["target_chembl_id"] for t in out] == ["CHEMBL230", "CHEMBL231"]


def test_crawl_refuses_pages_that_do_not_add_up(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "api", lambda path, params, pause: {"mechanisms": [], "page_meta": {"total_count": 3}})
    with pytest.raises(RuntimeError, match="empty page"):
        fetch.crawl(tmp_path, [], "mechanism", "mechanism.json", "mechanisms", {}, "mec_id", "x", fetch.trim_mechanism, 0)


def test_api_retries_throttling_and_network_errors_but_not_a_404(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    answers = [urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "7"}, None), urllib.error.URLError("reset"), io.BytesIO(b'{"ok": 1}')]
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda req, timeout: _next(answers))
    assert fetch.api("molecule.json", {"limit": 1}, 0) == {"ok": 1} and not answers

    answers[:] = [urllib.error.HTTPError("u", 404, "nf", {}, None), io.BytesIO(b"{}")]
    with pytest.raises(urllib.error.HTTPError):
        fetch.api("molecule.json", {}, 0)
    assert len(answers) == 1  # raised at once, no second attempt

    answers[:] = [urllib.error.URLError("down")] * fetch.TRIES
    with pytest.raises(RuntimeError, match="gave up"):
        fetch.api("molecule.json", {}, 0)


def _next(answers):
    a = answers.pop(0)
    if isinstance(a, Exception):
        raise a
    return a


# ------------------------------------------------------------------------------------------------ build_pool
def test_merge_unions_forms_and_parents_with_the_same_connectivity():
    # CHEMBL9 and CHEMBL10 are two parents of one connectivity (enantiomers); CHEMBL10 sorts first as a string, CHEMBL9 as a number
    molecules = [
        rec("CHEMBL10", smiles="C[C@@H](N)c1ccccc1", name="S-FORM", atc=["N06BA01"], year=1990, syn=[{"name": "s-form", "type": "INN"}]),
        rec("CHEMBL9", smiles="C[C@H](N)c1ccccc1", name="R-FORM", year=1980, oral=True),
        rec("CHEMBL11", parent="CHEMBL10", smiles="C[C@@H](N)c1ccccc1.Cl", atc=["N06BA02"], syn=[{"name": "s-form hcl", "type": "OTHER"}]),
        rec("CHEMBL20", parent="CHEMBL21", smiles=None),  # no SMILES of its own: the parent's is used
        rec("CHEMBL30", smiles=None),
        rec("CHEMBL31", smiles="not a smiles"),
        rec("CHEMBL32", smiles="*C"),  # a dummy atom is refused by chem.standardize
    ]
    parents = [rec("CHEMBL21", smiles="CCCCCCO", name="HEXANOL")]
    rows, mols, st = build.merge(molecules, parents)
    assert dict(st) == {"no_smiles": 1, "not_standardised": 2, "standardised": 4, "parents": 3}
    assert [r["chembl_id"] for r in rows] == ["CHEMBL9", "CHEMBL21"] and len(mols) == 2
    r = rows[0]
    assert r["chembl_ids"] == ["CHEMBL9", "CHEMBL10", "CHEMBL11"] and r["first_approval"] == 1980 and r["oral"] is True
    assert r["atc"] == ["N06BA01", "N06BA02"] and r["pref_name"] == "R-FORM"
    assert r["synonyms"] == ["s-form", "s-form hcl"] and r["smiles"] == chem.canonical(chem.standardize("C[C@H](N)c1ccccc1"))
    assert r["conn"] == chem.conn(mols[0]) and r["n_heavy"] == 9 and r["scaffold"] == "c1ccccc1"
    assert rows[1]["chembl_ids"] == ["CHEMBL20", "CHEMBL21"] and rows[1]["scaffold"] == ""


def test_merge_strips_a_salt_down_to_the_parent_structure():
    rows, _, st = build.merge([rec("CHEMBL1", smiles=ASPIRIN), rec("CHEMBL2", parent="CHEMBL1", smiles=ASPIRIN_NA)], [])
    assert len(rows) == 1 and st["standardised"] == 2 and st["parents"] == 1
    assert rows[0]["smiles"] == chem.canonical(chem.standardize(ASPIRIN))


def test_attach_facts_joins_by_molecule_or_parent_id_and_collapses_repeats():
    rows = [{"chembl_ids": ["CHEMBL1", "CHEMBL2"]}, {"chembl_ids": ["CHEMBL3"]}]
    mech = [{"mec_id": 1, "molecule_chembl_id": "CHEMBL2", "parent_molecule_chembl_id": "CHEMBL1", "mechanism_of_action": "a",
             "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL230"},
            {"mec_id": 2, "molecule_chembl_id": "CHEMBL1", "parent_molecule_chembl_id": "CHEMBL1", "mechanism_of_action": "a",
             "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL230"},  # the same fact recorded on the parent
            {"mec_id": 3, "molecule_chembl_id": "CHEMBL3", "parent_molecule_chembl_id": None, "mechanism_of_action": "b",
             "action_type": None, "target_chembl_id": None}]
    ind = [{"drugind_id": 1, "molecule_chembl_id": "CHEMBL2", "parent_molecule_chembl_id": "CHEMBL1", "mesh_heading": "Pain",
            "efo_term": "pain", "max_phase_for_ind": 4.0}]
    build.attach_facts(rows, mech, [TARGETS["CHEMBL230"]], ind)
    assert rows[0]["mechanisms"] == [{"mechanism_of_action": "a", "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL230",
                                      "target_name": "target CHEMBL230", "target_type": "SINGLE PROTEIN"}]
    assert rows[0]["indications"] == [{"mesh_heading": "Pain", "efo_term": "pain", "max_phase_for_ind": 4.0}]
    assert rows[1]["mechanisms"] == [{"mechanism_of_action": "b", "action_type": None, "target_chembl_id": None,
                                      "target_name": None, "target_type": None}] and rows[1]["indications"] == []


def test_leakage_rule_at_the_boundary():
    assert SIM == 0.4
    assert build.leak_reason(0.4, False) == "tanimoto" and build.leak_reason(0.4, True) == "tanimoto"
    assert build.leak_reason(np.nextafter(0.4, 0), False) is None
    assert build.leak_reason(0.0, True) == "scaffold" and build.leak_reason(0.39, False) is None

    # the same on real molecules: a pair at exactly 0.4 is leaked, a pair at 0.3846 is not
    a, b = (chem.standardize(s) for s in AT_THRESHOLD)
    assert chem.tanimoto_bits(a, b) == 0.4 and chem.scaffold(a) == chem.scaffold(b) == ""
    assert build.check(a, "", [b], set()) == "tanimoto"
    c, d = (chem.standardize(s) for s in JUST_BELOW)
    assert 0.38 < chem.tanimoto_bits(c, d) < 0.4
    assert build.check(c, "", [d], set()) is None  # two acyclic molecules never share a scaffold
    assert build.check(a, "", [b, c], set()) == "tanimoto"  # any held-out drug is enough


def test_leakage_rule_shared_scaffold_below_the_similarity_threshold():
    men, tra = chem.standardize(MENTHOL), chem.standardize(TRANEXAMIC)
    assert chem.scaffold(men) == chem.scaffold(tra) == "C1CCCCC1" and chem.tanimoto_bits(men, tra) < SIM
    assert build.check(tra, chem.scaffold(tra), [men], {chem.scaffold(men)}) == "scaffold"
    other = chem.standardize(CAFFEINE)
    assert build.check(tra, chem.scaffold(tra), [other], {chem.scaffold(other)}) is None
    assert build.check(tra, "", [men], {chem.scaffold(men)}) is None  # an empty scaffold matches nothing


def test_dataset_conns_are_rederived_and_include_alternates(tmp_path):
    ds = {"train": {"ibuprofen": (IBUPROFEN, [DIAZEPAM])}, "val": {"menthol": (MENTHOL, [])}, "test_r": {}, "test_t": {}}
    keys = build.dataset_conns(ds)
    assert keys[chem.conn(chem.standardize(IBUPROFEN))] == ("train", "ibuprofen")
    assert keys[chem.conn(chem.standardize(DIAZEPAM))] == ("train", "ibuprofen")  # an alternate belongs to its drug
    assert keys[chem.conn(chem.standardize(MENTHOL))] == ("val", "menthol") and len(keys) == 3
    with pytest.raises(ValueError):
        build.dataset_conns({"train": {"x": ("not a smiles", [])}})


def test_load_cache_refuses_a_file_that_differs_from_the_manifest(tmp_path):
    page = tmp_path / "molecule_000000.json"
    page.write_text(json.dumps({"records": [rec("CHEMBL1", smiles="CCO")]}), encoding="utf-8")
    manifest = {"files": {page.name: {"records": 1, "sha256": hashlib.sha256(page.read_bytes()).hexdigest()}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert build.load_cache(tmp_path)[1]["molecule"][0]["id"] == "CHEMBL1"
    page.write_text(json.dumps({"records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        build.load_cache(tmp_path)


def write_cache(cache, molecules, mechanisms=(), indications=(), targets=()):
    files = {"status.json": {"chembl_db_version": "ChEMBL_TEST"}, "molecule_000000.json": {"records": molecules},
             "mechanism_000000.json": {"records": list(mechanisms)}, "indication_000000.json": {"records": list(indications)},
             "target_000.json": {"records": list(targets)}, "parent_000.json": {"records": []}}
    cache.mkdir(parents=True, exist_ok=True)
    for n, obj in files.items():
        (cache / n).write_text(json.dumps(obj), encoding="utf-8")
    (cache / "manifest.json").write_text(json.dumps({"chembl_release": "ChEMBL_TEST", "files": {
        n: {"records": None, "sha256": hashlib.sha256((cache / n).read_bytes()).hexdigest()} for n in files}}), encoding="utf-8")


def write_dataset(data, **splits):
    data.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test_r", "test_t"):
        rows = [{"group": g, "ref_smiles": smi, "ref_alt_smiles": [], "prompt": "PRIVATE PROMPT TEXT", "id": f"{g}-{k}"}
                for g, smi in splits.get(s, {}).items() for k in range(2)]  # two rows per drug, like the real files
        (data / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_build_pool_end_to_end(tmp_path, capsys):
    mech = [{"mec_id": 1, "molecule_chembl_id": "CHEMBL2", "parent_molecule_chembl_id": "CHEMBL1", "mechanism_of_action": "COX inhibitor",
             "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL230"}]
    molecules = [
        rec("CHEMBL1", smiles=IBUPROFEN, name="IBUPROFEN"), rec("CHEMBL2", parent="CHEMBL1", smiles=IBUPROFEN + ".[Na]", atc=["M01AE01"]),
        rec("CHEMBL3", smiles=NORDAZEPAM, name="NORDAZEPAM"),  # Tanimoto >= 0.4 to a Val drug
        rec("CHEMBL4", smiles=TRANEXAMIC, name="TRANEXAMIC ACID"),  # shares the menthol scaffold
        rec("CHEMBL5", smiles=CAFFEINE, name="CAFFEINE", atc=["N06BC01"]), rec("CHEMBL6", smiles=THEOPHYLLINE, name="THEOPHYLLINE"),
        rec("CHEMBL7", smiles=METFORMIN, name="METFORMIN", atc=["A10BA02"]), rec("CHEMBL8", smiles="broken"),
        rec("CHEMBL9", smiles=WARFARIN, name="WARFARIN"),  # a Test-R drug itself: already in the dataset
    ]
    write_cache(tmp_path / "cache", molecules, mech, [], [TARGETS["CHEMBL230"]])
    write_dataset(tmp_path / "data", train={"ibuprofen": IBUPROFEN}, val={"diazepam": DIAZEPAM, "menthol": MENTHOL},
                  test_r={"warfarin": WARFARIN}, test_t={"morphine": MORPHINE})
    assert chem.tanimoto_bits(chem.standardize(NORDAZEPAM), chem.standardize(DIAZEPAM)) >= SIM

    build.main(["--cache", str(tmp_path / "cache"), "--data", str(tmp_path / "data")])
    out = capsys.readouterr().out
    pool = [json.loads(l) for l in (tmp_path / "cache" / "pool.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["chembl_id"] for r in pool] == ["CHEMBL5", "CHEMBL6", "CHEMBL7"]
    assert pool[0]["atc"] == ["N06BC01"] and pool[2]["conn"] == chem.conn(chem.standardize(METFORMIN))
    assert "fetched 9 records" in out and "(no SMILES 0, not standardised 1)" in out
    assert "duplicates 1 " in out and "already in the dataset 2" in out  # ibuprofen (merged with its salt) and warfarin
    assert "excluded by bit Tanimoto >= 0.4: 1" in out and "excluded by a shared scaffold only: 1" in out and "kept 3" in out
    assert "PRIVATE" not in out and "ibuprofen" not in out.lower()  # counts only

    train = [json.loads(l) for l in (tmp_path / "cache" / "train_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 1 and train[0]["group"] == "ibuprofen" and train[0]["chembl_ids"] == ["CHEMBL1", "CHEMBL2"]
    assert train[0]["atc"] == ["M01AE01"] and train[0]["mechanisms"][0]["target_name"] == "target CHEMBL230"
    assert train[0]["smiles"] == chem.canonical(chem.standardize(IBUPROFEN))


# ------------------------------------------------------------------------------------------------ class_ceiling
def test_features_build_atc_levels_mechanism_and_indication_facts():
    row = {"atc": ["N02BE01", "N02BE"], "indications": [{"mesh_heading": "Pain"}, {"mesh_heading": None}, {"efo_term": "x"}],
           "mechanisms": [{"action_type": "INHIBITOR", "target_chembl_id": "CHEMBL230", "target_type": "SINGLE PROTEIN"},
                          {"action_type": None, "target_chembl_id": None, "target_type": None}]}
    assert ceil.features(row) == {"atc3:N02B", "atc4:N02BE", "action:INHIBITOR", "target:CHEMBL230", "ttype:SINGLE PROTEIN", "mesh:Pain"}
    assert ceil.features({"atc": ["N02", "A10B"], "mechanisms": [], "indications": []}) == {"atc3:A10B"}  # too short for a level
    assert ceil.features({"atc": [], "mechanisms": [], "indications": []}) == frozenset()


def test_jaccard_matrix():
    sets = [frozenset("ab"), frozenset("bc"), frozenset("ab"), frozenset("d"), frozenset()]
    J = ceil.jaccard_matrix(sets)
    assert J[0, 1] == pytest.approx(1 / 3) and J[0, 2] == 1.0 and J[0, 3] == 0.0 and J[1, 0] == J[0, 1]
    assert (J[4] == 0).all() and (J[:, 4] == 0).all()  # an empty set is similar to nothing, not even to another empty set
    for i in range(4):
        for j in range(4):
            assert J[i, j] == pytest.approx(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))


def test_predictions_break_ties_by_row_order_and_pick_a_medoid():
    n = 8
    J = np.zeros((n, n))
    J[0, [3, 2, 5, 6, 7, 1]] = [0.9, 0.9, 0.5, 0.4, 0.3, 0.1]  # rows 2 and 3 tie for drug 0: the lower row (lower ChEMBL id) wins
    T = np.full((n, n), 0.25)
    np.fill_diagonal(T, 1.0)
    T[2, 3] = T[3, 2] = T[2, 5] = T[5, 2] = T[3, 5] = T[5, 3] = 0.5  # 2, 3 and 5 are mutual neighbours; 6 and 7 are strangers
    ok = ~np.eye(n, dtype=bool)
    knn, med = ceil.predictions(J, T, ok)
    assert knn[0] == 2
    assert med[0] == 2  # of {2,3,5,6,7}: 2, 3 and 5 average 0.375 to the others, 6 and 7 average 0.25; the tie goes to the more similar
    ok2 = ok.copy()
    ok2[0, 2] = False  # a ruled-out neighbour is never used
    knn2, med2 = ceil.predictions(J, T, ok2)
    assert knn2[0] == 3 and med2[0] == 3
    only_two = np.zeros((n, n), bool)
    only_two[:, 1] = True
    only_two[0, [5, 6]] = True  # drug 0 may use 5 and 6 only: k shrinks to 2, no phantom neighbour joins the medoid
    only_two[0, 1] = False
    knn3, med3 = ceil.predictions(J, T, only_two)
    assert knn3[0] == 5 and med3[0] == 5


def test_scores_constant_is_never_the_drugs_own_structure():
    T = np.array([[1.0, 0.5, 0.6, 0.1], [0.5, 1.0, 0.3, 0.2], [0.6, 0.3, 1.0, 0.0], [0.1, 0.2, 0.0, 1.0]])
    J = np.array([[0, .9, .1, 0], [.9, 0, .2, 0], [.1, .2, 0, .5], [0, 0, .5, 0]], float)
    ok = ~np.eye(4, dtype=bool)
    s = ceil.scores(J, T, ok)
    assert list(s["knn"]) == [0.5, 0.5, 0.0, 0.0]  # drug 0 -> 1, 1 -> 0, 2 -> 3, 3 -> 2
    assert (T.sum(0) - 1).argmax() == 0  # structure 0 is the most central
    assert list(s["constant"][1:]) == [T[1, 0], T[2, 0], T[3, 0]]
    assert s["constant"][0] == T[0, 1]  # the runner-up answers drug 0, never its own structure (Tanimoto 1)
    assert list(s["random"]) == pytest.approx([(0.5 + 0.6 + 0.1) / 3, (0.5 + 0.3 + 0.2) / 3, (0.6 + 0.3) / 3, (0.1 + 0.2) / 3])
    assert list(s["oracle"]) == [0.6, 0.5, 0.6, 0.2]
    ok2 = ok.copy()
    ok2[1, 0] = ok2[0, 1] = False  # the split rule keeps the pair 0/1 apart: drug 1 falls back to the next most central structure
    s2 = ceil.scores(J, T, ok2)
    assert s2["constant"][1] == T[1, 2] and s2["oracle"][1] == 0.3 and s2["random"][1] == pytest.approx(0.25)
    with pytest.raises(AssertionError):
        ceil.scores(J, T, np.zeros((4, 4), bool))


def test_tanimoto_matrix_is_the_public_tanimoto():
    mols = [chem.standardize(s) for s in (IBUPROFEN, DIAZEPAM, NORDAZEPAM, MENTHOL, CAFFEINE, "CCO")]
    for bits, f in ((False, chem.tanimoto), (True, chem.tanimoto_bits)):
        M = ceil.tanimoto_matrix(mols, bits)
        for i, a in enumerate(mols):
            for j, b in enumerate(mols):
                assert M[i, j] == f(a, b)


def test_bootstrap_is_seeded_paired_and_brackets_the_mean():
    a = np.random.default_rng(1).random(200)
    V = np.array([a, a + 0.25, np.full(200, 0.7)])
    mean, lo, hi = ceil.bootstrap(V, boot=500, seed=3)
    assert mean == pytest.approx(V.mean(1)) and lo[0] < mean[0] < hi[0] and hi[0] - lo[0] < 0.15
    assert lo[2] == pytest.approx(0.7) and hi[2] == pytest.approx(0.7)  # a constant has no sampling spread
    again = ceil.bootstrap(V, boot=500, seed=3)
    assert all((x == y).all() for x, y in zip(again, (mean, lo, hi)))
    assert not (ceil.bootstrap(V, boot=500, seed=4)[1] == lo).all()
    d = ceil.bootstrap(V[1:2] - V[:1], boot=500, seed=3)  # a paired difference that is exactly 0.25 for every drug
    assert d[1][0] == pytest.approx(0.25) and d[2][0] == pytest.approx(0.25)
    assert np.isnan(ceil.bootstrap(np.empty((2, 0)))[0]).all()
    big = ceil.bootstrap(np.array([a]), boot=2500, seed=0)  # three chunks of resamples, none left unfilled
    assert big[1][0] < a.mean() < big[2][0]


def test_verdict_reads_the_interval_against_zero():
    assert ceil.verdict(0.01, 0.05) == "beats" and ceil.verdict(-0.05, -0.01) == "falls below"
    assert ceil.verdict(-0.01, 0.05) == "cannot be told apart from" and ceil.verdict(0.0, 0.05) == "cannot be told apart from"


def test_run_on_a_small_population_and_report():
    smiles = [IBUPROFEN, DIAZEPAM, NORDAZEPAM, MENTHOL, TRANEXAMIC, CAFFEINE, THEOPHYLLINE, METFORMIN, WARFARIN, MORPHINE, ASPIRIN,
              "CCO", "CCCO", "CCCCO"]
    pop = []
    for k, s in enumerate(smiles):
        mol = chem.standardize(s)
        pop.append({"chembl_id": f"CHEMBL{k + 1}", "smiles": s, "conn": chem.conn(mol), "scaffold": chem.scaffold(mol),
                    "first_approval": 2000 + 2 * k if k % 4 else None, "source": "train" if k < 3 else "pool",
                    "atc": ["N06BC01"] if k in (5, 6) else ["M01AE01"] if k in (0, 4, 10) else ["V03AB16"],
                    "mechanisms": [{"action_type": "INHIBITOR", "target_chembl_id": "CHEMBL1", "target_type": "SINGLE PROTEIN"}] if k % 2 else [],
                    "indications": [{"mesh_heading": "Pain"}] if k < 5 else []})
    res = ceil.run(pop, boot=200, seed=0)
    assert res["population"]["drugs"] == 14 and res["population"]["train"] == 3 and res["population"]["unknown_year"] == 4
    assert set(res["rules"]) == {"any", "split-matched"}
    for rule, subsets in res["rules"].items():
        assert set(subsets) == {"all drugs", "at least 3 facts", "first approved 2015 or later", "Train drugs only"}
        r = subsets["all drugs"]
        assert r["n"] == 14 and subsets["Train drugs only"]["n"] == 3
        for m in ceil.METHODS:
            mean, lo, hi = r[f"mean:{m}"]
            assert 0 <= lo <= hi <= 1 + 1e-9 and 0 <= mean <= 1 + 1e-9
            assert r[f"share0.3:{m}"][0] >= r[f"share0.4:{m}"][0]
        assert r["mean:oracle"][0] >= max(r[f"mean:{m}"][0] for m in ("knn", "medoid", "constant", "random")) - 1e-12
    allowed = res["rules"]["any"]["all drugs"]["mean:oracle"][0]
    assert res["rules"]["split-matched"]["all drugs"]["mean:oracle"][0] <= allowed + 1e-12  # fewer neighbours, never a better oracle
    text = ceil.report(res)
    assert "facts-kNN" in text and "reading" in text and "split-matched" in text and "in short:" in text


def test_load_population_keeps_drugs_with_facts_one_per_connectivity(tmp_path):
    def row(cid, conn, **kw):
        return {"chembl_id": cid, "conn": conn, "atc": [], "mechanisms": [], "indications": [], "smiles": "CCO", "scaffold": "", **kw}

    pool = [row("CHEMBL100", "A", atc=["N02BE01"]), row("CHEMBL9", "B", mechanisms=[{"action_type": "X"}]), row("CHEMBL5", "C"),
            row("CHEMBL7", "D", atc=["A10BA02"])]
    train = [row("CHEMBL8", "D", atc=["A10BA02"])]  # the same connectivity as pool drug D: the Train row wins
    for name, rows in (("pool.jsonl", pool), ("train_facts.jsonl", train)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    pop, n_rows = ceil.load_population(tmp_path)
    assert n_rows == 5 and [(r["chembl_id"], r["source"]) for r in pop] == [("CHEMBL8", "train"), ("CHEMBL9", "pool"), ("CHEMBL100", "pool")]
