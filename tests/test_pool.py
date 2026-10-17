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
from rlexp import data_prep as dp

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


def test_fetch_keeps_the_derived_record_only_while_the_raw_pages_are_unchanged(tmp_path, small_pages, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(fetch, "api", api)
    m = fetch.fetch(tmp_path)
    assert "derived" not in m
    derived = {"built_by": "scripts/build_pool.py", "files": {"pool.jsonl": {"records": 3, "sha256": "ab" * 32}}, "inputs": {}}
    fetch.save(tmp_path / "manifest.json", {**m, "derived": derived})  # what build_pool.py leaves behind
    before, n_calls = (tmp_path / "manifest.json").read_bytes(), len(api.calls)
    assert fetch.fetch(tmp_path)["derived"] == derived and (tmp_path / "manifest.json").read_bytes() == before
    assert len(api.calls) == n_calls  # still offline, and byte-identical
    page = tmp_path / "indication_000002.json"  # a raw page that is no longer the one the derived files were built from
    page.write_text(page.read_text(encoding="utf-8").replace("Anesthesia", "Anaesthesia"), encoding="utf-8")
    assert "derived" not in fetch.fetch(tmp_path) and "derived" not in json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


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

    cache, data = tmp_path / "cache", tmp_path / "data"  # the manifest now records what was built and from which split files
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["derived"]["built_by"] == "scripts/build_pool.py"
    assert manifest["derived"]["files"] == {"pool.jsonl": {"records": 3, "sha256": sha(cache / "pool.jsonl")},
                                            "train_facts.jsonl": {"records": 1, "sha256": sha(cache / "train_facts.jsonl")}}
    assert manifest["derived"]["inputs"] == {f"{s}.jsonl": sha(data / f"{s}.jsonl") for s in ("train", "val", "test_r", "test_t")}
    assert sorted(manifest["files"]) == ["indication_000000.json", "mechanism_000000.json", "molecule_000000.json", "parent_000.json",
                                         "status.json", "target_000.json"]  # the raw entries are untouched
    first = (cache / "manifest.json").read_bytes()
    build.main(["--cache", str(cache), "--data", str(data)])  # a rerun verifies the raw files and leaves the same record
    assert (cache / "manifest.json").read_bytes() == first


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


def test_scores_central_constant_is_never_the_drugs_own_structure():
    T = np.array([[1.0, 0.5, 0.6, 0.1], [0.5, 1.0, 0.3, 0.2], [0.6, 0.3, 1.0, 0.0], [0.1, 0.2, 0.0, 1.0]])
    J = np.array([[0, .9, .1, 0], [.9, 0, .2, 0], [.1, .2, 0, .5], [0, 0, .5, 0]], float)
    ok, train = ~np.eye(4, dtype=bool), np.array([True, True, False, False])
    s, picks = ceil.scores(J, T, ok, train)
    assert list(s["knn"]) == [0.5, 0.5, 0.0, 0.0]  # drug 0 -> 1, 1 -> 0, 2 -> 3, 3 -> 2
    assert (T.sum(0) - 1).argmax() == 0  # structure 0 is the most central
    assert list(s["constant"][1:]) == [T[1, 0], T[2, 0], T[3, 0]]
    assert s["constant"][0] == T[0, 1] and list(picks["constant"]) == [1, 0, 0, 0]  # the runner-up answers drug 0, never its own structure
    assert list(s["random"]) == pytest.approx([(0.5 + 0.6 + 0.1) / 3, (0.5 + 0.3 + 0.2) / 3, (0.6 + 0.3) / 3, (0.1 + 0.2) / 3])
    assert list(s["oracle"]) == [0.6, 0.5, 0.6, 0.2]
    ok2 = ok.copy()
    ok2[1, 0] = ok2[0, 1] = False  # the split rule keeps the pair 0/1 apart: drug 1 falls back to the next most central structure
    s2, _ = ceil.scores(J, T, ok2, train)
    assert s2["constant"][1] == T[1, 2] and s2["oracle"][1] == 0.3 and s2["random"][1] == pytest.approx(0.25)
    with pytest.raises(AssertionError):
        ceil.scores(J, T, np.zeros((4, 4), bool), train)


def test_train_fixed_constant_is_ranked_by_the_train_references_only():
    # drugs 0 and 1 are the Train references. Drug 2 is the most central structure of the population (0.50) but poor for Train;
    # drug 3 has the best mean Tanimoto to the Train references (0.5)
    T = np.array([[1, .2, .1, .5, .3], [.2, 1, .1, .5, .3], [.1, .1, 1, .9, .9], [.5, .5, .9, 1, .05], [.3, .3, .9, .05, 1]])
    J, ok, train = np.zeros((5, 5)), ~np.eye(5, dtype=bool), np.array([True, True, False, False, False])
    assert (T.sum(0) - 1).argmax() == 2 and T[train].mean(0)[2:].argmax() == 1  # the premise: central 2, Train-best 3
    s, picks = ceil.scores(J, T, ok, train)
    assert list(picks["constant"]) == [2, 2, 3, 2, 2] and list(picks["constant_train"]) == [3, 3, 3, 4, 3]
    assert list(s["constant_train"]) == [T[i, c] for i, c in enumerate(picks["constant_train"])]
    ok2 = ok.copy()
    ok2[2, 3] = ok2[3, 2] = False  # drug 3 ineligible for drug 2: the next Train-ranked candidate answers it, the ranking itself is unchanged
    assert ceil.scores(J, T, ok2, train)[1]["constant_train"][2] == 4


def test_a_train_target_does_not_help_rank_its_own_constant():
    # Train references 0, 1, 2. Candidate 4 is Train-best (0.5) only thanks to drug 1 (0.9); candidate 3 is steady (0.4)
    T = np.array([[1, .1, .1, .4, .3], [.1, 1, .1, .4, .9], [.1, .1, 1, .4, .3], [.4, .4, .4, 1, .1], [.3, .9, .3, .1, 1]])
    J, ok, train = np.zeros((5, 5)), ~np.eye(5, dtype=bool), np.array([True, True, True, False, False])
    _, picks = ceil.scores(J, T, ok, train)
    assert list(picks["constant_train"]) == [4, 3, 4, 4, 3]  # drug 1 is answered with 3: its own 0.9 does not count for 4
    ok2 = ok.copy()
    ok2[0, 4] = False  # and an ineligible best candidate gives way to the next one
    assert ceil.scores(J, T, ok2, train)[1]["constant_train"][0] == 3
    with pytest.raises(AssertionError, match="Train reference"):
        ceil.scores(J, T, ok, np.zeros(5, bool))


def test_tanimoto_matrix_is_the_public_tanimoto():
    mols = [chem.standardize(s) for s in (IBUPROFEN, DIAZEPAM, NORDAZEPAM, MENTHOL, CAFFEINE, "CCO")]
    for bits, f in ((False, chem.tanimoto), (True, chem.tanimoto_bits)):
        M = ceil.tanimoto_matrix(mols, bits)
        for i, a in enumerate(mols):
            for j, b in enumerate(mols):
                assert M[i, j] == f(a, b)


FAMILIES = {  # drug -> SMILES. naproxen links to the methyl ester only through ibuprofen: 0.42 to ibuprofen, 0.33 to the ester
    "ibuprofen": IBUPROFEN, "ibuprofen methyl ester": IBUPROFEN + "OC", "naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "diazepam": DIAZEPAM, "nordazepam": NORDAZEPAM, "menthol": MENTHOL, "tranexamic acid": TRANEXAMIC, "gabapentin": "NCC1(CC(=O)O)CCCCC1",
    "caffeine": CAFFEINE, "theophylline": THEOPHYLLINE, "metformin": METFORMIN, "valproic acid": "CCCC(CCC)C(=O)O", "warfarin": WARFARIN}


def family_matrices():
    mols = [chem.standardize(s) for s in FAMILIES.values()]
    return list(FAMILIES), mols, [chem.scaffold(m) for m in mols], ceil.tanimoto_matrix(mols, bits=True)


def test_components_equal_the_partition_data_prep_splits_by():
    names, mols, scaffolds, B = family_matrices()
    comp = ceil.components(B, scaffolds)
    refs = {g: {"ref_smiles": s, "ref_scaffold": sc, "approval_year": 2018, "exclude_reason": None, "cluster_id": -1, "split": "excluded"}
            for (g, s), sc in zip(FAMILIES.items(), scaffolds)}
    dp.assign_splits(refs, 0, 0, 0)  # no Val or Test-R target: every component lands in Train and gets its cluster_id
    mine, theirs = {}, {}
    for g, c in zip(names, comp):
        mine.setdefault(int(c), set()).add(g)
    for g, r in refs.items():
        theirs.setdefault(r["cluster_id"], set()).add(g)
    assert sorted(map(sorted, mine.values())) == sorted(map(sorted, theirs.values()))
    assert {"ibuprofen", "ibuprofen methyl ester", "naproxen"} in mine.values()  # chained through ibuprofen
    assert {"menthol", "tranexamic acid", "gabapentin"} in mine.values()  # one cyclohexane scaffold, no Tanimoto edge
    assert {"metformin"} in mine.values()  # an acyclic molecule links to nothing by scaffold


def test_components_join_a_pair_at_exactly_the_similarity_threshold():
    at, below = [chem.standardize(s) for s in AT_THRESHOLD], [chem.standardize(s) for s in JUST_BELOW]
    B = ceil.tanimoto_matrix(at + below, bits=True)
    assert B[0, 1] == SIM and B[2, 3] < SIM  # acyclic pairs: only the Tanimoto edge can join them
    comp = ceil.components(B, [""] * 4)
    assert comp[0] == comp[1] and comp[2] != comp[3] and len(set(comp.tolist())) == 3


def test_component_matched_neighbours_exclude_the_whole_component():
    names, mols, scaffolds, B = family_matrices()
    rules, comp = ceil.rule_masks(B, scaffolds)
    i, j = names.index("naproxen"), names.index("ibuprofen methyl ester")
    assert B[i, j] < SIM and scaffolds[i] != scaffolds[j]  # no direct edge between the two ...
    assert comp[i] == comp[j]  # ... yet one component, through ibuprofen
    assert rules["pairwise-matched"][i, j] and rules["any"][i, j] and not rules[ceil.PRIMARY][i, j]
    assert not (rules[ceil.PRIMARY] & ~rules["pairwise-matched"]).any() and not (rules["pairwise-matched"] & ~rules["any"]).any()
    assert not rules[ceil.PRIMARY].diagonal().any()
    # the pairwise rule lets naproxen be answered by the ester; the primary rule does not
    sets = [frozenset({"atc4:M01AE"}) if g in ("naproxen", "ibuprofen methyl ester") else frozenset({f"atc4:{k}"}) for k, g in enumerate(names)]
    J, T = ceil.jaccard_matrix(sets), ceil.tanimoto_matrix(mols)
    train = np.array([g in ("caffeine", "theophylline", "metformin", "warfarin") for g in names])
    picks = {rule: ceil.scores(J, T, ok, train)[1]["knn"] for rule, ok in rules.items()}
    assert picks["pairwise-matched"][i] == j and picks["any"][i] == j
    assert comp[picks[ceil.PRIMARY][i]] != comp[i] and comp[picks[ceil.PRIMARY][j]] != comp[j]


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


def test_bootstrap_by_component_resamples_whole_components():
    per = np.random.default_rng(0).random(60)
    V = np.repeat(per, 5)[None, :]  # 60 components of 5 drugs that all carry their component's value
    groups = np.repeat(np.arange(60), 5)
    by_comp, by_drug = ceil.bootstrap(V, boot=2000, seed=1, groups=groups), ceil.bootstrap(V, boot=2000, seed=1)
    assert by_comp[0] == pytest.approx(by_drug[0])
    assert by_comp[2][0] - by_comp[1][0] > 1.6 * (by_drug[2][0] - by_drug[1][0])  # about sqrt(5) wider: the 5 are not independent
    singles = ceil.bootstrap(V, boot=500, seed=1, groups=np.arange(300))  # one drug per component is the drug bootstrap
    assert all((x == y).all() for x, y in zip(singles, ceil.bootstrap(V, boot=500, seed=1)))
    labels = ceil.bootstrap(V, boot=500, seed=1, groups=np.repeat(["x", "y", "z"], 100))  # any labels do
    assert labels[1][0] <= labels[0][0] <= labels[2][0]
    one = ceil.bootstrap(V, boot=200, seed=1, groups=np.zeros(300, int))  # a single component has nothing to resample
    assert one[1][0] == pytest.approx(V.mean()) and one[2][0] == pytest.approx(V.mean())
    # components of 1 drug (value 0) and 3 drugs (value 1): a resample's mean is over the drugs it holds, 0, 3/4 or 1
    W = np.array([[0.0, 1.0, 1.0, 1.0]])
    mean, lo, hi = ceil.bootstrap(W, boot=1000, seed=2, groups=np.array([0, 1, 1, 1]))
    assert mean[0] == 0.75 and lo[0] == 0.0 and hi[0] == 1.0


def test_summarize_reports_a_component_interval_and_a_drug_interval():
    per = np.random.default_rng(0).random(40)
    comp, v = np.repeat(np.arange(40), 5), np.repeat(per, 5)  # 40 components of 5 drugs with one shared value
    r = ceil.summarize({m: v for m in ceil.METHODS}, np.ones(200, bool), comp, 1000, 0)
    q = r["mean:knn"]
    assert r["n"] == 200 and r["n_components"] == 40 and q["mean"] == pytest.approx(v.mean())
    assert q["ci"][1] - q["ci"][0] > 1.6 * (q["ci_drug"][1] - q["ci_drug"][0])  # the primary interval counts components, not drugs
    lo, hi = ceil.bootstrap(v[None, :], 1000, 0)[1:]
    assert q["ci_drug"] == pytest.approx([lo[0], hi[0]])  # the drug interval is the plain drug bootstrap
    half = ceil.summarize({m: v for m in ceil.METHODS}, np.arange(200) < 100, comp, 100, 0)
    assert half["n"] == 100 and half["n_components"] == 20  # a subset keeps only its own components


def test_verdict_reads_the_interval_against_zero():
    assert ceil.verdict(0.01, 0.05) == "beats" and ceil.verdict(-0.05, -0.01) == "falls below"
    assert ceil.verdict(-0.01, 0.05) == "cannot be told apart from" and ceil.verdict(0.0, 0.05) == "cannot be told apart from"


def small_population():
    """Rows like pool.jsonl / train_facts.jsonl for 14 drugs; the first three are Train drugs."""
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
    return pop


def test_run_on_a_small_population_and_report():
    res = ceil.run(small_population(), boot=200, seed=0)
    p = res["population"]
    assert p["drugs"] == 14 and p["train"] == 3 and p["pool"] == 11 and p["unknown_year"] == 4
    assert p["components"]["n"] <= 14 and p["components"]["largest"] >= 2 and 0 <= p["pairwise_knn_pick_in_own_component"]["train"] <= 3
    assert list(res["rules"]) == [ceil.PRIMARY, "pairwise-matched", "any"]
    oracle = {}
    for rule, rr in res["rules"].items():
        assert set(rr["subsets"]) == {"all drugs", "at least 3 facts", "first approved 2015 or later", "Train drugs only"}
        r = rr["subsets"]["all drugs"]
        assert r["n"] == 14 and rr["subsets"]["Train drugs only"]["n"] == 3 and 1 <= r["n_components"] <= 14
        assert rr["distinct_constants"]["constant"] >= 1 and rr["distinct_constants"]["constant_train"] >= 1
        for m in ceil.METHODS:
            q = r[f"mean:{m}"]
            assert 0 <= q["ci"][0] <= q["ci"][1] <= 1 + 1e-9 and 0 <= q["mean"] <= 1 + 1e-9 and q["ci_drug"][0] <= q["ci_drug"][1]
            assert r[f"share0.3:{m}"]["mean"] >= r[f"share0.4:{m}"]["mean"]
        for a, b in ceil.DIFFS:
            assert r[f"diff:{a}-{b}"]["mean"] == pytest.approx(r[f"mean:{a}"]["mean"] - r[f"mean:{b}"]["mean"])
        assert r["mean:oracle"]["mean"] >= max(r[f"mean:{m}"]["mean"] for m in ("knn", "medoid", "constant", "constant_train", "random")) - 1e-12
        oracle[rule] = r["mean:oracle"]["mean"]
    assert oracle[ceil.PRIMARY] <= oracle["pairwise-matched"] + 1e-12 <= oracle["any"] + 2e-12  # fewer neighbours, never a better oracle
    text = ceil.report(res)
    assert "facts-kNN" in text and "reading" in text and "pairwise-matched" in text and "component-matched (primary)" in text
    assert "in short, with whole components kept apart" in text and "(c') constant, Train-fixed" in text and "over components" in text and "over drugs" in text


def set_diffs(cells, **ci):
    """Overwrite paired cells of a summarize() result: key 'knn-constant' -> (mean, lo, hi)."""
    for key, (mean, lo, hi) in ci.items():
        cells["diff:" + key] = {"mean": mean, "ci": [lo, hi], "ci_drug": [lo, hi]}


def test_reading_states_its_scope_and_reports_upper_bounds():
    res = ceil.run(small_population(), boot=100, seed=0)
    cells = res["rules"][ceil.PRIMARY]["subsets"]["Train drugs only"]
    set_diffs(cells, **{"knn-constant": (-0.014, -0.03, 0.004), "knn-constant_train": (0.001, -0.012, 0.015),
                        "medoid-constant": (-0.02, -0.04, -0.01), "medoid-constant_train": (0.03, 0.01, 0.05)})
    lines = ceil.reading(res)
    text = " ".join(lines)
    assert lines[0] == ceil.SCOPE
    for phrase in ("one-nearest-neighbour", "5-medoid", "learned predictors", "generated structures", "richer facts", "free-text profiles",
                   "failing to reject zero is not equivalence"):
        assert phrase in text
    line = next(x for x in lines if x.startswith(f"{ceil.PRIMARY}, Train drugs only"))
    knn, medoid = line.split("; 5-medoid ")
    assert "cannot be told apart from the central constant" in knn and "(paired -0.014 [-0.030, +0.004]; compatible with gains up to +0.004)" in knn
    assert "cannot be told apart from the Train-fixed constant" in knn and "(paired +0.001 [-0.012, +0.015]; compatible with gains up to +0.015)" in knn
    assert "falls below the central constant" in medoid and "beats the Train-fixed constant" in medoid and "compatible with" not in medoid
    assert "recall" not in text.lower() and " nil" not in text and "no gain" not in text
    set_diffs(cells, **{"knn-constant": (-0.04, -0.05, -0.02), "knn-constant_train": (-0.03, -0.05, -0.02)})  # wholly below 0: no upper bound
    below = next(x for x in ceil.reading(res) if x.startswith(f"{ceil.PRIMARY}, Train drugs only")).split("; 5-medoid ")[0]
    assert below.count("falls below") == 2 and "compatible with" not in below
    assert "over Train drugs it falls below both constants" in ceil.reading(res)[-1]


def test_summary_words_follow_the_two_intervals():
    def r(a, b):
        return {"diff:knn-constant": {"ci": a}, "diff:knn-constant_train": {"ci": b}}

    assert ceil._summary(r([0.01, 0.02], [0.01, 0.02])) == "beats both constants"
    assert ceil._summary(r([0.01, 0.02], [-0.01, 0.02])) == "beats one constant but not the other"
    assert ceil._summary(r([-0.02, -0.01], [-0.03, -0.01])) == "falls below both constants"
    assert ceil._summary(r([-0.02, -0.01], [-0.01, 0.02])) == "does not beat either constant"


def write_population(cache, rows, derived=True):
    """pool.jsonl, train_facts.jsonl and a manifest (with the build_pool.py `derived` record unless derived is False)."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, part in (("train_facts.jsonl", [r for r in rows if r["source"] == "train"]), ("pool.jsonl", [r for r in rows if r["source"] != "train"])):
        (cache / name).write_text("".join(json.dumps({k: v for k, v in r.items() if k != "source"}) + "\n" for r in part), encoding="utf-8")
    manifest = {"chembl_release": "ChEMBL_TEST", "release_date": "2026-01-01", "files": {}}
    if derived:
        manifest["derived"] = {"built_by": "scripts/build_pool.py", "inputs": {}, "files": {
            n: {"records": 0, "sha256": hashlib.sha256((cache / n).read_bytes()).hexdigest()} for n in ("pool.jsonl", "train_facts.jsonl")}}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_provenance_records_inputs_settings_and_versions(tmp_path):
    write_population(tmp_path, small_population())
    prov = ceil.provenance(tmp_path, 123, 7, "python scripts/class_ceiling.py --boot 123")
    sha = lambda n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
    assert prov["command"] == "python scripts/class_ceiling.py --boot 123" and prov["seed"] == 7 and prov["bootstrap_resamples"] == 123
    assert prov["chembl_release"] == "ChEMBL_TEST" and prov["chembl_release_date"] == "2026-01-01"
    assert prov["inputs"]["pool.jsonl"] == {"sha256": sha("pool.jsonl"), "rows": 11}
    assert prov["inputs"]["train_facts.jsonl"] == {"sha256": sha("train_facts.jsonl"), "rows": 3}
    assert prov["inputs"]["manifest.json"] == {"sha256": sha("manifest.json")} and prov["manifest_records_derived_files"] is True
    assert prov["git"]["revision"] is None or len(prov["git"]["revision"]) == 40
    assert isinstance(prov["git"]["differs_from_revision"], list)
    assert prov["git"]["script_sha256"] == hashlib.sha256((ROOT / "scripts" / "class_ceiling.py").read_bytes()).hexdigest()
    assert set(prov["versions"]) == {"python", "rdkit", "numpy"} and all(prov["versions"].values())
    json.dumps(prov)  # serialisable as is

    (tmp_path / "pool.jsonl").write_text("changed\n", encoding="utf-8")  # a pool that is not the one build_pool.py recorded
    with pytest.raises(ValueError, match="differs from the manifest"):
        ceil.provenance(tmp_path, 1, 0, "x")
    write_population(tmp_path, small_population(), derived=False)  # no derived record: nothing to check, and the result says so
    assert ceil.provenance(tmp_path, 1, 0, "x")["manifest_records_derived_files"] is False


def test_main_writes_the_json_with_numbers_reading_and_provenance(tmp_path, capsys):
    write_population(tmp_path / "cache", small_population())
    out = tmp_path / "result.json"
    ceil.main(["--cache", str(tmp_path / "cache"), "--boot", "100", "--seed", "5", "--json", str(out)])
    printed = capsys.readouterr().out
    res = json.loads(out.read_text(encoding="utf-8"))
    assert set(res) == {"population", "rules", "reading", "provenance"} and list(res["rules"]) == [ceil.PRIMARY, "pairwise-matched", "any"]
    assert res["provenance"]["seed"] == 5 and res["provenance"]["bootstrap_resamples"] == 100
    assert "class_ceiling.py" in res["provenance"]["command"] and "--boot 100" in res["provenance"]["command"] and "--seed 5" in res["provenance"]["command"]
    assert res["reading"][0] == ceil.SCOPE and "free-text profiles" in printed and "reading" in printed
    assert res["reading"] == ceil.reading({k: res[k] for k in ("population", "rules")})
    ceil.main(["--cache", str(tmp_path / "cache"), "--boot", "100", "--seed", "5", "--json", str(tmp_path / "again.json")])
    again = json.loads((tmp_path / "again.json").read_text(encoding="utf-8"))
    assert again["rules"] == res["rules"] and again["reading"] == res["reading"]  # seeded: the same numbers every run
    with pytest.raises(SystemExit):  # no cache, no result
        ceil.main(["--cache", str(tmp_path / "missing")])


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
