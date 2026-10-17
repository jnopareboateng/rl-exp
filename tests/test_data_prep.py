"""data_prep on small synthetic inputs: no network, no real dataset.

    .venv/bin/python -m pytest tests/test_data_prep.py
"""
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem

from rlexp import chem
from rlexp import data_prep as dp

THINK = "<think>\n"
DRUGS = {  # name -> SMILES. Scaffolds are unique except where a family below says otherwise.
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "ibuprofen methyl ester": "CC(C)Cc1ccc(cc1)C(C)C(=O)OC",
    "ibuprofen ethyl ester": "CC(C)Cc1ccc(cc1)C(C)C(=O)OCC",
    "naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",  # 0.42 to ibuprofen but 0.33 to its methyl ester: linked only through ibuprofen
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc12",
    "nordazepam": "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1",
    "morphine": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
    "codeine": "COc1ccc2CC3N(C)CCC45C(Oc1c24)C(O)C=CC35",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "omeprazole": "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
    "lansoprazole": "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1",
    "menthol": "CC(C)C1CCC(C)CC1O",  # the three cyclohexanes share a scaffold with Tanimoto 0.05 to 0.24
    "tranexamic acid": "NCC1CCC(CC1)C(=O)O",
    "gabapentin": "NCC1(CC(=O)O)CCCCC1",
    "sumatriptan": "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1",  # indole scaffold, Tanimoto 0.36 to melatonin
    "melatonin": "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
    "isoniazid": "NNC(=O)c1ccncc1",  # pyridine scaffold, Tanimoto 0.34 to nicotinamide
    "nicotinamide": "NC(=O)c1cccnc1",
    "metformin": "CN(C)C(=N)N=C(N)N",  # acyclic: empty scaffold, which must never link two drugs
    "valproic acid": "CCCC(CCC)C(=O)O",
    "sildenafil": "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "loratadine": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "tamoxifen": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "sertraline": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc12",
    "nicotine": "CN1CCCC1c1cccnc1",
    "warfarin": "CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O",
    "haloperidol": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
}
TANIMOTO_FAMILIES = [  # components held together by Tanimoto >= 0.4
    {"ibuprofen", "ibuprofen methyl ester", "ibuprofen ethyl ester", "naproxen"},
    {"diazepam", "nordazepam"}, {"morphine", "codeine"}, {"caffeine", "theophylline"}, {"omeprazole", "lansoprazole"},
]
SCAFFOLD_FAMILIES = [  # components held together only by an identical non-empty scaffold
    {"menthol", "tranexamic acid", "gabapentin"}, {"sumatriptan", "melatonin"}, {"isoniazid", "nicotinamide"},
]
FAMILIES = TANIMOTO_FAMILIES + SCAFFOLD_FAMILIES
YEARS = {g: 2018 for g in DRUGS} | {"ibuprofen ethyl ester": 2025, "sildenafil": 2026, "diazepam": 2025, "nordazepam": 2025,
                                    "menthol": 2025, "morphine": 2024}
TEST_T = {"ibuprofen ethyl ester", "sildenafil", "diazepam", "nordazepam", "menthol"}
BUFFER = {"ibuprofen", "ibuprofen methyl ester", "naproxen", "tranexamic acid", "gabapentin"}  # the rest of two Test-T components
ASPIRIN, CAFFEINE, THEOPHYLLINE = "CC(=O)Oc1ccccc1C(=O)O", DRUGS["caffeine"], DRUGS["theophylline"]
OSELTAMIVIR, FUROSEMIDE = "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl"  # unique scaffolds
PEPTIDE = "NCC(=O)" * 30 + "O"  # 30 glycines: 121 heavy atoms, acyclic
MALTOL = "Cc1occc(=O)c1O"  # the ligand of ferric maltol
L_ALANINE, D_ALANINE = "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"  # same connectivity, different SMILES
KETO, ENOL, ENOL2 = "CCOC(=O)CC(C)=O", "CCOC(=O)C=C(C)O", "CCOC(O)=CC(C)=O"  # ethyl acetoacetate tautomers: different conn, equal skeleton
DUMMY, OVERSIZE = "c1ccc(*)cc1CC(=O)O", "NCC(=O)" * 120 + "O"  # chem.standardize refuses a dummy atom and more than 400 atoms


def rows_for(group, *answers, tasks=("A", "B", "C", "MO")):
    """One prompt row per answer; the completion ends with the answer."""
    return [{"id": f"{group}#{i}", "corpus": "inverse", "task_type": tasks[i % len(tasks)], "drug": group, "group": group,
             "n_tokens": 30, "n_prompt_tokens": 10 + i, "n_completion_tokens": 20 + i, "text": "",
             "prompt": f"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nDescribe {group} {i}<|im_end|>\n<|im_start|>assistant\n",
             "completion": f"{THINK}reasoning {i}</think>\n\n{a}"} for i, a in enumerate(answers)]


def entry(name, smiles, year, **kw):
    return {"name": name, "status": "ok", "match": "exact", "chembl_id": f"CHEMBL_{name}",
            "first_approval": year, "approval_year": year, "parent_smiles": smiles} | kw


def synth(years=YEARS):
    rows = [r for g, s in DRUGS.items() for r in rows_for(g, s, s)]
    return rows, {g: entry(g, s, years[g]) for g, s in DRUGS.items()}


def refs_for(years=YEARS, seed=0, n_test_r=5, n_val=3):
    rows, pool = synth(years)
    refs, _ = dp.build_refs(rows, pool)
    return refs, dp.assign_splits(refs, seed, n_test_r, n_val)


def make_inputs(tmp_path):
    """Write synthetic row files and a pool file; returns (main args, every input row including the ones that get dropped)."""
    rows, pool = synth()
    rows += rows_for("mystery", "not a structure")  # no pool entry and no parsable answer
    rows += rows_for("undated", OSELTAMIVIR)
    pool["undated"] = entry("undated", OSELTAMIVIR, None)
    rows += rows_for("bare atom", "not a structure")  # a one-atom reference the answer extractor cannot recover
    pool["bare atom"] = entry("bare atom", "[129Xe]", 2022)
    rows += rows_for("peptide", PEPTIDE, PEPTIDE)  # over 100 heavy atoms
    pool["peptide"] = entry("peptide", PEPTIDE, 2018)
    rows += rows_for("acetoacetate", ENOL, ENOL)  # ChEMBL has the keto form, the SFT answers the enol form: agreement, one alternate
    pool["acetoacetate"] = entry("acetoacetate", KETO, 2018)
    rows += [r | {"task_type": "PD"} for r in rows_for("protein only", ASPIRIN)]
    rows += rows_for("caffeine | aspirin", ASPIRIN)
    cut = len(rows) // 2
    for name, part in (("train_think.jsonl", rows[:cut]), ("val_think.jsonl", rows[cut:])):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
    (tmp_path / "pool.json").write_text(json.dumps(list(pool.values())), encoding="utf-8")
    args = ["--rows", str(tmp_path / "train_think.jsonl"), str(tmp_path / "val_think.jsonl"),
            "--pool", str(tmp_path / "pool.json"), "--data", str(tmp_path / "data"), "--hand", str(tmp_path / "data"),
            "--n-test-r", "5", "--n-val", "3"]
    return args, rows


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_components_follow_tanimoto_and_identical_scaffold_edges():
    refs, _ = refs_for()
    assert not [g for g, r in refs.items() if r["exclude_reason"]]  # every fixture drug passes the screens
    mols = {g: chem.standardize(s) for g, s in DRUGS.items()}
    assert chem.tanimoto_bits(mols["naproxen"], mols["ibuprofen"]) >= 0.4 > chem.tanimoto_bits(mols["naproxen"], mols["ibuprofen methyl ester"])
    for fam in SCAFFOLD_FAMILIES:  # same scaffold, no Tanimoto edge: only the scaffold rule can join them
        scaffolds = {refs[g]["ref_scaffold"] for g in fam}
        assert len(scaffolds) == 1 and scaffolds != {""}, fam
        assert all(chem.tanimoto_bits(mols[a], mols[b]) < 0.4 for a, b in itertools.combinations(sorted(fam), 2)), fam
    assert refs["metformin"]["ref_scaffold"] == refs["valproic acid"]["ref_scaffold"] == ""  # acyclic: no scaffold, no edge
    clusters = {}
    for g, r in refs.items():
        clusters.setdefault(r["cluster_id"], set()).add(g)
    singles = [{g} for g in DRUGS if not any(g in f for f in FAMILIES)]
    assert sorted(map(sorted, clusters.values())) == sorted(map(sorted, FAMILIES + singles))
    assert sorted(clusters) == list(range(len(clusters)))  # ids are 0..n-1


@pytest.mark.parametrize("seed", range(6))
def test_no_pair_across_splits_is_similar_or_shares_a_scaffold_and_components_stay_whole(seed):
    refs, info = refs_for(seed=seed)
    used = [g for g, r in refs.items() if r["split"] in dp.SPLITS]
    mols = {g: chem.standardize(refs[g]["ref_smiles"]) for g in used}
    for a, b in itertools.combinations(used, 2):
        if refs[a]["split"] != refs[b]["split"]:
            assert chem.tanimoto_bits(mols[a], mols[b]) < 0.4, (a, b)
            sa, sb = chem.scaffold(mols[a]), chem.scaffold(mols[b])  # recomputed, not read back from refs
            assert not (sa and sa == sb), (a, b, sa)
    for fam in FAMILIES:
        splits = {refs[g]["split"] for g in fam}
        assert len(splits) == 1 or (fam & TEST_T and splits == {"test_t", "buffer"}), (fam, splits)
    assert len(info["max_cross_tanimoto"]) == 6  # all pairs among train, val, test_r, test_t
    for pair, v in info["max_cross_tanimoto"].items():  # the report numbers match an independent recomputation
        a, b = v["pair"]
        assert {refs[a]["split"], refs[b]["split"]} == set(pair.split("|"))
        assert v["max"] == round(chem.tanimoto_bits(mols[a], mols[b]), 4) < 0.4
    assert info["scaffold_overlap"] == {"drugs_sharing_a_scaffold_with_train": {"val": 0, "test_r": 0, "test_t": 0},
                                        "shared_scaffolds": dict.fromkeys(info["max_cross_tanimoto"], 0)}


def test_test_t_is_year_2025_plus_and_buffer_is_the_rest_of_its_component():
    refs, _ = refs_for()
    by = lambda s: {g for g, r in refs.items() if r["split"] == s}
    assert by("test_t") == TEST_T and by("buffer") == BUFFER and not by("excluded")
    for s in ("train", "val", "test_r"):  # nothing recent and nothing linked to a Test-T drug is used anywhere else
        assert not by(s) & (TEST_T | BUFFER)
        assert all(refs[g]["approval_year"] < 2025 for g in by(s))
    assert refs["diazepam"]["split"] == refs["nordazepam"]["split"] == "test_t"  # two Test-T drugs may share a component
    assert refs["morphine"]["split"] == refs["codeine"]["split"] != "test_t"  # 2024 is not Test-T
    # a Test-T drug buffers its scaffold-only relatives too, although their Tanimoto to it is below 0.25
    assert refs["menthol"]["split"] == "test_t" and refs["tranexamic acid"]["split"] == refs["gabapentin"]["split"] == "buffer"
    refs, _ = refs_for(years=YEARS | {"metformin": 2024, "valproic acid": 2025})  # empty scaffolds: no link to each other
    assert refs["metformin"]["split"] in ("train", "val", "test_r") and refs["valproic acid"]["split"] == "test_t"


def test_targets_are_hit_exactly_or_the_shortfall_is_reported():
    refs, info = refs_for(n_test_r=5, n_val=3)
    assert info["targets"] == {"test_r": {"target": 5, "got": 5, "exact": True}, "val": {"target": 3, "got": 3, "exact": True}}
    assert Counter(r["split"] for r in refs.values()) == {"train": 11, "test_r": 5, "val": 3, "test_t": 5, "buffer": 5}
    refs, info = refs_for(n_test_r=100, n_val=3)  # only 19 drugs are eligible
    assert info["targets"]["test_r"] == {"target": 100, "got": 19, "exact": False}
    assert info["targets"]["val"] == {"target": 3, "got": 0, "exact": False}


def test_the_seed_drives_the_draw():
    draws = {tuple(sorted(g for g, r in refs_for(seed=s)[0].items() if r["split"] == "test_r")) for s in range(6)}
    assert len(draws) > 1


def test_approval_year_precedence_manual_then_audit_then_chembl_then_openfda():
    names = ["manual", "audit", "audit_null", "openfda", "chembl", "none", "manual_without_pool"]
    rows = [r for g in names for r in rows_for(g, ASPIRIN)]
    pool = {g: entry(g, ASPIRIN, 2025) for g in ("manual", "audit", "audit_null", "chembl")}
    pool["openfda"] = entry("openfda", ASPIRIN, 2026, first_approval=None, fda_first_approval="20260115")
    pool["none"] = entry("none", ASPIRIN, None)
    manual = {"manual": {"smiles": ASPIRIN, "approval_year": 2019}, "manual_without_pool": {"smiles": ASPIRIN, "approval_year": 2021}}
    audit = {"manual": {"earliest_global_year": 2020}, "audit": {"earliest_global_year": 2023, "confidence": "high"},
             "audit_null": {"earliest_global_year": None}}
    refs, notes = dp.build_refs(rows, pool, manual, audit)
    assert {g: (r["approval_year"], r["approval_source"], r["exclude_reason"]) for g, r in refs.items()} == {
        "manual": (2019, "manual", None),
        "audit": (2023, "audit", None),
        "audit_null": (2025, "chembl", None),
        "openfda": (2026, "openfda", None),
        "chembl": (2025, "chembl", None),
        "none": (None, None, "no_approval_year"),
        "manual_without_pool": (2021, "manual", None),
    }
    assert {o["group"]: o["source"] for o in notes["year_overrides"]} == {
        "manual": "manual", "audit": "audit", "manual_without_pool": "manual"}
    assert refs["none"]["ref_smiles"] and refs["none"]["split"] == "excluded"  # keeps its structure, is just not used


def test_reference_precedence_and_the_disagreement_exclusion():
    def case(group, answers, smiles, manual=None, **kw):
        return group, (answers, entry(group, smiles, 2018, **kw), manual)

    cases = dict([
        case("agree", ["OC(=O)c1ccccc1OC(C)=O"], ASPIRIN + ".[Na+].[Cl-]"),  # same molecule written differently; salt stripped
        case("no_sft", ["no structure here"], CAFFEINE),
        case("no_chembl", [DRUGS["isoniazid"]], None, status="no_match"),
        case("disagree", [THEOPHYLLINE] * 2, CAFFEINE),
        case("disagree_manual", [THEOPHYLLINE] * 2, CAFFEINE, {"smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"}),
        case("nothing", ["no structure here"], None, status="no_match"),
        case("majority_outvotes_odd_row", [CAFFEINE, THEOPHYLLINE, CAFFEINE], CAFFEINE),
        case("majority_disagrees", [CAFFEINE, THEOPHYLLINE, CAFFEINE], THEOPHYLLINE),
    ])
    rows = [r for g, (answers, _, _) in cases.items() for r in rows_for(g, *answers)]
    refs, notes = dp.build_refs(rows, {g: p for g, (_, p, _) in cases.items()},
                                {g: m for g, (_, _, m) in cases.items() if m})
    assert {g: (r["ref_source"], r["sft_agrees"], r["exclude_reason"]) for g, r in refs.items()} == {
        "agree": ("chembl", True, None),
        "no_sft": ("chembl", None, None),
        "no_chembl": ("sft_answer", None, None),
        "disagree": (None, False, "unresolved_disagreement"),
        "disagree_manual": ("manual", False, None),
        "nothing": (None, None, "no_structure"),
        "majority_outvotes_odd_row": ("chembl", True, None),
        "majority_disagrees": (None, False, "unresolved_disagreement"),
    }
    canon = lambda s: chem.canonical(chem.standardize(s))
    assert refs["agree"]["ref_smiles"] == canon(ASPIRIN) and refs["agree"]["chembl_id"] == "CHEMBL_agree"
    assert refs["no_chembl"]["ref_smiles"] == canon(DRUGS["isoniazid"]) and refs["no_chembl"]["chembl_id"] is None
    assert refs["disagree_manual"]["ref_smiles"] == canon(CAFFEINE)
    assert refs["disagree"]["ref_smiles"] is None and refs["disagree"]["ref_conn"] is None
    ref = refs["no_sft"]
    mol = chem.standardize(ref["ref_smiles"])
    assert (ref["ref_conn"], ref["ref_scaffold"], ref["ref_inchikey"], ref["n_heavy"]) == (
        chem.conn(mol), chem.scaffold(mol), chem.inchikey(mol), mol.GetNumHeavyAtoms())
    assert refs["agree"]["n_heavy"] == 13 and refs["disagree"]["n_heavy"] is None  # aspirin; no structure
    assert {d["group"]: d["resolved_by"] for d in notes["disagreements"]} == {
        "disagree": None, "disagree_manual": "manual", "majority_disagrees": None}


def test_manual_smiles_that_does_not_parse_is_an_error_not_a_silent_fallback():
    rows, pool = synth()
    with pytest.raises(ValueError, match="gabapentin"):
        dp.build_refs(rows, pool, {"gabapentin": {"smiles": "C1CC(("}})


def test_a_manual_entry_with_null_smiles_is_unresolvable_and_never_falls_back():
    names = ["data agrees", "unresolvable", "unresolvable undated", "sft only"]
    rows = [r for g in names for r in rows_for(g, ASPIRIN)]
    pool = {g: entry(g, ASPIRIN, 2018) for g in names[:3]} | {"unresolvable undated": entry("unresolvable undated", ASPIRIN, None)}
    manual = {"unresolvable": {"smiles": None, "approval_year": 2014}, "unresolvable undated": {"smiles": None, "approval_year": None},
              "sft only": {"smiles": None, "approval_year": 2015}}
    refs, notes = dp.build_refs(rows, pool, manual)
    assert (refs["data agrees"]["ref_source"], refs["data agrees"]["exclude_reason"]) == ("chembl", None)
    for g in ("unresolvable", "sft only"):  # ChEMBL agrees with the SFT answer, or only the SFT answer exists: neither is used
        r = refs[g]
        assert (r["exclude_reason"], r["ref_source"], r["split"]) == ("manual_unresolvable", None, "excluded")
        assert all(r[k] is None for k in ("ref_smiles", "ref_inchikey", "ref_conn", "ref_scaffold", "n_heavy"))
    assert (refs["unresolvable"]["approval_year"], refs["unresolvable"]["approval_source"]) == (2014, "manual")  # still dated
    assert refs["unresolvable"]["chembl_id"] == "CHEMBL_unresolvable" and refs["unresolvable"]["sft_agrees"] is True  # still reported
    assert refs["unresolvable undated"]["exclude_reason"] == "manual_unresolvable"  # the structure problem comes first
    assert notes["manual_effect"]["unresolvable"] == ["sft only", "unresolvable", "unresolvable undated"]


def test_manual_effect_says_what_each_entry_did_to_the_reference_the_data_would_have_given():
    canon = lambda s: chem.canonical(chem.standardize(s))
    cases = {  # group -> (ChEMBL SMILES, SFT answers, manual SMILES, expected effect)
        "same": (CAFFEINE, [CAFFEINE], CAFFEINE, "identical"),
        "other tautomer or stereo": (L_ALANINE, [L_ALANINE], D_ALANINE, "changed_smiles_only"),
        "other molecule": (CAFFEINE, [CAFFEINE], THEOPHYLLINE, "changed_connectivity"),
        "gap": (CAFFEINE, [THEOPHYLLINE] * 2, CAFFEINE, "filled_gap"),  # ChEMBL and SFT disagree: excluded without the entry
        "no data": (None, ["no structure here"], THEOPHYLLINE, "filled_gap"),
    }
    rows = [r for g, (_, answers, _, _) in cases.items() for r in rows_for(g, *answers)]
    pool = {g: entry(g, c, 2018, **({} if c else {"status": "no_match"})) for g, (c, _, _, _) in cases.items()}
    refs, notes = dp.build_refs(rows, pool, {g: {"smiles": m} for g, (_, _, m, _) in cases.items()})
    for g, (_, _, manual_smiles, effect) in cases.items():
        assert g in notes["manual_effect"][effect], (g, notes["manual_effect"])
        assert refs[g]["ref_source"] == "manual" and refs[g]["ref_smiles"] == canon(manual_smiles)  # the entry always wins
    assert sum(map(len, notes["manual_effect"].values())) == len(cases)
    assert refs["other tautomer or stereo"]["ref_conn"] == chem.conn(chem.standardize(L_ALANINE))  # the stereo flip keeps the connectivity


def test_convention_is_copied_from_the_manual_entry_and_validated():
    rows = rows_for("ferric maltol", MALTOL) + rows_for("plain", ASPIRIN)
    pool = {"ferric maltol": entry("ferric maltol", None, 2016, status="no_match"), "plain": entry("plain", ASPIRIN, 2018)}
    refs, _ = dp.build_refs(rows, pool, {"ferric maltol": {"smiles": MALTOL, "approval_year": 2016, "convention": "ligand_only"}})
    assert (refs["ferric maltol"]["convention"], refs["ferric maltol"]["ref_source"]) == ("ligand_only", "manual")
    assert refs["plain"]["convention"] is None and refs["ferric maltol"]["n_heavy"] == 9
    assert list(refs["plain"])[-4:] == ["n_heavy", "convention", "pubchem_create_year", "pubmed_count"]  # the contract's order
    with pytest.raises(AssertionError, match="convention"):  # a typo must not silently become null
        dp.build_refs(rows, pool, {"ferric maltol": {"smiles": MALTOL, "approval_year": 2016, "convention": "ligand-only"}})


def test_alternates_are_other_registry_forms_with_an_equal_skeleton_and_a_different_conn():
    canon = lambda s: chem.canonical(chem.standardize(s))
    keto, enol, enol2 = canon(KETO), canon(ENOL), canon(ENOL2)
    cases = {  # group -> (ChEMBL SMILES, SFT answers, manual SMILES, expected (ref_source, sft_agrees, ref_alt_smiles))
        "chembl keto, sft enol": (KETO, [ENOL] * 2, None, ("chembl", True, [enol])),
        "chembl keto, sft keto": (KETO, [KETO], None, ("chembl", True, [])),
        "chembl keto, sft keto, manual enol": (KETO, [KETO], ENOL, ("manual", True, [keto])),  # the same form twice is listed once
        "chembl keto, sft enol, manual enol": (KETO, [ENOL], ENOL, ("manual", True, [keto])),  # the reference itself is never listed
        "chembl keto, sft enol, manual keto": (KETO, [ENOL], KETO, ("manual", True, [enol])),
        "chembl enol, sft enol2, manual keto": (ENOL, [ENOL2], KETO, ("manual", True, sorted([enol, enol2]))),
        "chembl salt, sft keto, manual keto": (ENOL + ".[Na+].[Cl-]", [KETO], KETO, ("manual", True, [enol])),  # standardised first
        "no chembl, sft enol": (None, [ENOL], None, ("sft_answer", None, [])),
        "no chembl, sft enol, manual keto": (None, [ENOL], KETO, ("manual", None, [enol])),
        "other skeleton under manual": (CAFFEINE, [CAFFEINE], THEOPHYLLINE, ("manual", True, [])),  # a different molecule is no alternate
    }
    rows = [r for g, (_, answers, _, _) in cases.items() for r in rows_for(g, *answers)]
    pool = {g: entry(g, c, 2018, **({} if c else {"status": "no_match"})) for g, (c, _, _, _) in cases.items()}
    refs, _ = dp.build_refs(rows, pool, {g: {"smiles": m} for g, (_, _, m, _) in cases.items() if m})
    for g, (_, _, _, want) in cases.items():
        r = refs[g]
        assert (r["ref_source"], r["sft_agrees"], r["ref_alt_smiles"]) == want and r["exclude_reason"] is None, g
        ref_mol = chem.standardize(r["ref_smiles"])
        alt_mols = [chem.standardize(a) for a in r["ref_alt_smiles"]]
        assert all(chem.conn(m) != r["ref_conn"] and chem.skeleton(m) == chem.skeleton(ref_mol) for m in alt_mols), g
        assert len({chem.conn(m) for m in alt_mols}) == len(alt_mols)
    assert {g for g, r in refs.items() if r["ref_alt_smiles"]} == {g for g, c in cases.items() if c[3][2]}


def test_a_tautomer_difference_is_agreement_and_only_a_different_skeleton_is_a_disagreement():
    rows = rows_for("tautomer", ENOL, ENOL) + rows_for("other molecule", THEOPHYLLINE, THEOPHYLLINE)
    pool = {"tautomer": entry("tautomer", KETO, 2018), "other molecule": entry("other molecule", CAFFEINE, 2018)}
    refs, notes = dp.build_refs(rows, pool)
    assert chem.conn(chem.standardize(KETO)) != chem.conn(chem.standardize(ENOL))  # strict conn does differ
    assert (refs["tautomer"]["sft_agrees"], refs["tautomer"]["exclude_reason"], refs["tautomer"]["ref_source"]) == (True, None, "chembl")
    assert (refs["other molecule"]["sft_agrees"], refs["other molecule"]["exclude_reason"]) == (False, "unresolved_disagreement")
    assert [d["group"] for d in notes["disagreements"]] == ["other molecule"]


def test_structures_chem_refuses_are_reported_not_dropped_silently_and_no_reference_has_a_dummy_atom(monkeypatch):
    rows = rows_for("chembl dummy", ASPIRIN) + rows_for("sft dummy", DUMMY, ASPIRIN) + rows_for("oversize", OVERSIZE)
    pool = {"chembl dummy": entry("chembl dummy", DUMMY, 2018), "sft dummy": entry("sft dummy", ASPIRIN, 2018),
            "oversize": entry("oversize", OVERSIZE, 2018)}
    refs, notes = dp.build_refs(rows, pool)
    assert notes["rejected_structures"] == [
        {"group": "chembl dummy", "source": "chembl", "why": "dummy_atom"},
        {"group": "oversize", "source": "chembl", "why": "over_max_atoms"},
        {"group": "oversize", "source": "sft_answer", "why": "over_max_atoms", "rows": 1},
        {"group": "sft dummy", "source": "sft_answer", "why": "dummy_atom", "rows": 1}]
    assert (refs["chembl dummy"]["ref_source"], refs["chembl dummy"]["exclude_reason"]) == ("sft_answer", None)  # the SFT answer is used
    assert (refs["sft dummy"]["ref_source"], refs["sft dummy"]["sft_agrees"]) == ("chembl", True)  # the other row still votes
    assert (refs["oversize"]["ref_smiles"], refs["oversize"]["exclude_reason"]) == (None, "no_structure")
    mols = [Chem.MolFromSmiles(r["ref_smiles"]) for r in refs.values() if r["ref_smiles"]]
    assert mols and not any(a.GetAtomicNum() == 0 for m in mols for a in m.GetAtoms())
    for bad, why in ((DUMMY, "dummy_atom"), (OVERSIZE, "over_max_atoms")):  # a hand-written structure is an error that says why
        with pytest.raises(ValueError, match=why):
            dp.build_refs(rows_for("x", ASPIRIN), {}, {"x": {"smiles": bad}})
    # the check itself: were a dummy atom ever to get past chem, the reference would be refused rather than written
    real = dp._mol
    monkeypatch.setattr(dp, "_mol", lambda s: (Chem.MolFromSmiles(s), "FAKEKEYFAKEKEY") if s == DUMMY else real(s))
    with pytest.raises(AssertionError, match="dummy atom"):
        dp.build_refs(rows_for("leaky", "no structure here"), {"leaky": entry("leaky", DUMMY, 2018)})


def test_references_the_answer_extractor_cannot_recover_are_excluded():
    cases = {"bare isotope": ("[129Xe]", 2022), "ethanol": ("CCO", 2018), "tiny and undated": ("CCO", None),
             "fine": (ASPIRIN, 2018), "fine and undated": (ASPIRIN, None)}
    rows = [r for g in cases for r in rows_for(g, "no structure here")]
    refs, _ = dp.build_refs(rows, {g: entry(g, s, y) for g, (s, y) in cases.items()})
    assert {g: r["exclude_reason"] for g, r in refs.items()} == {
        "bare isotope": "unextractable_reference", "ethanol": "unextractable_reference",
        "tiny and undated": "unextractable_reference",  # the structure problem is reported before the year problem
        "fine": None, "fine and undated": "no_approval_year"}
    assert refs["ethanol"]["ref_smiles"] == "CCO" and refs["ethanol"]["split"] == "excluded"  # recorded, just unusable
    refs, _ = dp.build_refs(rows_for("fine", "no structure here"), {"fine": entry("fine", ASPIRIN, 2018)}, {"fine": {"smiles": "CCO"}})
    assert refs["fine"]["ref_source"] == "manual" and refs["fine"]["exclude_reason"] == "unextractable_reference"
    refs, _ = dp.build_refs(rows_for("sft", "OC(=O)c1ccccc1OC(C)=O"), {})  # an SFT-answer reference goes through the same screen
    assert refs["sft"]["ref_source"] == "sft_answer" and refs["sft"]["exclude_reason"] == "no_approval_year"


def test_extractability_screen_tries_the_smiles_and_20_seeded_random_variants(monkeypatch):
    mol = chem.standardize(DRUGS["sildenafil"])
    conn, smiles = chem.conn(mol), chem.canonical(mol)
    real, seen = chem.extract_answer, []
    monkeypatch.setattr(chem, "extract_answer", lambda text: seen.append(text) or real(text))
    assert dp._extractable(mol, conn, smiles)
    first = list(seen)
    seen.clear()
    assert dp._extractable(mol, conn, smiles) and seen == first  # same variants every time
    assert len(first) == 1 + dp.N_VARIANTS == 21 and first[0] == "</think>\n" + smiles and len(set(first)) > 5
    monkeypatch.setattr(chem, "extract_answer", lambda text: real(text) if text == first[0] else None)  # fails on a variant only
    assert not dp._extractable(mol, conn, smiles)


def test_exposure_is_copied_into_refs_and_null_when_missing():
    rows, pool = synth()
    exposure = {"naproxen": {"pubchem_cid": 156391, "pubchem_create_year": 2005, "pubmed_count": 12000},
                "nicotine": {"pubchem_cid": 89594, "pubchem_create_year": None, "pubmed_count": 0},
                "not a group": {"pubchem_cid": 1, "pubchem_create_year": 2000, "pubmed_count": 1}}
    plain, _ = dp.build_refs(rows, pool)
    refs, notes = dp.build_refs(rows, pool, exposure=exposure)
    assert (refs["naproxen"]["pubchem_create_year"], refs["naproxen"]["pubmed_count"]) == (2005, 12000)
    assert (refs["nicotine"]["pubchem_create_year"], refs["nicotine"]["pubmed_count"]) == (None, 0)  # a zero count stays a zero
    assert notes["exposure_matched"] == 2
    assert all(r["pubchem_create_year"] is None and r["pubmed_count"] is None for g, r in plain.items())
    assert list(refs["naproxen"])[-2:] == ["pubchem_create_year", "pubmed_count"]
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("pub")}
    assert {g: strip(r) for g, r in refs.items()} == {g: strip(r) for g, r in plain.items()}  # nothing else changes


def test_load_rows_drops_pd_rows_and_combination_groups(tmp_path):
    keep = rows_for("aspirin", ASPIRIN, ASPIRIN)
    pd = [r | {"task_type": "PD", "id": "aspirin#pd"} for r in rows_for("aspirin", ASPIRIN)]  # PD row of a kept drug
    pd += [r | {"task_type": "PD"} for r in rows_for("seq only", ASPIRIN) + rows_for("nivolumab | other", ASPIRIN)]
    combo = rows_for("caffeine | aspirin", ASPIRIN, ASPIRIN)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in keep[:1] + pd + combo[:1]), encoding="utf-8")
    b.write_text("".join(json.dumps(r) + "\n" for r in keep[1:] + combo[1:]) + "\n", encoding="utf-8")  # trailing blank line
    kept, dropped = dp.load_rows([a, b])
    assert [r["id"] for r in kept] == ["aspirin#0", "aspirin#1"]
    assert dropped == {"pd": {"rows": 3, "groups": ["aspirin", "nivolumab | other", "seq only"]},
                       "combination": {"rows": 2, "groups": ["caffeine | aspirin"]}}  # a PD combination counts as PD only
    b.write_text(json.dumps(keep[0]) + "\n", encoding="utf-8")  # an id that is already in a.jsonl
    with pytest.raises(AssertionError, match="duplicate"):
        dp.load_rows([a, b])


def test_split_rows_move_the_think_prefix_into_the_prompt_and_carry_cluster_and_inchikey(tmp_path):
    args, rows = make_inputs(tmp_path)
    dp.main(args)
    data = tmp_path / "data"
    orig = {r["id"]: r for r in rows}
    refs = {r["group"]: r for r in read_jsonl(data / "refs.jsonl")}
    seen = 0
    for split in dp.SPLITS:
        for row in read_jsonl(data / f"{split}.jsonl"):
            o, ref = orig[row["id"]], refs[row["group"]]
            seen += 1
            assert list(row) == ["id", "group", "task_type", "split", "cluster_id", "prompt",
                                 *(["completion"] if split in ("train", "val") else []),
                                 "ref_smiles", "ref_alt_smiles", "ref_conn", "ref_scaffold", "ref_inchikey", "n_heavy",
                                 "approval_year", "n_prompt_tokens", "n_completion_tokens"]  # the contract's field order
            assert row["split"] == split == ref["split"]
            assert row["prompt"] == o["prompt"] + THINK and row["prompt"].endswith("<|im_start|>assistant\n<think>\n")
            if split in ("train", "val"):
                assert row["completion"] == o["completion"][len(THINK):] and not row["completion"].startswith("<think>")
                assert row["prompt"] + row["completion"] == o["prompt"] + o["completion"]  # same text, new cut point
            assert (row["cluster_id"], row["ref_inchikey"], row["n_heavy"], row["ref_alt_smiles"]) == (
                ref["cluster_id"], ref["ref_inchikey"], ref["n_heavy"], ref["ref_alt_smiles"])
            mol = chem.standardize(row["ref_smiles"])
            assert row["cluster_id"] >= 0 and row["ref_inchikey"] == chem.inchikey(mol) and row["n_heavy"] == mol.GetNumHeavyAtoms()
            assert row["ref_conn"] == chem.conn(mol)
            assert (row["ref_smiles"], row["ref_conn"], row["ref_scaffold"], row["approval_year"]) == (
                ref["ref_smiles"], ref["ref_conn"], ref["ref_scaffold"], ref["approval_year"])
            assert (row["n_prompt_tokens"], row["n_completion_tokens"], row["task_type"], row["group"]) == (
                o["n_prompt_tokens"], o["n_completion_tokens"], o["task_type"], o["group"])
    assert seen == sum(r["n_rows"] for r in refs.values() if r["split"] in dp.SPLITS) == 2 * 26
    assert any(row["ref_alt_smiles"] for split in dp.SPLITS for row in read_jsonl(data / f"{split}.jsonl"))  # the non-empty case is covered
    g = next(g for g, r in refs.items() if r["split"] == "train")
    good = orig[f"{g}#0"]
    assert dp.make_row(good, refs[g])["completion"] == good["completion"][len(THINK):]
    with pytest.raises(AssertionError):  # prompt does not end with the assistant tag
        dp.make_row(good | {"prompt": good["prompt"] + " "}, refs[g])
    with pytest.raises(AssertionError):  # completion does not start with <think>
        dp.make_row(good | {"completion": good["completion"][len(THINK):]}, refs[g])
    with pytest.raises(AssertionError):  # think closed twice
        dp.make_row(good | {"completion": good["completion"] + "</think>"}, refs[g])


def test_every_group_is_in_exactly_one_split_and_the_report_says_why_the_rest_are_out(tmp_path):
    args, rows = make_inputs(tmp_path)
    dp.main(args)
    data = tmp_path / "data"
    refs = {r["group"]: r for r in read_jsonl(data / "refs.jsonl")}
    assert set(refs) == set(DRUGS) | {"mystery", "undated", "bare atom", "peptide", "acetoacetate"}  # PD-only and combination groups never reach refs
    assert list(refs["peptide"]) == [  # the contract's field order
        "group", "components", "ref_smiles", "ref_inchikey", "ref_conn", "ref_scaffold", "ref_source", "chembl_id", "approval_year",
        "approval_source", "sft_agrees", "ref_alt_smiles", "cluster_id", "split", "exclude_reason", "n_rows", "n_heavy", "convention",
        "pubchem_create_year", "pubmed_count"]
    assert {g: r["exclude_reason"] for g, r in refs.items() if r["exclude_reason"]} == {
        "mystery": "no_structure", "undated": "no_approval_year", "bare atom": "unextractable_reference"}
    where = {}
    for split in dp.SPLITS:
        for row in read_jsonl(data / f"{split}.jsonl"):
            where.setdefault(row["group"], set()).add(split)
    assert all(len(s) == 1 for s in where.values())
    assert {g: next(iter(s)) for g, s in where.items()} == {g: r["split"] for g, r in refs.items() if r["split"] in dp.SPLITS}
    by_split = Counter(r["split"] for r in refs.values())
    assert by_split == {"train": 13, "val": 3, "test_r": 5, "test_t": 5, "buffer": 5, "excluded": 3}
    assert all(r["cluster_id"] == -1 for r in refs.values() if r["split"] == "excluded")
    n_rows = Counter(r["group"] for r in rows if r["task_type"] != "PD" and " | " not in r["group"])
    assert all(r["n_rows"] == n_rows[g] for g, r in refs.items())
    assert all(r["pubchem_create_year"] is None and r["pubmed_count"] is None for r in refs.values())  # no exposure.json here

    report = json.loads((data / "split_report.json").read_text(encoding="utf-8"))
    assert report["dropped"]["pd"]["rows"] == 1 and report["dropped"]["combination"]["rows"] == 1
    assert report["dropped"]["excluded"] == {  # every reason with the names of its drugs
        "no_approval_year": {"rows": 1, "groups": ["undated"]}, "no_structure": {"rows": 1, "groups": ["mystery"]},
        "unextractable_reference": {"rows": 1, "groups": ["bare atom"]}}
    assert {k: v["drugs"] for k, v in report["splits"].items()} == dict(by_split)
    assert report["splits"]["test_t"]["drugs_by_year"]["2025"] == 4 and report["splits"]["test_t"]["drugs_by_year"]["2026"] == 1
    assert refs["peptide"]["n_heavy"] == 121 and refs["peptide"]["convention"] is None
    big = {s: sum((r["n_heavy"] or 0) > 100 for r in refs.values() if r["split"] == s) for s in by_split}
    assert {s: v["drugs_n_heavy_over_100"] for s, v in report["splits"].items()} == big and sum(big.values()) == 1
    assert {v["drugs_ligand_only"] for v in report["splits"].values()} == {0}
    assert report["manual_effect"] == {k: [] for k in ("identical", "changed_smiles_only", "changed_tautomer", "changed_connectivity",
                                                       "filled_gap", "unresolvable")}
    keto, enol = (chem.canonical(chem.standardize(s)) for s in (KETO, ENOL))
    tauto = refs["acetoacetate"]  # ChEMBL keto form, SFT answers in the enol form: they agree, and the enol form rides along
    assert (tauto["ref_source"], tauto["sft_agrees"], tauto["ref_smiles"], tauto["ref_alt_smiles"]) == ("chembl", True, keto, [enol])
    assert {g for g, r in refs.items() if r["ref_alt_smiles"]} == {"acetoacetate"}
    assert report["references"]["with_alternates"] == {"count": 1, "groups": ["acetoacetate"]} and report["rejected_structures"] == []
    assert report["seed"] == 0 and all(t["exact"] for t in report["targets"].values())
    sizes = Counter(Counter(r["cluster_id"] for r in refs.values() if r["cluster_id"] >= 0).values())
    assert report["clusters"]["size_distribution"] == {str(k): v for k, v in sorted(sizes.items())} == {"1": 12, "2": 6, "3": 1, "4": 1}
    assert report["scaffold_overlap"]["drugs_sharing_a_scaffold_with_train"] == {"val": 0, "test_r": 0, "test_t": 0}
    assert set(report["scaffold_overlap"]["shared_scaffolds"].values()) == {0}
    assert report["inputs"]["sha256"] == {  # the first 12 hex of every input file's SHA-256 (no hand files exist here)
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        for p in (tmp_path / "train_think.jsonl", tmp_path / "val_think.jsonl", tmp_path / "pool.json")}
    assert report["inputs"]["exposure"] == {"present": False}


def test_two_processes_with_different_hash_seeds_write_identical_files(tmp_path):
    args, _ = make_inputs(tmp_path)
    i = args.index("--data")
    out = []
    for n, hashseed in enumerate(("1", "2")):
        subprocess.run([sys.executable, "-m", "rlexp.data_prep", *args[:i], "--data", str(tmp_path / f"run{n}"), *args[i + 2:]],
                       check=True, capture_output=True, env=os.environ | {"PYTHONHASHSEED": hashseed})
        out.append(tmp_path / f"run{n}")
    names = sorted(p.name for p in out[0].iterdir())
    assert names == ["refs.jsonl", "split_report.json", "test_r.jsonl", "test_t.jsonl", "train.jsonl", "val.jsonl"]
    for name in names:
        assert (out[0] / name).read_bytes() == (out[1] / name).read_bytes(), name


def test_hand_files_are_read_but_never_overwritten_or_deleted(tmp_path):
    args, _ = make_inputs(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    manual, audit, exposure = data / "manual_refs.json", data / "approval_audit.json", data / "exposure.json"
    notes = data / "manual_refs_notes.md"
    manual.write_text(json.dumps({
        "mystery": {"smiles": FUROSEMIDE, "approval_year": 2019, "source": "t", "note": "n"},
        "metformin": {"smiles": DRUGS["metformin"], "approval_year": 2018, "source": "t", "note": "n", "convention": "ligand_only"},
        "tamoxifen": {"smiles": None, "approval_year": 2018, "source": "t", "note": "unresolvable"},
        "acetoacetate": {"smiles": ENOL, "approval_year": 2018, "source": "t", "note": "the SFT form"},
        "typo group": {"smiles": None, "approval_year": 2000}}), encoding="utf-8")
    audit.write_text(json.dumps({"undated": {"fda_year": None, "earliest_global_year": 2020, "earliest_regulator": "EMA",
                                             "evidence": [], "confidence": "low"}}), encoding="utf-8")
    exposure.write_text(json.dumps({"warfarin": {"pubchem_cid": 54678486, "pubchem_create_year": 2005, "pubmed_count": 9000},
                                    "nicotine": {"pubchem_cid": None, "pubchem_create_year": None, "pubmed_count": None},
                                    "not in the data": {"pubchem_cid": 1, "pubchem_create_year": 1999, "pubmed_count": 2}}),
                        encoding="utf-8")
    notes.write_text("keep me\n", encoding="utf-8")
    hand = (manual, audit, exposure, notes)
    snap = lambda: {p: (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in hand}
    before = snap()
    report = dp.main(args)
    dp.main(args)  # a second run changes nothing either
    assert snap() == before
    assert sorted(p.name for p in data.iterdir()) == sorted(
        [p.name for p in hand] + ["refs.jsonl", "split_report.json", "test_r.jsonl", "test_t.jsonl", "train.jsonl", "val.jsonl"])
    refs = {r["group"]: r for r in read_jsonl(data / "refs.jsonl")}
    assert (refs["mystery"]["ref_source"], refs["mystery"]["approval_year"], refs["mystery"]["approval_source"]) == ("manual", 2019, "manual")
    assert (refs["undated"]["approval_year"], refs["undated"]["approval_source"], refs["undated"]["exclude_reason"]) == (2020, "audit", None)
    assert (refs["warfarin"]["pubchem_create_year"], refs["warfarin"]["pubmed_count"]) == (2005, 9000)
    assert (refs["nicotine"]["pubchem_create_year"], refs["nicotine"]["pubmed_count"]) == (None, None)
    assert refs["caffeine"]["pubchem_create_year"] is None and refs["caffeine"]["pubmed_count"] is None  # no entry for it
    assert (refs["tamoxifen"]["exclude_reason"], refs["tamoxifen"]["split"], refs["tamoxifen"]["approval_year"]) == ("manual_unresolvable", "excluded", 2018)
    assert (refs["metformin"]["convention"], refs["metformin"]["ref_source"]) == ("ligand_only", "manual")
    assert report["dropped"]["excluded"]["manual_unresolvable"] == {"rows": 2, "groups": ["tamoxifen"]}
    assert report["manual_effect"] == {"identical": ["metformin"], "changed_smiles_only": [], "changed_tautomer": ["acetoacetate"],
                                       "changed_connectivity": [], "filled_gap": ["mystery"], "unresolvable": ["tamoxifen"]}
    assert (refs["acetoacetate"]["ref_source"], refs["acetoacetate"]["sft_agrees"]) == ("manual", True)
    assert refs["acetoacetate"]["ref_alt_smiles"] == [chem.canonical(chem.standardize(KETO))]  # the ChEMBL form
    assert report["references"]["with_alternates"] == {"count": 1, "groups": ["acetoacetate"]}
    assert {s: v["drugs_ligand_only"] for s, v in report["splits"].items() if v["drugs_ligand_only"]} == {refs["metformin"]["split"]: 1}
    assert report["unused_hand_entries"] == {"manual_refs": ["typo group"], "approval_audit": []}
    assert report["references"]["exposure_matched"] == 2
    assert {k: report["inputs"][k]["entries"] for k in ("manual_refs", "approval_audit", "exposure")} == {
        "manual_refs": 5, "approval_audit": 1, "exposure": 3}
    assert {k: v for k, v in report["inputs"]["sha256"].items() if k in {p.name for p in hand}} == {  # hashes of what was read
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in (manual, audit, exposure)}
    assert {o["group"] for o in report["year_overrides"]} == {"mystery", "undated"}
    with pytest.raises(AssertionError):  # the writer itself refuses these names, and stages nothing
        dp._write_all(data, {"exposure.json": "{}", "refs.jsonl": "{}\n"})
    assert snap() == before and (data / "refs.jsonl").read_text(encoding="utf-8") != "{}\n"


@pytest.mark.parametrize("name, text, error, match", [
    ("manual_refs.json", "{not json", ValueError, "manual_refs.json"),  # half-written by another worker
    ("manual_refs.json", json.dumps({"tamoxifen": {"approval_year": 2018}}), AssertionError, "manual_refs.json"),  # no smiles key at all
    ("manual_refs.json", json.dumps({"tamoxifen": {"smiles": "C1CC((", "approval_year": 2018}}), ValueError, "tamoxifen"),
    ("manual_refs.json", json.dumps({"tamoxifen": {"smiles": None, "approval_year": "2018"}}), AssertionError, "manual_refs.json"),
    ("manual_refs.json", json.dumps([]), AssertionError, "manual_refs.json"),
    ("approval_audit.json", json.dumps({"undated": {"earliest_global_year": 2020.5}}), AssertionError, "approval_audit.json"),
    ("exposure.json", json.dumps({"warfarin": {"pubmed_count": "9000"}}), AssertionError, "exposure.json"),
])
def test_a_malformed_hand_file_stops_the_run_and_is_left_alone(tmp_path, name, text, error, match):
    args, _ = make_inputs(tmp_path)
    dp.main(args)  # outputs from a good run
    data = tmp_path / "data"
    outputs = {p: p.read_bytes() for p in data.glob("*.jsonl")}
    (data / name).write_text(text, encoding="utf-8")
    with pytest.raises(error, match=match):
        dp.main(args)
    assert (data / name).read_text(encoding="utf-8") == text  # never repaired or deleted
    assert {p: p.read_bytes() for p in data.glob("*.jsonl")} == outputs  # and nothing half-written replaced the last good outputs


def test_a_pair_at_exactly_the_threshold_is_linked_and_one_just_below_is_not(monkeypatch):
    assert dp.SIM == 0.4
    rows, pool = synth()
    refs, _ = dp.build_refs(rows, pool)
    key = lambda a, b: frozenset((refs[a]["ref_smiles"], refs[b]["ref_smiles"]))
    table = {key("nicotine", "warfarin"): 0.4, key("fluoxetine", "haloperidol"): math.nextafter(0.4, 0)}
    monkeypatch.setattr(chem, "tanimoto_bits", lambda a, b: table.get(frozenset((chem.canonical(a), chem.canonical(b))), 0.0))
    assert len({refs[g]["ref_scaffold"] for g in ("nicotine", "warfarin")}) == 2  # only the similarity rule can join the pair
    dp.assign_splits(refs, 0, 5, 3)
    assert refs["nicotine"]["cluster_id"] == refs["warfarin"]["cluster_id"]  # >= 0.4, not > 0.4
    assert refs["fluoxetine"]["cluster_id"] != refs["haloperidol"]["cluster_id"]  # the largest double below 0.4 is no link


def test_the_leakage_graph_and_the_report_use_the_bit_tanimoto_never_the_count_one(tmp_path, monkeypatch):
    def forbidden(a, b):
        raise AssertionError("chem.tanimoto (count vectors) must not feed the leakage graph or the report: 0.4 was set on bits")

    monkeypatch.setattr(chem, "tanimoto", forbidden)
    refs_for()
    args, _ = make_inputs(tmp_path)
    dp.main(args)


FILES = ("refs.jsonl", "train.jsonl", "val.jsonl", "test_r.jsonl", "test_t.jsonl", "split_report.json")


class Injected(OSError):
    pass


def snapshot(directory):
    """Every entry of the directory, hidden ones included, with its bytes: a stray staging or backup file shows up."""
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def fail_at_step(mp, n):
    """Make step n (1-based) of the next _write_all raise: a staging write, a backup copy or a rename. Returns the step log."""
    log = []

    def wrap(fn, label):
        def inner(*a, **k):
            log.append(label)
            if len(log) == n:
                raise Injected(f"step {n} ({label})")
            return fn(*a, **k)

        return inner

    mp.setattr(os, "replace", wrap(os.replace, "rename"))
    mp.setattr(shutil, "copy2", wrap(shutil.copy2, "backup"))
    mp.setattr(Path, "write_text", wrap(Path.write_text, "stage"))
    return log


@pytest.mark.parametrize("existing", [FILES, FILES[::2], ()], ids=["all six exist", "three exist", "none exist"])
def test_write_is_all_or_nothing_whichever_step_fails(tmp_path, monkeypatch, existing):
    new = {n: f"new {n}\n" * 3 for n in FILES}
    out = tmp_path / "out"
    out.mkdir()

    def reset():
        for p in out.iterdir():
            p.unlink()
        for n in existing:
            (out / n).write_text(f"old {n}\n", encoding="utf-8")
        (out / "unrelated.txt").write_text("not ours\n", encoding="utf-8")
        return snapshot(out)

    before = reset()
    with monkeypatch.context() as mp:  # a run that succeeds, to count its steps
        log = fail_at_step(mp, 0)
        dp._write_all(out, new)
    assert snapshot(out) == {**before, **{n: t.encode() for n, t in new.items()}}  # new files in, unrelated one kept, nothing stray
    steps = len(log)
    assert steps == len(FILES) + len(existing) + len(FILES)  # stage, back up what exists, rename
    for n in range(1, steps + 1):
        before = reset()
        with monkeypatch.context() as mp:
            fail_at_step(mp, n)
            with pytest.raises(Injected):
                dp._write_all(out, new)
        assert snapshot(out) == before, f"a failure at step {n} of {steps} left the directory changed"


def test_a_failure_on_the_second_rename_of_a_real_run_leaves_the_previous_outputs_untouched(tmp_path, monkeypatch):
    args, _ = make_inputs(tmp_path)
    dp.main(args)
    data = tmp_path / "data"
    before = snapshot(data)
    renames, real = [], os.replace

    def second_rename_fails(*a, **k):
        renames.append(a)
        if len(renames) == 2:
            raise Injected("second rename")
        return real(*a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(os, "replace", second_rename_fails)
        with pytest.raises(Injected):
            dp.main(args + ["--seed", "3"])
    assert snapshot(data) == before  # not refs.jsonl new while train.jsonl is old
    dp.main(args + ["--seed", "3"])
    after = snapshot(data)
    assert after != before and set(after) == set(before)
