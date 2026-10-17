"""Tests for rlexp.reward (contracts.md "reward.py (v2)"): formula, gating, edge cases, invariances, the answer-format
fixture and the offline-check helpers."""
import importlib.util
import json
import random
import string
import time
import warnings
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from rlexp import chem, reward

ROOT = Path(__file__).resolve().parents[1]
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
CAFFEINE_D9 = "[2H]C([2H])([2H])n1cnc2c1c(=O)n(C([2H])([2H])[2H])c(=O)n2C([2H])([2H])[2H]"
L_ALA, D_ALA = "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"
TETRACYCLINE = "CC1(O)C2CC3C(N(C)C)C(O)=C(C(N)=O)C(=O)C3(O)C(O)=C2C(=O)c2c(O)cccc21"
TETRACYCLINE_RING_A_SWAPPED = "CC1(O)C2CC3C(N(C)C)C(=O)C(C(N)=O)=C(O)C3(O)C(O)=C2C(=O)c2c(O)cccc21"
PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"
OCTANOIC_ACID = "CCCCCCCC(=O)O"                      # acyclic: empty Murcko scaffold

# Public approved drugs: name, SMILES, molecular formula (the formula guards the hard-coded SMILES against typos).
DRUGS = [
    ("aspirin", ASPIRIN, "C9H8O4"),
    ("ibuprofen", IBUPROFEN, "C13H18O2"),
    ("caffeine", CAFFEINE, "C8H10N4O2"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1", "C8H9NO2"),
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", "C33H35FN2O5"),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "C29H31N7O"),
    ("metformin", "CN(C)C(=N)N=C(N)N", "C4H11N5"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1", "C22H30N6O4S"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "C19H16O4"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "C16H13ClN2O"),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1", "C17H18F3NO"),
    ("omeprazole", "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C", "C17H19N3O3S"),
    ("simvastatin", "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21", "C25H38O5"),
    ("lisinopril", "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", "C21H31N3O5"),
    ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)-c1ccccc1-c1nnn[nH]1", "C22H23ClN6O"),
    ("amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC", "C20H25ClN2O5"),
    ("metoprolol", "COCCc1ccc(OCC(O)CNC(C)C)cc1", "C15H25NO3"),
    ("sertraline", "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc12", "C17H17Cl2N"),
    ("ciprofloxacin", "OC(=O)C1=CN(C2CC2)c2cc(N3CCNCC3)c(F)cc2C1=O", "C17H18FN3O3"),
    ("penicillin G", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "C16H18N2O4S"),
    ("morphine", "CN1CC[C@]23[C@H]4Oc5c(O)ccc(C[C@@H]1[C@@H]2C=C[C@@H]4O)c35", "C17H19NO3"),
    ("tamoxifen", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "C26H29NO"),
    ("propranolol", PROPRANOLOL, "C16H21NO2"),
    ("celecoxib", "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F", "C17H14F3N3O2S"),
    ("clopidogrel", "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1", "C16H16ClNO2S"),
    ("nicotine", "CN1CCC[C@H]1c1cccnc1", "C10H14N2"),
    ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "C14H11Cl2NO2"),
    ("naproxen", "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1", "C14H14O3"),
    ("dexamethasone", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4([C@]3([C@H](C[C@@]2([C@]1(C(=O)CO)O)C)O)F)C", "C22H29FO5"),
    ("tetracycline", TETRACYCLINE, "C22H24N2O8"),
    ("erlotinib", "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC", "C22H23N3O4"),
    ("tadalafil", "CN1CC(=O)N2[C@H](Cc3c([nH]c4ccccc34)[C@H]2c2ccc3OCOc3c2)C1=O", "C22H19N3O4"),
    ("ondansetron", "Cc1nccn1CC1CCc2c(C1=O)c1ccccc1n2C", "C18H19N3O"),
    ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C", "C14H22N2O"),
    ("oseltamivir", "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1", "C16H28N2O4"),
    ("acyclovir", "Nc1nc2n(COCCO)cnc2c(=O)[nH]1", "C8H11N5O3"),
    ("testosterone", "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O", "C19H28O2"),
    ("estradiol", "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O", "C18H24O2"),
]
IBUPROFEN_ANALOGUES = [IBUPROFEN.replace("C(=O)O", tail) for tail in ("C(=O)N", "C(=O)OC")] + [
    "CC(C)Cc1ccc(cc1)CC(=O)O", "Cc1ccc(cc1)C(C)C(=O)O", "c1ccccc1CC(=O)O", "OC(=O)c1ccccc1O"]      # benzene scaffold
POOL = [s for _, s, _ in DRUGS] + IBUPROFEN_ANALOGUES + ["CCCCCCCC(=O)N", "c1ccccc1CCCCCCC(=O)O"]
REFS = [ASPIRIN, IBUPROFEN, CAFFEINE, DRUGS[4][1], OCTANOIC_ACID]


def closed(answer, think="reasoning"):
    return f"{think}</think>\n{answer}"


def run(answer, ref_smiles, **ref_extra):
    return reward.score(closed(answer), {"ref_smiles": ref_smiles, **ref_extra})


def norm(smiles):
    mol = chem.standardize(smiles)
    return None if mol is None else chem.canonical(mol)


@pytest.fixture(scope="module")
def pool_results():
    """score() of every pool answer against every reference in REFS, plus a few invalid texts."""
    texts = [closed(s) for s in POOL] + ["no think tag " + ASPIRIN, "", closed("(" * 50), closed("C1CC(C(=O)N")]
    return [(ref, reward.score(t, {"ref_smiles": ref})) for ref in REFS for t in texts]


# ---------------------------------------------------------------- formula and weights
def test_weights_match_the_contract_and_leave_room_below_one():
    assert (reward.W_VALID, reward.W_TANI, reward.W_SCAF) == (0.05, 0.6, 0.3)
    assert reward.W_VALID + reward.W_TANI + reward.W_SCAF < 1.0


def test_exact_answer_scores_one_with_every_flag_set():
    assert run(ASPIRIN, ASPIRIN) == {"closed": True, "answer": ASPIRIN, "valid": True, "tanimoto": 1.0,
                                     "scaffold_match": True, "conn_match": True, "stereo_match": True, "reward": 1.0}


def test_near_miss_with_the_same_scaffold_follows_the_formula():
    res = run("CC(C)Cc1ccc(cc1)C(C)C(=O)N", IBUPROFEN)               # ibuprofen amide
    assert res["valid"] and res["scaffold_match"] and not res["conn_match"] and not res["stereo_match"]
    assert 0.5 < res["tanimoto"] < 1.0
    assert res["reward"] == pytest.approx(0.05 + 0.6 * res["tanimoto"] + 0.3)


def test_wrong_drug_without_a_shared_scaffold_gets_validity_plus_similarity():
    res = run(CAFFEINE, IBUPROFEN)
    assert res["valid"] and not res["scaffold_match"] and not res["conn_match"]
    assert res["reward"] == pytest.approx(0.05 + 0.6 * res["tanimoto"])
    assert 0.05 <= res["reward"] < 0.3


def test_valid_wrong_answer_ranks_above_invalid_and_below_exact():
    invalid, wrong, exact = run("C1CC(C(=O)N", IBUPROFEN), run(CAFFEINE, IBUPROFEN), run(IBUPROFEN, IBUPROFEN)
    assert invalid["reward"] == 0.0 < reward.W_VALID <= wrong["reward"] < exact["reward"] == 1.0


# ---------------------------------------------------------------- stereo, tautomers, isotopes
@pytest.mark.parametrize("answer, ref", [(D_ALA, L_ALA), ("C(=C\\c1ccccc1)\\c1ccccc1", "C(=C/c1ccccc1)\\c1ccccc1")],
                         ids=["enantiomer", "E-Z"])
def test_stereo_only_difference_is_exact_but_not_stereo_exact(answer, ref):
    res = run(answer, ref)
    assert res["conn_match"] and not res["stereo_match"] and res["reward"] == 1.0 and res["tanimoto"] == 1.0


@pytest.mark.parametrize("answer, ref", [("CC(O)=CC(C)=O", "CC(=O)CC(C)=O"), (TETRACYCLINE_RING_A_SWAPPED, TETRACYCLINE),
                                         (CAFFEINE_D9, CAFFEINE)], ids=["keto-enol", "tetracycline", "isotopologue"])
def test_tautomers_and_isotopologues_are_exact_matches_without_stereo_match(answer, ref):
    res = run(answer, ref)
    assert res["conn_match"] and not res["stereo_match"] and res["reward"] == 1.0


# ---------------------------------------------------------------- the gate
@pytest.mark.parametrize("text", [ASPIRIN, f"thinking {ASPIRIN}", f"<think>\n{ASPIRIN}", f"</thinking>\n{ASPIRIN}"])
def test_text_without_a_closing_think_tag_scores_zero(text):
    res = reward.score(text, {"ref_smiles": ASPIRIN})
    assert (res["closed"], res["answer"], res["valid"], res["reward"]) == (False, None, False, 0.0)


@pytest.mark.parametrize("text", [None, "", " \n\t", " "])
def test_empty_none_and_blank_text_score_zero(text):
    res = reward.score(text, {"ref_smiles": ASPIRIN})
    assert (res["closed"], res["answer"], res["valid"], res["reward"]) == (False, None, False, 0.0)


@pytest.mark.parametrize("tail", ["", "\n", "I could not determine the structure.", "SMILES: unknown", "C1CC(C(=O)N"])
def test_closed_think_without_a_parseable_answer_scores_zero(tail):
    res = run(tail, ASPIRIN)
    assert res["closed"] and res["answer"] is None and not res["valid"] and res["reward"] == 0.0


def test_the_last_think_tag_counts():
    ref = {"ref_smiles": ASPIRIN}
    assert reward.score(f"a</think>\n{IBUPROFEN}</think>\nb</think>\n{ASPIRIN}", ref)["reward"] == 1.0
    res = reward.score(f"{ASPIRIN}</think>\nthinking</think>\nno structure", ref)      # answer before the last tag
    assert res["closed"] and res["answer"] is None and res["reward"] == 0.0


@pytest.mark.parametrize("tail", [f"{ASPIRIN}<|im_end|>", f"{ASPIRIN}<|im_end|>\n", f"SMILES: {ASPIRIN}<|im_end|>"])
def test_literal_im_end_glued_to_the_answer_is_removed(tail):
    assert run(tail, ASPIRIN)["reward"] == 1.0


@pytest.mark.parametrize("text", [
    f"</think>  {ASPIRIN} ",                                      # non-breaking spaces
    f"</think>　{ASPIRIN}　",                                            # ideographic spaces
    f"思考结束</think>\n答案： {ASPIRIN}",           # Chinese label before the answer
    f"</think>\n\U0001f9ea {ASPIRIN} \U0001f9ea",                                # emoji around it
    f"</think>\r\n\t{ASPIRIN}\r\n",
])
def test_unicode_and_non_breaking_spaces_around_the_answer(text):
    assert reward.score(text, {"ref_smiles": ASPIRIN})["reward"] == 1.0


# ---------------------------------------------------------------- salts, charges, mixtures
@pytest.mark.parametrize("answer", [f"{ASPIRIN}.Cl", "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", f"[Na+].[Cl-].{ASPIRIN}",
                                    f"{ASPIRIN}.O"], ids=["hydrochloride", "sodium-salt", "counter-ions-first", "hydrate"])
def test_counter_ions_in_the_answer_are_dropped(answer):
    assert run(answer, ASPIRIN)["reward"] == 1.0


@pytest.mark.parametrize("answer", [PROPRANOLOL, f"{PROPRANOLOL}.Cl", f"{PROPRANOLOL}.Br"])
def test_counter_ion_in_the_reference_is_dropped(answer):
    assert run(answer, f"{PROPRANOLOL}.Cl")["reward"] == 1.0


@pytest.mark.parametrize("answer, ref", [
    ("C[NH+](C)CCOC(c1ccccc1)c1ccccc1", "CN(C)CCOC(c1ccccc1)c1ccccc1"),         # protonated amine
    ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),                                           # zwitterion
    ("CC(=O)Oc1ccccc1C(=O)[O-]", ASPIRIN),                                      # carboxylate
    ("C[N+](C)(C)CCO", "C[N+](C)(C)CCO"),                                       # permanent charge on both sides
], ids=["protonated", "zwitterion", "carboxylate", "quaternary"])
def test_charged_species_are_neutralised_where_possible(answer, ref):
    assert run(answer, ref)["reward"] == 1.0


def test_quaternary_ammonium_is_not_the_neutral_amine():
    assert run("C[N+](C)(C)CCO", "CN(C)CCO")["reward"] < 1.0


def test_mixture_scores_its_largest_fragment():
    assert run(f"CCO.{ASPIRIN}", ASPIRIN)["reward"] == run(f"{ASPIRIN}.CCO", ASPIRIN)["reward"] == 1.0
    assert run(f"{ASPIRIN}.{IBUPROFEN}", IBUPROFEN)["reward"] == 1.0          # ibuprofen has more heavy atoms
    assert run(f"{ASPIRIN}.{IBUPROFEN}", ASPIRIN)["reward"] < 1.0


# ---------------------------------------------------------------- hostile and oversized text
GARBAGE = {
    "letters": "x" * 5000,
    "printable": "".join(random.Random(0).choice(string.printable.strip()) for _ in range(5000)),
    "smiles alphabet": "".join(random.Random(1).choice("CNOcnos()[]=#@+-123456789%") for _ in range(5000)),
    "open branches": "C(" * 2500,
    "open parentheses": "(" * 5000,
    "open brackets": "[" * 5000,
    "ring labels": "C1" * 2500,
    "digits": "1" * 5000,
    "equals": "=" * 5000,
}


@pytest.mark.parametrize("token", GARBAGE.values(), ids=GARBAGE.keys())
def test_5000_character_garbage_token_scores_zero_quickly(token):
    start = time.perf_counter()
    res = run(token, ASPIRIN)
    assert time.perf_counter() - start < 5
    assert res["closed"] and not res["valid"] and res["reward"] == 0.0


def test_garbage_after_a_valid_answer_does_not_hide_it():
    assert run(f"{ASPIRIN}\n{'(' * 5000}", ASPIRIN)["reward"] == 1.0


@pytest.mark.parametrize("token", ["c1ccccc1" * 625, "C" * 399 + "(=O)O", "C1CC1" * 300, "O=C." * 400 + "O=C"],
                         ids=["5000-chars", "400-atoms", "900-atoms", "400-fragments"])
def test_valid_but_oversized_answer_is_invalid_and_cheap(token):
    start = time.perf_counter()
    res = run(token, ASPIRIN)
    assert time.perf_counter() - start < 5          # chem.scaffold alone would need minutes on 3,750 atoms
    assert res["answer"] in (token, None) and not res["valid"] and res["reward"] == 0.0   # over 1,500 characters is never extracted


def test_a_molecule_as_large_as_the_largest_reference_is_still_scored():
    peptide = "N" + "CC(=O)N" * 70 + "CC(=O)O"                                   # polyglycine, 285 heavy atoms
    res = run(peptide, peptide)
    assert chem.standardize(peptide).GetNumHeavyAtoms() > 280 and res["reward"] == 1.0


# ---------------------------------------------------------------- scaffold edge cases
@pytest.mark.parametrize("extra", [{}, {"ref_scaffold": ""}], ids=["derived", "given"])
@pytest.mark.parametrize("answer", ["CCCCCCCC(=O)N", "CCCCCCCCCC(=O)O"])
def test_acyclic_reference_never_earns_the_scaffold_bonus(answer, extra):
    assert chem.scaffold(chem.standardize(answer)) == "" == chem.scaffold(chem.standardize(OCTANOIC_ACID))
    res = run(answer, OCTANOIC_ACID, **extra)
    assert res["valid"] and res["scaffold_match"] is False and not res["conn_match"]
    assert res["reward"] == pytest.approx(0.05 + 0.6 * res["tanimoto"])


def test_cyclic_answer_for_an_acyclic_reference_has_no_scaffold_match():
    assert run("c1ccccc1CCCCCCC(=O)O", OCTANOIC_ACID)["scaffold_match"] is False


# ---------------------------------------------------------------- properties over a pool of answers
def test_reward_follows_the_contract_formula(pool_results):
    for _, res in pool_results:
        if not res["valid"]:
            assert res["reward"] == 0.0 and res["tanimoto"] == 0.0 and not res["conn_match"]
        elif res["conn_match"]:
            assert res["reward"] == 1.0
        else:
            assert res["reward"] == pytest.approx(0.05 + 0.6 * res["tanimoto"] + 0.3 * res["scaffold_match"])


def test_reward_is_in_the_unit_interval(pool_results):
    assert all(0.0 <= res["reward"] <= 1.0 and 0.0 <= res["tanimoto"] <= 1.0 for _, res in pool_results)


def test_reward_is_one_if_and_only_if_conn_matches(pool_results):
    assert any(res["conn_match"] for _, res in pool_results)
    assert all((res["reward"] == 1.0) == res["conn_match"] for _, res in pool_results)
    assert all(res["reward"] < 1.0 for _, res in pool_results if not res["conn_match"])


def test_reward_is_non_decreasing_in_tanimoto_at_a_fixed_scaffold_flag(pool_results):
    sizes = {False: 0, True: 0}
    for ref in REFS:
        for flag in (False, True):
            rows = sorted((r["tanimoto"], r["reward"]) for p, r in pool_results
                          if p == ref and r["valid"] and not r["conn_match"] and r["scaffold_match"] == flag)
            sizes[flag] = max(sizes[flag], len(rows))
            assert all(a[1] <= b[1] for a, b in zip(rows, rows[1:])), (ref, flag)
    assert min(sizes.values()) >= 5, sizes                  # both flags are exercised by some reference


def test_scaffold_flag_adds_exactly_the_bonus_at_equal_tanimoto():
    plain, bonus = run(CAFFEINE, IBUPROFEN), run("CC(C)Cc1ccc(cc1)C(C)C(=O)N", IBUPROFEN)
    assert bonus["scaffold_match"] and not plain["scaffold_match"]
    assert (bonus["reward"] - 0.6 * bonus["tanimoto"]) - (plain["reward"] - 0.6 * plain["tanimoto"]) == pytest.approx(0.3)


# ---------------------------------------------------------------- known drugs
@pytest.mark.parametrize("name, smiles, formula", DRUGS, ids=[d[0] for d in DRUGS])
def test_randomised_smiles_of_a_drug_score_exactly_one(name, smiles, formula):
    mol = Chem.MolFromSmiles(smiles)
    assert CalcMolFormula(mol) == formula
    std = chem.standardize(smiles)
    full_ref = {"ref_smiles": smiles, "ref_conn": chem.conn(std), "ref_scaffold": chem.scaffold(std),
                "ref_inchikey": chem.inchikey(std)}
    for ref in ({"ref_smiles": smiles}, full_ref):
        for _ in range(4):
            variant = Chem.MolToSmiles(mol, doRandom=True)
            res = reward.score(f"...</think>\n{variant}", ref)
            assert res["reward"] == 1.0 and res["conn_match"], variant


def test_known_drugs_are_distinct_for_the_scorer():
    keys = {chem.conn(chem.standardize(s)) for _, s, _ in DRUGS}
    assert len(keys) == len(DRUGS) >= 20


# ---------------------------------------------------------------- reference handling
def test_reference_keys_are_derived_when_missing():
    mol = chem.standardize(IBUPROFEN)
    full = {"ref_smiles": IBUPROFEN, "ref_conn": chem.conn(mol), "ref_scaffold": chem.scaffold(mol),
            "ref_inchikey": chem.inchikey(mol)}
    for answer in (IBUPROFEN, "CC(C)Cc1ccc(cc1)C(C)C(=O)N", CAFFEINE, "C1CC(C(=O)N"):
        assert reward.score(closed(answer), full) == reward.score(closed(answer), {"ref_smiles": IBUPROFEN})
    assert reward.score(closed(IBUPROFEN), full | {"ref_inchikey": None, "ref_scaffold": None})["reward"] == 1.0


def test_extra_keys_in_the_reference_row_are_ignored():
    row = {"id": "x", "group": "g", "prompt": "p", "ref_smiles": ASPIRIN, "extra": [1, 2]}
    assert reward.score(closed(ASPIRIN), row)["reward"] == 1.0


def test_stored_reference_keys_are_ignored_and_rederived_from_ref_smiles():
    caffeine_conn = chem.conn(chem.standardize(CAFFEINE))
    assert run(IBUPROFEN, IBUPROFEN, ref_conn=caffeine_conn)["conn_match"]              # a wrong stored key changes nothing
    assert not run(CAFFEINE, IBUPROFEN, ref_conn=caffeine_conn)["conn_match"]
    assert run("CC(C)Cc1ccc(cc1)C(C)C(=O)N", IBUPROFEN, ref_scaffold="")["scaffold_match"] is True   # stored "" is ignored


def test_structures_inchi_cannot_encode_never_match_by_connectivity():
    ref = "*CC(*)c1ccccc1"                                              # dummy atoms: chem.conn is None on both sides
    assert chem.conn(chem.standardize(ref)) is None
    res = run("*CCC(*)c1ccccc1", ref)
    assert res["valid"] and not res["conn_match"] and not res["stereo_match"] and res["reward"] < 1.0


@pytest.mark.parametrize("ref", [{}, {"ref_smiles": None}, {"ref_smiles": "nonsense"}, {"ref_smiles": "C" * 450}])
def test_score_raises_on_a_bad_reference_even_for_empty_text(ref):
    with pytest.raises((KeyError, ValueError)):
        reward.score("", ref)


def test_repeated_scoring_is_stable():
    first = [run(s, IBUPROFEN) for s in POOL]
    assert [run(s, IBUPROFEN) for s in POOL] == first
    assert run(IBUPROFEN, ASPIRIN)["reward"] < 1.0 and run(IBUPROFEN, IBUPROFEN)["reward"] == 1.0


# ---------------------------------------------------------------- interface
def test_score_returns_the_contract_keys_with_plain_types():
    for res in (run(ASPIRIN, ASPIRIN), run(CAFFEINE, ASPIRIN), reward.score(None, {"ref_smiles": ASPIRIN})):
        assert set(res) == {"closed", "answer", "valid", "tanimoto", "scaffold_match", "conn_match", "stereo_match",
                            "reward"}
        assert all(type(res[k]) is bool for k in ("closed", "valid", "scaffold_match", "conn_match", "stereo_match"))
        assert type(res["tanimoto"]) is float and type(res["reward"]) is float
        assert json.loads(json.dumps(res)) == res


def test_compute_score_takes_a_dict_or_a_json_string_and_ignores_extra_info():
    ref = {"ref_smiles": IBUPROFEN, "ref_conn": chem.conn(chem.standardize(IBUPROFEN))}
    text = closed("CC(C)Cc1ccc(cc1)C(C)C(=O)N")
    expected = reward.score(text, ref)["reward"]
    assert 0.0 < expected < 1.0
    got = {reward.compute_score("any", text, ref), reward.compute_score("any", text, json.dumps(ref)),
           reward.compute_score(data_source="other", solution_str=text, ground_truth=ref, extra_info={"k": 1})}
    assert got == {expected} and all(type(x) is float for x in got)


@pytest.mark.parametrize("solution, truth", [
    (closed(ASPIRIN), None), (closed(ASPIRIN), ""), (closed(ASPIRIN), "not json"), (closed(ASPIRIN), "[]"),
    (closed(ASPIRIN), {}), (closed(ASPIRIN), {"ref_smiles": "nonsense"}), (closed(ASPIRIN), {"ref_smiles": None}),
    (closed(ASPIRIN), 12345), (12345, {"ref_smiles": ASPIRIN}), (b"</think>", {"ref_smiles": ASPIRIN}),
    (None, None), (closed(ASPIRIN + "\ud800"), {"ref_smiles": ASPIRIN}),
])
def test_compute_score_never_raises(solution, truth):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert reward.compute_score("d", solution, truth) in (0.0, 1.0)


def test_compute_score_returns_zero_for_a_bad_reference_but_does_not_stay_silent():
    with pytest.warns(RuntimeWarning, match="returned 0.0"):
        assert reward.compute_score("d", closed(ASPIRIN), {"ref_smiles": "nonsense"}) == 0.0


def test_lone_surrogate_in_the_answer_scores_zero_without_raising():
    res = run(ASPIRIN + "\ud800", ASPIRIN)                     # chem.extract_answer raises UnicodeEncodeError on this
    assert res["closed"] and not res["valid"] and res["reward"] == 0.0


# ---------------------------------------------------------------- answer-format fixture (tests/fixtures/answer_formats.json)
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "answer_formats.json").read_text())


@pytest.mark.parametrize("case", FIXTURE, ids=[c["name"] for c in FIXTURE])
def test_answer_format_fixture(case):
    text, expected = case["text"], case["expected"]
    answer = chem.extract_answer(text)                                    # never raises, limitation or not
    res = reward.score(text, {"ref_smiles": ASPIRIN})
    assert res["answer"] == answer
    if case["limitation"]:
        return
    assert norm(answer) == norm(expected), case["why"]
    if expected is None:
        assert res["reward"] == 0.0
    else:
        assert reward.score(text, {"ref_smiles": expected})["conn_match"], case["why"]


def test_fixture_is_the_documented_one():
    assert len(FIXTURE) == 77
    assert sum(c["limitation"] for c in FIXTURE) == 24 and sum(c["expected"] is None for c in FIXTURE) == 11


# ---------------------------------------------------------------- scripts/offline_reward_checks.py
def _load_checks():
    spec = importlib.util.spec_from_file_location("offline_reward_checks", ROOT / "scripts" / "offline_reward_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checks = _load_checks()


def test_group_spread_counts_groups_with_enough_std():
    groups = {"a": [0.1, 0.5], "b": [0.3, 0.3, 0.3], "c": [0.0, 0.0, 1.0], "d": [0.2, 0.21]}
    assert checks.group_spread(groups) == 0.5
    assert checks.group_spread(groups, min_std=0.0) == 1.0
    assert checks.group_spread(groups, min_std=0.6) == 0.0


def test_group_spread_threshold_is_inclusive():
    assert checks.group_spread({"a": [0.0, 0.1]}, min_std=0.05) == 1.0          # population std is exactly 0.05
    assert checks.group_spread({"a": [0.0, 0.1]}, min_std=0.0501) == 0.0


def test_group_spread_with_valid_flags_ignores_variance_created_by_invalid_answers():
    rewards = {"p1": [0.0, 0.0, 0.3, 0.3], "p2": [0.0, 0.2, 0.4, 0.0], "p3": [0.1, 0.1, 0.1, 0.1]}
    flags = {"p1": [False, False, True, True], "p2": [False, True, True, False], "p3": [True] * 4}
    assert checks.group_spread(rewards) == pytest.approx(2 / 3)                 # p1 and p2 look spread out
    assert checks.group_spread(rewards, valid_only_flags=flags) == pytest.approx(1 / 3)     # only p2 has valid variance


def test_group_spread_needs_two_valid_samples_and_handles_empty_input():
    assert checks.group_spread({"a": [0.0, 0.9, 0.9]}, valid_only_flags={"a": [False, False, True]}) == 0.0
    assert checks.group_spread({"a": [0.0, 0.9]}, valid_only_flags={"a": [False, False]}) == 0.0
    assert checks.group_spread({}) == 0.0
    with pytest.raises(ValueError):
        checks.group_spread({"a": [0.0, 1.0]}, valid_only_flags={"a": [True]})


def test_group_spread_on_real_scores_separates_format_noise_from_chemistry():
    texts = ["no answer", closed("C1CC(C(=O)N"), closed(CAFFEINE), closed(CAFFEINE)]     # two invalid, two identical valid
    scored = [reward.score(t, {"ref_smiles": IBUPROFEN}) for t in texts]
    rewards, flags = {"p": [s["reward"] for s in scored]}, {"p": [s["valid"] for s in scored]}
    assert checks.group_spread(rewards) == 1.0 and checks.group_spread(rewards, valid_only_flags=flags) == 0.0


def _rows(split, names):
    smiles = {n: s for n, s, _ in DRUGS}
    out = []
    for name in names:
        mol = chem.standardize(smiles[name])
        row = {"id": f"{split}_{name}", "group": name, "split": split, "prompt": "p", "ref_smiles": smiles[name],
               "ref_conn": chem.conn(mol), "ref_scaffold": chem.scaffold(mol), "ref_inchikey": chem.inchikey(mol)}
        if split in ("train", "val"):
            row["completion"] = f"thinking</think>\n{smiles[name]}<|im_end|>"
        out.append(row)
    return out


def test_offline_script_runs_end_to_end_and_flags_stale_reference_fields(tmp_path, capsys):
    split_names = {"train": ["atorvastatin", "imatinib", "caffeine", "sildenafil", "morphine", "tamoxifen", "simvastatin"],
                   "val": ["losartan", "omeprazole"], "test_r": ["sertraline", "tadalafil"],
                   "test_t": ["nicotine", "diazepam"]}
    data = {split: _rows(split, names) for split, names in split_names.items()}
    for split, rows in data.items():
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert checks.main(["--data", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for name in ("REFS", "T1", "T2", "T3"):
        assert f"PASS {name}:" in out
    assert "INFO BAR:" in out and "prompt" not in out and "thinking" not in out
    stale = [dict(r, ref_conn="0123456789ABCD") for r in data["train"][:1]] + data["train"][1:]
    ok, message = checks.check_refs(stale)
    assert not ok and message.startswith("1 of 7 drugs")
