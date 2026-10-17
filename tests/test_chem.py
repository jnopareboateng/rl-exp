"""Unit tests for rlexp.chem: standardize, canonical, conn, skeleton, inchikey, scaffold, tanimoto, tanimoto_bits,
extract_answer."""
import re
import time

import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from rlexp import chem

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
L_ALA, D_ALA = "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"
KETO, ENOL = "CC(=O)CC(C)=O", "CC(O)=CC(C)=O"                                     # acetylacetone tautomers
TETRACYCLINE = "CC1(O)C2CC3C(N(C)C)C(O)=C(C(N)=O)C(=O)C3(O)C(O)=C2C(=O)c2c(O)cccc21"
TETRACYCLINE_RING_A_SWAPPED = "CC1(O)C2CC3C(N(C)C)C(=O)C(C(N)=O)=C(O)C3(O)C(O)=C2C(=O)c2c(O)cccc21"
THC_DELTA9 = "CCCCCc1cc(O)c2[C@@H]3C=C(C)CC[C@H]3C(C)(C)Oc2c1"
THC_DELTA8 = "CCCCCc1cc(O)c2[C@@H]3CC(C)=CC[C@H]3C(C)(C)Oc2c1"                   # double bond moved by one position
NORETHINDRONE = "C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@@H]4[C@H]3CC[C@@]21C"
NORETHYNODREL = "C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=C(CCC(=O)C4)[C@H]3CC[C@@]21C"


def std(smiles):
    return chem.standardize(smiles)


def can(smiles):
    return chem.canonical(chem.standardize(smiles))


# ---------------------------------------------------------------- standardize
@pytest.mark.parametrize("bad", ["", " ", "not a smiles", "C(C)(C)(C)(C)C", "c1ccccc", None, 42, b"CCO"])
def test_standardize_returns_none_for_unparseable_or_non_string(bad):
    assert chem.standardize(bad) is None


def test_standardize_keeps_the_largest_fragment_whatever_the_order():
    assert can("CCO." + ASPIRIN) == can(ASPIRIN) == can(ASPIRIN + ".CCO")


@pytest.mark.parametrize("charged, neutral", [
    ("CC(=O)[O-].[Na+]", "CC(=O)O"),                                              # sodium salt
    ("C[NH+](C)CCO", "CN(C)CCO"),                                                 # protonated amine
    ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),                                             # zwitterion
    ("CC(C)NCC(O)COc1cccc2ccccc12.Cl", "CC(C)NCC(O)COc1cccc2ccccc12"),            # hydrochloride
])
def test_standardize_neutralises_and_drops_counter_ions(charged, neutral):
    assert can(charged) == can(neutral)


def test_standardize_keeps_a_permanent_charge():
    assert can("C[N+](C)(C)CCO") == "C[N+](C)(C)CCO"


def test_standardize_is_idempotent():
    once = chem.canonical(std("CC(=O)[O-].[Na+].O"))
    assert can(once) == once


@pytest.mark.parametrize("smiles", ["*CC(*)c1ccccc1", "[*]CCCC", "CC(=O)O*", "c1ccc(*)cc1", "CC(=O)Oc1ccccc1C(=O)O.*"],
                         ids=["two", "bracketed", "terminal", "ring", "fragment"])
def test_standardize_rejects_dummy_atoms_even_in_a_discarded_fragment(smiles):
    assert chem.standardize(smiles) is None


def test_cost_guard_constants_match_the_contract():
    assert (chem.MAX_ATOMS, chem.MAX_LEN) == (400, 1500)


def test_standardize_accepts_exactly_max_atoms_and_rejects_one_more():
    limit = chem.MAX_ATOMS
    assert std("C" * (limit - 2) + "(=O)O").GetNumAtoms() == limit
    assert std("C" * (limit - 1) + "(=O)O") is None


def test_standardize_counts_atoms_of_all_fragments_not_only_the_largest():
    half = chem.MAX_ATOMS // 2
    assert std("C" * half + "." + "C" * half) is not None
    assert std("C" * half + "." + "C" * (half + 1)) is None


def test_standardize_rejects_a_many_fragment_answer_before_the_slow_steps():
    token = "O=C." * 374 + "O=C"                     # 750 atoms in 1,499 characters; the largest fragment has 3
    start = time.perf_counter()
    assert std(token) is None
    assert time.perf_counter() - start < 1


# ---------------------------------------------------------------- canonical
def test_canonical_is_independent_of_the_spelling():
    assert can("OC(=O)c1ccccc1OC(C)=O") == can(ASPIRIN) == chem.canonical(Chem.MolFromSmiles(ASPIRIN))
    assert can("C[C@H](N)C(=O)O") == can(L_ALA)


def test_canonical_is_isomeric():
    assert can(L_ALA) != can(D_ALA)


# ---------------------------------------------------------------- conn, skeleton and inchikey
@pytest.mark.parametrize("a, b", [
    (L_ALA, D_ALA),                                                                # enantiomers
    ("C(=C/c1ccccc1)\\c1ccccc1", "C(=C\\c1ccccc1)\\c1ccccc1"),                     # E and Z
    (CAFFEINE, "[2H]C([2H])([2H])n1cnc2c1c(=O)n(C([2H])([2H])[2H])c(=O)n2C([2H])([2H])[2H]"),   # isotopologue
], ids=["enantiomers", "E-Z", "isotopologue"])
def test_conn_ignores_stereo_and_isotopes_but_inchikey_does_not(a, b):
    assert chem.conn(std(a)) == chem.conn(std(b))
    assert chem.inchikey(std(a)) != chem.inchikey(std(b))


@pytest.mark.parametrize("a, b", [
    (KETO, ENOL),                                                                  # acetylacetone
    ("NC(=O)C1=C(O)CCCC1=O", "NC(=O)C1C(=O)CCCC1=O"),                              # tetracycline A-ring fragment
    (TETRACYCLINE, TETRACYCLINE_RING_A_SWAPPED),                                   # two tautomers of tetracycline
], ids=["keto-enol", "tetracycline-fragment", "tetracycline"])
def test_a_tautomer_written_differently_has_another_conn_but_the_same_skeleton(a, b):
    assert chem.conn(std(a)) != chem.conn(std(b))
    assert chem.skeleton(std(a)) == chem.skeleton(std(b))
    assert chem.inchikey(std(a)) != chem.inchikey(std(b))


def test_conn_equates_tautomers_that_standard_inchi_normalises():
    pyridone, hydroxypyridine = std("O=c1cccc[nH]1"), std("Oc1ccccn1")
    assert chem.conn(pyridone) == chem.conn(hydroxypyridine)
    assert chem.inchikey(pyridone) == chem.inchikey(hydroxypyridine)


@pytest.mark.parametrize("a, b", [
    ("C=CCC", "CC=CC"),                                                            # 1-butene and 2-butene
    (THC_DELTA9, THC_DELTA8),
    (NORETHINDRONE, NORETHYNODREL),
], ids=["butenes", "delta-9-delta-8-THC", "norethindrone-norethynodrel"])
def test_conn_separates_isomers_that_differ_only_in_where_a_double_bond_sits(a, b):
    ma, mb = std(a), std(b)
    assert CalcMolFormula(ma) == CalcMolFormula(mb)
    assert chem.skeleton(ma) == chem.skeleton(mb)                                  # why a skeleton key may not match answers
    assert chem.conn(ma) != chem.conn(mb)


@pytest.mark.parametrize("a, b", [
    ("CCO", "COC"),                                                                # same formula, other skeleton
    ("CCCC", "CC(C)C"),
    ("Cc1ccccc1O", "Cc1ccc(O)cc1"),                                                # ortho and para
    (ASPIRIN, IBUPROFEN),
    ("C[N+](C)(C)CCO", "CN(C)CCO"),                                                # other formula
])
def test_conn_and_skeleton_separate_other_skeletons_and_formulas(a, b):
    assert chem.conn(std(a)) != chem.conn(std(b))
    assert chem.skeleton(std(a)) != chem.skeleton(std(b))


def test_conn_is_the_same_for_every_spelling_and_counter_ion():
    spellings = {chem.conn(std(Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN), doRandom=True))) for _ in range(10)}
    assert spellings == {chem.conn(std(ASPIRIN)), chem.conn(std("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"))}


def test_conn_is_the_first_block_of_the_inchikey():
    mol = std(ASPIRIN)
    assert re.fullmatch(r"[A-Z]{14}", chem.conn(mol))
    assert chem.inchikey(mol).split("-")[0] == chem.conn(mol)


def test_skeleton_format_is_14_uppercase_hex_characters():
    assert re.fullmatch(r"[0-9A-F]{14}", chem.skeleton(std(ASPIRIN)))


def test_inchikey_format_and_sensitivity():
    key = chem.inchikey(std(ASPIRIN))
    assert re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key)
    assert key == chem.inchikey(std("OC(=O)c1ccccc1OC(C)=O"))
    assert chem.inchikey(std(L_ALA)) != chem.inchikey(std(D_ALA))


@pytest.mark.parametrize("smiles", ["C" * 1100, "*"], ids=["beyond-1024-atoms", "dummy-atom"])
def test_keys_are_none_when_inchi_fails(smiles):
    mol = Chem.MolFromSmiles(smiles)                # a raw molecule: standardize would already have rejected both
    assert chem.conn(mol) is None and chem.skeleton(mol) is None and chem.inchikey(mol) is None


# ---------------------------------------------------------------- scaffold
@pytest.mark.parametrize("smiles, expected", [
    ("Cc1ccccc1", "c1ccccc1"),                                                     # side chain removed
    (IBUPROFEN, "c1ccccc1"),
    ("c1ccccc1CCc1ccccc1", "c1ccc(CCc2ccccc2)cc1"),                                # linker between rings kept
    ("O=C1CCCCC1C", "O=C1CCCCC1"),                                                 # exocyclic double bond on a ring kept
    ("CCCCCCCC(=O)O", ""),                                                         # acyclic: empty string
    ("CCO", ""),
])
def test_scaffold(smiles, expected):
    assert chem.scaffold(std(smiles)) == expected


def test_scaffold_ignores_stereo():
    # contracts.md: non-isomeric scaffold SMILES; stereo is scored separately. cis and trans decalin share a scaffold.
    cis, trans = chem.scaffold(std("C1CC[C@H]2CCCC[C@@H]2C1")), chem.scaffold(std("C1CC[C@H]2CCCC[C@H]2C1"))
    assert cis == trans and "@" not in cis


def test_scaffold_is_shared_by_different_substituents_on_one_ring_system():
    assert chem.scaffold(std(ASPIRIN)) == chem.scaffold(std(IBUPROFEN))
    assert chem.scaffold(std(ASPIRIN)) != chem.scaffold(std("C1CCCCC1C(=O)N"))


# ---------------------------------------------------------------- tanimoto (count vectors) and tanimoto_bits
PHENYL_ACID_16 = "c1ccccc1" + "C" * 7 + "C(=O)O"          # phenyl fatty acids of 16 and 108 heavy atoms
PHENYL_ACID_108 = "c1ccccc1" + "C" * 99 + "C(=O)O"


@pytest.mark.parametrize("similarity", [chem.tanimoto, chem.tanimoto_bits])
def test_tanimoto_identity_symmetry_and_range(similarity):
    a, b = std(ASPIRIN), std(IBUPROFEN)
    assert similarity(a, a) == 1.0
    assert similarity(a, b) == similarity(b, a)
    assert 0.0 <= similarity(a, b) < 1.0


def test_tanimoto_is_the_morgan_radius_2_count_vector_similarity():
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    a, b = std(ASPIRIN), std("OC(=O)c1ccccc1O")
    by_hand = DataStructs.TanimotoSimilarity(gen.GetCountFingerprint(a), gen.GetCountFingerprint(b))
    assert chem.tanimoto(a, b) == pytest.approx(by_hand) == pytest.approx(0.55)
    radius_3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    by_hand_r3 = DataStructs.TanimotoSimilarity(radius_3.GetCountFingerprint(a), radius_3.GetCountFingerprint(b))
    assert by_hand_r3 != pytest.approx(0.55)                        # the pair is sensitive to the radius


def test_tanimoto_bits_is_morgan_radius_2_with_2048_bits():
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    a, b = std(ASPIRIN), std("OC(=O)c1ccccc1O")            # a pair on which radius 1, 3 and 1024 bits all differ
    by_hand = DataStructs.TanimotoSimilarity(gen.GetFingerprint(a), gen.GetFingerprint(b))
    assert chem.tanimoto_bits(a, b) == pytest.approx(by_hand) == pytest.approx(0.4482758620689655)


def test_count_tanimoto_separates_repeated_units_that_the_bit_version_cannot():
    short, long = std(PHENYL_ACID_16), std(PHENYL_ACID_108)
    assert (short.GetNumHeavyAtoms(), long.GetNumHeavyAtoms()) == (16, 108)
    assert chem.tanimoto_bits(short, long) == 1.0
    assert chem.tanimoto(short, long) < 0.2


def test_tanimoto_ranks_an_analogue_above_an_unrelated_molecule():
    aspirin = std(ASPIRIN)
    for similarity in (chem.tanimoto, chem.tanimoto_bits):
        assert similarity(aspirin, std("OC(=O)c1ccccc1O")) > similarity(aspirin, std(CAFFEINE))


# ---------------------------------------------------------------- extract_answer
@pytest.mark.parametrize("text", [ASPIRIN, "", "</think>", "</think>\n  \n"],
                         ids=["no-think", "empty", "bare-tag", "blank-tail"])
def test_extract_needs_a_closed_think_block_and_a_tail(text):
    assert chem.extract_answer(text) is None


@pytest.mark.parametrize("not_text", [None, 42, b"</think>\nCC(=O)Oc1ccccc1C(=O)O"])
def test_extract_returns_none_for_non_strings(not_text):
    assert chem.extract_answer(not_text) is None


def test_extract_reads_only_after_the_last_think_tag():
    assert chem.extract_answer(f"{IBUPROFEN}</think>\nthinking</think>\n{ASPIRIN}") == ASPIRIN
    assert chem.extract_answer(f"</think>\n{ASPIRIN}\n</think>\nno structure here") is None


def test_extract_takes_the_last_smiles_like_token():
    assert chem.extract_answer(f"</think>\n{IBUPROFEN} and then {ASPIRIN}") == ASPIRIN


@pytest.mark.parametrize("tail", [
    f"SMILES: {ASPIRIN}", f"**{ASPIRIN}**", f"`{ASPIRIN}`", f"{ASPIRIN}.", f"{ASPIRIN},", f"{ASPIRIN};",
    f"{ASPIRIN}<|im_end|>", f"{ASPIRIN} ", f"　{ASPIRIN}　", f"\t{ASPIRIN}\r\n",
])
def test_extract_strips_decoration_and_unicode_spaces(tail):
    assert chem.extract_answer("reasoning</think>\n" + tail) == ASPIRIN


def test_extract_length_and_hint_gates():
    assert chem.extract_answer("</think>\nCC(=O)O") is None            # 7 characters
    assert chem.extract_answer("</think>\nCCCCCCCCO") is None          # long enough but no ring, branch, bond or digit
    assert chem.extract_answer("</think>\nCCCCCCC=O") == "CCCCCCC=O"


def test_extract_ignores_tokens_longer_than_max_len():
    at_limit = "C" * (chem.MAX_LEN - 5) + "(=O)O"
    assert chem.extract_answer(f"</think>\n{ASPIRIN} {at_limit}") == at_limit
    assert chem.extract_answer(f"</think>\n{ASPIRIN} C{at_limit}") == ASPIRIN


def test_extract_replaces_im_end_by_a_space_and_keeps_a_closing_parenthesis():
    assert chem.extract_answer(f"</think>\n{ASPIRIN}<|im_end|>trailing") == ASPIRIN
    assert chem.extract_answer("</think>\nCC(=O)OC(C)(C)") == "CC(=O)OC(C)(C)"


def test_extract_skips_tokens_that_do_not_parse():
    assert chem.extract_answer(f"</think>\n{ASPIRIN} C1CC(C(=O)N") == ASPIRIN


def test_extract_skips_a_token_with_a_lone_surrogate_instead_of_raising():
    assert chem.extract_answer(f"</think>\n{ASPIRIN}\ud800") is None
    assert chem.extract_answer(f"</think>\n{IBUPROFEN} {ASPIRIN}\ud800") == IBUPROFEN
