"""Unit tests for rlexp.chem: standardize, canonical, conn, inchikey, scaffold, tanimoto, extract_answer."""
import re

import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from rlexp import chem

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
L_ALA, D_ALA = "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"
TETRACYCLINE = "CC1(O)C2CC3C(N(C)C)C(O)=C(C(N)=O)C(=O)C3(O)C(O)=C2C(=O)c2c(O)cccc21"
TETRACYCLINE_RING_A_SWAPPED = "CC1(O)C2CC3C(N(C)C)C(=O)C(C(N)=O)=C(O)C3(O)C(O)=C2C(=O)c2c(O)cccc21"


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


# ---------------------------------------------------------------- canonical
def test_canonical_is_independent_of_the_spelling():
    assert can("OC(=O)c1ccccc1OC(C)=O") == can(ASPIRIN) == chem.canonical(Chem.MolFromSmiles(ASPIRIN))
    assert can("C[C@H](N)C(=O)O") == can(L_ALA)


def test_canonical_is_isomeric():
    assert can(L_ALA) != can(D_ALA)


# ---------------------------------------------------------------- conn and inchikey
@pytest.mark.parametrize("a, b", [
    (L_ALA, D_ALA),                                                                # enantiomers
    ("C(=C/c1ccccc1)\\c1ccccc1", "C(=C\\c1ccccc1)\\c1ccccc1"),                     # E and Z
    (CAFFEINE, "[2H]C([2H])([2H])n1cnc2c1c(=O)n(C([2H])([2H])[2H])c(=O)n2C([2H])([2H])[2H]"),   # isotopologue
    ("CC(=O)CC(C)=O", "CC(O)=CC(C)=O"),                                            # acetylacetone keto and enol
    ("NC(=O)C1=C(O)CCCC1=O", "NC(=O)C1C(=O)CCCC1=O"),                              # tetracycline A-ring fragment
    (TETRACYCLINE, TETRACYCLINE_RING_A_SWAPPED),                                   # two tautomers of tetracycline
], ids=["enantiomers", "E-Z", "isotopologue", "keto-enol", "tetracycline-fragment", "tetracycline"])
def test_conn_ignores_stereo_isotopes_and_hydrogen_placement_but_inchikey_does_not(a, b):
    assert chem.conn(std(a)) == chem.conn(std(b))
    assert chem.inchikey(std(a)) != chem.inchikey(std(b))


@pytest.mark.parametrize("a, b", [
    ("CCO", "COC"),                                                                # same formula, other skeleton
    ("CCCC", "CC(C)C"),
    ("Cc1ccccc1O", "Cc1ccc(O)cc1"),                                                # ortho and para
    (ASPIRIN, IBUPROFEN),
    ("C[N+](C)(C)CCO", "CN(C)CCO"),                                                # other formula
])
def test_conn_separates_other_skeletons_and_formulas(a, b):
    assert chem.conn(std(a)) != chem.conn(std(b))


def test_conn_is_the_same_for_every_spelling_and_counter_ion():
    spellings = {chem.conn(std(Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN), doRandom=True))) for _ in range(10)}
    assert spellings == {chem.conn(std(ASPIRIN)), chem.conn(std("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"))}


def test_conn_format_is_14_uppercase_hex_characters():
    assert re.fullmatch(r"[0-9A-F]{14}", chem.conn(std(ASPIRIN)))


def test_inchikey_format_and_sensitivity():
    key = chem.inchikey(std(ASPIRIN))
    assert re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key)
    assert key == chem.inchikey(std("OC(=O)c1ccccc1OC(C)=O"))
    assert chem.inchikey(std(L_ALA)) != chem.inchikey(std(D_ALA))


@pytest.mark.parametrize("smiles", ["C" * 1100, "*"], ids=["beyond-1024-atoms", "dummy-atom"])
def test_conn_and_inchikey_are_none_when_inchi_fails(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert chem.conn(mol) is None and chem.inchikey(mol) is None


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


# ---------------------------------------------------------------- tanimoto
def test_tanimoto_identity_symmetry_and_range():
    a, b = std(ASPIRIN), std(IBUPROFEN)
    assert chem.tanimoto(a, a) == 1.0
    assert chem.tanimoto(a, b) == chem.tanimoto(b, a)
    assert 0.0 <= chem.tanimoto(a, b) < 1.0


def test_tanimoto_is_morgan_radius_2_with_2048_bits():
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    a, b = std(ASPIRIN), std("OC(=O)c1ccccc1O")            # a pair on which radius 1, 3 and 1024 bits all differ
    by_hand = DataStructs.TanimotoSimilarity(gen.GetFingerprint(a), gen.GetFingerprint(b))
    assert chem.tanimoto(a, b) == pytest.approx(by_hand)
    assert chem.tanimoto(a, b) == pytest.approx(0.4482758620689655)


def test_tanimoto_ranks_an_analogue_above_an_unrelated_molecule():
    aspirin = std(ASPIRIN)
    assert chem.tanimoto(aspirin, std("OC(=O)c1ccccc1O")) > chem.tanimoto(aspirin, std(CAFFEINE))


# ---------------------------------------------------------------- extract_answer
@pytest.mark.parametrize("text", [ASPIRIN, "", "</think>", "</think>\n  \n"],
                         ids=["no-think", "empty", "bare-tag", "blank-tail"])
def test_extract_needs_a_closed_think_block_and_a_tail(text):
    assert chem.extract_answer(text) is None


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


def test_extract_replaces_im_end_by_a_space_and_keeps_a_closing_parenthesis():
    assert chem.extract_answer(f"</think>\n{ASPIRIN}<|im_end|>trailing") == ASPIRIN
    assert chem.extract_answer("</think>\nCC(=O)OC(C)(C)") == "CC(=O)OC(C)(C)"


def test_extract_skips_tokens_that_do_not_parse():
    assert chem.extract_answer(f"</think>\n{ASPIRIN} C1CC(C(=O)N") == ASPIRIN
