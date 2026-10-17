"""Shared chemistry primitives: the one implementation of standardise, connectivity, scaffold,
similarity and answer extraction. reward.py, data_prep.py, metrics.py and controls.py import from here.
Definitions are in CONTRACTS.md; change them there first."""
import hashlib
import re

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGE = rdMolStandardize.Uncharger()
_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_STRIP = "`*.,;:"
_HINT = re.compile(r"[=()\[\]@\d]")
# Cost guards for generated text: scaffold extraction is cubic in heavy atoms (143 s at 3,750), so one runaway answer could
# stall a reward worker. Longest reference: 715 characters, 295 heavy atoms.
MAX_LEN, MAX_ATOMS = 1500, 400
_LAYER = re.compile(r"/[hqpbtmsifr]")  # first InChI layer after formula and /c


def standardize(smiles):
    """Largest fragment, neutralised. None if the string does not parse, has more than MAX_ATOMS atoms in total
    (dummy atoms and extra fragments included, so the later steps stay cheap) or contains a dummy atom."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    if mol is None or mol.GetNumAtoms() > MAX_ATOMS or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None
    return _UNCHARGE.uncharge(_LARGEST.choose(mol))


def canonical(mol):
    return Chem.MolToSmiles(mol)


def conn(mol):
    """Identity key: first block of the standard InChIKey (formula, connectivity and hydrogen placement; no stereo).
    None if InChI fails. A tautomer written differently is a different key; accepted alternates live in the data
    (`ref_alt_smiles`) instead of in a looser key, which would equate real isomers such as delta-8 and delta-9 THC."""
    key = Chem.MolToInchiKey(mol)
    return key.split("-")[0] if key else None


def skeleton(mol):
    """Coarse key: formula plus heavy-atom skeleton (InChI layers before hydrogens, charge, stereo, isotopes), sha1 to 14 hex.
    Never used to match answers; only to check that a registry alternate of a reference has the same skeleton."""
    inchi = Chem.MolToInchi(mol)
    return hashlib.sha1(_LAYER.split(inchi, 1)[0].encode()).hexdigest()[:14].upper() if inchi else None


def inchikey(mol):
    """Full standard InChIKey (stereo and hydrogen-placement sensitive). None if InChI fails."""
    return Chem.MolToInchiKey(mol) or None


def scaffold(mol):
    """Murcko scaffold SMILES without stereo; empty string for acyclic molecules."""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf.GetNumAtoms() else ""   # stereo is scored separately


def tanimoto(a, b):
    """Morgan radius 2 count-vector Tanimoto (repeated units change the score). The reward uses this one."""
    return DataStructs.TanimotoSimilarity(_FP.GetCountFingerprint(a), _FP.GetCountFingerprint(b))


def tanimoto_bits(a, b):
    """Morgan radius 2, 2048-bit Tanimoto. Used for the leakage graph (threshold 0.4) and the controls."""
    return DataStructs.TanimotoSimilarity(_FP.GetFingerprint(a), _FP.GetFingerprint(b))


def _parses(token):
    try:  # a lone surrogate or another malformed generated string must not crash a reward worker
        return Chem.MolFromSmiles(token) is not None
    except Exception:
        return False


def extract_answer(text):
    """Last SMILES-like token after the last </think>; None if think is not closed or nothing parses."""
    if not isinstance(text, str) or "</think>" not in text:
        return None
    tail = text.rsplit("</think>", 1)[1].replace("<|im_end|>", " ")
    for tok in reversed(tail.split()):
        tok = tok.strip(_STRIP)
        # length and a ring/branch/bond/stereo/digit char keep words like "CCO" or "NO" out
        if 8 <= len(tok) <= MAX_LEN and _HINT.search(tok) and _parses(tok):
            return tok
    return None
