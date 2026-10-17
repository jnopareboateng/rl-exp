"""Reward v2: recover the structure of the named drug (contracts.md, "reward.py (v2)").

    reward = 0                                                      no </think>, no answer, or it does not standardize
    reward = 1.0                                                    connectivity matches the reference
    reward = W_VALID + W_TANI * tanimoto + W_SCAF * scaffold_match  otherwise (below 1.0)

No similarity floor: group-relative advantages need only the rank order of the rewards inside a group, and a floor
would collapse the many low-similarity answers into a tie. A scaffold bonus because two random drugs share an exact
Murcko scaffold only about 0.1% of the time, so a hit is rare and informative. Stereo is reported, never rewarded.
"""
import json
import warnings
from functools import lru_cache

from rlexp import chem

W_VALID, W_TANI, W_SCAF = 0.05, 0.6, 0.3    # provisional: tune on Train rollouts and Val only
MAX_LEN, MAX_ATOMS = chem.MAX_LEN, chem.MAX_ATOMS   # cost guards for generated text (see chem.py)


@lru_cache(maxsize=2048)
def _features(smiles):
    """(mol, conn, scaffold, inchikey) of the standardized structure; None if it does not standardize or is oversized."""
    mol = chem.standardize(smiles)
    if mol is None or mol.GetNumHeavyAtoms() > MAX_ATOMS:
        return None
    return mol, chem.conn(mol), chem.scaffold(mol), chem.inchikey(mol)


def _quiet(fn, *args):
    """fn(*args), or None when chem/RDKit raises on a malformed generated string (a lone surrogate, say)."""
    try:
        return fn(*args)
    except Exception:
        return None


def score(solution_str: str | None, ref: dict) -> dict:
    """Score one generated text against one reference row. Only ref_smiles is used: every key is derived from it in this
    process, so the reward does not depend on keys stored by another machine or RDKit version. Raises only on a bad
    reference, never on generated text."""
    text = solution_str or ""
    answer = _quiet(chem.extract_answer, text)
    out = {"closed": "</think>" in text, "answer": answer, "valid": False, "tanimoto": 0.0, "scaffold_match": False,
           "conn_match": False, "stereo_match": False, "reward": 0.0}
    r = _features(ref["ref_smiles"])
    if r is None:
        raise ValueError(f"reference does not standardize or exceeds {MAX_ATOMS} heavy atoms")
    a = _quiet(_features, answer) if answer and len(answer) <= MAX_LEN else None
    if a is None:
        return out
    mol, conn, scaf, key = a
    r_mol, r_conn, r_scaf, r_key = r
    tani = chem.tanimoto(mol, r_mol)
    conn_match = conn is not None and conn == r_conn
    scaffold_match = bool(scaf) and scaf == r_scaf          # two acyclic molecules do not share a scaffold
    out.update(valid=True, tanimoto=tani, scaffold_match=scaffold_match, conn_match=conn_match,
               stereo_match=key is not None and key == r_key,
               reward=1.0 if conn_match else W_VALID + W_TANI * tani + W_SCAF * scaffold_match)
    return out


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """verl entry point. ground_truth is the ref dict or its JSON string. Returns the reward; never raises."""
    try:
        ref = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        return float(score(solution_str, ref)["reward"])
    except Exception as e:
        warnings.warn(f"reward.compute_score returned 0.0 after {type(e).__name__}: {str(e)[:100]}", RuntimeWarning)
        return 0.0
