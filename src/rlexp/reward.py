"""Reward v2: recover the structure of the named drug (contracts.md, "reward.py (v2)").

    reward = 0                                                      no </think>, no answer, or it does not standardize
    reward = 1.0                                                    connectivity matches the reference or an alternate
    reward = W_VALID + W_TANI * tanimoto + W_SCAF * scaffold_match  otherwise (below 1.0)

No similarity floor: group-relative advantages need only the rank order of the rewards inside a group, and a floor
would collapse the many low-similarity answers into a tie. A scaffold bonus because two random drugs share an exact
Murcko scaffold only about 0.1% of the time, so a hit is rare and informative. Stereo is reported, never rewarded.

v3 (2026-10-17): a sample that did not end by itself (`finished=False`: cut by the length cap, or aborted) is a failed answer,
because a deployed system cannot use text that never ended: valid False, every score 0. `answer` and `closed` are still reported
for diagnostics.
"""
import json
import warnings
from functools import lru_cache

from rlexp import chem

W_VALID, W_TANI, W_SCAF = 0.05, 0.6, 0.3    # provisional: tune on Train rollouts and Val only
MAX_LEN = chem.MAX_LEN                       # longer answers are invalid; chem.standardize bounds atoms, rejects dummy atoms


@lru_cache(maxsize=2048)
def _features(smiles):
    """(mol, conn, scaffold, inchikey) of the standardized structure; None if chem.standardize rejects the string."""
    mol = chem.standardize(smiles)
    return None if mol is None else (mol, chem.conn(mol), chem.scaffold(mol), chem.inchikey(mol))


def _quiet(fn, *args):
    """fn(*args), or None when chem/RDKit raises on a malformed string (a lone surrogate, an unhashable alternate)."""
    try:
        return fn(*args)
    except Exception:
        return None


def score(solution_str: str | None, ref: dict, finished: bool = True) -> dict:
    """Score one generated text against one reference row. ref_smiles is required; ref_alt_smiles (other registry
    SMILES of the same compound, usually empty) is optional. Every key is derived from them in this process, so stored
    ref_conn, ref_scaffold and ref_inchikey are ignored. finished=False marks a sample that did not end by itself (cut by the
    length cap, or aborted): it scores as a failure (valid False, reward 0, nothing matched) with `answer` and `closed` still reported.
    Raises only on a bad reference, never on generated text."""
    text = solution_str or ""
    answer = _quiet(chem.extract_answer, text)
    out = {"closed": "</think>" in text, "answer": answer, "valid": False, "tanimoto": 0.0, "scaffold_match": False,
           "conn_match": False, "stereo_match": False, "reward": 0.0}
    r = _features(ref["ref_smiles"])
    if r is None:
        raise ValueError("reference does not standardize (unparseable, dummy atom or too many atoms)")
    if not finished:
        return out
    alts = ref.get("ref_alt_smiles") or []
    alts = [alts] if isinstance(alts, str) else alts
    conns = {f[1] for f in [r, *(_quiet(_features, s) for s in alts)] if f and f[1]}    # alternates that fail are ignored
    a = _quiet(_features, answer) if answer and len(answer) <= MAX_LEN else None
    if a is None:
        return out
    mol, conn, scaf, key = a
    tani = chem.tanimoto(mol, r[0])
    conn_match = conn in conns
    scaffold_match = bool(scaf) and scaf == r[2]            # two acyclic molecules do not share a scaffold
    out.update(valid=True, tanimoto=tani, scaffold_match=scaffold_match, conn_match=conn_match,
               stereo_match=key is not None and key == r[3],
               reward=1.0 if conn_match else W_VALID + W_TANI * tani + W_SCAF * scaffold_match)
    return out


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """verl entry point. ground_truth is the ref dict or its JSON string. Returns the reward; never raises. It has no completion
    status (the verl signature carries none): it scores the text as if finished. A caller that knows the sample was cut by the
    length cap must call score(..., finished=False) or zero the reward itself, or the old training signal comes back."""
    try:
        ref = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        return float(score(solution_str, ref)["reward"])
    except Exception as e:
        warnings.warn(f"reward.compute_score returned 0.0 after {type(e).__name__}: {str(e)[:100]}", RuntimeWarning)
        return 0.0
