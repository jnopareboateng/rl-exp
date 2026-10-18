#!/usr/bin/env python3
"""Two audits on real Val samples before prereg-v1 (docs/plan.md section 4). Counts and rates only: no text is printed or written.

    .venv/bin/python scripts/offline_audit.py [--runs runs/colab-s1] [--refs data/processed/refs.jsonl]
        [--extractor-out results/extractor_audit.json] [--t5-out results/t5.json]

Extractor audit. For every Val sample of a model run: `rlexp.chem.extract_answer` (the scorer's reader, A: the last whitespace-
separated token after the last </think> that looks like a SMILES) against the stricter reader the plan names (B: the last LINE
after the last </think> that is a SMILES by itself, one token, with the characters chem.extract_answer strips from a token (` * . , ; :) stripped
from both ends, standardised like references). They disagree when the connectivity keys of what they read differ, or one reads a molecule and the other none.
Plan rule: more than 1% of outputs differ -> switch to B before prereg-v1. The verdict uses the stricter of two rates: over all
samples and over finished samples (a sample cut by the length cap scores 0 whatever is read from it). To judge WHICH reader is
right in a disagreement, a_only_small and b_only_small count the molecules read that have fewer heavy atoms than the smallest
reference (a fragment, not a drug); b_only_guard counts the b_only answers that A cannot read even alone (its length or
character guard, a known limit).

T5. Per run with k >= 4 samples per prompt row: the share of prompt rows whose reward std over their VALID finished samples is
>= 0.05 (offline_reward_checks.group_spread, population std; fewer than two valid samples count as no spread). Plan rule: >= 30%.
Rewards come from rlexp.reward.score on the samples and the references of refs.jsonl, the length-cap rule applied.
share_all_samples is the same threshold over all k samples with invalid ones at reward 0: the spread an RL update sees.
Only Val is audited (--split refuses the Test splits, which belong to the locked evaluation).

Skipped (SKIP, matched on the run folder's name): ident- at the start (the answer is a drug name); control_ anywhere (CPU controls, val-control_ and
val8-control_, not model output); strict- at the start (a rescoring of another run, same samples); and the Val diagnostic families made after the committed
audit, so that the audited run list stays the 29 runs of results/extractor_audit.json: -shuffle- and -blank- anywhere, r1s and r3s at the start (RL
checkpoint families), a4-t at the start (extra A4 temperatures), -seed<n> at the end (seed repeats), rlfull-s10- at the start and rlfull-s<n>-t0.3 (the RL
step-10 and T=0.3 sweeps), rlfull<n>- at the start (a later RL replicate: rlfull5-...), rlvalidity- at the start (the RL validity runs). The population is pinned by tests/test_offline_audit.py: a run folder that none of these names must be added to SKIP, or the audit
artefacts regenerated and the numbers quoted in docs/prereg-v1.md section 8 restated. Also skipped: a run whose every sample (same model, mode, temperature, top-p, token cap,
seed, id, sample index and text) is also in another run: a partial copy adds nothing.
"""
import argparse
import hashlib
import json
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import rdkit

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ is not a package
from offline_reward_checks import group_spread                # noqa: E402  the T5 helper, one definition of "spread"
from rlexp import chem, metrics, reward                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SKIP = ((r"^ident-", "the answer is a drug name, not a structure"),
        (r"control_", "CPU control, not model output"),
        (r"^strict-", "a strict-reader rescoring of another run, not new samples"),
        (r"-shuffle-|-blank-|^r1s|^r3s|^a4-t", "Val diagnostic family made after the committed audit"),
        (r"-seed\d+$|^rlfull-s10-|^rlfull-s\d+-t0\.3$|^rlfull\d+-|^rlvalidity-", "Val run made after the committed audit: a seed repeat, the RL step-10 and T=0.3 sweeps, a later RL replicate or the RL validity runs"))   # (regex on the run folder's name, why)
SWITCH_RATE, MIN_STD, T5_BAR, T5_MIN_K = 0.01, 0.05, 0.30, 4
STRIP = chem._STRIP                                            # ` * . , ; : : what chem.extract_answer strips from a token (code fences, bold, a closing full stop)
SAMPLING = ("model", "mode", "temperature", "top_p", "max_tokens", "seed")     # runs that differ in any of these are not copies


# ---------------------------------------------------------------------------------------------------- extractor audit
@lru_cache(maxsize=None)
def _key(smiles):
    """Identity of a read answer: the connectivity key the scorer matches on (the canonical SMILES if InChI fails); None when
    nothing was read or the string does not standardise the way the scorer's references do."""
    if not smiles:
        return None
    try:
        mol = chem.standardize(smiles)
        return None if mol is None else chem.conn(mol) or chem.canonical(mol)
    except Exception:                                          # a malformed generated string must not stop an audit
        return None


def last_line_answer(text):
    """Reader B: the last line after the last </think> that is a SMILES by itself; None if the block is not closed or no line is.
    No length or character guard (reader A has one): a short SMILES on its own line is read."""
    if not isinstance(text, str) or "</think>" not in text:
        return None
    tail = text.rsplit("</think>", 1)[1].replace("<|im_end|>", "\n")
    for line in reversed(re.split(r"\r?\n", tail)):             # not str.splitlines: it also cuts at U+2028 and U+0085
        tok = line.strip().strip(STRIP).strip()
        if tok and len(tok.split()) == 1 and len(tok) <= chem.MAX_LEN and _key(tok) is not None:
            return tok
    return None


def classify(a, b):
    """How the two readers differ, given what A (chem.extract_answer) and B (last_line_answer) read from one text: a tuple of
    kinds, "agree" (same molecule), "agree_none" (neither reads one), "a_only", "b_only" (the other reads nothing) or
    "differ" (two molecules). A b_only answer that A cannot read even alone (its length or character guard, a known limit)
    is also counted as "b_only_guard"."""
    ka, kb = _key(a), _key(b)
    if ka == kb:
        return ("agree_none" if ka is None else "agree",)
    if kb is None:
        return ("a_only",)
    if ka is None:
        return ("b_only", "b_only_guard") if chem.extract_answer("</think>\n" + b) is None else ("b_only",)
    return ("differ",)


KINDS = ("agree", "agree_none", "a_only", "b_only", "b_only_guard", "differ")
COUNTS = ("samples", "finished", "a_reads", "b_reads", *KINDS, *(f"{k}_finished" for k in KINDS), "a_only_small", "b_only_small")


@lru_cache(maxsize=None)
def _heavy(smiles):
    mol = chem.standardize(smiles) if smiles else None
    return None if mol is None else mol.GetNumHeavyAtoms()


def audit_extractor(samples, min_heavy=0):
    """Counts for one run's samples: samples, finished (metrics.finished: finish_reason "stop" or none recorded; "length" and "abort" are
    not finished), how many each reader reads, the KINDS over all samples and over finished ones, and how many a_only / b_only molecules
    have fewer than min_heavy heavy atoms (the smallest reference's size: too small to be the answer). disagree = a_only + b_only + differ."""
    out = dict.fromkeys(COUNTS, 0)
    for s in samples:
        text, fin = s.get("text") or "", metrics.finished(s)
        a, b = chem.extract_answer(text), last_line_answer(text)
        out["samples"] += 1
        out["finished"] += fin
        out["a_reads"] += _key(a) is not None
        out["b_reads"] += _key(b) is not None
        for k in classify(a, b):
            out[k] += 1
            out[f"{k}_finished"] += fin
            if k == "a_only":
                out["a_only_small"] += (_heavy(a) or 0) < min_heavy
            elif k == "b_only":
                out["b_only_small"] += (_heavy(b) or 0) < min_heavy
    return finish_counts(out)


def finish_counts(c):
    """Add disagree, disagree_finished and the two rates to a dict of counts (rates are 0.0 for no samples)."""
    c["disagree"] = c["a_only"] + c["b_only"] + c["differ"]
    c["disagree_finished"] = c["a_only_finished"] + c["b_only_finished"] + c["differ_finished"]
    c["rate"] = round(c["disagree"] / c["samples"], 6) if c["samples"] else 0.0
    c["rate_finished"] = round(c["disagree_finished"] / c["finished"], 6) if c["finished"] else 0.0
    return c


def extractor_verdict(overall):
    """Plan rule: switch to reader B before prereg-v1 if more than 1% of outputs differ (the stricter of the two rates)."""
    rate = max(overall["rate"], overall["rate_finished"])
    return {"rule": "switch to the last SMILES-only line before prereg-v1 if more than 1% of outputs differ", "threshold": SWITCH_RATE,
            "rate_all_samples": overall["rate"], "rate_finished_samples": overall["rate_finished"], "decided_on": round(rate, 6),
            "verdict": "switch" if rate > SWITCH_RATE else "keep"}


# ---------------------------------------------------------------------------------------------------- T5
def t5_run(samples, refs):
    """T5 counts for one run's samples (all from one split): k (samples per prompt row: the most common count, so one over-long or
    short row cannot change it; the larger count on a tie), rows with exactly k samples, rows with at least two valid finished
    samples, rows whose reward std over those is >= MIN_STD, and the shares. share is the plan's T5 (valid samples only);
    share_all_samples is the same threshold over all k samples with invalid ones at reward 0, which is the spread RL updates see.
    A row with another count than k is left out and counted in rows_incomplete.
    refs: {group: {"ref_smiles": ..., "ref_alt_smiles": [...]}}."""
    by_id, group = {}, {}
    for s in samples:
        by_id.setdefault(s["id"], []).append(s)
        group[s["id"]] = s["group"]
    counts = Counter(len(v) for v in by_id.values())
    k = max(counts, key=lambda n: (counts[n], n), default=0)
    rewards, flags = {}, {}
    for i, rs in by_id.items():
        if len(rs) != k:
            continue
        res = [reward.score(s.get("text") or "", refs[group[i]], finished=metrics.finished(s)) for s in rs]
        rewards[i], flags[i] = [r["reward"] for r in res], [bool(r["valid"]) for r in res]
    ge2 = sum(sum(f) >= 2 for f in flags.values())
    spread = sum(group_spread({i: rewards[i]}, MIN_STD, {i: flags[i]}) > 0 for i in rewards)
    spread_all = sum(group_spread({i: rewards[i]}, MIN_STD) > 0 for i in rewards)       # invalid samples (reward 0) included
    n = len(rewards)
    return {"k": k, "rows": n, "rows_incomplete": len(by_id) - n, "rows_ge2_valid": ge2, "rows_spread": spread,
            "share": round(spread / n, 6) if n else 0.0, "share_among_ge2_valid": round(spread / ge2, 6) if ge2 else 0.0,
            "meets_bar": bool(n) and spread / n >= T5_BAR,
            "rows_spread_all_samples": spread_all, "share_all_samples": round(spread_all / n, 6) if n else 0.0}


# ---------------------------------------------------------------------------------------------------- files and provenance
def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True, timeout=30).stdout.rstrip("\n")
    except (OSError, subprocess.SubprocessError):
        return None


def load_refs(path):
    """({group: {"ref_smiles", "ref_alt_smiles"}} of the included drugs of refs.jsonl, heavy atoms of the smallest reference)."""
    rows = [r for r in metrics.read_jsonl(path) if r.get("exclude_reason") is None and r.get("ref_smiles")]
    refs = {r["group"]: {"ref_smiles": r["ref_smiles"], "ref_alt_smiles": r.get("ref_alt_smiles") or []} for r in rows}
    return refs, min((r["n_heavy"] for r in rows if r.get("n_heavy")), default=0)


def provenance(command, refs_path, inputs):
    return {"command": command, "inputs": {"refs.jsonl": {"sha256": _sha256(refs_path)}, **inputs},
            "git": {"revision": _git("rev-parse", "HEAD"), "script_sha256": _sha256(__file__)},
            "versions": {"python": platform.python_version(), "rdkit": rdkit.__version__, "numpy": np.__version__}}


def _sample_key(r):
    return (*(r.get(f) for f in SAMPLING), r.get("id"), r.get("sample_idx"), hashlib.sha1((r.get("text") or "").encode()).hexdigest())


def collect(runs_dir, split="val"):
    """({run: samples of `split`}, {run: sha256 of its samples.jsonl}, {run: why it was left out}) for each run folder. A run
    whose every sample is also in another kept run (a partial copy) is left out, larger runs first."""
    loaded, notes = {}, {}
    for d in sorted(p for p in Path(runs_dir).iterdir() if p.is_dir()):
        why = next((w for pattern, w in SKIP if re.search(pattern, d.name)), None)
        f = d / "samples.jsonl"
        if why or not f.is_file():
            notes[d.name] = why or "no samples.jsonl"
            continue
        rows = [r for r in metrics.read_jsonl(f) if r.get("split") == split]
        if rows:
            loaded[d.name] = (rows, _sha256(f), {_sample_key(r) for r in rows})
        else:
            notes[d.name] = f"no {split} samples"
    kept = []
    for name in sorted(loaded, key=lambda n: (-len(loaded[n][2]), n)):
        copy_of = next((m for m in kept if loaded[name][2] <= loaded[m][2]), None)
        if copy_of:
            notes[name] = f"every sample is also in run {copy_of}"
        else:
            kept.append(name)
    kept = sorted(kept)
    return {n: loaded[n][0] for n in kept}, {n: loaded[n][1] for n in kept}, notes


def run_audits(runs_dir, refs_path, split="val"):
    """(extractor result, T5 result, input hashes, printed lines) from the run folders under runs_dir."""
    runs, hashes, skipped = collect(runs_dir, split)
    refs, min_heavy = load_refs(refs_path)
    unknown = {s["group"] for rows in runs.values() for s in rows} - set(refs)
    if unknown:
        raise ValueError(f"{len(unknown)} sample group(s) have no included reference in {refs_path}")
    ext, t5, t5_skipped, lines = {}, {}, {}, []
    for name, rows in runs.items():
        ext[name] = audit_extractor(rows, min_heavy)
        t = t5_run(rows, refs)
        if t["k"] >= T5_MIN_K:
            t5[name] = t
        else:
            t5_skipped[name] = f"k = {t['k']} < {T5_MIN_K}"
    overall = finish_counts({k: sum(r[k] for r in ext.values()) for k in COUNTS})
    for name, r in ext.items():
        lines.append(f"extractor {name:<16} samples {r['samples']:>4} disagree {r['disagree']:>3} ({r['rate']:.4f}); finished "
                     f"{r['finished']:>4} disagree {r['disagree_finished']:>3} ({r['rate_finished']:.4f}); a_only {r['a_only']} "
                     f"b_only {r['b_only']} (guard {r['b_only_guard']}, small {r['b_only_small']}) differ {r['differ']}; a_only small {r['a_only_small']}")
    verdict = extractor_verdict(overall)
    lines.append(f"extractor overall: samples {overall['samples']}, disagree {overall['disagree']} ({overall['rate']:.4f}); finished "
                 f"{overall['finished']}, disagree {overall['disagree_finished']} ({overall['rate_finished']:.4f}) -> {verdict['verdict']}")
    for name, t in t5.items():
        lines.append(f"T5 {name:<16} k {t['k']} rows {t['rows']:>3} with >=2 valid {t['rows_ge2_valid']:>3} spread {t['rows_spread']:>3} "
                     f"share {t['share']:.3f} (bar {T5_BAR}) {'meets' if t['meets_bar'] else 'below'}; over all samples {t['share_all_samples']:.3f}")
    inputs = {f"{n}/samples.jsonl": {"sha256": h} for n, h in hashes.items()}
    return ({"split": split, "min_reference_heavy_atoms": min_heavy, "runs": ext, "overall": overall, "verdict": verdict, "notes": skipped},
            {"split": split, "bar": T5_BAR, "min_std": MIN_STD, "min_k": T5_MIN_K, "runs": t5, "notes": {**skipped, **t5_skipped},
             "definition": "share of prompt rows whose reward std over valid finished samples is >= min_std; rows are the prompt "
                           "ids with all k samples; reward = rlexp.reward.score with the length-cap rule"}, inputs, lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs/colab-s1", help="folder of run folders, each with samples.jsonl")
    ap.add_argument("--refs", type=Path, default=ROOT / "data/processed/refs.jsonl")
    ap.add_argument("--split", default="val", help="the only split audited (Test splits are for the locked evaluation)")
    ap.add_argument("--extractor-out", type=Path, help="write the extractor audit here (JSON)")
    ap.add_argument("--t5-out", type=Path, help="write the T5 result here (JSON)")
    a = ap.parse_args(argv)
    if a.split in ("test_t", "test_r"):
        sys.exit("error: Test splits are scored only in the locked evaluation")
    exe = Path(sys.executable)
    exe = exe.relative_to(ROOT) if exe.is_relative_to(ROOT) else exe
    command = shlex.join([str(exe), "scripts/offline_audit.py", *(sys.argv[1:] if argv is None else argv)])
    try:
        ext, t5, inputs, lines = run_audits(a.runs, a.refs, a.split)
    except (OSError, ValueError, KeyError, metrics.InputError) as e:
        sys.exit(f"error: {type(e).__name__}: {str(e)[:200]}")
    prov = provenance(command, a.refs, inputs)
    print("\n".join(lines))
    for path, res in ((a.extractor_out, ext), (a.t5_out, t5)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**res, "provenance": prov}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
