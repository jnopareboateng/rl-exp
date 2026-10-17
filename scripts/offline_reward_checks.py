#!/usr/bin/env python
"""Offline reward checks before any RL (docs/plan.md section 4): REFS, T1, T2, T3 and the constant-answer bar.

    python scripts/offline_reward_checks.py [--data data/processed] [--seed 0]

Reads train.jsonl and val.jsonl only: test_r and test_t are scored in the locked evaluation and nowhere else. Prints
PASS/FAIL/INFO/SKIP lines with counts, rates and ids only (never prompt or completion text) and exits 1 when a check
fails. group_spread() is the T5 helper for real rollouts (Phase 2).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from rlexp import chem, reward

SPLITS = ("train", "val")
CLOSED = "</think>\n"        # prefix that turns a bare SMILES into a closed answer


def group_spread(rewards_by_prompt, min_std=0.05, valid_only_flags=None):
    """Share of prompt groups whose within-group reward std (population) is >= min_std; 0.0 for no groups.
    valid_only_flags {prompt: [bool per sample]}: take the std over the valid samples only, because a group of valid
    answers carries the chemical signal while invalid ones only add format variance. Fewer than 2 valid samples: std 0."""
    n = 0
    for prompt, rewards in rewards_by_prompt.items():
        if valid_only_flags is not None:
            flags = valid_only_flags[prompt]
            if len(flags) != len(rewards):
                raise ValueError(f"{len(flags)} valid flags for {len(rewards)} rewards")
            rewards = [x for x, ok in zip(rewards, flags) if ok]
        n += len(rewards) > 1 and float(np.std(rewards)) >= min_std - 1e-12      # tolerance for float noise at the edge
    return n / len(rewards_by_prompt) if rewards_by_prompt else 0.0


def read_jsonl(path):
    """Rows of a JSONL file, one per line. Iterates the file: str.splitlines would also cut at U+2028, U+2029 and U+0085,
    which are legal inside a JSON string."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(data):
    return {s: read_jsonl(Path(data) / f"{s}.jsonl") for s in SPLITS}


def verdict(ok):
    return "PASS" if ok else "FAIL"


def check_refs(rows):
    """Stored ref_conn, ref_scaffold (and ref_inchikey when present) equal what chem derives from ref_smiles, and every
    ref_alt_smiles standardizes to the reference's skeleton (an alternate that is another compound would score 1.0)."""
    derived, bad = {}, set()
    for r in rows:
        if r["ref_smiles"] not in derived:
            m = chem.standardize(r["ref_smiles"])
            derived[r["ref_smiles"]] = (chem.conn(m), chem.scaffold(m), chem.inchikey(m), chem.skeleton(m)) if m else None
        d = derived[r["ref_smiles"]]
        alts = [chem.standardize(a) for a in r.get("ref_alt_smiles") or []]
        if (d is None or (r["ref_conn"], r["ref_scaffold"]) != d[:2] or r.get("ref_inchikey", d[2]) != d[2]
                or any(m is None or chem.skeleton(m) != d[3] for m in alts)):
            bad.add(r["group"])
    return verdict(not bad), (f"{len(bad)} of {len({r['group'] for r in rows})} drugs have stored reference fields or "
                              f"alternates that disagree with chem (stale data_prep output?)")


def check_t1(rows, need=0.98):
    """A reference completion scored against its own row must recover the reference structure."""
    res = [(r["id"], reward.score(r["completion"], r)) for r in rows if r.get("completion")]
    miss = [(i, s) for i, s in res if not s["conn_match"]]
    share = 1 - len(miss) / len(res)
    msg = (f"{share:.4f} of {len(res)} train+val completions give conn_match (need >= {need}); {len(miss)} misses, "
           f"{sum(s['answer'] is None for _, s in miss)} without an extractable answer")
    return verdict(share >= need), msg + (f"\n  failing ids: {', '.join(i for i, _ in miss[:30])}" if miss else "")


def check_t2(rows, rng, n=2000, limit=0.2):
    """Another drug's reference, offered as the answer, must earn little."""
    drugs = {r["group"]: r["ref_smiles"] for r in rows if chem.extract_answer(CLOSED + r["ref_smiles"])}
    names, rewards, scaffold_hits, same, tries = sorted(drugs), [], 0, 0, 0
    while len(names) > 1 and len(rewards) < n and tries < 50 * n:         # the cap ends a draw that cannot succeed
        tries += 1
        row, other = rng.choice(rows), rng.choice(names)
        if other == row["group"]:
            continue
        s = reward.score(CLOSED + drugs[other], row)
        if s["conn_match"]:
            same += 1               # two names, one structure: not a wrong drug
            continue
        rewards.append(s["reward"])
        scaffold_hits += s["scaffold_match"]
    if not rewards:
        return "SKIP", (f"no wrong-identity pair: {len(names)} drug(s) with an extractable reference, {same} same-structure "
                        f"draw(s) in {tries} tries; T2 needs two drugs with different structures")
    mean = float(np.mean(rewards))
    msg = (f"{len(rewards)} wrong-drug pairs: mean {mean:.4f} (need <= {limit}), p99 {np.percentile(rewards, 99):.4f}, "
           f"scaffold hits {scaffold_hits} ({scaffold_hits / len(rewards):.4f}), skipped {same} same-structure pairs")
    return verdict(mean <= limit), msg + (f"; only {len(rewards)} of {n} drawn in {tries} tries" if len(rewards) < n else "")


def check_t3(rows, rng, n=200):
    """Randomised spellings of the reference must all score exactly 1.0."""
    picked, bad = rng.sample(rows, min(n, len(rows))), []
    for r in picked:
        variant = Chem.MolToSmiles(Chem.MolFromSmiles(r["ref_smiles"]), doRandom=True)
        s = reward.score(CLOSED + variant, r)
        if s["reward"] != 1.0:
            bad.append((r["id"], round(s["reward"], 4), variant[:120]))      # random spellings do not repeat: show it
    return verdict(not bad), f"{len(picked) - len(bad)} of {len(picked)} randomised spellings score 1.0; failures {bad[:5]}"


def best_constant(rows):
    """(group, mean reward) of the Train reference that scores highest as the answer to the rows of the other drugs
    (its own drug left out); None with fewer than two drugs."""
    groups = {}                                                # group -> [first row, number of rows]
    for r in rows:
        groups.setdefault(r["group"], [r, 0])[1] += 1
    best = None
    for g, (cand, _) in groups.items():
        text = CLOSED + cand["ref_smiles"]
        others = [(row, k) for h, (row, k) in groups.items() if h != g]
        if not others or not chem.extract_answer(text):
            continue
        mean = sum(reward.score(text, row)["reward"] * k for row, k in others) / sum(k for _, k in others)
        if best is None or mean > best[1]:
            best = (g, mean)
    return best


def constant_bar(data):
    """The do-nothing bar: the best constant answer is chosen on Train references only and reported on Val only."""
    pick = best_constant(data["train"])
    if pick is None or not data["val"]:
        return "SKIP", "needs two Train drugs with an extractable reference and at least one Val row"
    group, train_mean = pick
    smiles = next(r["ref_smiles"] for r in data["train"] if r["group"] == group)
    per_drug = {}                                              # group -> [reward, number of rows]
    for r in data["val"]:
        if r["group"] not in per_drug:
            per_drug[r["group"]] = [reward.score(CLOSED + smiles, r)["reward"], 0]
        per_drug[r["group"]][1] += 1
    row_mean = sum(x * k for x, k in per_drug.values()) / sum(k for _, k in per_drug.values())
    return "INFO", (f"best of {len({r['group'] for r in data['train']})} Train references is '{group}' (Train mean "
                    f"{train_mean:.4f}, own drug left out); on Val: row mean {row_mean:.4f}, drug mean "
                    f"{np.mean([x for x, _ in per_drug.values()]):.4f} ({len(per_drug)} drugs). Test is scored only in "
                    f"the locked evaluation")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/processed", help="directory with train.jsonl and val.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0, data, rng = time.perf_counter(), load(args.data), random.Random(args.seed)
    rows = data["train"] + data["val"]
    print(f"data: train {len(data['train'])} rows, val {len(data['val'])} rows; {len({r['group'] for r in rows})} drugs "
          f"(test_r and test_t are not read: they are scored only in the locked evaluation)")
    failed = 0
    for name, check in (("REFS", lambda: check_refs(rows)), ("T1", lambda: check_t1(rows)),
                        ("T2", lambda: check_t2(rows, rng)), ("T3", lambda: check_t3(rows, rng)),
                        ("BAR", lambda: constant_bar(data))):
        try:
            status, msg = check()
        except Exception as e:          # one broken row must not hide the other checks
            status, msg = "FAIL", f"crashed with {type(e).__name__}: {str(e)[:100]}"
        failed += status == "FAIL"
        print(f"{status} {name}: {msg}")
    print(f"done in {time.perf_counter() - t0:.1f}s, {failed} failed")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
