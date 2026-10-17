#!/usr/bin/env python
"""Offline reward checks before any RL (docs/plan.md section 4): REFS, T1, T2, T3 and the constant-answer bar.

    python scripts/offline_reward_checks.py [--data data/processed] [--seed 0]

Reads {train,val,test_r,test_t}.jsonl, prints PASS/FAIL/INFO lines with counts, rates and ids only (never prompt or
completion text) and exits 1 when a check fails. group_spread() is the T5 helper for real rollouts (Phase 2).
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

SPLITS = ("train", "val", "test_r", "test_t")
HELD = ("val", "test_r", "test_t")
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


def load(data):
    return {s: [json.loads(line) for line in (Path(data) / f"{s}.jsonl").read_text().splitlines() if line.strip()]
            for s in SPLITS}


def check_refs(rows):
    """Stored ref_conn, ref_scaffold (and ref_inchikey when present) equal what chem derives from ref_smiles."""
    derived, bad = {}, set()
    for r in rows:
        if r["ref_smiles"] not in derived:
            m = chem.standardize(r["ref_smiles"])
            derived[r["ref_smiles"]] = (chem.conn(m), chem.scaffold(m), chem.inchikey(m)) if m else None
        d = derived[r["ref_smiles"]]
        if d is None or (r["ref_conn"], r["ref_scaffold"]) != d[:2] or r.get("ref_inchikey", d[2]) != d[2]:
            bad.add(r["group"])
    return not bad, (f"{len(bad)} of {len({r['group'] for r in rows})} drugs have stored reference fields that "
                     f"disagree with chem (stale data_prep output?)")


def check_t1(rows, need=0.98):
    """A reference completion scored against its own row must recover the reference structure."""
    res = [(r["id"], reward.score(r["completion"], r)) for r in rows if r.get("completion")]
    miss = [(i, s) for i, s in res if not s["conn_match"]]
    share = 1 - len(miss) / len(res)
    msg = (f"{share:.4f} of {len(res)} train+val completions give conn_match (need >= {need}); {len(miss)} misses, "
           f"{sum(s['answer'] is None for _, s in miss)} without an extractable answer")
    return share >= need, msg + (f"\n  failing ids: {', '.join(i for i, _ in miss[:30])}" if miss else "")


def check_t2(rows, rng, n=2000, limit=0.2):
    """Another drug's reference, offered as the answer, must earn little."""
    drugs = {r["group"]: r["ref_smiles"] for r in rows if chem.extract_answer(CLOSED + r["ref_smiles"])}
    names = sorted(drugs)
    if len(names) < 2:
        raise ValueError("T2 needs at least two drugs")
    rewards, scaffold_hits, same = [], 0, 0
    while len(rewards) < n:
        row, other = rng.choice(rows), rng.choice(names)
        if other == row["group"]:
            continue
        s = reward.score(CLOSED + drugs[other], row)
        if s["conn_match"]:
            same += 1               # two names, one structure: not a wrong drug
            continue
        rewards.append(s["reward"])
        scaffold_hits += s["scaffold_match"]
    mean = float(np.mean(rewards))
    return mean <= limit, (f"{n} wrong-drug pairs: mean {mean:.4f} (need <= {limit}), p99 {np.percentile(rewards, 99):.4f}, "
                           f"scaffold hits {scaffold_hits} ({scaffold_hits / n:.4f}), skipped {same} same-structure pairs")


def check_t3(rows, rng, n=200):
    """Randomised spellings of the reference must all score exactly 1.0."""
    picked, bad = rng.sample(rows, min(n, len(rows))), []
    for r in picked:
        variant = Chem.MolToSmiles(Chem.MolFromSmiles(r["ref_smiles"]), doRandom=True)
        s = reward.score(CLOSED + variant, r)
        if s["reward"] != 1.0:
            bad.append((r["id"], round(s["reward"], 4), variant[:120]))      # random spellings do not repeat: show it
    return not bad, f"{len(picked) - len(bad)} of {len(picked)} randomised spellings score 1.0; failures {bad[:5]}"


def constant_bar(data):
    """The Train reference with the highest mean reward as the answer to every val + test_r + test_t row."""
    held = {}                                                # group -> [row, split, n_rows]
    for s in HELD:
        for r in data[s]:
            held.setdefault(r["group"], [r, s, 0])[2] += 1
    total = sum(n for _, _, n in held.values())
    train = {r["group"]: r["ref_smiles"] for r in data["train"] if chem.extract_answer(CLOSED + r["ref_smiles"])}
    best = None
    for g, smiles in train.items():
        rew = {h: reward.score(CLOSED + smiles, row)["reward"] for h, (row, _, _) in held.items()}
        pooled = sum(rew[h] * n for h, (_, _, n) in held.items()) / total
        if best is None or pooled > best[0]:
            best = (pooled, g, rew)
    pooled, g, rew = best
    per = []
    for s in HELD:
        hs = [h for h in held if held[h][1] == s]
        rows_mean = sum(rew[h] * held[h][2] for h in hs) / sum(held[h][2] for h in hs)
        per.append(f"{s} {rows_mean:.4f} (drug mean {np.mean([rew[h] for h in hs]):.4f}, {len(hs)} drugs)")
    return None, f"best of {len(train)} Train references is '{g}': pooled row mean {pooled:.4f}; " + "; ".join(per)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/processed", help="directory with train/val/test_r/test_t .jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0, data, rng = time.perf_counter(), load(args.data), random.Random(args.seed)
    rows = [r for s in SPLITS for r in data[s]]
    print("data: " + ", ".join(f"{s} {len(data[s])} rows" for s in SPLITS) + f"; {len({r['group'] for r in rows})} drugs")
    failed = 0
    for name, check in (("REFS", lambda: check_refs(rows)), ("T1", lambda: check_t1(data["train"] + data["val"])),
                        ("T2", lambda: check_t2(rows, rng)), ("T3", lambda: check_t3(rows, rng)),
                        ("BAR", lambda: constant_bar(data))):
        try:
            ok, msg = check()
        except Exception as e:          # one broken row must not hide the other checks
            ok, msg = False, f"crashed with {type(e).__name__}: {str(e)[:100]}"
        failed += ok is False
        print(f"{'INFO' if ok is None else 'PASS' if ok else 'FAIL'} {name}: {msg}")
    print(f"done in {time.perf_counter() - t0:.1f}s, {failed} failed")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
