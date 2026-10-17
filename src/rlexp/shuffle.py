#!/usr/bin/env python3
"""Shuffled-profile control (docs/plan.md section 2 (a) and (d); docs/contracts.md "shuffle.py"): every row is shown the prompt of a
DONOR row from another drug, and its samples are scored against its own reference. A frozen arm that uses the pharmacology loses
delivered similarity (fv_tanimoto) and lift here. One that only emits a good constant does not, and neither does one that keys on a
nuisance cue that the donor shares with the row: its task type or its prompt length. The control keeps task type and length, not
format beyond that (section order and wording still differ between two prompts of one task type).

  python -m rlexp.shuffle write --data data/processed --out data/processed/arms/shuffle [--seed 0] [--splits val]
                                (Val only unless the Test splits are named, which is for after the lock)

Per split, the donors are a ROW-level derangement inside strata:
  stratum    the rows of one task_type, cut into terciles of prompt characters (rank terciles, ties by id). Each stratum is a
             bijection: every prompt is shown to exactly one row, and no row sees its own.
  donor      same task_type (always), a different drug (always) and, whenever the stratum allows it, a different component
             (cluster_id; a null id makes the drug its own component), with a prompt from the same length tercile.
  fallback   if a tercile cannot give every row such a donor, it is merged with the adjacent tercile, and if the merged group
             cannot either, the length bin is dropped (the whole task type is one stratum). Where even that leaves rows with a donor
             from their own component, the number of such rows is the minimum possible. If a task type cannot give every row a donor
             from another drug (one row, or one drug owns more than half of its rows), the split cannot be shuffled and the write
             raises. The strata that needed a fallback are recorded.
  method     an exact assignment (Hungarian algorithm on integer costs: a donor from the own drug is forbidden, one from the own
             component costs one unit) with seeded random tie-breaking. A valid derangement is found whenever one exists, with no
             search over random permutations; the same seed gives the same mapping, another seed gives another valid one.

Every field except `prompt` is kept, in the input order, so gpu/sample_vllm.py --data out reads the files unchanged and the scorer
sees the original ids and references (n_prompt_tokens keeps the row's own count, which the shown prompt matches only within a length
tercile). <split>.meta.json holds the strata sizes, the fallbacks used, balance statistics (mean absolute length gap between the own
and the shown prompt against two random baselines, the share of rows whose shown prompt has the same task type, which is 1.0), the
mapping at drug level and the row mapping by row id; it holds no prompt text.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

from rlexp import arms

SPLITS = ("val", "test_r", "test_t")
BINS = 3                                    # terciles of prompt characters within a task type
UNIT, FORBID, NOISE = 10 ** 7, 10 ** 12, 1000     # integer costs: a donor from the own component costs UNIT, one from the own drug is
                                                  # forbidden, and tie-breaking noise below NOISE never outweighs a UNIT (groups < 10^4 rows)


def hungarian(cost):
    """Minimum-cost perfect assignment of a square integer matrix (rows to columns): the column of each row. Classic O(n^3) shortest
    augmenting path algorithm with potentials; exact, so a feasible assignment is never missed."""
    n = len(cost)
    u, v, p, way = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def _donors(drug, comp, rng):
    """Donor index for each of the rows (given by the drug and component of each) of one stratum, plus the number of rows whose donor
    is from the own drug (must be 0) and from the own component (the minimum possible). A min-cost perfect matching."""
    n = len(drug)
    cost = [[(FORBID if drug[i] == drug[j] else UNIT if comp[i] == comp[j] else 0) + rng.randrange(NOISE) for j in range(n)]
            for i in range(n)]
    donor = hungarian(cost)
    return donor, sum(drug[i] == drug[j] for i, j in enumerate(donor)), sum(comp[i] == comp[j] for i, j in enumerate(donor))


def _terciles(rows, members):
    """The row indices of one task type in BINS lists by rank of prompt length (ties by id), shortest first."""
    order = sorted(members, key=lambda i: (len(rows[i]["prompt"]), rows[i]["id"]))
    bins = [[] for _ in range(BINS)]
    for rank, i in enumerate(order):
        bins[rank * BINS // len(order)].append(i)
    return bins


def _stratify(rows, drug, comp, task_type, seed):
    """({row index: donor row index}, {row index: tercile}, record) for one task type: the tercile ladder of the module docstring."""
    bins = _terciles(rows, [i for i, r in enumerate(rows) if r["task_type"] == task_type])
    groups = [[b] for b in range(BINS) if bins[b]]                  # each group: the tercile numbers it holds
    solved = {}

    def solve(group):
        idx = sorted(i for b in group for i in bins[b])
        rng = random.Random(f"{seed}:{task_type}:{group}")           # one stream per stratum: a fallback elsewhere cannot move it
        donor, same_drug, same_comp = _donors([drug[i] for i in idx], [comp[i] for i in idx], rng)
        return {idx[k]: idx[j] for k, j in enumerate(donor)}, same_drug, same_comp

    i = 0
    while i < len(groups):
        solved[tuple(groups[i])] = solve(groups[i])
        if solved[tuple(groups[i])][1:] == (0, 0) or len(groups) == 1:
            i += 1
            continue
        j = i + 1 if i + 1 < len(groups) else i - 1                  # merge with the adjacent length bin and solve again
        lo, hi = min(i, j), max(i, j)
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
        i = lo
    donor, records = {}, []
    for g in groups:
        mapping, same_drug, same_comp = solved[tuple(g)]
        if same_drug:
            raise ValueError(f"task type {task_type!r}: {sum(len(bins[b]) for b in g)} row(s) in the last stratum cannot all be shown a prompt "
                             f"of another drug ({same_drug} would keep their own drug: one row, or one drug owns more than half of them)")
        donor.update(mapping)
        level = "tercile" if len(g) == 1 else "merged" if len(g) < sum(bool(b) for b in bins) else "task_type"
        records.append({"bins": g, "rows": len(mapping), "level": level, "component_overlap_rows": same_comp})
    return donor, {i: b for b, members in enumerate(bins) for i in members}, \
        {"task_type": task_type, "terciles": [len(b) for b in bins], "groups": records}


def _balance(rows, donor, drug, comp, tercile):
    """Statistics of a donor mapping: the length gap between the own and the shown prompt against two random baselines (a random
    other drug, which is what swapping whole prompts gives, and a random other drug of the same task type), the shares of rows
    that keep the task type, the tercile and the component, and the largest number of rows that share one donor prompt."""
    n = len(rows)
    length = np.array([len(r["prompt"]) for r in rows], float)
    gap = np.abs(length[:, None] - length[None, :])
    other = np.array([[drug[i] != drug[j] for j in range(n)] for i in range(n)])
    kind = np.array([[rows[i]["task_type"] == rows[j]["task_type"] for j in range(n)] for i in range(n)])
    mine = np.array([donor[i] for i in range(n)])
    shown = length[mine]
    return {
        "mean_abs_length_gap": round(float(np.abs(length - shown).mean()), 1),
        "baseline_gap_any_other_drug": round(float(gap[other].mean()), 1),
        "baseline_gap_same_task_type": round(float(gap[other & kind].mean()), 1),
        "length_correlation": round(float(np.corrcoef(length, shown)[0, 1]), 4) if n > 2 and length.std() and shown.std() else None,
        "same_task_type_share": round(float(np.mean([rows[i]["task_type"] == rows[j]["task_type"] for i, j in enumerate(mine)])), 4),
        "same_tercile_share": round(float(np.mean([tercile[i] == tercile[j] for i, j in enumerate(mine)])), 4),
        "same_component_rows": int(sum(comp[i] == comp[j] for i, j in enumerate(mine))),
        "same_drug_rows": int(sum(drug[i] == drug[j] for i, j in enumerate(mine))),
        "max_rows_per_donor": int(np.bincount(mine, minlength=n).max()),
    }


def shuffle_rows(rows, seed=0):
    """(new rows, report): `rows` with `prompt` replaced by the prompt of the donor row, in the input order. The report holds `strata`
    (per task type: tercile sizes and the groups that were solved, with their level and component overlap), `fallbacks` (the groups that
    needed a merge or still share a component), `balance`, `mapping` (drug to the sorted donor drugs) and `row_mapping` (row id to
    donor row id). Raises ValueError for a row without task_type, a repeated id, a drug in two components or an unshuffleable task type."""
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("the rows repeat an id")
    if any("task_type" not in r for r in rows):
        raise ValueError("a row lacks task_type")
    drug = [r["group"] for r in rows]
    own = {}
    for r in rows:                                                    # a null cluster_id makes the drug its own component
        cid = r.get("cluster_id")
        c = ("drug", r["group"]) if cid is None else ("cluster", cid)
        if own.setdefault(r["group"], c) != c:
            raise ValueError(f"drug {r['group'][:60]!r} has rows in two components")
    comp = [own[d] for d in drug]
    donor, tercile, strata = {}, {}, []
    for task_type in sorted({r["task_type"] for r in rows}):
        part, bins, record = _stratify(rows, drug, comp, task_type, seed)
        donor.update(part)
        tercile.update(bins)
        strata.append(record)
    new = [{**r, "prompt": rows[donor[i]]["prompt"]} for i, r in enumerate(rows)]
    report = {"strata": strata,
              "fallbacks": [{"task_type": s["task_type"], **g} for s in strata for g in s["groups"]
                            if g["level"] != "tercile" or g["component_overlap_rows"]],
              "balance": _balance(rows, donor, drug, comp, tercile),
              "mapping": {d: sorted({drug[donor[i]] for i in range(len(rows)) if drug[i] == d}) for d in sorted(set(drug))},
              "row_mapping": {rows[i]["id"]: rows[donor[i]]["id"] for i in range(len(rows))}}
    check(rows, new, report)
    return new, report


def check(rows, new, report):
    """Raise ValueError unless `new` is a valid shuffle of `rows` (the invariants of the module docstring), so a bug can never write a
    control that looks like one: only the prompt changed, each prompt is the donor row's, every donor is another drug of the same task
    type, no donor prompt is used twice, and the report's balance numbers agree. One message per invariant."""
    by_id = {r["id"]: r for r in rows}
    mapping = report["row_mapping"]
    if [r["id"] for r in new] != [r["id"] for r in rows] or sorted(mapping.values()) != sorted(by_id):
        raise ValueError("the shuffle is not a bijection of the rows")
    for old, now in zip(rows, new):
        donor = by_id[mapping[old["id"]]]
        if {k: v for k, v in old.items() if k != "prompt"} != {k: v for k, v in now.items() if k != "prompt"}:
            raise ValueError("a field other than the prompt changed")
        if now["prompt"] != donor["prompt"]:
            raise ValueError("a row was not shown the prompt of its donor row")
        if donor["group"] == old["group"]:
            raise ValueError("a row was shown a prompt of its own drug")
        if donor["task_type"] != old["task_type"]:
            raise ValueError("a row was shown a prompt of another task type")
    b = report["balance"]
    if b["same_task_type_share"] != 1.0 or b["same_drug_rows"] or b["max_rows_per_donor"] != 1:
        raise ValueError("the balance statistics contradict the shuffle")


def write(data_dir, out_dir, seed=0, splits=SPLITS):
    """Shuffle every named split, then write them: a split that cannot be shuffled raises before any file of the call is written."""
    done = []
    for split in splits:
        done.append((split, *shuffle_rows(arms._read(Path(data_dir) / f"{split}.jsonl"), seed)))
    for split, new, report in done:
        arms._write(Path(out_dir) / f"{split}.jsonl", new, lines=True)
        arms._write(Path(out_dir) / f"{split}.meta.json", {
            "file": f"{split}.jsonl", "rows": len(new), "control": "shuffled_profile", "seed": seed,
            "design": "row-level derangement within task_type x prompt-length tercile: a donor row of another drug and, when the "
                      "stratum allows it, another component; keeps task type and length, not format beyond that", **report})
        b = report["balance"]
        print(f"{split}.jsonl: {len(new)} rows | same task type {b['same_task_type_share']:.3f} | mean length gap {b['mean_abs_length_gap']:.0f} "
              f"chars (random other drug {b['baseline_gap_any_other_drug']:.0f}, same task type {b['baseline_gap_same_task_type']:.0f}) | "
              f"same tercile {b['same_tercile_share']:.3f} | length correlation {b['length_correlation']} | rows with a donor from their own "
              f"component {b['same_component_rows']} | strata needing a fallback {len(report['fallbacks'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write the shuffled-profile prompt files")
    w.add_argument("--data", default="data/processed")
    w.add_argument("--out", default="data/processed/arms/shuffle")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--splits", default="val", help="comma-separated; the default is Val only: name test_r or test_t only after the lock")
    a = ap.parse_args(argv)
    try:
        write(a.data, a.out, a.seed, tuple(s for s in a.splits.split(",") if s))
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
