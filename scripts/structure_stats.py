#!/usr/bin/env python3
"""Approval years, similarity noise floor, neighbour structure and clusters, from resolve.py output.

    python scripts/structure_stats.py data/curated/chembl_refs.json data/raw/train_think.jsonl data/raw/val_think.jsonl
"""
import json
import random
import sys
from collections import Counter, defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
refs = {r["name"]: r for r in json.load(open(sys.argv[1]))}
rows = [json.loads(l) for src in sys.argv[2:] for l in open(src)]
rows = [r for r in rows if r["task_type"] != "PD"]

gy = {}
for g in {r["group"] for r in rows}:
    ys = [refs[c.strip()].get("approval_year") for c in g.split(" | ") if c.strip() in refs]
    gy[g] = max([y for y in ys if y] or [0]) or None
rows_per = Counter(r["group"] for r in rows)
bucket = lambda y: "unknown" if y is None else "<2015" if y < 2015 else "2015-2021" if y < 2022 else str(y)
gb, rb = Counter(bucket(y) for y in gy.values()), Counter()
for g, y in gy.items():
    rb[bucket(y)] += rows_per[g]
print("approval year -> (groups, rows); combo groups use their latest member:", {k: (gb[k], rb[k]) for k in sorted(gb)})
per = defaultdict(set)
for r in rows:
    per[r["group"]].add(r["task_type"])
print("groups:", len(per), "| task types per group:", dict(Counter(len(v) for v in per.values())),
      "| rows/group:", dict(Counter(Counter(r["group"] for r in rows).values())))

names = [n for n, r in refs.items() if r.get("parent_smiles")]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(refs[n]["parent_smiles"])) for n in names]
random.seed(0)
pairs = sorted(DataStructs.TanimotoSimilarity(*random.sample(fps, 2)) for _ in range(20000))
q = lambda v, p: v[int(p * (len(v) - 1))]
print(f"null (random drug pair) Tanimoto: mean {sum(pairs) / len(pairs):.3f} p50 {q(pairs, .5):.2f} p95 {q(pairs, .95):.2f} p99 {q(pairs, .99):.2f}")
nn = []
for i, f in enumerate(fps):
    s = DataStructs.BulkTanimotoSimilarity(f, fps)
    s[i] = 0
    nn.append(max(s))
nn.sort()
print(f"nearest-other-drug Tanimoto: p25 {q(nn, .25):.2f} p50 {q(nn, .5):.2f} p75 {q(nn, .75):.2f} p90 {q(nn, .9):.2f} | share with a neighbour >=0.5: {sum(x >= .5 for x in nn) / len(nn):.0%}")
n, d = len(fps), []
for i in range(1, n):
    d.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
for thr in (0.6, 0.7):
    cl = Butina.ClusterData(d, n, thr, isDistData=True)
    print(f"Butina clusters at sim>={1 - thr:.1f}: {len(cl)} for {n} drugs | singletons {sum(len(c) == 1 for c in cl)} | largest {max(len(c) for c in cl)}")
