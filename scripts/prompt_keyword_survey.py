#!/usr/bin/env python3
"""Keyword survey: how many prompts mention formula, MW, logP, TPSA, HBD/HBA or related terms at all?
Presence of a word only (an upper bound on usable numeric constraints, not a count of checkable ones). Counts only, prints no prompt text.

    python scripts/prompt_keyword_survey.py data/raw/train_think.jsonl
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]


def split(p):
    m = re.match(r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>", p, re.S)
    return (m.group(1), m.group(2)) if m else ("", p)


CATS = {
    "formula": r"\bC\d{1,3}H\d{1,3}(?:[A-Z][a-z]?\d{0,3})*\b",
    "mol_weight": r"molecular weight|\bMW\b|g/mol|\bDa\b|daltons?",
    "logP/logD": r"log\s?[PD]\b|logD|lipophilic",
    "pKa/ionis": r"pKa|ionis|ioniz|zwitterion|basic centre|basic center|acidic",
    "TPSA/polar": r"TPSA|polar surface",
    "HBD/HBA": r"hydrogen[- ]bond|H-bond|donor|acceptor",
    "rot_bonds": r"rotatable",
    "stereo": r"stereo|chiral|enantiomer|isomeric|diastereo|\(R\)|\(S\)",  # 'isomeric' also matches the answer-format line
    "ring/arom": r"\bring\b|aromatic|heterocycl|bicyclic|macrocycl",
    "func_group": r"\bester|\bamide|carboxyl|sulfonamide|\bnitrile|\bamine|\bphenol|halogen|fluor|chlor|\bketone|\bhydroxyl|\bphosph",
    "isotope/deut": r"deuter|isotop|\b13C|\b14C",
    "salt/solvate": r"hydrochloride|mesylate|sodium|potassium|\bsalt\b|hydrate|solvate",
    "solubility": r"solubil|mg/mL|\bmM\b",
    "potency": r"IC50|\bKi\b|\bKd\b|\bEC50|\bnM\b|\bµM|\buM\b|pIC50",
    "PK(t1/2,CL,Vd,F)": r"half-life|t1/2|clearance|volume of distribution|bioavailab|AUC|\bCmax",
    "metabolism/CYP": r"CYP\d|cytochrome|glucuronid|metaboli",
    "target_named": r"inhibitor|agonist|antagonist|modulator|degrader|binder",
    "permeab/BBB": r"permeab|blood[- ]brain|\bBBB\b|P-gp|efflux",
}
pat = {k: re.compile(v, re.I) for k, v in CATS.items()}
by, sysh = defaultdict(list), defaultdict(Counter)
for r in rows:
    s, u = split(r["prompt"])
    by[r["task_type"]].append(u)
    sysh[r["task_type"]][hashlib.md5(s.encode()).hexdigest()[:6] + f":{len(s)}ch"] += 1
print("distinct system prompts per task:", {k: dict(v) for k, v in sysh.items()})
TT = ["A", "B", "C", "MO", "PD"]
print("\n% of user prompts containing each category")
print("category".ljust(20) + "".join(f"{t}(n={len(by[t])})".rjust(11) for t in TT))
for k in CATS:
    print(k.ljust(20) + "".join(f"{100 * sum(1 for u in by[t] if pat[k].search(u)) / len(by[t]):10.0f}%" for t in TT))
anch = ["formula", "mol_weight", "logP/logD", "TPSA/polar", "HBD/HBA", "rot_bonds"]
for t in ["A", "B", "C", "MO"]:
    c = Counter(sum(bool(pat[k].search(u)) for k in anch) for u in by[t])
    print(t, "number of hard anchors present (formula, MW, logP, TPSA, HBD/HBA, rot bonds):", dict(sorted(c.items())))
