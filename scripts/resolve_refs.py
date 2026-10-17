#!/usr/bin/env python3
"""Resolve each non-PD drug name to a ChEMBL structure and compare it with the SFT answer.

    python scripts/resolve_refs.py data/curated/chembl_refs.json data/raw/train_think.jsonl data/raw/val_think.jsonl

Needs network (ChEMBL REST, public). Sends only drug names. Prints statistics only.
Parent structure = largest fragment, uncharged. 'Same connectivity' = InChIKey first block.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from rlexp.chem import extract_answer as _extract_smiles

RDLogger.DisableLog("rdApp.*")
out, srcs = sys.argv[1], sys.argv[2:]
rows = [json.loads(l) for src in srcs for l in open(src)]
rows = [r for r in rows if r["task_type"] != "PD"]
groups = sorted({r["group"] for r in rows})
comps = sorted({c.strip() for g in groups for c in g.split(" | ")})
print("non-PD rows", len(rows), "groups", len(groups), "components", len(comps))
SALT = (r"\b(hydrochloride|dihydrochloride|hydrobromide|potassium|sodium|calcium|magnesium|mesylate|fumarate|maleate|"
        r"tartrate|citrate|succinate|sulfate|phosphate|tosylate|besylate|acetate|hemihydrate|monohydrate|hydrate|"
        r"disodium|dipotassium|mesilate|bromide|chloride|oxalate|lactate)\b")


def get(u):
    err = None
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"Accept": "application/json"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:
            err = e
            time.sleep(1.5 * (i + 1))
    return {"error": str(err)}


def lookup(name):
    d = get("https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?limit=6&q=" + urllib.parse.quote(name))
    if "error" in d:
        return {"name": name, "status": "api_error"}
    ms = d.get("molecules", [])
    low = name.lower()
    stripped = re.sub(SALT, "", low).strip()
    pn = lambda m: (m.get("pref_name") or "").lower()
    pick = q = None
    for m in ms:
        if pn(m) == low:
            pick, q = m, "exact"
            break
    if not pick:
        for m in ms:
            if pn(m) == stripped:
                pick, q = m, "stripped"
                break
    if not pick:
        for m in ms:
            if pn(m).split(" ")[0] == stripped.split(" ")[0] and m.get("molecule_structures"):
                pick, q = m, "first_word"
                break
    if not pick:
        return {"name": name, "status": "no_match", "n_hits": len(ms)}
    s = (pick.get("molecule_structures") or {}).get("canonical_smiles")
    return {"name": name, "status": "ok", "match": q, "chembl_id": pick.get("molecule_chembl_id"),
            "pref_name": pick.get("pref_name"), "first_approval": pick.get("first_approval"),
            "max_phase": pick.get("max_phase"), "smiles_raw": s}


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lookup, comps))
lg, un = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
for r in res:
    r["parent_smiles"] = None
    if r.get("smiles_raw"):
        m = Chem.MolFromSmiles(r["smiles_raw"])
        if m:
            r["parent_smiles"] = Chem.MolToSmiles(un.uncharge(lg.choose(m)))


def fda_date(name):
    """Earliest FDA original approval (YYYYMMDD) over every Drugs@FDA application listing this drug as an
    active ingredient (both search fields, up to 1,000 applications each). None if none found or unreachable."""
    base = re.sub(SALT, "", name.lower()).strip().upper()
    best = None
    for field in ("openfda.generic_name", "products.active_ingredients.name"):
        for q in dict.fromkeys([base, name.upper()]):
            time.sleep(0.35)  # openFDA allows 240 requests/min without a key
            u = ("https://api.fda.gov/drug/drugsfda.json?search=" + field + ":"
                 + urllib.parse.quote('"' + q + '"') + "&limit=1000")
            try:
                d = json.load(urllib.request.urlopen(u, timeout=60))
            except (urllib.error.URLError, TimeoutError):  # 404 = no match
                continue
            for app in d.get("results", []):
                names = [i.get("name", "").upper() for p in app.get("products", []) for i in p.get("active_ingredients", [])]
                if not any(base in n for n in names):  # identity check: the drug must be an active ingredient
                    continue
                for s in app.get("submissions", []):
                    when = s.get("submission_status_date")
                    if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP" and when:
                        best = when if best is None or when < best else best
    return best


# ChEMBL leaves recent approvals undated (and some phase 3); fill those from openFDA
undated = [r for r in res if r["status"] == "ok" and r.get("first_approval") is None]
with ThreadPoolExecutor(3) as ex:
    for r, d in zip(undated, ex.map(lambda r: fda_date(r["name"]), undated)):
        r["fda_first_approval"] = d
for r in res:
    fda = r.get("fda_first_approval")
    r["approval_year"] = r.get("first_approval") or (int(fda[:4]) if fda else None)
json.dump(res, open(out, "w"), indent=1)
print("status", dict(Counter(r["status"] for r in res)))
print("match type", dict(Counter(r.get("match") for r in res if r["status"] == "ok")))
ok = [r for r in res if r.get("parent_smiles")]
print("components with parent structure:", len(ok), "of", len(comps))
bucket = lambda y: "unknown" if y is None else "<2015" if y < 2015 else "2015-2021" if y < 2022 else str(y)
print("approval year (ChEMBL, else openFDA):", dict(sorted(Counter(bucket(r["approval_year"]) for r in ok).items())))

ref = {r["name"]: r for r in res}
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
sim, exact, n, resolved, combo = defaultdict(list), Counter(), Counter(), Counter(), Counter()
for r in rows:
    t = r["task_type"]
    n[t] += 1
    if " | " in r["group"]:
        combo[t] += 1
        continue
    rr = ref.get(r["group"])
    if not rr or not rr.get("parent_smiles"):
        continue
    resolved[t] += 1
    ans = _extract_smiles(r["completion"])
    am = Chem.MolFromSmiles(ans) if ans else None
    if am is None:
        continue
    am = un.uncharge(lg.choose(am))
    rm = Chem.MolFromSmiles(rr["parent_smiles"])
    sim[t].append(DataStructs.TanimotoSimilarity(gen.GetFingerprint(am), gen.GetFingerprint(rm)))
    exact[t] += Chem.MolToInchiKey(am).split("-")[0] == Chem.MolToInchiKey(rm).split("-")[0]
print("\nSFT answer vs ChEMBL structure (single-drug groups only)")
for t in ["A", "B", "C", "MO"]:
    v = sorted(sim[t])
    if v:
        q = lambda p: v[int(p * (len(v) - 1))]
        print(f"{t}: rows {n[t]} combos-excluded {combo[t]} resolved {resolved[t]} compared {len(v)} | Tanimoto p10 {q(.1):.2f} "
              f"p50 {q(.5):.2f} | same connectivity: {exact[t] / len(v):.0%} of compared, {exact[t] / (n[t] - combo[t]):.0%} of all single-drug rows")
