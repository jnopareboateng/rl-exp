#!/usr/bin/env python3
"""Approved-drug pool for the class-information check: standardise, merge duplicates, drop dataset drugs, filter leakage.

    .venv/bin/python scripts/build_pool.py [--cache data/raw/chembl_pool] [--data data/processed]

Reads the cache of scripts/fetch_chembl_pool.py (every file checked against manifest.json) and, from
data/processed/{train,val,test_r,test_t}.jsonl, only group, ref_smiles and ref_alt_smiles (no prompt text is kept).
Writes <cache>/pool.jsonl (one row per surviving drug) and <cache>/train_facts.jsonl (the same facts for the Train
drugs ChEMBL knows, with the Train reference as `smiles`; input of scripts/class_ceiling.py), and records their sha256, row
counts and the sha256 of the four processed files it read under `derived` in <cache>/manifest.json (the raw `files` entries
stay as fetch_chembl_pool.py wrote them). Prints counts only.

  1. A record's structure is its hierarchy parent's SMILES (the parent record's when it has one, else its own) through
     chem.standardize (largest fragment, neutral). Records with no SMILES or that do not standardise are counted and skipped.
  2. Records with the same chem.conn merge into one drug (salt forms, enantiomer parents): the lowest ChEMBL parent id
     represents it, flags are OR-ed, first_approval is the earliest, list facts are unions.
  3. Drugs whose conn equals that of any reference (or alternate) in the four processed files are dropped.
  4. Leakage filter against the held-out drugs (Val, Test-R, Test-T), the rule of rlexp.data_prep: bit Tanimoto
     (chem.tanimoto_bits) >= data_prep.SIM to a held-out drug, else the same non-empty Murcko scaffold.
     data/processed/split_report.json gives counts, not names, for the buffer drugs, so the buffer is not a reference
     set; the number of kept drugs a buffer check would also remove is printed from data/processed/refs.jsonl.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rlexp import chem
from rlexp.data_prep import SIM, SPLITS

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = ("val", "test_r", "test_t")
FLAGS = ("withdrawn_flag", "therapeutic_flag", "oral", "parenteral", "topical")


def num(chembl_id):
    """CHEMBL25 -> 25: ChEMBL ids order numerically, not as strings."""
    return int(chembl_id.removeprefix("CHEMBL"))


def load_cache(cache):
    """(manifest, {endpoint: records}) of the cached pages, each file checked against its sha256 in manifest.json."""
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    records = defaultdict(list)
    for name, meta in manifest["files"].items():
        raw = (cache / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError(f"{name}: sha256 differs from manifest.json; run scripts/fetch_chembl_pool.py again")
        if name != "status.json":
            records[name.rsplit("_", 1)[0]] += json.loads(raw)["records"]
    return manifest, records


def _uniq(items):
    """Distinct dicts in a stable order."""
    return [json.loads(s) for s in sorted({json.dumps(d, sort_keys=True) for d in items})]


def merge(molecules, parents):
    """Merge fetched records into drugs by connectivity (steps 1 and 2 of the module docstring).
    Returns (rows, mols, stats): rows in ChEMBL-id order without mechanisms and indications, mols parallel to rows, and
    stats = counts of records that failed ("no_smiles", "not_standardised"), standardised records and their parents."""
    by_id = {r["id"]: r for r in parents + molecules}
    seen, groups, stats = {}, defaultdict(list), Counter()
    for rec in molecules:
        p = by_id.get(rec["parent_id"])
        smiles = p["smiles"] if p and p["smiles"] else rec["smiles"]
        if not smiles:
            stats["no_smiles"] += 1
            continue
        if smiles not in seen:
            mol = chem.standardize(smiles)
            seen[smiles] = (mol, chem.conn(mol) if mol is not None else None)
        mol, key = seen[smiles]
        if key is None:
            stats["not_standardised"] += 1
            continue
        groups[key].append((rec, mol))
    stats["standardised"] = sum(len(g) for g in groups.values())
    stats["parents"] = len({rec["parent_id"] for g in groups.values() for rec, _ in g})
    out = []
    for key, group in groups.items():
        group.sort(key=lambda t: (num(t[0]["parent_id"]), num(t[0]["id"])))
        mol, members = group[0][1], [rec for rec, _ in group]
        pids = sorted({r["parent_id"] for r in members}, key=num)
        recs = list({r["id"]: r for r in [by_id[p] for p in pids if p in by_id] + members}.values())  # parents first
        years = [r["first_approval"] for r in recs if r["first_approval"] is not None]
        row = {"chembl_id": pids[0], "chembl_ids": sorted({r["id"] for r in recs} | set(pids), key=num),
               "pref_name": next((r["pref_name"] for r in recs if r["pref_name"]), None),
               "synonyms": list(dict.fromkeys(s["name"] for r in recs for s in r["synonyms"])),
               "smiles": chem.canonical(mol), "conn": key, "scaffold": chem.scaffold(mol), "n_heavy": mol.GetNumHeavyAtoms(),
               "first_approval": min(years) if years else None, **{k: any(r[k] for r in recs) for k in FLAGS},
               "atc": sorted({c for r in recs for c in r["atc"]})}
        out.append((row, mol))
    out.sort(key=lambda t: num(t[0]["chembl_id"]))
    return [r for r, _ in out], [m for _, m in out], stats


def attach_facts(rows, mechanisms, targets, indications):
    """Add `mechanisms` and `indications` to every row, from the records whose molecule or parent id is one of its ids."""
    tgt = {t["target_chembl_id"]: t for t in targets}
    mech, ind = defaultdict(list), defaultdict(list)
    for m in mechanisms:
        for i in {m["molecule_chembl_id"], m["parent_molecule_chembl_id"]} - {None}:
            mech[i].append({"mechanism_of_action": m["mechanism_of_action"], "action_type": m["action_type"],
                            "target_chembl_id": m["target_chembl_id"],
                            "target_name": (tgt.get(m["target_chembl_id"]) or {}).get("pref_name"),
                            "target_type": (tgt.get(m["target_chembl_id"]) or {}).get("target_type")})
    for x in indications:
        for i in {x["molecule_chembl_id"], x["parent_molecule_chembl_id"]} - {None}:
            ind[i].append({k: x[k] for k in ("mesh_heading", "efo_term", "max_phase_for_ind")})
    for r in rows:
        r["mechanisms"] = _uniq(m for i in r["chembl_ids"] for m in mech.get(i, ()))
        r["indications"] = _uniq(x for i in r["chembl_ids"] for x in ind.get(i, ()))


def read_dataset(data_dir):
    """{split: {group: (ref_smiles, [alternate smiles])}} from the processed row files; prompt text is never kept."""
    out = {}
    for split in SPLITS:
        out[split] = {}
        with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[split][r["group"]] = (r["ref_smiles"], r["ref_alt_smiles"])
    return out


def dataset_conns(dataset):
    """{conn: (split, group)} for every reference in the dataset and its alternates, re-derived with rlexp.chem."""
    out = {}
    for split, groups in dataset.items():
        for group, (ref, alts) in groups.items():
            for smiles in (ref, *alts):
                mol = chem.standardize(smiles)
                key = chem.conn(mol) if mol is not None else None
                if key is None:
                    raise ValueError(f"a {split} reference does not standardize")
                out.setdefault(key, (split, group))
    return out


def held_out(dataset):
    """(mols, non-empty Murcko scaffolds) of the Val, Test-R and Test-T references."""
    mols = [chem.standardize(ref) for s in HELD_OUT for ref, _ in dataset[s].values()]
    return mols, {chem.scaffold(m) for m in mols} - {""}


def leak_reason(sim, shares_scaffold):
    """The rule of rlexp.data_prep for one candidate: "tanimoto" when its highest bit Tanimoto to a held-out drug is >= SIM,
    else "scaffold" when it shares a non-empty Murcko scaffold with one, else None."""
    return "tanimoto" if sim >= SIM else "scaffold" if shares_scaffold else None


def check(mol, scaffold, held_mols, held_scaffolds):
    """leak_reason of one candidate against a set of held-out drugs (their mols and non-empty scaffolds)."""
    top = max((chem.tanimoto_bits(mol, h) for h in held_mols), default=0.0)  # ponytail: one chem.tanimoto_bits call per held-out drug (about 1 min for 1,800 x 91); cache fingerprints if the held-out set grows
    return leak_reason(top, bool(scaffold) and scaffold in held_scaffolds)


def train_row(group, ref_smiles, row):
    """A pool-style row for a Train drug: ChEMBL's facts, the Train reference as the structure."""
    mol = chem.standardize(ref_smiles)
    return {**row, "group": group, "smiles": chem.canonical(mol), "conn": chem.conn(mol), "scaffold": chem.scaffold(mol),
            "n_heavy": mol.GetNumHeavyAtoms()}


def write_jsonl(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_derived(cache, data_dir, manifest, n_pool, n_train):
    """Add the `derived` record to manifest.json (same atomic, compact format fetch_chembl_pool.py writes it in)."""
    manifest["derived"] = {
        "built_by": "scripts/build_pool.py",
        "files": {n: {"records": k, "sha256": _sha256(cache / n)} for n, k in (("pool.jsonl", n_pool), ("train_facts.jsonl", n_train))},
        "inputs": {f"{s}.jsonl": _sha256(data_dir / f"{s}.jsonl") for s in SPLITS}}
    tmp = cache / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache / "manifest.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data/raw/chembl_pool")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed")
    a = ap.parse_args(argv)
    try:
        manifest, rec = load_cache(a.cache)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    rows, mols, st = merge(rec["molecule"], rec["parent"])
    attach_facts(rows, rec["mechanism"], rec["target"], rec["indication"])
    n_fetched, dup = len(rec["molecule"]), st["standardised"] - len(rows)
    print(f"fetched {n_fetched} records ({manifest['chembl_release']}); standardised {st['standardised']} "
          f"(no SMILES {st['no_smiles']}, not standardised {st['not_standardised']})")
    print(f"duplicates {dup} (further forms of one parent {st['standardised'] - st['parents']}, other parents with the same "
          f"connectivity {st['parents'] - len(rows)}) -> {len(rows)} drugs")

    dataset = read_dataset(a.data)
    keys = dataset_conns(dataset)
    hits = defaultdict(lambda: defaultdict(list))  # split -> group -> indices of the rows that carry its connectivity
    for i, r in enumerate(rows):
        if r["conn"] in keys:
            split, group = keys[r["conn"]]
            hits[split][group].append(i)
    in_ds = {i for groups in hits.values() for idx in groups.values() for i in idx}
    train = {}
    for group, idx in hits["train"].items():
        ref = dataset["train"][group][0]
        best = next((i for i in idx if rows[i]["conn"] == chem.conn(chem.standardize(ref))), idx[0])  # the reference itself over an alternate
        train[group] = train_row(group, ref, rows[best])
    print(f"already in the dataset {len(in_ds)}; dataset drugs found in ChEMBL: "
          + ", ".join(f"{s} {len(hits[s])} of {len(dataset[s])}" for s in SPLITS))

    held_mols, held_scaffolds = held_out(dataset)
    reason = {i: check(mols[i], rows[i]["scaffold"], held_mols, held_scaffolds) for i in range(len(rows)) if i not in in_ds}
    n_t, n_s = sum(v == "tanimoto" for v in reason.values()), sum(v == "scaffold" for v in reason.values())
    kept = [i for i, v in reason.items() if v is None]
    pool = [rows[i] for i in kept]
    print(f"held-out reference set: Val, Test-R, Test-T ({len(held_mols)} drugs; buffer not used); excluded by bit Tanimoto >= {SIM}: "
          f"{n_t}; excluded by a shared scaffold only: {n_s}; kept {len(pool)}")
    print(f"kept with a mechanism {sum(bool(r['mechanisms']) for r in pool)}, with an ATC code {sum(bool(r['atc']) for r in pool)}, "
          f"with an indication {sum(bool(r['indications']) for r in pool)}, with a mechanism or an ATC code "
          f"{sum(bool(r['mechanisms'] or r['atc']) for r in pool)}; withdrawn {sum(r['withdrawn_flag'] for r in pool)}, "
          f"no first_approval {sum(r['first_approval'] is None for r in pool)}, SMILES under 8 characters "
          f"{sum(len(r['smiles']) < 8 for r in pool)}")

    refs = a.data / "refs.jsonl"  # sensitivity only: how many kept drugs a buffer check would also remove
    if refs.exists():
        lines = [json.loads(l) for l in refs.read_text(encoding="utf-8").splitlines() if l.strip()]
        buf = [chem.standardize(r["ref_smiles"]) for r in lines if r["split"] == "buffer"]
        buf_scaffolds = {chem.scaffold(m) for m in buf} - {""}
        extra = sum(check(mols[i], rows[i]["scaffold"], buf, buf_scaffolds) is not None for i in kept)
        print(f"buffer drugs in refs.jsonl {len(buf)}; kept drugs a buffer check would also remove: {extra}")

    print(f"Train drugs with ChEMBL facts {len(train)} of {len(dataset['train'])}; with a mechanism or an ATC code "
          f"{sum(bool(r['mechanisms'] or r['atc']) for r in train.values())}")
    write_jsonl(a.cache / "pool.jsonl", pool)
    write_jsonl(a.cache / "train_facts.jsonl", [train[g] for g in sorted(train)])
    record_derived(a.cache, a.data, manifest, len(pool), len(train))
    print(f"wrote {a.cache / 'pool.jsonl'} ({len(pool)} rows) and {a.cache / 'train_facts.jsonl'} ({len(train)} rows); "
          f"their sha256 are in the manifest")


if __name__ == "__main__":
    main()
