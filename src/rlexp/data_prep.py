#!/usr/bin/env python3
"""Data foundation (docs/contracts.md "Data products"): references, approval years, leakage-safe splits, prompt files.

    python -m rlexp.data_prep [--seed 0]

Reads data/raw/train_think.jsonl + val_think.jsonl (pooled), data/curated/chembl_refs.json and, if present, the hand files
data/curated/{manual_refs,approval_audit,exposure}.json (read here, never written).
Writes data/processed/{refs,train,val,test_r,test_t}.jsonl and split_report.json, staged first and renamed together.
Deterministic: the same inputs and seed give byte-identical outputs.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

from rlexp import chem

ROOT = Path(__file__).resolve().parents[2]
SEED, SIM, N_TEST_R, N_VAL, TEST_T_YEAR = 0, 0.4, 30, 20, 2025
N_VARIANTS, VARIANT_SEED = 20, 1  # extractability screen; RDKit re-seeds MolToRandomSmilesVect only for seeds > 0
THINK, ASSISTANT = "<think>\n", "<|im_start|>assistant\n"
SPLITS = ("train", "val", "test_r", "test_t")
HAND = ("manual_refs.json", "approval_audit.json", "exposure.json")  # read-only inputs, owned by other workers
BUCKETS = ("<2015", "2015-2021", "2022", "2023", "2024", "2025", "2026")


def load_rows(paths) -> tuple[list[dict], dict]:
    """Pool the row files. Drop PD rows, then rows of combination groups (" | "). Returns (kept, dropped)."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:  # iterate lines, never str.splitlines (it splits on U+2028 etc.)
            rows += [json.loads(line) for line in f if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows), "duplicate row ids across the input files"
    pd = [r for r in rows if r["task_type"] == "PD"]
    rest = [r for r in rows if r["task_type"] != "PD"]
    combo = [r for r in rest if " | " in r["group"]]
    kept = [r for r in rest if " | " not in r["group"]]
    names = lambda rs: {"rows": len(rs), "groups": sorted({r["group"] for r in rs})}
    return kept, {"pd": names(pd), "combination": names(combo)}


def _mol(smiles):
    """(standardised Mol, connectivity key), or (None, None) when the string does not parse or has no key."""
    m = chem.standardize(smiles)
    c = chem.conn(m) if m is not None else None
    return (m, c) if c else (None, None)


def _sft(rows):
    """(connectivity, Mol) of the majority standardised answer over a group's completions; (None, None) if none parse."""
    votes, mols = Counter(), {}
    for r in rows:
        a = chem.extract_answer(r["completion"])
        m, c = _mol(a) if a else (None, None)
        if m is not None:
            key = (c, chem.canonical(m))
            votes[key] += 1
            mols[key] = m
    per_conn = Counter()
    for (c, _), n in votes.items():
        per_conn[c] += n
    if not votes:
        return None, None
    c = min(per_conn, key=lambda k: (-per_conn[k], k))  # ties: smallest key, so the order of rows never matters
    return c, mols[min((k for k in votes if k[0] == c), key=lambda k: (-votes[k], k[1]))]


def _year(pool_entry, manual, audit):
    """Final approval year and its source: manual > audit (earliest global) > pool (ChEMBL, else openFDA)."""
    for y, src in ((manual.get("approval_year"), "manual"), (audit.get("earliest_global_year"), "audit"),
                   (pool_entry.get("approval_year"), "chembl" if pool_entry.get("first_approval") else "openfda")):
        if y is not None:
            return int(y), src
    return None, None


def _extractable(mol, conn, smiles) -> bool:
    """chem.extract_answer recovers the molecule from its canonical SMILES and from N_VARIANTS seeded random variants of it."""
    for s in (smiles, *Chem.MolToRandomSmilesVect(mol, N_VARIANTS, randomSeed=VARIANT_SEED)):
        a = chem.extract_answer("</think>\n" + s)
        m = chem.standardize(a) if a else None
        if m is None or (chem.canonical(m) != smiles and chem.conn(m) != conn):  # same canonical SMILES needs no InChI
            return False
    return True


def build_refs(rows, pool, manual=None, audit=None, exposure=None) -> tuple[dict, dict]:
    """One reference row per group in `rows` (single-drug, non-PD). Returns ({group: ref}, notes for the report).

    Structure: manual > ChEMBL (when the SFT majority answer agrees or is missing) > SFT answer (when ChEMBL has none).
    A manual entry with smiles null declares the group unresolvable: excluded, never falling back to ChEMBL or the SFT
    answer. ChEMBL and SFT disagreeing with no manual entry excludes the group, as does a reference the answer extractor
    cannot recover (see _extractable). Year: see _year. Exposure (PubChem record year, PubMed count) and the manual
    `convention` are copied through. cluster_id and split are filled in by assign_splits."""
    manual, audit, exposure = manual or {}, audit or {}, exposure or {}
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    notes = {"disagreements": [], "year_overrides": [], "exposure_matched": len(set(exposure) & set(by)),
             "manual_effect": {k: [] for k in ("identical", "changed_smiles_only", "changed_connectivity", "filled_gap", "unresolvable")},
             "unused_manual": sorted(set(manual) - set(by)), "unused_audit": sorted(set(audit) - set(by))}
    refs = {}
    for g in sorted(by):
        e, aud, man = pool.get(g) or {}, audit.get(g) or {}, manual.get(g)  # man: None, or {"smiles": str | None, ...}
        ok = e.get("status") == "ok"
        ch, ch_conn = _mol(e.get("parent_smiles")) if ok else (None, None)
        sft_conn, sft = _sft(by[g])
        agrees = None if ch is None or sft is None else ch_conn == sft_conn
        man_mol, man_conn = _mol(man["smiles"]) if man is not None else (None, None)
        if man is not None and man["smiles"] is not None and man_mol is None:
            raise ValueError(f"manual_refs[{g!r}].smiles does not parse: {man['smiles']!r}")
        unresolvable = man is not None and man["smiles"] is None
        convention = (man or {}).get("convention")
        assert convention in (None, "ligand_only"), f"manual_refs[{g!r}].convention must be 'ligand_only' or absent: {convention!r}"
        if ch is not None and agrees is not False:  # what the data alone give
            base = ch, ch_conn, "chembl"
        elif ch is None and sft is not None:
            base = sft, sft_conn, "sft_answer"
        else:
            base = None, None, None
        mol, conn, src = (None,) * 3 if unresolvable else (man_mol, man_conn, "manual") if man_mol is not None else base
        if man is not None:  # what the entry did to the reference the data alone would have given
            how = ("unresolvable" if unresolvable else "filled_gap" if base[0] is None
                   else "changed_connectivity" if base[1] != man_conn
                   else "changed_smiles_only" if chem.canonical(base[0]) != chem.canonical(man_mol) else "identical")
            notes["manual_effect"][how].append(g)
        year, ysrc = _year(e, man or {}, aud)
        ref = dict.fromkeys(("ref_smiles", "ref_inchikey", "ref_conn", "ref_scaffold"))
        if mol is not None:
            ref = {"ref_smiles": chem.canonical(mol), "ref_inchikey": chem.inchikey(mol), "ref_conn": conn,
                   "ref_scaffold": chem.scaffold(mol)}
            assert _mol(ref["ref_smiles"])[1] == conn, f"{g!r}: the reference does not keep its connectivity through its SMILES"
        if mol is None:
            reason = "manual_unresolvable" if unresolvable else "unresolved_disagreement" if ch is not None else "no_structure"
        elif not _extractable(mol, conn, ref["ref_smiles"]):
            reason = "unextractable_reference"
        else:
            reason = "no_approval_year" if year is None else None
        if agrees is False:
            notes["disagreements"].append({
                "group": g, "chembl_smiles": chem.canonical(ch), "sft_smiles": chem.canonical(sft),
                "tanimoto": round(chem.tanimoto(ch, sft), 3), "resolved_by": "manual" if man_mol is not None else None})
        if ysrc in ("manual", "audit") and year != e.get("approval_year"):
            notes["year_overrides"].append({"group": g, "pool_year": e.get("approval_year"), "year": year, "source": ysrc,
                                            **({"confidence": aud.get("confidence")} if ysrc == "audit" else {})})
        refs[g] = {"group": g, "components": g.split(" | "), **ref, "ref_source": src,
                   "chembl_id": e.get("chembl_id") if ok else None, "approval_year": year, "approval_source": ysrc,
                   "sft_agrees": agrees, "cluster_id": -1, "split": "excluded", "exclude_reason": reason, "n_rows": len(by[g]),
                   "n_heavy": mol.GetNumHeavyAtoms() if mol is not None else None, "convention": convention,
                   "pubchem_create_year": (exposure.get(g) or {}).get("pubchem_create_year"),
                   "pubmed_count": (exposure.get(g) or {}).get("pubmed_count")}
    return refs, notes


def assign_splits(refs, seed=SEED, n_test_r=N_TEST_R, n_val=N_VAL) -> dict:
    """Set cluster_id and split on every ref in place. Returns the split facts for the report.

    Clusters = connected components over the included drugs, with an edge when Tanimoto >= SIM or when the two drugs share
    an identical non-empty Murcko scaffold (an answer that earns scaffold credit on one earns it on the other).
    Test-T = final year >= TEST_T_YEAR; the rest of a Test-T drug's component is buffer. Remaining whole components, in a
    seeded shuffle, fill Test-R then Val greedily (a component that would overshoot is skipped); everything left is Train."""
    inc = [g for g in sorted(refs) if refs[g]["exclude_reason"] is None]
    mols = [chem.standardize(refs[g]["ref_smiles"]) for g in inc]  # the same parse every downstream scorer does
    assert all(m is not None for m in mols), [g for g, m in zip(inc, mols) if m is None]
    n = len(inc)
    sim = np.eye(n)
    for i in range(n):  # ponytail: O(n^2) chem.tanimoto calls (~12 s at 340 drugs); cache fingerprints if n reaches thousands
        for j in range(i):
            sim[i, j] = sim[j, i] = chem.tanimoto(mols[i], mols[j])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    ii, jj = np.nonzero(np.triu(sim >= SIM, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        union(i, j)
    first = {}
    for i, g in enumerate(inc):
        if refs[g]["ref_scaffold"]:  # acyclic molecules have no scaffold and link to nothing
            union(first.setdefault(refs[g]["ref_scaffold"], i), i)
    members = defaultdict(list)
    for i, g in enumerate(inc):
        members[find(i)].append(g)
    comps = sorted(members.values())  # members are sorted, so components are ordered by their first drug
    for cid, c in enumerate(comps):
        for g in c:
            refs[g]["cluster_id"] = cid

    rest = []
    for c in comps:
        if any(refs[g]["approval_year"] >= TEST_T_YEAR for g in c):
            for g in c:
                refs[g]["split"] = "test_t" if refs[g]["approval_year"] >= TEST_T_YEAR else "buffer"
        else:
            rest.append(c)
    random.Random(seed).shuffle(rest)
    targets = {}
    for name, size in (("test_r", n_test_r), ("val", n_val)):
        got, skipped = 0, []
        for c in rest:
            if got + len(c) <= size:
                got += len(c)
                for g in c:
                    refs[g]["split"] = name
            else:
                skipped.append(c)
        rest = skipped
        targets[name] = {"target": size, "got": got, "exact": got == size}
    for c in rest:
        for g in c:
            refs[g]["split"] = "train"

    at = {s: [i for i, g in enumerate(inc) if refs[g]["split"] == s] for s in SPLITS}
    scaf = {s: {refs[inc[i]]["ref_scaffold"] for i in at[s]} - {""} for s in SPLITS}
    leak, shared = {}, {}
    for a, b in itertools.combinations(SPLITS, 2):
        shared[f"{a}|{b}"] = len(scaf[a] & scaf[b])
        assert not shared[f"{a}|{b}"], f"leakage: {a} and {b} share scaffolds {sorted(scaf[a] & scaf[b])}"
        if at[a] and at[b]:
            sub = sim[np.ix_(at[a], at[b])]
            i, j = np.unravel_index(sub.argmax(), sub.shape)
            leak[f"{a}|{b}"] = {"max": round(float(sub[i, j]), 4), "pair": [inc[at[a][i]], inc[at[b][j]]]}
            assert sub[i, j] < SIM, f"leakage: {a} and {b} hold a pair at Tanimoto {sub[i, j]:.3f}: {leak[f'{a}|{b}']['pair']}"
    big = sorted(comps, key=lambda c: (-len(c), c))[:3]
    with_train = {s: sum(refs[inc[i]]["ref_scaffold"] in scaf["train"] for i in at[s]) for s in ("val", "test_r", "test_t")}
    return {"seed": seed, "targets": targets, "max_cross_tanimoto": leak,
            "scaffold_overlap": {"drugs_sharing_a_scaffold_with_train": with_train, "shared_scaffolds": shared},
            "clusters": {"n": len(comps), "size_distribution": dict(sorted(Counter(len(c) for c in comps).items())),
                         "largest": [{"size": len(c), "splits": dict(sorted(Counter(refs[g]["split"] for g in c).items())),
                                      "members": c} for c in big]}}


def make_row(r, ref) -> dict:
    """Split-file row: the opening "<think>\\n" moves from the completion to the prompt (completion kept for train and val)."""
    assert r["prompt"].endswith(ASSISTANT), f"{r['id']}: prompt must end with the assistant tag"
    assert r["completion"].startswith(THINK), f"{r['id']}: completion must start with {THINK!r}"
    assert r["completion"].count("</think>") == 1, f"{r['id']}: completion must close think exactly once"
    row = {"id": r["id"], "group": r["group"], "task_type": r["task_type"], "split": ref["split"],
           "cluster_id": ref["cluster_id"], "prompt": r["prompt"] + THINK}
    if ref["split"] in ("train", "val"):
        row["completion"] = r["completion"][len(THINK):]
    row |= {k: ref[k] for k in ("ref_smiles", "ref_conn", "ref_scaffold", "ref_inchikey", "n_heavy", "approval_year")}
    return row | {k: r[k] for k in ("n_prompt_tokens", "n_completion_tokens")}


def _bucket(y):
    return "unknown" if y is None else "<2015" if y < 2015 else "2015-2021" if y < 2022 else str(y)


def _by_bucket(counts):
    return {**{b: counts.get(b, 0) for b in BUCKETS}, **{k: v for k, v in sorted(counts.items()) if k not in BUCKETS}}


def make_report(refs, rows, dropped, notes, info, inputs) -> dict:
    tasks = defaultdict(Counter)
    for r in rows:
        tasks[r["group"]][r["task_type"]] += 1
    splits = {}
    for s in (*SPLITS, "buffer", "excluded"):
        gs = [r for r in refs.values() if r["split"] == s]
        drugs, nrows, by_task = Counter(), Counter(), Counter()
        for r in gs:
            drugs[_bucket(r["approval_year"])] += 1
            nrows[_bucket(r["approval_year"])] += r["n_rows"]
            by_task.update(tasks[r["group"]])
        splits[s] = {"drugs": len(gs), "rows": sum(r["n_rows"] for r in gs), "rows_by_task": dict(sorted(by_task.items())),
                     "drugs_by_year": _by_bucket(drugs), "rows_by_year": _by_bucket(nrows),
                     "drugs_n_heavy_over_100": sum((r["n_heavy"] or 0) > 100 for r in gs),
                     "drugs_ligand_only": sum(r["convention"] == "ligand_only" for r in gs)}
    why = defaultdict(list)
    for g, r in refs.items():
        if r["exclude_reason"]:
            why[r["exclude_reason"]].append(g)
    count = lambda it: dict(sorted(Counter(it).items()))
    return {
        "seed": info["seed"], "similarity_threshold": SIM, "test_t_min_year": TEST_T_YEAR, "inputs": inputs,
        "dropped": {**dropped, "excluded": {k: {"rows": sum(refs[g]["n_rows"] for g in gs), "groups": gs}
                                            for k, gs in sorted(why.items())}},
        "splits": splits, "targets": info["targets"], "clusters": info["clusters"],
        "max_cross_tanimoto": info["max_cross_tanimoto"], "scaffold_overlap": info["scaffold_overlap"],
        "references": {"exposure_matched": notes["exposure_matched"], "ref_source": count(r["ref_source"] for r in refs.values() if r["ref_source"]),
                       "sft_agrees": count(json.dumps(r["sft_agrees"]) for r in refs.values()),
                       "approval_source": count(r["approval_source"] for r in refs.values() if r["approval_source"]),
                       "sft_answer_groups": [g for g, r in refs.items() if r["ref_source"] == "sft_answer"]},
        "manual_effect": notes["manual_effect"], "disagreements": notes["disagreements"], "year_overrides": notes["year_overrides"],
        "unused_hand_entries": {"manual_refs": notes["unused_manual"], "approval_audit": notes["unused_audit"]},
    }


def _write_all(out, files):
    """Stage every file beside its target, then rename them all; a failure replaces nothing. Hand files are refused."""
    assert not set(files) & set(HAND), f"data_prep never writes {HAND}"
    staged = {}
    try:
        for name, text in files.items():
            staged[name] = out / f".{name}.{os.getpid()}.tmp"
            staged[name].write_text(text, encoding="utf-8")
        for name, tmp in staged.items():
            os.replace(tmp, out / name)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)  # only our own staging files


def write_outputs(refs, rows, report, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump = lambda o: json.dumps(o, ensure_ascii=False) + "\n"
    files = {"refs.jsonl": "".join(dump(r) for r in refs.values())}
    parts = {s: [] for s in SPLITS}
    for r in rows:  # input order is kept inside each split file
        ref = refs[r["group"]]
        if ref["split"] in parts:
            parts[ref["split"]].append(dump(make_row(r, ref)))
    files |= {f"{s}.jsonl": "".join(ls) for s, ls in parts.items()}
    files["split_report.json"] = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    _write_all(out, files)


def _read_hand(path, int_keys, required=()):
    """(entries, meta) of an optional hand file; read-only. meta = {"present": False} when absent, else
    {"present": True, "entries": n, "sha256": 12 hex of the bytes parsed}. Every entry must be an object that has the
    `required` keys, and its `int_keys` must be int or null."""
    if not path.exists():
        return {}, {"present": False}
    raw = path.read_bytes()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e}); data_prep never edits it, fix it and re-run") from e
    assert isinstance(d, dict), f"{path.name}: expected an object keyed by group"
    for g, v in d.items():
        assert isinstance(v, dict) and all(k in v for k in required), f"{path.name}[{g!r}]: expected an object with keys {required}"
        assert all(type(v.get(k)) in (int, type(None)) for k in int_keys), \
            f"{path.name}[{g!r}]: {', '.join(int_keys)} must be int or null"
    return d, {"present": True, "entries": len(d), "sha256": _sha12(raw)}


def _sha12(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:12]


def summary(rep) -> str:
    inp, drop, sp = rep["inputs"], rep["dropped"], rep["splits"]
    tasks = sorted({t for s in sp.values() for t in s["rows_by_task"]})
    hand = ", ".join(f"{k} {inp[k].get('entries', 'absent')}" for k in ("manual_refs", "approval_audit", "exposure"))
    lines = [f"rows read {inp['rows_read']}: dropped PD {drop['pd']['rows']} + combination {drop['combination']['rows']}"
             f" -> {inp['rows_kept']} rows, {inp['groups_kept']} drugs | hand files: {hand}",
             f"ref_source {rep['references']['ref_source']} sft_agrees {rep['references']['sft_agrees']}",
             "excluded " + (", ".join(f"{k} {len(v['groups'])}" for k, v in drop["excluded"].items()) or "none"),
             f"{'split':8}{'drugs':>6}{'rows':>6}  " + " ".join(f"{t:>4}" for t in tasks) + "   by year " + " ".join(BUCKETS)]
    for s, v in sp.items():
        lines.append(f"{s:8}{v['drugs']:>6}{v['rows']:>6}  " + " ".join(f"{v['rows_by_task'].get(t, 0):>4}" for t in tasks)
                     + "   " + " ".join(str(v["drugs_by_year"][b]) for b in BUCKETS))
    lines.append("manual entries: " + ", ".join(f"{k} {len(v)}" for k, v in rep["manual_effect"].items())
                 + " | drugs with n_heavy > 100 / ligand_only per split: "
                 + ", ".join(f"{k} {v['drugs_n_heavy_over_100']}/{v['drugs_ligand_only']}" for k, v in sp.items()))
    lines.append("targets " + ", ".join(f"{k} {v['got']}/{v['target']}{'' if v['exact'] else ' (NOT EXACT)'}"
                                        for k, v in rep["targets"].items())
                 + f" | clusters {rep['clusters']['n']}, sizes {rep['clusters']['size_distribution']}")
    lines.append("max cross-split Tanimoto (< %.1f) " % SIM + ", ".join(f"{k} {v['max']}" for k, v in rep["max_cross_tanimoto"].items()))
    lines.append("drugs sharing a scaffold with a Train drug (must be 0) " + str(rep["scaffold_overlap"]["drugs_sharing_a_scaffold_with_train"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", nargs="+", type=Path, default=[ROOT / "data/raw/train_think.jsonl", ROOT / "data/raw/val_think.jsonl"])
    ap.add_argument("--pool", type=Path, default=ROOT / "data/curated/chembl_refs.json")
    ap.add_argument("--hand", type=Path, default=ROOT / "data/curated", help="folder holding the hand files (read only)")
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-test-r", type=int, default=N_TEST_R)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    a = ap.parse_args(argv)
    rows, dropped = load_rows(a.rows)
    sha = {p.name: _sha12(p.read_bytes()) for p in a.rows}
    raw = a.pool.read_bytes()  # parse and hash the same bytes: another worker may be rewriting the cache
    pool, sha[a.pool.name] = {e["name"]: e for e in json.loads(raw)}, _sha12(raw)
    specs = (  # (int-or-null keys, required keys) of each hand file; a manual entry must say smiles, null meaning unresolvable
        (("approval_year",), ("smiles",)), (("earliest_global_year",), ()),
        (("pubchem_cid", "pubchem_create_year", "pubmed_count"), ()))
    (manual, m_meta), (audit, a_meta), (exposure, x_meta) = (_read_hand(a.hand / f, *k) for f, k in zip(HAND, specs))
    for f, m in zip(HAND, (m_meta, a_meta, x_meta)):
        if m["present"]:
            sha[f] = m.pop("sha256")
    refs, notes = build_refs(rows, pool, manual, audit, exposure)
    info = assign_splits(refs, a.seed, a.n_test_r, a.n_val)
    inputs = {"rows_read": len(rows) + dropped["pd"]["rows"] + dropped["combination"]["rows"], "rows_kept": len(rows),
              "groups_kept": len(refs), "sha256": sha, "manual_refs": m_meta, "approval_audit": a_meta, "exposure": x_meta}
    report = make_report(refs, rows, dropped, notes, info, inputs)
    write_outputs(refs, rows, report, a.data)
    print(summary(report))
    return report


if __name__ == "__main__":
    main()
