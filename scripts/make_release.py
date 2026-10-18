#!/usr/bin/env python3
"""Public release package: release/{splits,controls,values}.csv and release/README.md, from public or derived sources only.

  .venv/bin/python scripts/make_release.py [--root .] [--out release]

Reads data/processed/{refs.jsonl,split_report.json}, data/curated/{chembl_refs,approval_audit,exposure,manual_refs}.json, results/{constant_ceiling,val_baselines}.json
and, if present, results/random_pair_calibration.json. The C3 answer needs the ChEMBL pool of scripts/build_pool.py (data/raw/chembl_pool, hash-checked against
results/constant_ceiling.json); runs/colab-s1/val-control_constant_pool/meta.json is a cross-check only. No prompt file, no Test split file and nothing under
runs/final is opened. The output is deterministic (sorted rows, no timestamps). Every table passes a guard before anything is written: a field outside its allow-list
or a string over 2,000 characters (a guard against prompt text) refuses the whole run. A value that is not known stays blank, never invented; the run prints how many
rows lack each field, and counts, never row text.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path

from rlexp import chem, controls, metrics

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
POOL = "data/raw/chembl_pool/pool.jsonl"
META = "runs/colab-s1/val-control_constant_pool/meta.json"
MAX_STR = 2000
SPLIT_NAMES = ("train", "val", "test_r", "test_t", "buffer", "excluded")
FIELDS = {
    "splits.csv": ("chembl_id", "drug_name", "split", "cluster_id", "approval_year", "approval_source", "heavy_atoms", "pubchem_cid", "pubchem_create_year",
                   "pubmed_count", "smiles", "inchikey_first_block"),
    "controls.csv": ("control", "chembl_id", "smiles", "selection_rule"),
    "values.csv": ("table", "name", "split", "statistic", "value", "source_file"),
}
CAPS = {"chembl_id": 20, "drug_name": 100, "approval_source": 80, "inchikey_first_block": 14, "control": 4, "selection_rule": 500, "table": 30, "name": 60,
        "split": 60, "statistic": 60, "value": 30, "source_file": 100}
RUNS = {"C0": "val8-control_constant", "C1": "val8-control_copy_nearest", "C2": "val8-control_constant_scaffold", "C3": "val8-control_constant_pool"}
STATS = ("fv_tanimoto", "mean_reward", "lift", "exact@1", "core@1", "valid_closed", "fv_samples", "fv_tokens", "drugs")
CI = ("fv_tanimoto", "lift")
RULES = {
    "C0": "Train reference with the highest mean 2048-bit Morgan Tanimoto to the other Train references (ties: first in Train order)",
    "C1": "Copy the reference of the Train drug whose profile text is nearest by TF-IDF cosine; needs the private profile text, so no structure is released",
    "C2": "Shortest Train reference among those whose non-empty Murcko scaffold is the most common among the Train references (each drug counted once)",
    "C3": ("Highest mean count-vector Morgan Tanimoto to the Train references among the Train references (own entry left out) and the leakage-filtered ChEMBL "
           "approved-drug pool ({release}, {date}, {n} drugs, not copied, sha256 {sha}); rebuild: scripts/fetch_chembl_pool.py then scripts/build_pool.py"),
}


class ReleaseError(ValueError):
    pass


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path, optional=False):
    path = Path(path)
    if optional and not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v):
    """A number as text: ints as they are, floats rounded to 6 decimals (no negative zero)."""
    return str(v) if isinstance(v, int) else repr(round(v, 6) + 0.0)


def show(counts):
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def leaves(obj, path=()):
    """(path, number) of every numeric leaf; lists of up to 10 are enumerated from 1; longer lists, strings and booleans are skipped."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(obj, list) and len(obj) <= 10:
        for i, v in enumerate(obj, 1):
            yield from leaves(v, path + (str(i),))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield "/".join(path), obj


def guard(name, rows):
    """Refuse a table with a field outside its allow-list, a value that is not text, a number or blank, a string over its cap (2,000 characters at most) or an unknown split."""
    for i, row in enumerate(rows, 1):
        if extra := set(row) - set(FIELDS[name]):
            raise ReleaseError(f"{name} row {i}: field(s) outside the allow-list: {sorted(extra)}")
        for k, v in row.items():
            cap = min(MAX_STR, CAPS.get(k, MAX_STR))
            if not isinstance(v, (str, int, float, type(None))):
                raise ReleaseError(f"{name} row {i}: {k} is a {type(v).__name__}, not text or a number")
            if isinstance(v, str) and len(v) > cap:
                raise ReleaseError(f"{name} row {i}: {k} holds {len(v)} characters, over its limit of {cap} (a guard against prompt text)")
        if name == "splits.csv" and row.get("split") not in SPLIT_NAMES:
            raise ReleaseError(f"{name} row {i}: unknown split {str(row.get('split'))[:30]!r}")


def csv_text(name, rows):
    buf = io.StringIO()
    out = csv.DictWriter(buf, fieldnames=FIELDS[name], lineterminator="\n")
    out.writeheader()
    out.writerows(rows)                     # the csv module writes None as an empty cell
    return buf.getvalue()


def blanks(name, rows):
    """{field: rows where it is blank} for the fields that have any."""
    n = {f: sum(r.get(f) in (None, "") for r in rows) for f in FIELDS[name]}
    return {f: c for f, c in n.items() if c}


def _id_agrees(entry, ref):
    """False only when ChEMBL has a structure for the id and its connectivity is neither the reference's nor a recorded alternate's; None when there is nothing to compare."""
    mol = chem.standardize((entry or {}).get("parent_smiles"))
    if mol is None:
        return None
    keys = {ref["ref_conn"], *(chem.conn(chem.standardize(s)) for s in ref["ref_alt_smiles"])}
    return chem.conn(mol) in keys


def splits_table(refs, chembl, audit, exposure):
    """(rows, notes): one row per usable drug (split other than excluded), sorted by name. The structure is the reference as the scorer parses it (rlexp.chem).
    approval_source is the regulator the approval audit names where it set the year, else the label refs.jsonl records. A ChEMBL id whose own structure
    disagrees with the reference is withheld (notes["withheld"]); ids that cannot be compared are listed in notes["unchecked"]."""
    rows, notes = [], {"withheld": [], "unchecked": []}
    for g in sorted(refs):
        r = refs[g]
        if r["split"] == "excluded":
            continue
        mol = chem.standardize(r["ref_smiles"])
        cid = r.get("chembl_id")
        if cid:
            agrees = _id_agrees(chembl.get(g), r)
            if agrees is None:
                notes["unchecked"].append(g)
            elif not agrees:
                notes["withheld"].append(g)
                cid = None
        src = r["approval_source"]
        if src == "audit":
            src = (audit.get(g) or {}).get("earliest_regulator") or src
        rows.append({"chembl_id": cid, "drug_name": g, "split": r["split"], "cluster_id": r["cluster_id"], "approval_year": r["approval_year"],
                     "approval_source": src, "heavy_atoms": mol.GetNumHeavyAtoms() if mol else None,
                     "pubchem_cid": (exposure.get(g) or {}).get("pubchem_cid"), "pubchem_create_year": r["pubchem_create_year"],
                     "pubmed_count": r["pubmed_count"], "smiles": chem.canonical(mol) if mol else None,
                     "inchikey_first_block": chem.conn(mol) if mol else None})
    return rows, notes


def check_counts(rows, report):
    """The table and split_report.json must agree on the drugs per split."""
    got = Counter(r["split"] for r in rows)
    bad = {s: (got.get(s, 0), v["drugs"]) for s, v in report["splits"].items() if s != "excluded" and got.get(s, 0) != v["drugs"]}
    if bad:
        raise ReleaseError(f"refs.jsonl and split_report.json disagree on drugs per split (table, report): {bad}")


def year_issues(refs, chembl, audit, manual, test_t_min_year):
    """Every usable drug whose approval year or source looks inconsistent with the curated files, and every audit entry whose years contradict its regulator label."""
    out = []
    for g in sorted(refs):
        r = refs[g]
        if r["split"] == "excluded":
            continue
        y, s, c = r["approval_year"], r["approval_source"], chembl.get(g) or {}
        want = {"audit": (audit.get(g) or {}).get("earliest_global_year"), "manual": (manual.get(g) or {}).get("approval_year"),
                "chembl": c.get("approval_year"), "openfda": c.get("approval_year")}.get(s)
        if want != y:
            out.append(f"{g}: year {y} but its source {s} gives {want}")
        if s == "chembl" and c.get("first_approval") != y:
            out.append(f"{g}: source chembl but ChEMBL first_approval is {c.get('first_approval')}")
        if (g in manual and s != "manual") or (g in audit and s not in ("manual", "audit")):
            out.append(f"{g}: a manual or audit year exists but the source is {s}")
        if (y >= test_t_min_year) != (r["split"] == "test_t"):
            out.append(f"{g}: year {y} and split {r['split']} contradict the Test-T rule")
    for g, a in sorted(audit.items()):
        if a["earliest_global_year"] > a["fda_year"] or (a["earliest_regulator"] == "FDA" and a["earliest_global_year"] != a["fda_year"]):
            out.append(f"{g}: audit years ({a['earliest_global_year']}, FDA {a['fda_year']}) contradict the regulator {a['earliest_regulator']}")
    return out


def c3_answer(root, train, ceiling):
    """C3 as scripts/best_constant.py chooses it: {smiles, chembl_id (None for a Train winner), score, candidates, source, manifest, sha256}."""
    bc = load_script("best_constant")
    path = Path(root) / POOL
    try:
        pool, sha = bc.read_pool(path)
    except ValueError as e:
        raise ReleaseError(str(e)) from None
    if sha is None:
        raise ReleaseError(f"C3 needs the ChEMBL pool {path}: run scripts/fetch_chembl_pool.py and scripts/build_pool.py")
    if sha != ceiling.get("pool_sha256"):
        raise ReleaseError("the pool differs from the one results/constant_ceiling.json was made with (pool_sha256)")
    by_smiles = {}
    for row in pool:
        if bc.row_smiles(row):
            by_smiles.setdefault(bc.row_smiles(row), row)
    smiles, score, source = bc.choose(train, list(by_smiles))
    return {"smiles": smiles, "chembl_id": by_smiles[smiles]["chembl_id"] if source == "pool" else None, "score": score, "source": source,
            "candidates": len(train) + len(by_smiles), "manifest": read_json(path.with_name("manifest.json")), "sha256": sha}


def check_meta(path, c3):
    """Compare C3 with the run record of the Val control (train_score, answer_chars, candidates, and pool_winner when it has one). Returns a line for the log."""
    meta = read_json(path, optional=True)
    if meta is None:
        return "run meta.json absent: C3 not cross-checked against it"
    bad = [k for k, v in (("train_score", round(c3["score"], 4)), ("answer_chars", len(c3["smiles"])), ("candidates", c3["candidates"])) if meta.get(k) != v]
    if "pool_winner" in meta and meta["pool_winner"].get("chembl_id") != c3["chembl_id"]:
        bad.append("pool_winner")
    if bad:
        raise ReleaseError(f"C3 disagrees with the run meta.json on {bad}")
    return "C3 agrees with the run meta.json (train_score, answer_chars, candidates" + ("" if "pool_winner" in meta else "; it has no pool_winner, an older best_constant.py wrote it") + ")"


def check_val(answers, val_refs, baselines):
    """Each released control answer must reproduce the Val mean Tanimoto that results/val_baselines.json records for it (a constant answers every row alike)."""
    refs = [chem.standardize(s) for s in val_refs]
    for control, smiles in answers.items():
        mol = chem.standardize(smiles)
        got, want = sum(chem.tanimoto(mol, m) for m in refs) / len(refs), baselines[RUNS[control]]["mean_tanimoto"]
        if abs(got - want) > 1e-9:
            raise ReleaseError(f"{control}: the released answer scores {got:.9f} on Val but results/val_baselines.json records {want:.9f}")


def controls_table(train, by_group, c3):
    """(rows, {control: smiles}) for C0 to C3; C1 has a rule only."""
    c0, _ = metrics.best_constant(train)
    c2, _, _ = controls.scaffold_constant(train)
    owner = {s: g for g, s in reversed(list(train.items()))}      # the first Train drug with that structure
    ident = lambda smiles: (by_group[owner[smiles]].get("chembl_id") if smiles in owner else None)
    m = c3["manifest"]
    rule = RULES["C3"].format(release=m.get("chembl_release", "release not recorded"), date=m.get("release_date", "date not recorded"), sha=c3["sha256"],
                              n=m.get("derived", {}).get("files", {}).get("pool.jsonl", {}).get("records", "an unrecorded number of"))
    rows = [{"control": "C0", "chembl_id": ident(c0), "smiles": c0, "selection_rule": RULES["C0"]},
            {"control": "C1", "chembl_id": None, "smiles": None, "selection_rule": RULES["C1"]},
            {"control": "C2", "chembl_id": ident(c2), "smiles": c2, "selection_rule": RULES["C2"]},
            {"control": "C3", "chembl_id": c3["chembl_id"] or ident(c3["smiles"]), "smiles": c3["smiles"], "selection_rule": rule}]
    return rows, {"C0": c0, "C2": c2, "C3": c3["smiles"]}


def values_table(ceiling, baselines, calibration):
    """(rows, notes): the oracle constant per split, C0 to C3 on Val, and the random-pair similarity distribution when its file exists. Entries named test* are skipped."""
    rows, skipped = [], 0

    def add(table, name, split, statistic, value, source):
        nonlocal skipped
        if any(p.lower().startswith("test") for p in (split, *statistic.split("/"))):
            skipped += 1
        else:
            rows.append({"table": table, "name": name, "split": split, "statistic": statistic, "value": fmt(value), "source_file": source})

    if calibration is not None:
        body = {k: v for k, v in calibration.items() if k not in ("refs", "splits")}
        for stat, v in leaves(body):
            add("random_pair", "pairs of reference structures", "+".join(calibration.get("splits", ())), stat, v, "results/random_pair_calibration.json")
    for split, d in ceiling["splits"].items():
        for stat, v in leaves(d):
            add("oracle_constant", "best constant over Train plus pool", split, stat, v, "results/constant_ceiling.json")
    by_run = {r["run"]: r for r in baselines["rows"]}
    for control, run in RUNS.items():
        for stat in STATS:
            add("control", control, baselines["split"], stat, by_run[run][stat], "results/val_baselines.json")
            if stat in CI:
                for end, v in zip(("low", "high"), by_run[run][f"{stat}_ci"]):
                    add("control", control, baselines["split"], f"{stat}_ci_{end}", v, "results/val_baselines.json")
    return rows, {"skipped": skipped, "random_pair": calibration is not None}


def readme(rows, notes, n_excluded, c3, sums):
    """release/README.md: at most 40 lines. The checksum list leaves out this file, which cannot hold its own hash."""
    found = sum(r["approval_source"] in ("chembl", "openfda", "manual") for r in rows)
    man = c3["manifest"]
    release = f"{man.get('chembl_release', 'release not recorded')}, {man.get('release_date', 'date not recorded')}"
    lines = [
        "# Release tables: drug structures, splits and constant controls", "",
        "Built only from public or derived information: drug identifiers, public structures of approved drugs, approval years with their sources, split and component assignments, control answers and the",
        "study's reference numbers. No prompt, completion, reasoning trace, sample or adapter is included. Read docs/benchmark_card.md first, then docs/controls_checklist.md.", "",
        "## Files",
        f"- splits.csv: one row per usable drug ({len(rows)}): chembl_id, drug_name, split, cluster_id (structure component), approval_year, approval_source, heavy_atoms,",
        f"  pubchem_cid, pubchem_create_year, pubmed_count, smiles (the reference the scorer uses), inchikey_first_block. The {n_excluded} excluded groups (no usable reference) are not listed.",
        "- split: train; val; test_r (random whole components); test_t (first approved anywhere in 2025-26); buffer (other drugs in a test_t drug's component, used nowhere).",
        "- controls.csv: the constant answers C0, C2 and C3 with their selection rule, and the rule for C1, which needs the private profile text (no structure).",
        "- values.csv: the benchmark's own reference numbers and the file each comes from: oracle constant, C0 to C3 on Val, random-pair similarity when that file exists.",
        "  No Test score and no Test oracle value; the random-pair rows use only the reference structures that splits.csv lists.",
        f"- approval_source: the regulator the approval audit names where it set the year; for the other {found} drugs how the year was found: chembl (ChEMBL first approval),",
        "  openfda (first US marketing date in openFDA) or manual (hand-resolved, with its sources in data/curated/manual_refs.json).",
        f"- Blank means unknown, never a guess. Blank cells in splits.csv: {show(blanks('splits.csv', rows))}.", "",
        "## Licences and caveats",
        "- No licence has been chosen yet for this repository or these files.",
        "- References and ids are mostly ChEMBL data. The ChEMBL release behind the per-drug table was not recorded; the C3 pool is ChEMBL_37. The repository records no ChEMBL licence:",
        "  verify the ChEMBL licence terms before redistributing. The terms of openFDA, regulator pages, PubChem and PubMed were not checked either.",
        "- Test-R and Test-T structures are listed so the splits can be reproduced. A public label is no longer a held-out label: do not train on them.",
        f"- ChEMBL ids withheld because ChEMBL's structure for the id is a different compound: {', '.join(notes['withheld']) or 'none'}.",
        "", "## Rebuilding the C3 pool (not copied)",
        f"Pool {release}, sha256 {c3['sha256']}. From the repository root:", "```",
        ".venv/bin/python scripts/fetch_chembl_pool.py   # network: ChEMBL web API, approved small molecules; writes data/raw/chembl_pool",
        ".venv/bin/python scripts/build_pool.py          # reads group and reference structures from the four private row files; writes pool.jsonl",
        ".venv/bin/python scripts/best_constant.py --splits val --out <dir>   # checks the pool hash; prints C3's train_score and pool_winner", "```",
        "The API serves its current release, so a later fetch may not equal the one above: compare the sha256. Not verified: a rebuild from the public tables alone (the row files are private).", "",
        "## Generated by",
        "`python scripts/make_release.py` from the repository root. Deterministic: sorted rows, no timestamps. Numbers in values.csv are rounded to 6 decimals.", "",
        "## Checksums (sha256; this file cannot hold its own)",
        *(f"{h}  {name}" for name, h in sorted(sums.items())),
    ]
    return "\n".join(lines) + "\n"


def build(root):
    """({file name: text}, log lines) for the whole package. Raises ReleaseError before any text exists if a guard or a cross-check fails."""
    root = Path(root)
    read = lambda rel, **kw: read_json(root / rel, **kw)
    refs = {}
    for line in (root / "data/processed/refs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            refs[r["group"]] = r
    report = read("data/processed/split_report.json")
    chembl = {e["name"]: e for e in read("data/curated/chembl_refs.json")}
    audit, exposure, manual = (read(f"data/curated/{n}.json") for n in ("approval_audit", "exposure", "manual_refs"))
    ceiling, baselines = read("results/constant_ceiling.json"), read("results/val_baselines.json")
    if missing := [run for run in RUNS.values() if run not in {r["run"] for r in baselines["rows"]}]:
        raise ReleaseError(f"results/val_baselines.json has no row for {missing}")
    calibration = read("results/random_pair_calibration.json", optional=True)

    rows, notes = splits_table(refs, chembl, audit, exposure)
    check_counts(rows, report)
    issues = year_issues(refs, chembl, audit, manual, report["test_t_min_year"])
    log = [f"splits.csv: {len(rows)} rows ({show(Counter(r['split'] for r in rows))}); blank fields: {show(blanks('splits.csv', rows))}",
           f"  ChEMBL ids withheld (its structure for the id disagrees with the reference): {len(notes['withheld'])} {notes['withheld']}; "
           f"ids with no ChEMBL structure to compare: {len(notes['unchecked'])}",
           f"  approval year and source checks: {len(issues)} issue(s)" + "".join(f"\n    {i}" for i in issues)]

    train = {g: r["ref_smiles"] for g, r in refs.items() if r["split"] == "train"}
    c3 = c3_answer(root, train, ceiling)
    ctl_rows, answers = controls_table(train, refs, c3)
    check_val(answers, [r["ref_smiles"] for r in refs.values() if r["split"] == baselines["split"]], {r["run"]: r for r in baselines["rows"]})
    log += [f"controls.csv: {len(ctl_rows)} rows; blank fields: {show(blanks('controls.csv', ctl_rows))} (C1 by design: it needs the private profile text)",
            f"  C3 from {'the pool' if c3['source'] == 'pool' else 'Train'}, mean Tanimoto to Train {c3['score']:.4f}; " + check_meta(root / META, c3),
            "  C0, C2 and C3 reproduce their Val means in results/val_baselines.json"]

    val_rows, vnotes = values_table(ceiling, baselines, calibration)
    log.append(f"values.csv: {len(val_rows)} rows ({show(Counter(r['table'] for r in val_rows))})"
               + ("" if vnotes["random_pair"] else "; results/random_pair_calibration.json absent: no random-pair rows")
               + (f"; {vnotes['skipped']} Test-named entries skipped" if vnotes["skipped"] else ""))

    tables = {"splits.csv": rows, "controls.csv": ctl_rows, "values.csv": val_rows}
    for name, table in tables.items():
        guard(name, table)
    files = {name: csv_text(name, table) for name, table in tables.items()}
    sums = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in files.items()}
    files["README.md"] = readme(rows, notes, report["splits"]["excluded"]["drugs"], c3, sums)
    log.append(f"README.md: {len(files['README.md'].splitlines())} lines")
    return files, log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="the repository root the inputs are read from")
    ap.add_argument("--out", type=Path, help="default: <root>/release")
    args = ap.parse_args(argv)
    try:
        files, log = build(args.root)
    except (ReleaseError, metrics.InputError, OSError) as e:
        sys.exit(f"error: {e}")
    out = args.out or args.root / "release"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    print("\n".join(log))
    print(f"wrote {', '.join(files)} to {out}")


if __name__ == "__main__":
    main()
