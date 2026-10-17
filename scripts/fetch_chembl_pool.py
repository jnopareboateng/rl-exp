#!/usr/bin/env python3
"""Download ChEMBL approved small molecules (max_phase 4) with their mechanisms, targets and indications.

    .venv/bin/python scripts/fetch_chembl_pool.py [--out data/raw/chembl_pool] [--pause 0.5]

Public ChEMBL REST API only (https://www.ebi.ac.uk/chembl/api/data), one request at a time with a pause between
requests, and retries with backoff on throttling (429, 5xx) and network errors; if every retry of a page fails the page
size is halved (down to 100). Pages of up to 1,000 records (ids in batches of 100) are trimmed to the fields the pool needs
and written to <out>/<name>.json as soon as they arrive, so an interrupted run resumes at the first missing page and a
complete cache needs no network. manifest.json is written last: ChEMBL release, record counts, and the sha256 of every
cached file (a `derived` record added by build_pool.py is kept while these hashes are unchanged). Prints counts only.

  molecule_NNNNNN.json    every small molecule with max_phase 4 (page starting at offset NNNNNN, molecule_chembl_id order)
  parent_NNN.json         hierarchy parents that are not themselves in that set (batches of 100 ids)
  mechanism_NNNNNN.json   the whole drug-mechanism table (mec_id order); build_pool.py picks the pool's rows
  target_NNN.json         every target named by a mechanism (batches of 100 ids): name and type
  indication_NNNNNN.json  the whole drug-indication table (drugind_id order)
  status.json             ChEMBL release the cache was built from
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://www.ebi.ac.uk/chembl/api/data/"
PAGE, PAGE_MIN, BATCH, TRIES, TIMEOUT = 1000, 100, 100, 6, 180
RETRY_HTTP = (429, 500, 502, 503, 504)
MOLECULE = ("molecule_chembl_id,pref_name,molecule_hierarchy,molecule_structures,first_approval,withdrawn_flag,"
            "therapeutic_flag,oral,parenteral,topical,atc_classifications,molecule_synonyms,max_phase,molecule_type")
MECHANISM = "mec_id,molecule_chembl_id,parent_molecule_chembl_id,mechanism_of_action,action_type,target_chembl_id"
TARGET = "target_chembl_id,pref_name,target_type"
INDICATION = "drugind_id,molecule_chembl_id,parent_molecule_chembl_id,mesh_heading,efo_term,max_phase_for_ind"


def api(path, params, pause):
    """GET one JSON document; sleeps `pause` seconds afterwards. Retries 429/5xx and network errors with backoff."""
    url = API + path + "?" + urllib.parse.urlencode(params)
    err = None
    for attempt in range(TRIES):
        after = 0
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "rl-exp-chembl-pool/1 (research; sequential paging)"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.load(r)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:  # before URLError: HTTPError is a subclass
            if e.code not in RETRY_HTTP:
                raise
            err, header = e, e.headers.get("Retry-After", "")
            after = int(header) if header.isdigit() else 0
        except (urllib.error.URLError, http.client.HTTPException, OSError, json.JSONDecodeError) as e:
            err = e
        if attempt < TRIES - 1:
            wait = max(5 * 2 ** attempt, after)
            print(f"  retry {attempt + 1}/{TRIES - 1} in {wait}s after {type(err).__name__} {getattr(err, 'code', '')}".rstrip(), flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{path}: gave up after {TRIES} attempts ({type(err).__name__})")


def save(path, obj):
    """Write atomically: a killed run never leaves a half-written page behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _phase(x):
    return None if x is None else float(x)


def trim_molecule(m):
    h, s = m.get("molecule_hierarchy") or {}, m.get("molecule_structures") or {}
    return {"id": m["molecule_chembl_id"], "parent_id": h.get("parent_chembl_id") or m["molecule_chembl_id"],
            "pref_name": m.get("pref_name"), "smiles": s.get("canonical_smiles") or None, "first_approval": m.get("first_approval"),
            "withdrawn_flag": bool(m.get("withdrawn_flag")), "therapeutic_flag": bool(m.get("therapeutic_flag")),
            "oral": bool(m.get("oral")), "parenteral": bool(m.get("parenteral")), "topical": bool(m.get("topical")),
            "atc": sorted(set(m.get("atc_classifications") or [])),
            "synonyms": [{"name": x["molecule_synonym"], "type": x.get("syn_type")}
                         for x in m.get("molecule_synonyms") or [] if x.get("molecule_synonym")],
            "max_phase": _phase(m.get("max_phase")), "molecule_type": m.get("molecule_type")}


def trim_mechanism(m):
    return {k: m.get(k) for k in ("mec_id", "molecule_chembl_id", "parent_molecule_chembl_id", "mechanism_of_action",
                                  "action_type", "target_chembl_id")}


def trim_target(t):
    return {k: t.get(k) for k in ("target_chembl_id", "pref_name", "target_type")}


def trim_indication(i):
    return {**{k: i.get(k) for k in ("drugind_id", "molecule_chembl_id", "parent_molecule_chembl_id", "mesh_heading", "efo_term")},
            "max_phase_for_ind": _phase(i.get("max_phase_for_ind"))}


def crawl(out, used, name, path, listkey, params, order, only, trim, pause):
    """All records of a table, up to PAGE per request in `order`. The page starting at offset N is cached as <name>_<N>.json
    with its own record count. If the server gives up on a page (it answers a slow 1,000-row page with a 500), the page size
    is halved, down to PAGE_MIN, for the rest of the table. Returns (records, the API's total_count); raises if the pages
    do not add up to the total."""
    records, offset, total, size = [], 0, None, PAGE
    while total is None or offset < total:
        f = out / f"{name}_{offset:06d}.json"
        if f.exists():
            page = json.loads(f.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            try:
                d = api(path, {**params, "limit": size, "offset": offset, "order_by": order, "only": only}, pause)
            except RuntimeError:
                if size <= PAGE_MIN:
                    raise
                size = max(PAGE_MIN, size // 2)
                print(f"{name} offset {offset}: the server gave up, page size now {size}", flush=True)
                continue
            page = {"total_count": d["page_meta"]["total_count"], "offset": offset, "records": [trim(r) for r in d[listkey]]}
            if page["records"]:  # an empty page is an API fault, not data: never cached
                save(f, page)
            print(f"{name} offset {offset}: {len(page['records'])} of {page['total_count']} records, {time.time() - t0:.0f}s", flush=True)
        if not page["records"]:
            raise RuntimeError(f"{name}: empty page at offset {offset} of {page['total_count']}")
        used.append((f.name, len(page["records"])))
        total = page["total_count"]
        records += page["records"]
        offset += len(page["records"])
    if len(records) != total:
        raise RuntimeError(f"{name}: collected {len(records)} records but the API reports {total}")
    return records, total


def lookup(out, used, name, path, listkey, idkey, ids, only, trim, pause):
    """Records for `ids`, BATCH ids per request. Batch k is cached as <name>_<k>.json with the ids it asked for, so a
    changed id list refetches it. Ids the API does not know are simply absent from the result."""
    ids, records = sorted(ids), []
    for k in range(0, len(ids), BATCH):
        chunk, f = ids[k:k + BATCH], out / f"{name}_{k // BATCH:03d}.json"
        page = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        if page is None or page["ids"] != chunk:
            t0 = time.time()
            d = api(path, {idkey: ",".join(chunk), "limit": PAGE, "only": only}, pause)
            page = {"ids": chunk, "records": [trim(r) for r in d[listkey]]}
            save(f, page)
            print(f"{name} batch {k // BATCH}: {len(page['records'])} of {len(chunk)} ids, {time.time() - t0:.0f}s", flush=True)
        used.append((f.name, len(page["records"])))
        records += page["records"]
    return records


def fetch(out, pause=0.5):
    """Fill (or complete) the cache in `out`, write manifest.json, return it."""
    out.mkdir(parents=True, exist_ok=True)
    used = []
    f = out / "status.json"
    if not f.exists():
        d = api("status.json", {}, pause)
        save(f, {k: d.get(k) for k in ("chembl_db_version", "chembl_release_date")})
    used.append((f.name, None))
    release = json.loads(f.read_text(encoding="utf-8"))
    mols, n_mol = crawl(out, used, "molecule", "molecule.json", "molecules", {"max_phase": 4, "molecule_type": "Small molecule"},
                        "molecule_chembl_id", MOLECULE, trim_molecule, pause)
    have = {m["id"] for m in mols}
    parents = lookup(out, used, "parent", "molecule.json", "molecules", "molecule_chembl_id__in",
                     {m["parent_id"] for m in mols} - have, MOLECULE, trim_molecule, pause)
    mechs, n_mec = crawl(out, used, "mechanism", "mechanism.json", "mechanisms", {}, "mec_id", MECHANISM, trim_mechanism, pause)
    targets = lookup(out, used, "target", "target.json", "targets", "target_chembl_id__in",
                     {m["target_chembl_id"] for m in mechs if m["target_chembl_id"]}, TARGET, trim_target, pause)
    inds, n_ind = crawl(out, used, "indication", "drug_indication.json", "drug_indications", {}, "drugind_id", INDICATION,
                        trim_indication, pause)
    manifest = {
        "chembl_release": release["chembl_db_version"], "release_date": release["chembl_release_date"], "api": API,
        "records": {"molecule": len(mols), "molecule_api_total": n_mol, "parent": len(parents), "mechanism": len(mechs),
                    "mechanism_api_total": n_mec, "target": len(targets), "indication": len(inds), "indication_api_total": n_ind},
        "files": {n: {"records": r, "sha256": hashlib.sha256((out / n).read_bytes()).hexdigest()} for n, r in sorted(used)}}
    path = out / "manifest.json"
    if path.exists():  # build_pool.py's `derived` record stays while the raw pages are the very ones it was built from
        prev = json.loads(path.read_text(encoding="utf-8"))
        if "derived" in prev and prev.get("files") == manifest["files"]:
            manifest["derived"] = prev["derived"]
    save(path, manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data/raw/chembl_pool")
    ap.add_argument("--pause", type=float, default=0.5, help="seconds to wait after every request")
    a = ap.parse_args(argv)
    try:
        m = fetch(a.out, a.pause)
    except (RuntimeError, urllib.error.HTTPError, OSError) as e:
        sys.exit(f"error: {e}")
    print(f"{m['chembl_release']} ({m['release_date']}): " + ", ".join(f"{k} {v}" for k, v in m["records"].items())
          + f" | {len(m['files'])} cached files in {a.out}")


if __name__ == "__main__":
    main()
