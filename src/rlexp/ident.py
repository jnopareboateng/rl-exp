#!/usr/bin/env python3
"""Identification diagnostic (docs/plan.md section 5): can the model NAME the drug a profile describes?

  python -m rlexp.ident write --data data/processed --out data/processed/arms/ident
  python -m rlexp.ident score --samples runs/X/samples.jsonl --data data/processed [--curated data/curated/chembl_refs.json]

Not an arm. With the name-given control (rlexp.arms name) it splits the structure-recovery gap into identification (profile to
name) and recall (name to SMILES). A sample matches when its first answer line, normalised (case, accents, salt and form words,
punctuation), equals the normalised drug name or the ChEMBL preferred name of the row's drug. Rates are drug-balanced like
rlexp.metrics: mean over a drug's rows of the mean over each row's samples; pass@8 is "any of the first 8 samples".
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from rlexp import arms

IDENT_SYSTEM = "Reply with only the international nonproprietary name of the drug described. No explanation."
SPLITS = ("val", "test_r", "test_t")
_FORMS = {"hydrochloride", "hcl", "sodium", "potassium", "calcium", "mesylate", "maleate", "sulfate", "sulphate", "tartrate",
          "citrate", "phosphate", "acetate", "succinate", "besylate", "fumarate", "bromide", "chloride", "hydrate", "monohydrate",
          "dihydrate", "tosylate", "hydrobromide", "disodium", "dipotassium", "anhydrous"}
_DECOR = " \t\r\n`*_#>-•.,;:\"'"


def ident_prompt(row, system=IDENT_SYSTEM):
    """The row's prompt with the profile kept and the system turn asking for the name."""
    return arms.with_system(row["prompt"], system)


def normalize(name):
    """Comparison key of a drug name: accents folded, lower case, parentheses and form/salt words dropped, letters and digits only."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    return "".join(w for w in re.findall(r"[a-z0-9]+", text) if w not in _FORMS)


def answer_name(text):
    """The first non-empty line after the last '</think>', cut at the first comma, semicolon, slash or ' or ', markdown and quotes stripped."""
    tail = (text or "").rsplit("</think>", 1)[-1]
    line = next((x.strip(_DECOR) for x in tail.splitlines() if x.strip(_DECOR)), "")
    return re.split(r"[,;/]| or ", line, maxsplit=1)[0].strip(_DECOR)


def accepted_names(group, curated):
    """Normalised names that count as the drug: its dataset name and, from the curated ChEMBL file, its preferred name."""
    names = {normalize(group)}
    names |= {normalize(x["pref_name"]) for x in curated if x.get("pref_name") and normalize(x.get("name", "")) == normalize(group)}
    return names - {""}


def score(samples, rows, curated):
    """Aggregate over samples (dicts with id, sample_idx, text) of data rows `rows`: name_match@1, pass@8, answered; all drug-balanced."""
    by_id = {r["id"]: r for r in rows}
    per_row = defaultdict(list)
    for s in sorted(samples, key=lambda s: (s["id"], s["sample_idx"])):
        if s["id"] in by_id:
            per_row[s["id"]].append(s)
    drug = defaultdict(lambda: defaultdict(list))
    for rid, group in per_row.items():
        ok = accepted_names(by_id[rid]["group"], curated)
        hits = [normalize(answer_name(s["text"])) in ok for s in group]
        said = [bool(answer_name(s["text"])) for s in group]
        d = drug[by_id[rid]["group"]]
        d["match"].append(sum(hits) / len(hits))
        d["answered"].append(sum(said) / len(said))
        d["pass8"].append(float(any(hits[:8])))
    out = {"drugs": len(drug), "rows": len(per_row), "samples": sum(map(len, per_row.values()))}
    for key, label in (("match", "name_match@1"), ("pass8", "name_pass@8"), ("answered", "answered")):
        means = [sum(v[key]) / len(v[key]) for v in drug.values()]
        out[label] = round(sum(means) / len(means), 4) if means else None
    return out


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(data_dir, out_dir):
    """Write val.jsonl, test_r.jsonl and test_t.jsonl under out_dir: the processed rows with the identification prompt."""
    for split in SPLITS:
        rows = [{**r, "prompt": ident_prompt(r)} for r in _read(Path(data_dir) / f"{split}.jsonl")]
        arms._write(Path(out_dir) / f"{split}.jsonl", rows, lines=True)
        arms._sidecar(Path(out_dir) / f"{split}.jsonl", rows, IDENT_SYSTEM, control="identification")
        print(f"{split}.jsonl: {len(rows)} rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write the identification prompt files")
    w.add_argument("--data", default="data/processed")
    w.add_argument("--out", default="data/processed/arms/ident")
    s = sub.add_parser("score", help="name match rates of a samples.jsonl")
    s.add_argument("--samples", required=True)
    s.add_argument("--data", default="data/processed")
    s.add_argument("--curated", default="data/curated/chembl_refs.json")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "write":
            write(a.data, a.out)
        else:
            samples = _read(a.samples)
            rows = [r for split in sorted({x["split"] for x in samples}) for r in _read(Path(a.data) / f"{split}.jsonl")]
            print(json.dumps(score(samples, rows, json.load(open(a.curated, encoding="utf-8")))))
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
