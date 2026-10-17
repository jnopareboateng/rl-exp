#!/usr/bin/env python3
"""Identification diagnostic (docs/plan.md section 5): can the model NAME the drug a profile describes?

  python -m rlexp.ident write --data data/processed --out data/processed/arms/ident
  python -m rlexp.ident score --samples runs/X/samples.jsonl --data data/processed [--curated data/curated/chembl_refs.json]

Not an arm. With the name-given control (rlexp.arms name) it splits the structure-recovery gap into identification (profile to
name) and recall (name to SMILES). A sample matches when its first answer line, normalised (case, accents, salt and form words,
punctuation), equals the normalised drug name or the ChEMBL preferred name of the row's drug. Only a usable sample can match or
count as answered, by the common failure rule of rlexp.metrics: its think block closed ('</think>' in the text) and the sample ended
by itself (finish_reason "stop"; a cut or aborted one is a failure). Rates are drug-balanced like rlexp.metrics: mean over a drug's
rows of the mean over each row's samples. name_pass@k is "any of the first k samples", with k the largest number up to 8 that every
row supplies, and k is in the key (a run of 2 samples per row reports name_pass@2, never a pass@8) and in `pass_k`. The samples are checked (a repeated (id, sample_idx), an id the data does not know, a split that is
not the data row's, samples of several splits all raise) and `score` scores one split at a time, printing one JSON line per split
that the samples name and opening only those data files.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from rlexp import arms, metrics

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


def usable(sample):
    """The common failure rule (metrics.finished and a closed think block): a sample that was cut by the length cap or aborted, or
    whose think block never closed, is not an answer, so it cannot match and is not 'answered'."""
    return metrics.finished(sample) and "</think>" in (sample.get("text") or "")


def _by_id(samples, rows):
    """id -> data row, after the checks a score would otherwise hide: data ids that repeat (one row would overwrite the other),
    sample ids the data does not know (dropped without a word), a repeated (id, sample_idx) (counted twice), a sample whose split
    is not its row's, and samples of several splits (never pooled)."""
    if (miss := sum(not {"id", "sample_idx", "text"} <= s.keys() for s in samples)):
        raise metrics.InputError(f"{miss} sample(s) lack id, sample_idx or text")
    for name, ok, want in (("id", lambda v: isinstance(v, str), "non-string"), ("sample_idx", metrics.is_number, "non-finite or non-numeric")):
        if (bad := sum(not ok(s[name]) for s in samples)):         # before any key is built: a list or NaN must not reach a set
            raise metrics.InputError(f"{bad} sample(s) with a {want} {name}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise metrics.InputError(f"the data rows repeat {len(ids) - len(set(ids))} id(s)")
    by_id = {r["id"]: r for r in rows}
    unknown = {s["id"] for s in samples} - by_id.keys()
    if unknown:
        raise metrics.InputError(f"{len(unknown)} unknown sample id(s) not in the data rows, e.g. {[i[:60] for i in sorted(unknown)[:3]]}")
    keys = [(s["id"], s["sample_idx"]) for s in samples]
    if len(set(keys)) != len(keys):
        raise metrics.InputError(f"{len(keys) - len(set(keys))} repeated (id, sample_idx) pair(s); was a resumed run appended twice?")
    clash = [s["id"] for s in samples if "split" in s and "split" in by_id[s["id"]] and s["split"] != by_id[s["id"]]["split"]]
    if clash:
        raise metrics.InputError(f"{len(clash)} sample(s) whose split is not their data row's, e.g. {[i[:60] for i in clash[:3]]}")
    splits = {by_id[s["id"]].get("split") for s in samples}
    if len(splits) > 1:
        raise metrics.InputError(f"the samples span the splits {sorted(map(str, splits))}: score one split at a time")
    return by_id


def score(samples, rows, curated):
    """Aggregate over samples (dicts with id, sample_idx, text) of data rows `rows`: name_match@1, name_pass@k (k in the key and in
    pass_k), answered; all drug-balanced. Only usable samples match or count as answered. Raises metrics.InputError on the input
    problems _by_id lists."""
    by_id = _by_id(samples, rows)
    per_row = defaultdict(list)
    for s in sorted(samples, key=lambda s: (s["id"], s["sample_idx"])):
        per_row[s["id"]].append(s)
    k = min([8] + [len(g) for g in per_row.values()])              # the largest k <= 8 that every row supplies
    drug = defaultdict(lambda: defaultdict(list))
    for rid, group in per_row.items():
        ok = accepted_names(by_id[rid]["group"], curated)
        hits = [usable(s) and normalize(answer_name(s["text"])) in ok for s in group]
        said = [usable(s) and bool(answer_name(s["text"])) for s in group]
        d = drug[by_id[rid]["group"]]
        d["match"].append(sum(hits) / len(hits))
        d["answered"].append(sum(said) / len(said))
        d["passk"].append(float(any(hits[:k])))
    out = {"drugs": len(drug), "rows": len(per_row), "samples": sum(map(len, per_row.values())), "pass_k": k}
    for key, label in (("match", "name_match@1"), ("passk", f"name_pass@{k}"), ("answered", "answered")):
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
            if not samples:
                raise metrics.InputError(f"{a.samples} holds no samples")
            if (miss := sum("split" not in x for x in samples)):
                raise metrics.InputError(f"{miss} sample(s) lack split, so the data file to read cannot be chosen")
            with open(a.curated, encoding="utf-8") as f:
                curated = json.load(f)
            for split in sorted({x["split"] for x in samples}):         # one split at a time, only the files that are named
                path = Path(a.data) / f"{split}.jsonl"
                if not path.exists():
                    raise metrics.InputError(f"the samples name split {split!r} but {path} does not exist")
                part = [x for x in samples if x["split"] == split]
                print(json.dumps({"split": split, **score(part, _read(path), curated)}))
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
