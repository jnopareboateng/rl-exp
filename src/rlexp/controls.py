#!/usr/bin/env python3
"""CPU baselines scored by the same path as model samples (docs/contracts.md "controls.py").

  python -m rlexp.controls --data data/processed --out runs

Writes runs/control_copy_nearest/, runs/control_constant/ and runs/control_constant_scaffold/ (samples.jsonl,
scored.jsonl, meta.json), one answer per prompt row of val, test_r and test_t, as text '</think>\\n' + smiles, then
scores them with metrics.score_samples.
  control_copy_nearest       TF-IDF cosine between the user part of a prompt and the user part of every Train prompt;
                             the answer is the nearest Train row's ref_smiles, copied. An answer-copying control, not
                             retrieval-augmented prompting. Train rows are the only copy targets.
  control_constant           the Train reference with the highest mean bit Tanimoto to the other Train references
                             (2048-bit Morgan vectors, chem.tanimoto_bits; the scorer itself now uses count vectors).
  control_constant_scaffold  the shortest Train reference among those whose non-empty Murcko scaffold is the most
                             common among the Train references (each drug counted once): a do-nothing bar for core@1.
Ties go to the first in Train file order.
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rdkit

from rlexp import chem, metrics

CONTROL_SPLITS = ("val", "test_r", "test_t")
_TOKEN = re.compile(r"[a-z0-9]+")


def user_part(prompt):
    """Text between '<|im_start|>user' and the following '<|im_end|>' (the system and assistant parts are ignored)."""
    _, found, rest = prompt.partition("<|im_start|>user")
    if not found or "<|im_end|>" not in rest:
        raise metrics.InputError("prompt has no '<|im_start|>user ... <|im_end|>' part")
    return rest.split("<|im_end|>", 1)[0]


def nearest(corpus, queries):
    """Nearest corpus text per query by TF-IDF cosine: lowercase alphanumeric tokens, sublinear tf (1 + ln count),
    idf = ln((1 + N) / (1 + df)) + 1 and the vocabulary from the corpus only, L2-normalised rows. Returns
    (index into corpus, cosine) per query; ties go to the lowest index and a query with no known token gets 0."""
    def count(texts):
        return [Counter(_TOKEN.findall(t.lower())) for t in texts]

    docs = count(corpus)
    df = Counter(w for d in docs for w in d)
    vocab = {w: j for j, w in enumerate(sorted(df))}
    idf = np.log((1 + len(docs)) / (1 + np.array([df[w] for w in vocab]))) + 1.0

    def matrix(counters):
        X = np.zeros((len(counters), len(vocab)))
        for i, c in enumerate(counters):
            for w, k in c.items():
                if w in vocab:
                    X[i, vocab[w]] = 1.0 + np.log(k)
        X *= idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)

    S = matrix(count(queries)) @ matrix(docs).T
    return S.argmax(axis=1), S.max(axis=1)


def scaffold_constant(refs):
    """(smiles, scaffold, number of drugs with it): the shortest reference among those whose non-empty Murcko scaffold
    is the most common among the references. refs: {group: ref_smiles}, one per drug; ties go to the first."""
    by_scaffold, bad = {}, 0
    for smiles in refs.values():
        mol = chem.standardize(smiles)
        bad += mol is None
        if mol is not None and (scaf := chem.scaffold(mol)):
            by_scaffold.setdefault(scaf, []).append(smiles)
    if bad:
        raise metrics.InputError(f"{bad} reference(s) do not standardize")
    if not by_scaffold:
        raise metrics.InputError("no Train reference has a ring scaffold")
    scaf = max(by_scaffold, key=lambda k: len(by_scaffold[k]))      # max keeps the first of equals
    return min(by_scaffold[scaf], key=len), scaf, len(by_scaffold[scaf])


def _user(row):
    try:
        return user_part(row["prompt"])
    except metrics.InputError as e:
        raise metrics.InputError(f"row {row['id'][:60]}: {e}") from None


def run(data_dir, out_dir, scorer=None):
    """Write and score both controls under out_dir; returns {control name: meta dict}."""
    t0 = time.time()
    data = metrics.load_data(data_dir, keep=("prompt",))
    by = {s: [r for r in data if r["split"] == s] for s in ("train",) + CONTROL_SPLITS}
    empty = [s for s, rs in by.items() if not rs]
    if empty:
        raise metrics.InputError(f"no rows for split(s) {empty} under {data_dir}")
    train, ev = by["train"], [r for s in CONTROL_SPLITS for r in by[s]]
    idx, cos = nearest([_user(r) for r in train], [_user(r) for r in ev])
    refs = {r["group"]: r["ref_smiles"] for r in train}
    smiles, centrality = metrics.best_constant(refs)
    scaf_smiles, scaf, scaf_drugs = scaffold_constant(refs)
    common = {"train_rows": len(train), "train_drugs": len({r["group"] for r in train}),
              "rows_by_split": {s: len(by[s]) for s in CONTROL_SPLITS}, "command": " ".join(sys.argv),
              "numpy": np.__version__, "rdkit": rdkit.__version__}
    controls = {
        "control_copy_nearest": ([train[i]["ref_smiles"] for i in idx],
                              {"mean_cosine": float(cos.mean()), "zero_cosine_rows": int((cos == 0).sum())}),
        "control_constant": ([smiles] * len(ev), {"constant_smiles": smiles, "mean_bit_tanimoto_to_train": centrality}),
        "control_constant_scaffold": ([scaf_smiles] * len(ev), {"constant_smiles": scaf_smiles, "scaffold": scaf,
                                                                  "scaffold_drugs": scaf_drugs,
                                                                  "scaffold_share": scaf_drugs / len(refs)}),
    }
    out = {}
    for name, (answers, extra) in controls.items():
        samples = metrics.answer_samples(ev, answers, name)
        scored = metrics.score_samples(samples, ev, scorer)         # score first: a failure leaves no half-written run
        d = Path(out_dir) / name
        metrics.write_jsonl(d / "samples.jsonl", samples)
        metrics.write_jsonl(d / "scored.jsonl", scored)
        out[name] = {"model": name, "n_samples": len(samples), **common, **extra,
                     "wall_time_s": round(time.time() - t0, 2)}
        metrics.write_json(d / "meta.json", out[name])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    try:
        for name, meta in run(a.data, a.out).items():
            note = f", {meta['zero_cosine_rows']} rows with no known token" if "zero_cosine_rows" in meta else ""
            print(f"{name}: {meta['n_samples']} samples -> {Path(a.out) / name}{note}")
    except (metrics.InputError, OSError, RuntimeError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
