#!/usr/bin/env python3
"""Training-arm data (docs/plan.md section 2; row schema in docs/contracts.md). Pure functions plus a small CLI.

  python -m rlexp.arms build --data data/processed --out data/processed/arms
  python -m rlexp.arms rft --scored runs/X/scored.jsonl --data data/processed --out data/processed/arms/a3_rft_train.jsonl

  A1  trace_sft        the reference trace cut right after the final answer (a description after it on its line goes too)
  A2  answer_only_sft  an empty think block, then the canonical reference SMILES
  A3  rft_dataset      Train rollouts that reach conn_match when re-scored now; rft_report counts drugs covered and not
  A4  nearest_train + icl_prompt   the 3 nearest Train prompts as worked examples inside the user turn
Prompts end with '<think>\\n' and completions end with '<|im_end|>' (contracts.md); SFT rows are {id, prompt, completion}.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rlexp import chem, reward

END, THINK, USER = "<|im_end|>", "</think>", "<|im_start|>user\n"
ICL_HEADER = "Worked examples from similar profiles (each followed by its known answer):"
SPLITS = ("train", "val", "test_r", "test_t")
_TOKEN = re.compile(r"[a-z0-9]+")
_DECOR = " \t\r\n`*_#>-•.,;:"   # markdown and punctuation that may surround an answer token
_LABEL_WORDS = 4               # 'Proposed structure:', 'I propose', 'SMILES:' are labels; a longer lead-in is prose
_GLOSS = re.compile(r"\s+[—–-]\s|\s+\(|:")   # a description may follow the SMILES after ' — ', ' – ', ' - ', ' (' or ':'
_CLOSERS = re.compile(r"[`*_]*")                # markdown hugging the token: `S`, **S**


# ---------------------------------------------------------------- prompts
def split_prompt(prompt):
    """(prompt up to and including '<|im_start|>user\\n', user text, rest from that turn's '<|im_end|>' on)."""
    head, sep, rest = prompt.partition(USER)
    user, end, tail = rest.partition(END)
    if not (sep and end):
        raise ValueError("prompt has no '<|im_start|>user' turn closed by '<|im_end|>'")
    return head + sep, user, end + tail


def user_text(prompt):
    return split_prompt(prompt)[1]


# ---------------------------------------------------------------- A1 and A2
def trace_cut(completion):
    """(completion, status). Cut after the final answer: the last occurrence, after the last '</think>', of
    chem.extract_answer(completion). The answer must be the first thing on its line once markdown and a short label
    ('Proposed structure:', 'I propose') are ignored. Nothing else on the line: the whole line is kept. A description
    after it (separator ' — ', ' – ', ' - ', ' (' or ':') is dropped, so the SMILES and the markdown closing it are
    the last text before '<|im_end|>'. Everything after that line is dropped; inside a code fence the closing fence
    is kept. status is
      'cut'            rule applied, even if nothing needed dropping;
      'uncut'          the answer is inline in a sentence: the full completion is kept;
      'unextractable'  extract_answer finds nothing: the full completion is kept.
    The cut never changes what extract_answer returns (enforced below), and '</think>' stays unique."""
    if completion.count(THINK) != 1:
        raise ValueError(f"completion must contain exactly one {THINK}")
    ans = chem.extract_answer(completion)
    if ans is None:
        return completion, "unextractable"
    t0 = completion.index(THINK) + len(THINK)
    tail = completion[t0:]
    p = tail.rfind(ans)
    q = p + len(ans)
    le = tail.find("\n", q)
    le = len(tail) if le < 0 else le                         # end of the answer line
    if len(tail[tail.rfind("\n", 0, p) + 1:p].strip(_DECOR).split()) > _LABEL_WORDS:
        return completion, "uncut"                           # the token ends a sentence, it is not a labelled answer
    cut, after = le, tail[q:le]
    if after.replace(END, "").strip(_DECOR):                 # something follows the token on its line
        closers = _CLOSERS.match(after).end()
        if not _GLOSS.match(after, closers):
            return completion, "uncut"                       # inline in a sentence
        cut = q + closers                                    # drop the description
    keep = ""                                                # an open code fence keeps its closing fence
    if tail[:cut].count("```") % 2:
        j = tail.find("```", le)
        if j >= 0 and not tail[le:j].strip():
            keep = tail[le:j + 3]
    out = (completion[:t0] + tail[:cut] + keep).removesuffix(END) + END
    return (out, "cut") if chem.extract_answer(out) == ans else (completion, "uncut")


def trace_sft(row):
    """A1 row: {id, prompt, completion, cut}; cut is False when the full completion was kept (see trace_cut)."""
    try:
        completion, status = trace_cut(row["completion"])
    except ValueError as e:
        raise ValueError(f"row {row['id'][:60]}: {e}") from None
    return {"id": row["id"], "prompt": row["prompt"], "completion": completion, "cut": status == "cut"}


def answer_only_sft(row):
    """A2 row. prompt + completion ends in '<think>\\n\\n</think>\\n\\n<smiles>', the native thinking-off layout."""
    return {"id": row["id"], "prompt": row["prompt"], "completion": f"\n{THINK}\n\n{row['ref_smiles']}{END}"}


# ---------------------------------------------------------------- A4
def icl_prompt(row, neighbours):
    """The row's prompt with worked examples prepended to the user text. neighbours: [(user_text, ref_smiles)], in order.
    The system turn and the assistant opener are untouched."""
    head, user, tail = split_prompt(row["prompt"])
    shots = [part for u, smiles in neighbours for part in (u.strip(), f"Answer SMILES: {smiles}")]
    return head + "\n\n".join([ICL_HEADER, *shots, "Now the task:", user]) + tail


def neighbour_set(train_by_id, ids):
    """[(user_text, ref_smiles)] for icl_prompt from Train row ids (as nearest_train returns them)."""
    return [(user_text(train_by_id[i]["prompt"]), train_by_id[i]["ref_smiles"]) for i in ids]


def nearest_train(query_rows, train_rows, k=3):
    """{query id: [ids of the k nearest Train rows]}, best first. TF-IDF cosine over the user part of the prompt:
    lowercase alphanumeric tokens, sublinear tf (1 + ln count), idf = ln((1 + N) / (1 + df)) + 1 with N and the
    vocabulary from Train only, L2-normalised. A Train row is never a neighbour of itself or of a query with the same
    cluster_id (when both have one) or the same group. Ties go to the lowest id. Fewer than k ids come back only
    when fewer than k eligible Train rows exist."""
    ids = [r["id"] for r in train_rows]
    if len(set(ids)) != len(ids) or len({r["id"] for r in query_rows}) != len(query_rows):
        raise ValueError("duplicate row ids")

    def count(rows):
        return [Counter(_TOKEN.findall(user_text(r["prompt"]).lower())) for r in rows]

    docs = count(train_rows)
    df = Counter(w for d in docs for w in d)
    vocab = {w: j for j, w in enumerate(sorted(df))}
    idf = np.log((1 + len(docs)) / (1 + np.array([df[w] for w in vocab]))) + 1.0

    def matrix(counters):
        X = np.zeros((len(counters), len(vocab)))
        for i, c in enumerate(counters):
            for w, n in c.items():
                if w in vocab:
                    X[i, vocab[w]] = 1.0 + np.log(n)
        X *= idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)

    # rounding makes any float noise between identical documents a true tie, which the id rank then breaks
    S = np.round(matrix(count(query_rows)) @ matrix(docs).T, 12)
    rank = np.empty(len(ids), dtype=int)
    rank[sorted(range(len(ids)), key=ids.__getitem__)] = np.arange(len(ids))   # position in id order: the tie-break

    def keys(r):
        cluster = [("cluster", r["cluster_id"])] if r.get("cluster_id") is not None else []
        return [("id", r["id"]), ("group", r["group"])] + cluster

    banned = defaultdict(list)                                       # Train row indices that share an id, group or cluster
    for j, r in enumerate(train_rows):
        for key in keys(r):
            banned[key].append(j)
    out = {}
    for i, q in enumerate(query_rows):
        s = S[i].copy()
        for key in keys(q):
            s[banned.get(key, [])] = -1.0
        out[q["id"]] = [ids[j] for j in np.lexsort((rank, -s))[:k] if s[j] >= 0]
    return out


# ---------------------------------------------------------------- A3
def rft_dataset(scored_rows, data_rows, max_per_drug=2, require_closed=True, stats=None):
    """A3 rows {id, prompt, completion} from scored samples (contracts.md "Sampling and scoring products").
    Success is decided here, never by the stored conn_match/closed (they go stale when a reference or the scorer
    changes): each sample with finish_reason == 'stop' whose prompt row is in data_rows is re-scored with
    reward.score(text, that row) and kept if the fresh result has conn_match and closed (closed is implied by
    conn_match today; require_closed=False drops that test). Passing the Train rows drops samples of other splits.
    Per drug at most max_per_drug: preference is fewer n_tokens, then lower sample_idx, then id; the best sample of
    each prompt row comes first, the rest fill. completion = text + '<|im_end|>' unless it ends with it. A nothink
    sample's text starts '</think>', so the sequence the model saw ('<think>\\n\\n</think>\\n\\n') is restored with a
    leading newline. A reference that does not standardize raises ValueError. If stats is a dict it receives
    rescored, kept (fresh successes before the per-drug cap), stale_kept (the stored flags said success, the fresh
    score says no) and stale_dropped (the reverse)."""
    rows = {r["id"]: r for r in data_rows}
    by_drug = defaultdict(list)
    n = dict.fromkeys(("rescored", "kept", "stale_kept", "stale_dropped"), 0)
    for s in scored_rows:
        row = rows.get(s["id"])
        if row is None or s["finish_reason"] != "stop":
            continue
        try:
            fresh = reward.score(s["text"], row)
        except ValueError as e:
            raise ValueError(f"row {s['id'][:60]}: {e}") from None
        ok = bool(fresh["conn_match"] and (fresh["closed"] or not require_closed))
        was = bool(s["conn_match"] and (s["closed"] or not require_closed))
        n["rescored"] += 1
        n["kept"] += ok
        n["stale_kept"] += was and not ok
        n["stale_dropped"] += ok and not was
        if ok:
            by_drug[row["group"]].append(s)
    if stats is not None:
        stats.update(n)
    out = []
    for group in sorted(by_drug):
        ranked = sorted(by_drug[group], key=lambda s: (s["n_tokens"], s["sample_idx"], s["id"]))
        best = {}                                        # each prompt row's best sample, in rank order
        for s in ranked:
            best.setdefault(s["id"], s)
        taken = {id(s) for s in best.values()}
        for s in [*best.values(), *(s for s in ranked if id(s) not in taken)][:max_per_drug]:
            text = ("\n" if s.get("mode") == "nothink" else "") + s["text"]
            out.append({"id": s["id"], "prompt": rows[s["id"]]["prompt"], "completion": text if text.endswith(END) else text + END})
    return out


def rft_report(selected, data_rows):
    """Counts for an rft_dataset result against the data rows it was built for (the Train rows)."""
    group = {r["id"]: r["group"] for r in data_rows}
    drugs, covered = set(group.values()), {group[s["id"]] for s in selected}
    return {"examples": len(selected), "drugs_covered": len(covered), "drugs_total": len(drugs),
            "drugs_without_success": len(drugs - covered)}


# ---------------------------------------------------------------- files and CLI
def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path, obj, lines=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in obj)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=1)


def _chars(texts):
    """'p50 N p90 M' over character counts (tokens are not available here)."""
    xs = [len(t) for t in texts]
    return f"p50 {np.percentile(xs, 50):.0f} p90 {np.percentile(xs, 90):.0f}" if xs else "n/a"


def build(data_dir, out_dir):
    """Write the A1, A2 and A4 files under out_dir and print counts and character-length statistics (no text)."""
    data = {s: _read(Path(data_dir) / f"{s}.jsonl") for s in SPLITS}
    out = Path(out_dir)
    for s in ("train", "val"):
        rows, n = data[s], len(data[s])
        a1 = [trace_sft(r) for r in rows]
        _write(out / f"a1_trace_{s}.jsonl", a1, lines=True)
        cut = sum(a["cut"] for a in a1)
        trimmed = sum(a["cut"] and a["completion"] != r["completion"] for a, r in zip(a1, rows))
        unx = sum(not a["cut"] and chem.extract_answer(r["completion"]) is None for a, r in zip(a1, rows))
        print(f"a1_trace_{s}.jsonl: {n} rows | cut {cut} ({cut / n:.1%}, text dropped on {trimmed}) | "
              f"uncut {n - cut - unx} ({(n - cut - unx) / n:.1%}) | unextractable {unx} ({unx / n:.1%}) | "
              f"completion chars {_chars([a['completion'] for a in a1])}, before the cut {_chars([r['completion'] for r in rows])}")
        a2 = [answer_only_sft(r) for r in rows]
        _write(out / f"a2_answer_{s}.jsonl", a2, lines=True)
        ok = sum(chem.extract_answer(a["completion"]) == r["ref_smiles"] for a, r in zip(a2, rows))
        print(f"a2_answer_{s}.jsonl: {n} rows | extract_answer recovers ref_smiles on {ok} | "
              f"completion chars {_chars([a['completion'] for a in a2])}")
    train = data["train"]
    by_id = {r["id"]: r for r in train}
    for s in ("val", "test_r", "test_t"):
        near = nearest_train(data[s], train)
        _write(out / f"a4_neighbours_{s}.json", near)
        prompts = [icl_prompt(r, neighbour_set(by_id, near[r["id"]])) for r in data[s]]
        distinct = sum(len({by_id[i]["group"] for i in ids}) == len(ids) for ids in near.values())
        print(f"a4_neighbours_{s}.json: {len(near)} queries, {sum(map(len, near.values()))} neighbours, "
              f"{distinct} queries with all-distinct drugs | icl prompt chars {_chars(prompts)}")


def rft(scored_path, data_dir, out_path, max_per_drug=2):
    """Write the A3 file from a scored.jsonl of Train rollouts and print the coverage report."""
    train = _read(Path(data_dir) / "train.jsonl")
    stats = {}
    selected = rft_dataset(_read(scored_path), train, max_per_drug=max_per_drug, stats=stats)
    _write(out_path, selected, lines=True)
    rep = rft_report(selected, train)
    print(f"{Path(out_path).name}: {rep['examples']} examples | drugs covered {rep['drugs_covered']} of {rep['drugs_total']}, "
          f"{rep['drugs_without_success']} Train drugs without a successful sample | rescored {stats['rescored']} samples, "
          f"{stats['kept']} succeed now; the stored flags would have wrongly kept {stats['stale_kept']} and wrongly dropped "
          f"{stats['stale_dropped']} | completion chars {_chars([a['completion'] for a in selected])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the A1, A2 and A4 files")
    b.add_argument("--data", default="data/processed")
    b.add_argument("--out", default="data/processed/arms")
    r = sub.add_parser("rft", help="write the A3 file from scored Train rollouts")
    r.add_argument("--scored", required=True)
    r.add_argument("--data", default="data/processed")
    r.add_argument("--out", default="data/processed/arms/a3_rft_train.jsonl")
    r.add_argument("--max-per-drug", type=int, default=2)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            build(a.data, a.out)
        else:
            rft(a.scored, a.data, a.out, a.max_per_drug)
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
