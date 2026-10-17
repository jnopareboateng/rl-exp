#!/usr/bin/env python3
"""Training-arm data (docs/plan.md section 2; row schema in docs/contracts.md). Pure functions plus a small CLI.

  python -m rlexp.arms build --data data/processed --out data/processed/arms [--system-text TEXT | --keep-system]
  python -m rlexp.arms a4 --data data/processed --out data/processed/arms/a4 [--system-text TEXT]
  python -m rlexp.arms name --data data/processed --out data/processed/arms/name [--system-text TEXT]
  python -m rlexp.arms blank --data data/processed --out data/processed/arms/blank [--system-text TEXT] [--splits val]
  python -m rlexp.arms rft --scored runs/X/scored.jsonl --data data/processed --out data/processed/arms/a3_rft_train.jsonl

  A1  trace_sft        the reference trace cut right after the final answer (a description after it on its line goes too)
  A2  answer_only_sft  an empty think block, then the canonical reference SMILES; system turn = DIRECT_SYSTEM unless overridden
  A3  rft_dataset      Train rollouts that reach conn_match when re-scored now; rft_report counts drugs covered and not
  A4  nearest_train + icl_prompt   the 3 nearest Train prompts as worked examples inside the user turn; `a4` writes val, test_r
                       and test_t in the processed-row schema (prompt replaced, system turn = DIRECT_SYSTEM) for sample_vllm.py
  N0  name_prompt      diagnostic control, not an arm: the user turn is only "Drug: <name>". Its score is the recall ceiling for a
                       model that is told which drug it is; `name` writes val, test_r and test_t like `a4`
  B0  blank_prompt     diagnostic control, not an arm: the profile REMOVED, the user turn is only "Drug: unknown." (the text of
                       gpu/rl_grpo.py --profile-mode none). What an arm scores with no pharmacology at all, the other side of the
                       shuffled-profile control (rlexp.shuffle); `blank` writes the splits named by --splits (default val only: the
                       Test files are read only when asked) like `name`
Answer-only prompts use a short direct-answer system text: under the dataset's long 'reasoning effort' system text the base
model rambles instead of answering. Every written file gets a <name>.meta.json sidecar recording the system text.
Prompts end with '<think>\\n' and completions end with '<|im_end|>' (contracts.md); SFT rows are {id, prompt, completion}.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rlexp import chem, metrics, reward

END, THINK, USER = "<|im_end|>", "</think>", "<|im_start|>user\n"
ICL_HEADER = "Worked examples from similar profiles (each followed by its known answer):"
DIRECT_SYSTEM = "Reply with only the SMILES string of the drug described. No explanation."
K = 3                                                      # neighbours per A4 prompt
BLANK_USER = "Drug: unknown."      # the user turn of the profile-removed control; gpu/rl_grpo.py PROFILE_FREE_USER (a test pins equality)
SPLITS = ("train", "val", "test_r", "test_t")
_TOKEN = re.compile(r"[a-z0-9]+")
_SYSTEM = re.compile(r"(<\|im_start\|>system\n)(.*?)(<\|im_end\|>)", re.S)    # the match gpu/sample_vllm.py --system-text uses
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


def with_system(prompt, text):
    """The prompt with the text of its system turn replaced: the first '<|im_start|>system\\n' ... '<|im_end|>', matched
    exactly like gpu/sample_vllm.py --system-text. ValueError if the prompt has no system turn."""
    out, n = _SYSTEM.subn(lambda m: m.group(1) + text + m.group(3), prompt, count=1)
    if n != 1:
        raise ValueError("prompt has no system turn to replace")
    return out


def name_prompt(row, system=DIRECT_SYSTEM):
    """N0: the row's prompt with the profile replaced by the drug name, so a model that knows the structure of a named drug
    can answer. The system turn is `system`; the generation prefix is kept."""
    head, _, tail = split_prompt(row["prompt"])
    return with_system(head + f"Drug: {row['group']}" + tail, system)


def blank_prompt(row, system=DIRECT_SYSTEM):
    """B0: the row's prompt with its user turn replaced by BLANK_USER: no profile and no drug name, the same user turn for every row,
    so what an arm scores on it is what it answers with no pharmacology at all. The system turn is `system`; the generation prefix
    is kept. This is the prompt that gpu/rl_grpo.py --profile-mode none trains and evaluates on."""
    head, _, tail = split_prompt(row["prompt"])
    return with_system(head + BLANK_USER + tail, system)


def system_text(prompt):
    """The text of the prompt's system turn (the match with_system replaces)."""
    m = _SYSTEM.search(prompt)
    if m is None:
        raise ValueError("prompt has no system turn")
    return m.group(2)


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


def answer_only_sft(row, system=DIRECT_SYSTEM):
    """A2 row. prompt + completion ends in '<think>\\n\\n</think>\\n\\n<smiles>', the native thinking-off layout. The text of
    the prompt's system turn becomes `system` (DIRECT_SYSTEM by default: under the dataset's long 'reasoning effort' text
    the base model rambles instead of answering); None keeps the dataset's text."""
    prompt = row["prompt"] if system is None else with_system(row["prompt"], system)
    return {"id": row["id"], "prompt": prompt, "completion": f"\n{THINK}\n\n{row['ref_smiles']}{END}"}


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


def nearest_train(query_rows, train_rows, k=K):
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
def rft_dataset(scored_rows, data_rows, max_per_drug=2, require_closed=True, stats=None, system=None, min_reward=None):
    """A3 rows {id, prompt, completion} from scored samples (contracts.md "Sampling and scoring products").
    Success is decided here, never by the stored conn_match/closed (they go stale when a reference or the scorer
    changes): each sample with finish_reason == 'stop' (a length-terminated or aborted one is a failure, the rule of
    rlexp.metrics.finished) whose prompt row is in data_rows is re-scored with
    reward.score(text, that row) and kept if the fresh result has conn_match and closed (closed is implied by
    conn_match today; require_closed=False drops that test). Passing the Train rows drops samples of other splits.
    Per drug at most max_per_drug: preference is fewer n_tokens, then lower sample_idx, then id; the best sample of
    each prompt row comes first, the rest fill. completion = text + '<|im_end|>' unless it ends with it. A nothink
    sample's text starts '</think>', so the sequence the model saw ('<think>\\n\\n</think>\\n\\n') is restored with a
    leading newline. The row id is '<prompt row id>#<sample_idx>': two samples of one prompt row must not share an id,
    or the SFT reader rejects the file. system, if given, replaces the system-turn text of every prompt (pass the text
    the rollouts were sampled with, so the training prompt is the one the model saw). A reference that does not
    standardize raises ValueError, and so do scored rows that repeat an (id, sample_idx) pair. If stats is a dict it receives
    rescored, kept (fresh successes before the per-drug cap), stale_kept (the stored flags said success, the fresh
    score says no) and stale_dropped (the reverse).
    min_reward (dense mode, the reward-ranked A3): success is instead a fresh valid, closed sample whose reward is at least
    min_reward (identity hits are rare, so the identity-only rule starves); the preference is then higher reward, then the
    order above. The (id, sample_idx) pairs are checked first, before anything is scored or capped, so a pair that the
    per-drug cap would have dropped cannot hide a resumed run appended twice."""
    for field in ("sample_idx", "n_tokens"):
        if (bad := sum(not metrics.is_number(s[field]) for s in scored_rows)):
            raise ValueError(f"{bad} scored row(s) with a non-finite or non-numeric {field}")
    pairs = [(s["id"], s["sample_idx"]) for s in scored_rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"scored rows repeat {len(pairs) - len(set(pairs))} (id, sample_idx) pair(s); was a resumed run appended twice?")
    rows = {r["id"]: r for r in data_rows}
    reward_of = {}
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
        if min_reward is None:
            ok = bool(fresh["conn_match"] and (fresh["closed"] or not require_closed))
            was = bool(s["conn_match"] and (s["closed"] or not require_closed))
        else:
            ok = bool(fresh["valid"] and fresh["closed"] and fresh["reward"] >= min_reward)
            was = bool(s.get("valid") and s.get("closed") and s.get("reward", 0.0) >= min_reward)
        n["rescored"] += 1
        n["kept"] += ok
        n["stale_kept"] += was and not ok
        n["stale_dropped"] += ok and not was
        if ok:
            by_drug[row["group"]].append(s)
            reward_of[id(s)] = fresh.get("reward", 0.0)
    if stats is not None:
        stats.update(n)
    out = []
    for group in sorted(by_drug):
        ranked = sorted(by_drug[group], key=lambda s: (-reward_of[id(s)] if min_reward is not None else 0,
                                                       s["n_tokens"], s["sample_idx"], s["id"]))
        best = {}                                        # each prompt row's best sample, in rank order
        for s in ranked:
            best.setdefault(s["id"], s)
        taken = {id(s) for s in best.values()}
        for s in [*best.values(), *(s for s in ranked if id(s) not in taken)][:max_per_drug]:
            text = ("\n" if s.get("mode") == "nothink" else "") + s["text"]
            prompt = rows[s["id"]]["prompt"]
            out.append({"id": f"{s['id']}#{s['sample_idx']}", "prompt": prompt if system is None else with_system(prompt, system),
                        "completion": text if text.endswith(END) else text + END})
    return out


def rft_report(selected, data_rows):
    """Counts for an rft_dataset result against the data rows it was built for (the Train rows)."""
    group = {r["id"]: r["group"] for r in data_rows}
    drugs, covered = set(group.values()), {group[s["id"].rpartition("#")[0]] for s in selected}     # id = row id # sample_idx
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
    """'p50 N p90 M max K' over character counts (tokens are not available here)."""
    xs = [len(t) for t in texts]
    return f"p50 {np.percentile(xs, 50):.0f} p90 {np.percentile(xs, 90):.0f} max {max(xs)}" if xs else "n/a"


def _system_info(rows, system):
    """(system_source, system_text) of files whose prompts had their system turn set to `system`. None means the dataset's
    own text: the text is returned when every row has the same one, else None."""
    if system is None:
        texts = {system_text(r["prompt"]) for r in rows}
        return "dataset", texts.pop() if len(texts) == 1 else None
    return ("direct" if system == DIRECT_SYSTEM else "custom"), system


def _sidecar(path, rows, system, **extra):
    """<name>.meta.json next to a written file: the row count and the system text its prompts carry, so a training run
    can record it. `system` as for _system_info."""
    source, text = _system_info(rows, system)
    meta = {"file": Path(path).name, "rows": len(rows), "system_source": source, "system_text": text, **extra}
    _write(Path(path).with_suffix(".meta.json"), meta)


def build(data_dir, out_dir, system=DIRECT_SYSTEM):
    """Write the A1, A2 and A4-neighbour files under out_dir, a .meta.json beside each A1 and A2 file, and print counts
    and character-length statistics (no text). system is the system-turn text of the A2 prompts (None keeps the
    dataset's); A1 always keeps the dataset's."""
    data = {s: _read(Path(data_dir) / f"{s}.jsonl") for s in SPLITS}
    out = Path(out_dir)
    for s in ("train", "val"):
        rows, n = data[s], len(data[s])
        a1 = [trace_sft(r) for r in rows]
        _write(out / f"a1_trace_{s}.jsonl", a1, lines=True)
        _sidecar(out / f"a1_trace_{s}.jsonl", a1, None)
        cut = sum(a["cut"] for a in a1)
        trimmed = sum(a["cut"] and a["completion"] != r["completion"] for a, r in zip(a1, rows))
        unx = sum(not a["cut"] and chem.extract_answer(r["completion"]) is None for a, r in zip(a1, rows))
        print(f"a1_trace_{s}.jsonl: {n} rows | cut {cut} ({cut / n:.1%}, text dropped on {trimmed}) | "
              f"uncut {n - cut - unx} ({(n - cut - unx) / n:.1%}) | unextractable {unx} ({unx / n:.1%}) | "
              f"completion chars {_chars([a['completion'] for a in a1])}, before the cut {_chars([r['completion'] for r in rows])}")
        a2 = [answer_only_sft(r, system) for r in rows]
        _write(out / f"a2_answer_{s}.jsonl", a2, lines=True)
        _sidecar(out / f"a2_answer_{s}.jsonl", a2, system)
        ok = sum(chem.extract_answer(a["completion"]) == r["ref_smiles"] for a, r in zip(a2, rows))
        print(f"a2_answer_{s}.jsonl: {n} rows | system text {_system_info(a2, system)[0]} | "
              f"extract_answer recovers ref_smiles on {ok} | completion chars {_chars([a['completion'] for a in a2])}")
    train = data["train"]
    by_id = {r["id"]: r for r in train}
    for s in ("val", "test_r", "test_t"):
        near = nearest_train(data[s], train)
        _write(out / f"a4_neighbours_{s}.json", near)
        distinct = sum(len({by_id[i]["group"] for i in ids}) == len(ids) for ids in near.values())
        print(f"a4_neighbours_{s}.json: {len(near)} queries, {sum(map(len, near.values()))} neighbours, "
              f"{distinct} queries with all-distinct drugs")


def a4(data_dir, out_dir, system=DIRECT_SYSTEM):
    """Write val.jsonl, test_r.jsonl and test_t.jsonl under out_dir (plus a .meta.json each): the processed rows with every
    field kept and `prompt` replaced by the in-context-retrieval prompt (the K nearest Train prompts, cluster-excluded,
    then the task) whose system-turn text is `system`, so gpu/sample_vllm.py --data out_dir reads them unchanged.
    n_prompt_tokens is left as in the source row (it counts the original prompt). Prints counts and character lengths."""
    data = {s: _read(Path(data_dir) / f"{s}.jsonl") for s in SPLITS}
    train = data["train"]
    by_id = {r["id"]: r for r in train}
    out, longest = Path(out_dir), 0
    for s in ("val", "test_r", "test_t"):
        near = nearest_train(data[s], train)
        rows = [{**r, "prompt": with_system(icl_prompt(r, neighbour_set(by_id, near[r["id"]])), system)} for r in data[s]]
        _write(out / f"{s}.jsonl", rows, lines=True)
        _sidecar(out / f"{s}.jsonl", rows, system, neighbours=K)
        prompts = [r["prompt"] for r in rows]
        longest = max([longest, *map(len, prompts)])
        print(f"{s}.jsonl: {len(rows)} rows, {K} neighbours each | prompt chars {_chars(prompts)}")
    print(f"longest ICL prompt over all splits: {longest} chars, about {longest // 4} tokens at 4 chars/token (8k = 8192)")


def name_oracle(data_dir, out_dir, system=DIRECT_SYSTEM):
    """Write val.jsonl, test_r.jsonl and test_t.jsonl under out_dir (plus a .meta.json each): the processed rows with
    `prompt` replaced by name_prompt, for gpu/sample_vllm.py --data out_dir. Prints counts and character lengths."""
    out = Path(out_dir)
    for s in ("val", "test_r", "test_t"):
        rows = [{**r, "prompt": name_prompt(r, system)} for r in _read(Path(data_dir) / f"{s}.jsonl")]
        _write(out / f"{s}.jsonl", rows, lines=True)
        _sidecar(out / f"{s}.jsonl", rows, system, control="name_oracle")
        print(f"{s}.jsonl: {len(rows)} rows | prompt chars {_chars([r['prompt'] for r in rows])}")


def blank(data_dir, out_dir, system=DIRECT_SYSTEM, splits=("val",)):
    """Write <split>.jsonl and a .meta.json each under out_dir for the named splits (val, test_r, test_t): the processed rows with
    `prompt` replaced by blank_prompt, for gpu/sample_vllm.py --data out_dir. Only the named split files are read, so a Val run
    never opens a Test file. Prints counts and character lengths."""
    bad = [s for s in splits if s not in ("val", "test_r", "test_t")]
    if bad or not splits:
        raise ValueError(f"splits must be a non-empty subset of val, test_r, test_t; got {list(splits)}")
    out = Path(out_dir)
    for s in splits:
        rows = [{**r, "prompt": blank_prompt(r, system)} for r in _read(Path(data_dir) / f"{s}.jsonl")]
        _write(out / f"{s}.jsonl", rows, lines=True)
        _sidecar(out / f"{s}.jsonl", rows, system, control="blank_profile")
        print(f"{s}.jsonl: {len(rows)} rows | prompt chars {_chars([r['prompt'] for r in rows])}")


def rft(scored_path, data_dir, out_path, max_per_drug=2, system=None, min_reward=None):
    """Write the A3 file from a scored.jsonl of Train rollouts and print the coverage report. system: the system text the
    rollouts were sampled with, if it replaced the dataset's (sample_vllm.py --system-text)."""
    train = _read(Path(data_dir) / "train.jsonl")
    stats = {}
    selected = rft_dataset(_read(scored_path), train, max_per_drug=max_per_drug, stats=stats, system=system, min_reward=min_reward)
    _write(out_path, selected, lines=True)
    _sidecar(out_path, selected, system)
    rep = rft_report(selected, train)
    print(f"{Path(out_path).name}: {rep['examples']} examples | drugs covered {rep['drugs_covered']} of {rep['drugs_total']}, "
          f"{rep['drugs_without_success']} Train drugs without a successful sample | rescored {stats['rescored']} samples, "
          f"{stats['kept']} succeed now; the stored flags would have wrongly kept {stats['stale_kept']} and wrongly dropped "
          f"{stats['stale_dropped']} | completion chars {_chars([a['completion'] for a in selected])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the A1, A2 and A4-neighbour files")
    b.add_argument("--data", default="data/processed")
    b.add_argument("--out", default="data/processed/arms")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--system-text", help="system-turn text of the A2 prompts (default: the direct-answer text)")
    g.add_argument("--keep-system", action="store_true", help="keep the dataset's system text in the A2 prompts")
    c = sub.add_parser("a4", help="write the in-context-retrieval prompt files (processed-row schema) for sample_vllm.py")
    c.add_argument("--data", default="data/processed")
    c.add_argument("--out", default="data/processed/arms/a4")
    c.add_argument("--system-text", default=DIRECT_SYSTEM, help="system-turn text of the prompts (default: the direct-answer text)")
    n = sub.add_parser("name", help="write the name-only diagnostic prompt files (processed-row schema) for sample_vllm.py")
    n.add_argument("--data", default="data/processed")
    n.add_argument("--out", default="data/processed/arms/name")
    n.add_argument("--system-text", default=DIRECT_SYSTEM, help="system-turn text of the prompts (default: the direct-answer text)")
    k = sub.add_parser("blank", help="write the profile-removed control prompt files (processed-row schema) for sample_vllm.py")
    k.add_argument("--data", default="data/processed")
    k.add_argument("--out", default="data/processed/arms/blank")
    k.add_argument("--system-text", default=DIRECT_SYSTEM, help="system-turn text of the prompts (default: the direct-answer text)")
    k.add_argument("--splits", default="val", help="comma-separated subset of val,test_r,test_t; the default is Val only: name a Test split only after the lock")
    r = sub.add_parser("rft", help="write the A3 file from scored Train rollouts")
    r.add_argument("--scored", required=True)
    r.add_argument("--data", default="data/processed")
    r.add_argument("--out", default="data/processed/arms/a3_rft_train.jsonl")
    r.add_argument("--max-per-drug", type=int, default=2)
    r.add_argument("--min-reward", type=float, help="dense mode: keep fresh valid closed samples with at least this reward instead of identity hits")
    r.add_argument("--system-text", help="system text the rollouts were sampled with, if it replaced the dataset's")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            system = DIRECT_SYSTEM if a.system_text is None else a.system_text
            build(a.data, a.out, None if a.keep_system else system)
        elif a.cmd == "a4":
            a4(a.data, a.out, a.system_text)
        elif a.cmd == "name":
            name_oracle(a.data, a.out, a.system_text)
        elif a.cmd == "blank":
            blank(a.data, a.out, a.system_text, tuple(x for x in a.splits.split(",") if x))
        else:
            rft(a.scored, a.data, a.out, a.max_per_drug, a.system_text, a.min_reward)
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
