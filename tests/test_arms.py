"""arms.py on synthetic rows: A1 trace cut, A2 answer-only, A3 rejection sampling, A4 neighbours + in-context prompt, CLI."""
import importlib.util
import json
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rlexp import arms, chem

ASPIRIN, IBUPROFEN = "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
PARACETAMOL, CAFFEINE = "CC(=O)Nc1ccc(O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
NAPROXEN, NICOTINE = "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CN1CCCC1c1cccnc1"
SMILES = [ASPIRIN, IBUPROFEN, PARACETAMOL, CAFFEINE, NAPROXEN, NICOTINE]
SYSTEM = "You are a pharmacology assistant."
GPU = Path(__file__).resolve().parents[1] / "gpu"
REASON = "Step by step reasoning about the profile."


def prompt(user, system=SYSTEM):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def comp(answer_section, reasoning=REASON):
    return f"{reasoning}\n</think>\n\n{answer_section}<|im_end|>"


def row(rid, user="a profile", smiles=ASPIRIN, completion=None, group=None, **extra):
    r = {"id": rid, "group": group or rid.split("/")[0], "prompt": prompt(user), "ref_smiles": smiles}
    if completion is not None:
        r["completion"] = completion
    return {**r, **extra}


def test_fixture_smiles_are_extractable():
    assert all(chem.extract_answer("</think>\n\n" + s) == s for s in SMILES)


# ---------------------------------------------------------------- A1
def test_trace_cut_drops_trailing_prose_and_keeps_one_think():
    r = row("d/0", completion=comp(f"Proposed structure: {ASPIRIN}\n\nThis fits the profile.\nIt is an acid."))
    out = arms.trace_sft(r)
    assert out["cut"] is True
    assert out["completion"] == f"{REASON}\n</think>\n\nProposed structure: {ASPIRIN}<|im_end|>"
    assert out["completion"].count("</think>") == 1 and out["completion"].count("<|im_end|>") == 1
    assert out["completion"].endswith("<|im_end|>")
    assert (out["id"], out["prompt"]) == (r["id"], r["prompt"]) and set(out) == {"id", "prompt", "completion", "cut"}


def test_trace_cut_is_a_noop_when_the_answer_line_is_last():
    for section in (ASPIRIN, f"Proposed structure: {ASPIRIN}"):
        r = row("d/0", completion=comp(section))
        assert arms.trace_sft(r)["completion"] == r["completion"] and arms.trace_sft(r)["cut"] is True


@pytest.mark.parametrize("line", ["{s}", "`{s}`", "SMILES: {s}", "SMILES: `{s}`", "**Proposed structure:** `{s}`",
                                  "I propose {s}.", "- {s}", "> **{s}**", "  {s}  ", "The final answer is {s}"])
def test_trace_cut_label_and_markdown_forms(line):
    r = row("d/0", completion=comp(line.format(s=ASPIRIN) + "\n\nThe reasoning above supports this.\nAnother sentence."))
    out = arms.trace_sft(r)
    assert out["cut"] is True
    assert out["completion"] == comp(line.format(s=ASPIRIN))
    assert chem.extract_answer(out["completion"]) == chem.extract_answer(r["completion"]) == ASPIRIN


@pytest.mark.parametrize("section", [
    f"The molecule {ASPIRIN} is the parent compound of this class.",
    f"The molecule {ASPIRIN} is a parent drug — a known acid.",                        # a dash later in the sentence is no separator
    f"Proposed structure: {ASPIRIN}\n\nAs shown, {ASPIRIN} fits the profile.",       # the last occurrence is inline
    f"After weighing all of the evidence the proposed structure is {ASPIRIN}",         # a sentence, not a label
    f"After weighing all of the evidence the proposed structure is {ASPIRIN} (aspirin)",
    f"I propose {ASPIRIN}. This fits the profile.",                                    # a period starts a new sentence
])
def test_trace_uncut_when_the_answer_is_inline_in_prose(section):
    r = row("d/0", completion=comp(section + "\n\nTrailing prose."))
    out = arms.trace_sft(r)
    assert out["cut"] is False and out["completion"] == r["completion"]
    assert arms.trace_cut(r["completion"])[1] == "uncut"


@pytest.mark.parametrize("line, kept", [
    ("Proposed structure: {s} — an acetylated salicylate that fits the profile", "Proposed structure: {s}"),
    ("Proposed structure: {s} – en dash gloss", "Proposed structure: {s}"),
    ("Proposed structure: {s} - hyphen gloss", "Proposed structure: {s}"),
    ("Proposed structure: {s} (aspirin, the parent drug)", "Proposed structure: {s}"),
    ("SMILES: {s}: aspirin, the parent drug", "SMILES: {s}"),
    ("{s} — gloss", "{s}"),
    ("I propose {s} — gloss.", "I propose {s}"),
    ("**Proposed structure:** `{s}` — gloss", "**Proposed structure:** `{s}`"),             # markdown closing the token stays
    ("**{s}** (gloss)", "**{s}**"),
])
def test_trace_cut_drops_a_description_after_the_smiles(line, kept):
    for trailing in ("", "\n\nTrailing prose.\nMore."):
        r = row("d/0", completion=comp(line.format(s=ASPIRIN) + trailing))
        out = arms.trace_sft(r)
        assert out["cut"] is True
        assert out["completion"] == comp(kept.format(s=ASPIRIN))                         # the SMILES ends the text before <|im_end|>
        assert out["completion"].count("</think>") == 1 and out["completion"].count("<|im_end|>") == 1
        assert chem.extract_answer(out["completion"]) == chem.extract_answer(r["completion"]) == ASPIRIN


def test_trace_cut_follows_the_last_occurrence():
    r = row("d/0", completion=comp(f"Candidates include {IBUPROFEN} among others.\nProposed: {ASPIRIN}\nBecause of the profile.\nMore."))
    out = arms.trace_sft(r)
    assert out["cut"] is True
    assert out["completion"] == comp(f"Candidates include {IBUPROFEN} among others.\nProposed: {ASPIRIN}")


def test_trace_cut_is_refused_if_it_would_change_the_extracted_answer(monkeypatch):
    r = row("d/0", completion=comp(f"Proposed structure: {ASPIRIN}\n\nTrailing prose."))
    real = chem.extract_answer
    monkeypatch.setattr(chem, "extract_answer", lambda t: real(t) if "Trailing" in t else IBUPROFEN)    # an extractor the cut changes
    out = arms.trace_sft(r)
    assert out["cut"] is False and out["completion"] == r["completion"]


def test_trace_unextractable_keeps_the_full_completion():
    r = row("d/0", completion=comp("I could not determine a structure.\nMore text."))
    out = arms.trace_sft(r)
    assert out["cut"] is False and out["completion"] == r["completion"]
    assert arms.trace_cut(r["completion"])[1] == "unextractable"


def test_trace_cut_keeps_a_code_fence_closed():
    r = row("d/0", completion=comp(f"Proposed structure:\n```\n{ASPIRIN}\n```\n\nTrailing prose."))
    assert arms.trace_sft(r)["completion"] == comp(f"Proposed structure:\n```\n{ASPIRIN}\n```")
    r = row("d/0", completion=comp(f"```smiles\n{ASPIRIN}\n```"))               # fence is already the last line
    assert arms.trace_sft(r)["completion"] == r["completion"]
    r = row("d/0", completion=comp(f"Proposed structure: {ASPIRIN}\n\n```\nunrelated block\n```"))   # a later fence is not ours
    assert arms.trace_sft(r)["completion"] == comp(f"Proposed structure: {ASPIRIN}")


def test_trace_cut_closes_the_fence_after_dropping_a_description():
    r = row("d/0", completion=comp(f"Proposed structure:\n```smiles\n{ASPIRIN} — aspirin, the parent drug\n```\n\nTrailing prose."))
    out = arms.trace_sft(r)
    assert out["cut"] is True and out["completion"] == comp(f"Proposed structure:\n```smiles\n{ASPIRIN}\n```")
    r = row("d/0", completion=comp(f"```\n{ASPIRIN} (aspirin)\n```"))                       # the fence is the last line
    assert arms.trace_sft(r)["completion"] == comp(f"```\n{ASPIRIN}\n```")
    r = row("d/0", completion=comp(f"`{ASPIRIN}` — gloss\n\n```\nunrelated block\n```"))     # a later fence is not ours
    assert arms.trace_sft(r)["completion"] == comp(f"`{ASPIRIN}`")
    r = row("d/0", completion=comp(f"```\n{ASPIRIN} — see ```x``` here\n```"))              # fences inside the dropped description do not count
    assert arms.trace_sft(r)["completion"] == comp(f"```\n{ASPIRIN}\n```")


def test_trace_requires_exactly_one_think_close():
    for bad in ("no close<|im_end|>", "a </think> b </think> c<|im_end|>"):
        with pytest.raises(ValueError, match=r"d/9.*exactly one"):
            arms.trace_sft(row("d/9", completion=bad))


# ---------------------------------------------------------------- A2
def test_answer_only_format_and_extractability():
    r = row("d/0", smiles=IBUPROFEN, completion=comp(IBUPROFEN))
    out = arms.answer_only_sft(r)
    assert arms.DIRECT_SYSTEM == "Reply with only the SMILES string of the drug described. No explanation."
    assert out == {"id": "d/0", "prompt": arms.with_system(r["prompt"], arms.DIRECT_SYSTEM),
                   "completion": f"\n</think>\n\n{IBUPROFEN}<|im_end|>"}
    assert chem.extract_answer(out["completion"]) == IBUPROFEN
    assert chem.extract_answer("</think>\n\n" + IBUPROFEN) == IBUPROFEN                    # the nothink-mode text form
    assert chem.extract_answer("reasoning </think>\n\n" + IBUPROFEN) == IBUPROFEN
    assert "<think>\n\n</think>\n\n" + IBUPROFEN in out["prompt"] + out["completion"]       # the native thinking-off layout


def test_answer_only_system_text_options():
    r = row("d/0", smiles=IBUPROFEN, completion=comp(IBUPROFEN))
    kept, custom = arms.answer_only_sft(r, system=None), arms.answer_only_sft(r, system="Answer with a SMILES.")
    assert kept["prompt"] == r["prompt"]                                                   # the dataset's own text
    assert arms.system_text(custom["prompt"]) == "Answer with a SMILES."
    assert arms.split_prompt(custom["prompt"])[1:] == arms.split_prompt(r["prompt"])[1:]   # user turn and opener untouched
    assert kept["completion"] == custom["completion"] == arms.answer_only_sft(r)["completion"]


def test_with_system_replaces_the_text_of_the_first_system_turn_only():
    p = prompt("the user text", system="old system\nover two lines")
    assert arms.system_text(p) == "old system\nover two lines"
    assert arms.with_system(p, "NEW \\1 \\g<0> text") == prompt("the user text", system="NEW \\1 \\g<0> text")   # no regex escapes
    assert arms.with_system(p, "") == prompt("the user text", system="")
    out = arms.with_system(p + "<|im_start|>system\nsecond<|im_end|>\n", "X")
    assert out.startswith("<|im_start|>system\nX<|im_end|>") and out.endswith("second<|im_end|>\n")
    for bad in ("<|im_start|>user\nhi<|im_end|>\n", "<|im_start|>system without a newline<|im_end|>"):
        with pytest.raises(ValueError):
            arms.with_system(bad, "X")
        with pytest.raises(ValueError):
            arms.system_text(bad)


def load_sampler(monkeypatch):
    monkeypatch.syspath_prepend(str(GPU))
    spec = importlib.util.spec_from_file_location("sample_vllm_under_arms_test", GPU / "sample_vllm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_with_system_equals_the_samplers_system_text_flag(tmp_path, monkeypatch):
    pack = load_sampler(monkeypatch)
    double = "<|im_start|>system\nfirst<|im_end|>\n" + prompt("u4", "second")       # two system turns: only the first is replaced
    prompts = [prompt("u1", "old"), prompt("u2", "multi\nline system"), prompt("u3", ""), double]
    rows = [{"id": f"v{i}", "group": f"g{i}", "split": "val", "prompt": p} for i, p in enumerate(prompts)]
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for text in (arms.DIRECT_SYSTEM, "X \\1 \\g<0>\nmore", ""):
        args = types.SimpleNamespace(splits="val", data=tmp_path, system_text=text, limit_drugs=None, prompt_subset_seed=0)
        assert [r["prompt"] for r in pack.load_rows(args)] == [arms.with_system(p, text) for p in prompts]
    rows[0]["prompt"] = "<|im_start|>user\nno system turn<|im_end|>\n<|im_start|>assistant\n<think>\n"
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):                                                        # the sampler refuses it, as with_system does
        pack.load_rows(types.SimpleNamespace(splits="val", data=tmp_path, system_text="X", limit_drugs=None, prompt_subset_seed=0))
    with pytest.raises(ValueError):
        arms.with_system(rows[0]["prompt"], "X")


# ---------------------------------------------------------------- A4
TOPICS = ["kinase tyrosine phosphorylation oncology", "beta adrenergic heart rate hypertension",
          "bacterial ribosome antibiotic gram positive", "xanthine adenosine receptor stimulant caffeine",
          "cyclooxygenase prostaglandin analgesic fever", "hepatic acetaminophen glucuronidation overdose"]


def train_rows(clusters=None):
    """6 drugs x 2 views; clusters maps a drug index to its cluster_id (default: its own index)."""
    clusters = clusters or {}
    return [row(f"t{k}/{i}", f"{TOPICS[k]} view {i} plasma half life", SMILES[k], cluster_id=clusters.get(k, k))
            for k in range(6) for i in range(2)]


def test_split_prompt_roundtrip_and_error():
    p = prompt("the user text")
    head, user, tail = arms.split_prompt(p)
    assert user == "the user text" and head + user + tail == p
    assert head.endswith("<|im_start|>user\n") and tail.startswith("<|im_end|>") and tail.endswith("<think>\n")
    with pytest.raises(ValueError):
        arms.split_prompt("<|im_start|>system\nonly a system turn<|im_end|>\n")


def test_icl_prompt_structure():
    r = row("q/0", "QUERY TEXT about a kinase")
    nb = [("NEIGHBOUR ONE text", "SMILES-ONE"), ("NEIGHBOUR TWO text", "SMILES-TWO"), ("NEIGHBOUR THREE text", "SMILES-THREE")]
    p = arms.icl_prompt(r, nb)
    head, user, tail = arms.split_prompt(p)
    head0, user0, tail0 = arms.split_prompt(r["prompt"])
    assert (head, tail) == (head0, tail0)                       # system turn and assistant opener unchanged
    assert p.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n") and p.endswith("<|im_start|>assistant\n<think>\n")
    assert p.count("<|im_end|>") == 2 and p.count("<|im_start|>user") == 1
    assert user == "\n\n".join([arms.ICL_HEADER, "NEIGHBOUR ONE text", "Answer SMILES: SMILES-ONE", "NEIGHBOUR TWO text",
                                "Answer SMILES: SMILES-TWO", "NEIGHBOUR THREE text", "Answer SMILES: SMILES-THREE",
                                "Now the task:", "QUERY TEXT about a kinase"])
    assert user.startswith("Worked examples from similar profiles (each followed by its known answer):")
    assert user.endswith(user0)                                 # the original task text comes last, verbatim
    assert user.index("ONE") < user.index("TWO") < user.index("THREE") < user.index("Now the task:") < user.index("QUERY")


def test_nearest_train_ranks_by_tfidf_cosine():
    train = train_rows()
    queries = [row("v0/0", "tyrosine kinase oncology marker", group="v0"), row("v1/0", "xanthine adenosine stimulant", group="v1"),
               row("v2/0", "cyclooxygenase prostaglandin fever", group="v2")]
    near = arms.nearest_train(queries, train)
    assert list(near) == ["v0/0", "v1/0", "v2/0"]
    assert [near[q][0][:2] for q in near] == ["t0", "t3", "t4"]
    assert all(len(v) == 3 and len(set(v)) == 3 for v in near.values())
    assert arms.nearest_train(queries, train, k=1)["v0/0"] == near["v0/0"][:1]


def test_nearest_train_is_deterministic_and_breaks_ties_by_lowest_id():
    train = [row("b/0", "same words here", group="b"), row("a/0", "same words here", group="a"),
             row("c/0", "same words here", group="c"), row("z/0", "different vocabulary entirely", group="z")]
    q = [row("v/0", "same words here", group="v")]
    expect = ["a/0", "b/0", "c/0"]
    assert arms.nearest_train(q, train) == {"v/0": expect}
    for seed in range(5):
        shuffled = random.Random(seed).sample(train, len(train))
        assert arms.nearest_train(q, shuffled) == {"v/0": expect}


def test_nearest_train_excludes_the_query_cluster():
    train = train_rows({1: 0})                  # drugs 0 and 1 share cluster 0
    q = [row("v/0", "tyrosine kinase oncology phosphorylation", group="v", cluster_id=0)]
    near = arms.nearest_train(q, train, k=12)["v/0"]
    assert not any(i.startswith(("t0/", "t1/")) for i in near) and len(near) == 8
    other = [row("v/0", "tyrosine kinase oncology phosphorylation", group="v", cluster_id=99)]
    assert arms.nearest_train(other, train)["v/0"][0].startswith("t0/")      # the same text from another cluster does retrieve it


def test_nearest_train_falls_back_to_group_without_cluster_id():
    train = [row(f"t{k}/{i}", f"{TOPICS[k]} view {i}", SMILES[k]) for k in range(3) for i in range(2)]
    q = [row("x/0", TOPICS[0], group="t0")]                       # same group as t0/*, no cluster_id anywhere
    near = arms.nearest_train(q, train, k=6)["x/0"]
    assert not any(i.startswith("t0/") for i in near) and len(near) == 4
    mixed = [row("x/0", TOPICS[0], group="x", cluster_id=0)]       # query has a cluster, Train rows do not: only the group rule applies
    assert arms.nearest_train(mixed, train)["x/0"][0].startswith("t0/")


def test_nearest_train_for_train_queries_excludes_self_and_own_drug():
    train = train_rows({3: 0})
    near = arms.nearest_train(train, train, k=3)
    assert len(near) == len(train)
    by_id = {r["id"]: r for r in train}
    for qid, ids in near.items():
        q = by_id[qid]
        assert qid not in ids and len(ids) == 3
        assert all(by_id[i]["group"] != q["group"] and by_id[i]["cluster_id"] != q["cluster_id"] for i in ids)


def test_nearest_train_never_returns_the_query_row_itself():
    train = train_rows()
    twin = {**train[0], "group": "renamed", "cluster_id": 999}            # same id, different group and cluster: only the id rule applies
    near = arms.nearest_train([twin], train, k=11)[twin["id"]]
    assert twin["id"] not in near and len(near) == 11


def test_nearest_train_matches_an_independent_tfidf():
    import math
    import re
    from collections import Counter
    rng = random.Random(7)
    words = [f"w{i}" for i in range(30)]
    texts = [" ".join(rng.choice(words[:5 + 25 * (rng.random() < .5)]) for _ in range(rng.randint(3, 40))) for _ in range(40)]
    train = [row(f"t{i:02d}/0", t, group=f"t{i:02d}") for i, t in enumerate(texts)]
    docs = [Counter(re.findall(r"[a-z0-9]+", t)) for t in texts]
    df = Counter(w for d in docs for w in d)
    idf = {w: math.log((1 + len(docs)) / (1 + df[w])) + 1 for w in df}

    def vec(c):
        v = {w: (1 + math.log(n)) * idf[w] for w, n in c.items() if w in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    queries = [row(f"q{i}/0", " ".join(rng.choice(words) for _ in range(rng.randint(1, 25))), group=f"q{i}") for i in range(15)]
    near = arms.nearest_train(queries, train, k=5)
    for q in queries:
        qv = vec(Counter(re.findall(r"[a-z0-9]+", q["prompt"].split("user\n")[1].split("<|im_end|>")[0].lower())))
        score = [(-round(sum(qv.get(w, 0.0) * x for w, x in vec(d).items()), 9), r["id"]) for d, r in zip(docs, train)]
        assert near[q["id"]] == [i for _, i in sorted(score)[:5]]


def test_nearest_train_returns_fewer_than_k_when_train_is_small_and_rejects_duplicate_ids():
    train = [row("a/0", "alpha beta", group="a"), row("b/0", "alpha gamma", group="b")]
    assert arms.nearest_train([row("v/0", "alpha", group="v")], train, k=5) == {"v/0": ["a/0", "b/0"]}
    assert arms.nearest_train([row("a/1", "alpha", group="a")], train) == {"a/1": ["b/0"]}
    with pytest.raises(ValueError, match="duplicate"):
        arms.nearest_train([row("v/0")], train + [row("a/0", group="a2")])


def test_icl_pipeline_never_shows_the_query_cluster():
    train = [row(f"t{k}/{i}", f"{TOPICS[k]} MARK{k} view {i}", SMILES[k], cluster_id=0 if k < 2 else k) for k in range(6) for i in range(2)]
    q = row("v/0", "kinase tyrosine oncology phosphorylation adrenergic heart rate", group="v", cluster_id=0)
    near = arms.nearest_train([q], train)["v/0"]
    assert len(near) == 3 and not any(i.startswith(("t0/", "t1/")) for i in near)
    p = arms.icl_prompt(q, arms.neighbour_set({r["id"]: r for r in train}, near))
    assert "MARK0" not in p and "MARK1" not in p and ASPIRIN not in p and IBUPROFEN not in p
    assert all(f"MARK{i[1]}" in p and f"Answer SMILES: {SMILES[int(i[1])]}" in p for i in near)
    assert [p.index(f"MARK{i[1]}") for i in near] == sorted(p.index(f"MARK{i[1]}") for i in near)     # neighbours in order


# ---------------------------------------------------------------- A3
IBU_ALT = "OC(=O)C(C)c1ccc(CC(C)C)cc1"                    # ibuprofen written differently
REFS = {"d0": IBUPROFEN, "d1": ASPIRIN, "d2": PARACETAMOL, "d3": CAFFEINE}


def data_rows():
    return [row(f"{g}/{v}", smiles=REFS[g], group=g) for g, v in (("d0", "a"), ("d0", "b"), ("d1", "a"), ("d2", "a"), ("d3", "a"))]


def sample(rid, idx, n, ok=True, answer=None, **kw):
    """A scored sample. Its text answers the drug's reference (ok) or another molecule, and the stored flags agree with
    the text unless a keyword overrides them (a stale scored.jsonl)."""
    answer = answer or (REFS[rid.split("/")[0]] if ok else NAPROXEN)
    s = {"id": rid, "group": rid.split("/")[0], "split": "train", "sample_idx": idx, "text": f"T-{rid}-{idx}</think>\n\n{answer}",
         "finish_reason": "stop", "n_tokens": n, "closed": True, "conn_match": ok, "mode": "think"}
    return {**s, **kw}


def picked(out):
    return sorted(o["completion"].removeprefix("T-").split("</think>")[0] for o in out)


def test_rft_keeps_only_closed_conn_match_stop_samples():
    scored = [sample("d0/a", 0, 10, ok=False), sample("d0/a", 1, 11, text="T-d0/a-1 never closes " + IBUPROFEN, conn_match=False, closed=False),
              sample("d0/a", 2, 12, finish_reason="length"), sample("d0/a", 3, 13), sample("d0/a", 4, 14, answer=IBU_ALT),
              sample("d2/a", 0, 10, ok=False), sample("d2/a", 1, 10, finish_reason="length")]
    assert picked(arms.rft_dataset(scored, data_rows())) == ["d0/a-3", "d0/a-4"]       # a different spelling of the molecule counts


def test_rft_require_closed_follows_the_fresh_closed_flag(monkeypatch):
    monkeypatch.setattr(arms.reward, "score", lambda text, ref: {"conn_match": True, "closed": False})
    scored = [sample("d0/a", 0, 10)]
    assert arms.rft_dataset(scored, data_rows()) == []
    assert picked(arms.rft_dataset(scored, data_rows(), require_closed=False)) == ["d0/a-0"]


def test_rft_scores_each_stop_finished_sample_against_its_own_row(monkeypatch):
    seen = []
    monkeypatch.setattr(arms.reward, "score", lambda text, ref: seen.append((text, ref["id"])) or {"conn_match": True, "closed": True})
    scored = [sample("d0/b", 0, 10), sample("d1/a", 0, 10, finish_reason="length"), sample("zz/a", 0, 10, answer=ASPIRIN)]
    arms.rft_dataset(scored, data_rows())
    assert seen == [(f"T-d0/b-0</think>\n\n{IBUPROFEN}", "d0/b")]          # no scoring for a truncated sample or an unknown row


def test_rft_drops_a_sample_whose_reference_changed_after_scoring():
    scored = [sample("d1/a", 0, 10)]                                          # scored while d1's reference was aspirin
    rows = data_rows()
    assert picked(arms.rft_dataset(scored, rows)) == ["d1/a-0"]
    rows[2] = {**rows[2], "ref_smiles": PARACETAMOL}                          # the reference was corrected since
    stats = {}
    assert arms.rft_dataset(scored, rows, stats=stats) == []
    assert stats == {"rescored": 1, "kept": 0, "stale_kept": 1, "stale_dropped": 0}


def test_rft_rescoring_overrides_every_kind_of_stale_flag():
    scored = [sample("d0/a", 0, 10, answer=ASPIRIN),                                          # stored success for an old reference
              sample("d0/a", 1, 11, conn_match=False),                                        # stored failure, answers the current one
              sample("d0/a", 2, 12),                                                          # consistent success
              sample("d0/b", 0, 13, answer=NAPROXEN),                                         # stored success, wrong molecule
              sample("d0/b", 1, 14, text="T-d0/b-1 no think " + IBUPROFEN),                   # stored closed and conn_match, no </think>
              sample("d0/b", 2, 15, closed=False),                                            # stored not closed, the text closes and answers
              sample("d1/a", 0, 16, finish_reason="length"),                                  # correct but truncated: never a candidate
              sample("d1/a", 1, 17, ok=False)]                                                # consistent failure
    stored = [s for s in scored if s["conn_match"] and s["closed"] and s["finish_reason"] == "stop"]
    assert len(stored) == 4                                                                   # what the stored flags alone would keep
    stats = {}
    out = arms.rft_dataset(scored, data_rows(), max_per_drug=9, stats=stats)
    assert picked(out) == ["d0/a-1", "d0/a-2", "d0/b-2"]
    assert stats == {"rescored": 7, "kept": 3, "stale_kept": 3, "stale_dropped": 2}


def test_rft_raises_with_the_row_id_on_a_reference_that_does_not_standardize():
    with pytest.raises(ValueError, match="d9/a"):
        arms.rft_dataset([sample("d9/a", 0, 10, answer=ASPIRIN)], [row("d9/a", smiles="not a smiles", group="d9")])


def test_rft_max_per_drug_prefers_short_samples_spread_over_prompt_rows():
    scored = [sample("d0/a", 0, 100), sample("d0/a", 1, 90), sample("d0/a", 2, 95), sample("d0/b", 0, 200), sample("d0/b", 1, 300)]
    assert picked(arms.rft_dataset(scored, data_rows(), max_per_drug=1)) == ["d0/a-1"]
    assert picked(arms.rft_dataset(scored, data_rows(), max_per_drug=2)) == ["d0/a-1", "d0/b-0"]      # one per row before filling
    assert picked(arms.rft_dataset(scored, data_rows(), max_per_drug=3)) == ["d0/a-1", "d0/a-2", "d0/b-0"]
    assert picked(arms.rft_dataset(scored, data_rows(), max_per_drug=9)) == ["d0/a-0", "d0/a-1", "d0/a-2", "d0/b-0", "d0/b-1"]
    only_a = [s for s in scored if s["id"] == "d0/a"]
    assert picked(arms.rft_dataset(only_a, data_rows(), max_per_drug=2)) == ["d0/a-1", "d0/a-2"]       # fills from the one row


def test_rft_ties_go_to_the_lowest_sample_idx_then_id():
    scored = [sample("d0/a", 2, 50), sample("d0/a", 1, 50), sample("d0/b", 0, 50)]
    assert picked(arms.rft_dataset(scored, data_rows(), max_per_drug=2)) == ["d0/a-1", "d0/b-0"]
    assert arms.rft_dataset(scored[::-1], data_rows(), max_per_drug=2) == arms.rft_dataset(scored, data_rows(), max_per_drug=2)


def test_rft_rows_and_completion_end_token():
    rows = data_rows()
    scored = [sample("d0/a", 0, 10, text=f"why</think>\n\n{IBUPROFEN}"), sample("d1/a", 0, 10, text=f"why</think>\n\n{ASPIRIN}<|im_end|>")]
    out = arms.rft_dataset(scored, rows)
    assert [o["completion"] for o in out] == [f"why</think>\n\n{IBUPROFEN}<|im_end|>", f"why</think>\n\n{ASPIRIN}<|im_end|>"]
    assert [(o["id"], o["prompt"]) for o in out] == [("d0/a#0", rows[0]["prompt"]), ("d1/a#0", rows[2]["prompt"])]   # row id # sample_idx
    assert all(set(o) == {"id", "prompt", "completion"} for o in out)
    nothink = arms.rft_dataset([sample("d0/a", 0, 10, text=f"</think>\n\n{IBUPROFEN}", mode="nothink")], rows)[0]
    assert nothink["completion"] == f"\n</think>\n\n{IBUPROFEN}<|im_end|>"        # same layout as the A2 completion


def test_rft_rows_of_one_prompt_row_get_unique_ids_that_the_sft_reader_accepts(tmp_path):
    rows = data_rows()
    scored = [sample("d0/a", 1, 10), sample("d0/a", 2, 12)]                    # two successes of one Train prompt, both kept at max_per_drug=2
    out = arms.rft_dataset(scored, rows, max_per_drug=2)
    assert [o["id"] for o in out] == ["d0/a#1", "d0/a#2"]                       # the source row and the sample index stay in the id
    assert [o["prompt"] for o in out] == [rows[0]["prompt"]] * 2 and picked(out) == ["d0/a-1", "d0/a-2"]
    path = tmp_path / "a3_rft_train.jsonl"
    path.write_text("".join(json.dumps(o) + "\n" for o in out))
    spec = importlib.util.spec_from_file_location("sft_lora_under_arms_test", GPU / "sft_lora.py")
    sft_lora = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sft_lora)
    assert sft_lora.read_rows(path, "train") == out                             # a duplicate id makes the reader raise
    assert arms.rft_report(out, rows)["drugs_covered"] == 1                     # the ids still map back to their drug
    with pytest.raises(ValueError, match="repeat"):
        arms.rft_dataset(scored + scored, rows, max_per_drug=9)                 # a repeated (id, sample_idx) pair is refused


def test_rft_prompt_carries_the_system_text_the_rollouts_were_sampled_with():
    rows, scored = data_rows(), [sample("d0/a", 0, 10)]
    assert arms.rft_dataset(scored, rows)[0]["prompt"] == rows[0]["prompt"]
    out = arms.rft_dataset(scored, rows, system=arms.DIRECT_SYSTEM)[0]
    assert out["prompt"] == arms.with_system(rows[0]["prompt"], arms.DIRECT_SYSTEM) and out["prompt"] != rows[0]["prompt"]


def test_rft_ignores_samples_of_rows_outside_the_data_and_reports_coverage():
    rows = data_rows()
    scored = [sample("d0/a", 0, 10), sample("d1/a", 0, 10, ok=False), sample("d3/a", 0, 10), sample("val9/a", 0, 10, answer=IBUPROFEN)]
    out = arms.rft_dataset(scored, rows)
    assert picked(out) == ["d0/a-0", "d3/a-0"]                                     # the val9 sample has no Train row
    assert arms.rft_report(out, rows) == {"examples": 2, "drugs_covered": 2, "drugs_total": 4, "drugs_without_success": 2}
    assert arms.rft_report([], rows)["drugs_without_success"] == 4


# ---------------------------------------------------------------- CLI
def write_data(path):
    def split(name, n, with_completion, offset=0):
        rows = []
        for k in range(n):
            smiles = SMILES[(k + offset) % 6]
            r = row(f"{name}{k}/0", f"{TOPICS[(k + offset) % 6]} SECRETWORD view", smiles, split=name, cluster_id=offset * 10 + k)
            if with_completion:
                r["completion"] = comp(f"Proposed structure: {smiles}\n\nTrailing prose that is dropped." if k % 3 == 0 else
                                       smiles if k % 3 == 1 else f"The molecule {smiles} is the answer here.")
            rows.append(r)
        return rows
    data = {"train": split("t", 6, True), "val": split("v", 3, True, 1), "test_r": split("r", 2, False, 2), "test_t": split("x", 2, False, 3)}
    path.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return data


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cli_build_writes_the_arm_files_and_prints_only_counts(tmp_path, capsys):
    data = write_data(tmp_path / "data")
    out = tmp_path / "out"
    arms.main(["build", "--data", str(tmp_path / "data"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["a1_trace_train.jsonl", "a1_trace_val.jsonl", "a2_answer_train.jsonl", "a2_answer_val.jsonl",
         "a1_trace_train.meta.json", "a1_trace_val.meta.json", "a2_answer_train.meta.json", "a2_answer_val.meta.json",
         "a4_neighbours_val.json", "a4_neighbours_test_r.json", "a4_neighbours_test_t.json"])
    a1, a2 = jsonl(out / "a1_trace_train.jsonl"), jsonl(out / "a2_answer_train.jsonl")
    assert [a["id"] for a in a1] == [r["id"] for r in data["train"]] == [a["id"] for a in a2]
    assert all(a["prompt"] == r["prompt"] for a, r in zip(a1, data["train"]))                   # A1 keeps the dataset's system text
    assert all(arms.system_text(a["prompt"]) == arms.DIRECT_SYSTEM for a in a2)                 # A2 uses the direct-answer text
    assert json.loads((out / "a2_answer_train.meta.json").read_text()) == {
        "file": "a2_answer_train.jsonl", "rows": 6, "system_source": "direct", "system_text": arms.DIRECT_SYSTEM}
    assert json.loads((out / "a1_trace_val.meta.json").read_text()) == {
        "file": "a1_trace_val.jsonl", "rows": 3, "system_source": "dataset", "system_text": SYSTEM}
    assert [a["cut"] for a in a1] == [True, True, False] * 2                       # k%3==2 rows are inline
    assert a1[0]["completion"].endswith(f"Proposed structure: {data['train'][0]['ref_smiles']}<|im_end|>")
    assert all(a["completion"].startswith("\n</think>\n\n") for a in a2)
    near = json.loads((out / "a4_neighbours_val.json").read_text())
    assert list(near) == [r["id"] for r in data["val"]] and all(len(v) == 3 and set(v) <= {r["id"] for r in data["train"]} for v in near.values())
    assert "a1_trace_train.jsonl: 6 rows | cut 4 (66.7%, text dropped on 2) | uncut 2 (33.3%) | unextractable 0" in printed
    assert "a2_answer_val.jsonl: 3 rows | system text direct | extract_answer recovers ref_smiles on 3" in printed
    assert "a4_neighbours_test_t.json: 2 queries, 6 neighbours" in printed and "p50" in printed and "p90" in printed and " max " in printed
    assert "SECRETWORD" not in printed and "Trailing prose" not in printed       # no prompt or completion text


def test_cli_build_system_text_flags(tmp_path):
    data = write_data(tmp_path / "data")
    for name, flags, source, text in (("keep", ["--keep-system"], "dataset", SYSTEM),
                                      ("custom", ["--system-text", "Custom text."], "custom", "Custom text.")):
        out = tmp_path / name
        arms.main(["build", "--data", str(tmp_path / "data"), "--out", str(out), *flags])
        assert all(arms.system_text(a["prompt"]) == text for s in ("train", "val") for a in jsonl(out / f"a2_answer_{s}.jsonl"))
        assert json.loads((out / "a2_answer_val.meta.json").read_text())["system_source"] == source
        assert json.loads((out / "a2_answer_val.meta.json").read_text())["system_text"] == text
        assert [a["prompt"] for a in jsonl(out / "a1_trace_train.jsonl")] == [r["prompt"] for r in data["train"]]   # A1 never changes it
    with pytest.raises(SystemExit) as e:                                                   # the two flags exclude each other
        arms.main(["build", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "x"), "--keep-system", "--system-text", "T"])
    assert e.value.code == 2 and not (tmp_path / "x").exists()


def a4_data(path):
    """Train: 6 drugs x 2 views with markers. v0 shares cluster 0 with Train drug 0 (its best textual match) and must not see it.
    The val rows (and the two test_t rows) are near-duplicates of each other, so retrieving from the own split would show."""
    extra = {"approval_year": 2020, "n_prompt_tokens": 17, "ref_conn": "CONN", "ref_inchikey": "KEY"}
    data = {"train": [row(f"t{k}/{i}", f"{TOPICS[k]} TRAINMARK{k}{i} plasma", SMILES[k], split="train", cluster_id=k, completion=comp(SMILES[k]), **extra)
                      for k in range(6) for i in range(2)],
            "val": [row("v0/0", "tyrosine kinase oncology phosphorylation inhibitor VALMARK0", NAPROXEN, split="val", cluster_id=0, completion=comp(NAPROXEN), **extra),
                    row("v1/0", "tyrosine kinase oncology phosphorylation inhibitor VALMARK1", NICOTINE, split="val", cluster_id=11, completion=comp(NICOTINE), **extra)],
            "test_r": [row("r0/0", "bacterial ribosome antibiotic TESTRMARK0", NAPROXEN, split="test_r", cluster_id=12, **extra)],
            "test_t": [row("x0/0", "xanthine adenosine caffeine stimulant TESTTMARK0", NICOTINE, split="test_t", cluster_id=13, **extra),
                       row("x1/0", "xanthine adenosine caffeine stimulant TESTTMARK1", NAPROXEN, split="test_t", cluster_id=14, **extra)]}
    path.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return data


def test_cli_a4_writes_processed_row_files_with_icl_prompts(tmp_path, capsys):
    src = a4_data(tmp_path / "data")
    out = tmp_path / "a4"
    arms.main(["a4", "--data", str(tmp_path / "data"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{s}.{ext}" for s in ("val", "test_r", "test_t") for ext in ("jsonl", "meta.json"))
    by_id = {r["id"]: r for r in src["train"]}
    for split in ("val", "test_r", "test_t"):
        got = jsonl(out / f"{split}.jsonl")
        near = arms.nearest_train(src[split], src["train"])
        assert len(got) == len(src[split])
        for g, r in zip(got, src[split]):
            assert list(g) == list(r)                                                      # every field, in the same order
            assert {k: v for k, v in g.items() if k != "prompt"} == {k: v for k, v in r.items() if k != "prompt"}
            expect = arms.with_system(arms.icl_prompt(r, arms.neighbour_set(by_id, near[r["id"]])), arms.DIRECT_SYSTEM)
            assert g["prompt"] == expect and g["prompt"].endswith("<|im_start|>assistant\n<think>\n")
            assert arms.system_text(g["prompt"]) == arms.DIRECT_SYSTEM
            examples, task = arms.user_text(g["prompt"]).split("Now the task:\n\n")
            assert task == arms.user_text(r["prompt"]) and examples.startswith(arms.ICL_HEADER)
            assert examples.count("Answer SMILES:") == 3 and examples.count("TRAINMARK") == 3        # Train rows only
            assert not any(m in examples for m in ("VALMARK", "TESTRMARK", "TESTTMARK"))
    v0 = arms.user_text(jsonl(out / "val.jsonl")[0]["prompt"])
    assert "TRAINMARK0" not in v0                                                          # v0's cluster holds Train drug 0: excluded
    assert arms.nearest_train([{**src["val"][0], "cluster_id": 99}], src["train"])["v0/0"][0].startswith("t0/")   # ...its best match
    assert json.loads((out / "val.meta.json").read_text()) == {
        "file": "val.jsonl", "rows": 2, "system_source": "direct", "system_text": arms.DIRECT_SYSTEM, "neighbours": 3}
    assert "val.jsonl: 2 rows, 3 neighbours each | prompt chars p50" in printed and " p90 " in printed and " max " in printed
    assert "longest ICL prompt over all splits:" in printed and "tokens at 4 chars/token" in printed
    assert not any(m in printed for m in ("TRAINMARK", "VALMARK", "TESTRMARK", "TESTTMARK"))         # counts only


def test_cli_a4_system_text_override_and_sampler_consumes_the_output(tmp_path):
    src = a4_data(tmp_path / "data")
    out = tmp_path / "a4"
    arms.main(["a4", "--data", str(tmp_path / "data"), "--out", str(out), "--system-text", "Custom text."])
    assert all(arms.system_text(r["prompt"]) == "Custom text." for s in ("val", "test_r", "test_t") for r in jsonl(out / f"{s}.jsonl"))
    assert json.loads((out / "test_r.meta.json").read_text())["system_source"] == "custom"
    arms.main(["a4", "--data", str(tmp_path / "data"), "--out", str(out)])                 # back to the default text
    run = subprocess.run([sys.executable, "-S", str(GPU / "sample_vllm.py"), "--model", "fake", "--data", str(out),
                          "--splits", "val,test_r,test_t", "--k", "2", "--max-tokens", "7", "--out", str(tmp_path / "run"), "--dry-run"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    plan = json.loads(run.stdout)
    assert plan["prompts"] == 5 and plan["drugs"] == len({r["group"] for s in ("val", "test_r", "test_t") for r in src[s]})


def test_cli_rft_writes_the_file_and_reports_coverage(tmp_path, capsys):
    data = write_data(tmp_path / "data")
    ref = {r["id"]: r["ref_smiles"] for r in data["train"] + data["val"]}
    scored = [sample("t0/0", 0, 10, answer=ref["t0/0"]), sample("t1/0", 0, 10, answer=NAPROXEN, conn_match=False),
              sample("t2/0", 0, 10, answer=ref["t2/0"]), sample("t3/0", 0, 10, answer=NAPROXEN),   # t3: stored success, wrong now
              sample("v0/0", 0, 10, answer=ref["v0/0"])]                                          # a val sample: not a Train row
    (tmp_path / "scored.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scored))
    out = tmp_path / "a3.jsonl"
    arms.main(["rft", "--scored", str(tmp_path / "scored.jsonl"), "--data", str(tmp_path / "data"), "--out", str(out)])
    assert [o["id"] for o in jsonl(out)] == ["t0/0#0", "t2/0#0"]
    printed = capsys.readouterr().out
    assert "2 examples | drugs covered 2 of 6, 4 Train drugs without a successful sample" in printed
    assert "rescored 4 samples, 2 succeed now; the stored flags would have wrongly kept 1 and wrongly dropped 0" in printed
    assert [o["prompt"] for o in jsonl(out)] == [r["prompt"] for r in data["train"] if r["id"] in ("t0/0", "t2/0")]
    assert json.loads((tmp_path / "a3.meta.json").read_text()) == {
        "file": "a3.jsonl", "rows": 2, "system_source": "dataset", "system_text": SYSTEM}
    arms.main(["rft", "--scored", str(tmp_path / "scored.jsonl"), "--data", str(tmp_path / "data"), "--out", str(out), "--system-text", "Custom."])
    assert all(arms.system_text(o["prompt"]) == "Custom." for o in jsonl(out))
    assert json.loads((tmp_path / "a3.meta.json").read_text())["system_source"] == "custom"


def test_cli_fails_cleanly_on_bad_input(tmp_path):
    with pytest.raises(SystemExit, match="error"):
        arms.main(["build", "--data", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
    data = write_data(tmp_path / "data")
    bad = jsonl(tmp_path / "data" / "train.jsonl")
    bad[0]["completion"] = "no think close<|im_end|>"
    (tmp_path / "data" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(SystemExit, match="t0/0.*exactly one"):
        arms.main(["build", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")])


def test_name_prompt_is_the_name_only_under_the_direct_system_text():
    r = row("v0/0", "SECRETPROFILE kinase inhibitor", NAPROXEN, group="naproxen")
    p = arms.name_prompt(r)
    assert arms.user_text(p) == "Drug: naproxen" and "SECRETPROFILE" not in p
    assert arms.system_text(p) == arms.DIRECT_SYSTEM and p.endswith("<|im_start|>assistant\n<think>\n")
    assert arms.system_text(arms.name_prompt(r, "Custom.")) == "Custom."


def test_cli_name_writes_processed_row_files_and_prints_only_counts(tmp_path, capsys):
    src = a4_data(tmp_path / "data")
    out = tmp_path / "name"
    arms.main(["name", "--data", str(tmp_path / "data"), "--out", str(out)])
    printed = capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{s}.{ext}" for s in ("val", "test_r", "test_t") for ext in ("jsonl", "meta.json"))
    for split in ("val", "test_r", "test_t"):
        got = jsonl(out / f"{split}.jsonl")
        assert len(got) == len(src[split])
        for g, r in zip(got, src[split]):
            assert list(g) == list(r) and {k: v for k, v in g.items() if k != "prompt"} == {k: v for k, v in r.items() if k != "prompt"}
            assert g["prompt"] == arms.name_prompt(r)
    assert json.loads((out / "val.meta.json").read_text())["control"] == "name_oracle"
    assert "val.jsonl: 2 rows | prompt chars p50" in printed and not any(m in printed for m in ("VALMARK", "TESTRMARK", "TESTTMARK"))


def test_rft_dense_mode_keeps_valid_closed_samples_above_the_reward_bar_best_first():
    ref = REFS["d1"]
    near = ref + "C"                      # a valid molecule that is not the reference: partial reward only
    rows = data_rows()
    s_hit, s_near, s_junk = (sample("d1/a", 0, 30, answer=ref), sample("d1/a", 1, 10, answer=near),
                             sample("d1/a", 2, 5, answer="not a molecule"))
    scored = [s_junk, s_near, s_hit]
    fresh = {s["sample_idx"]: arms.reward.score(s["text"], rows[2]) for s in scored}
    assert fresh[0]["reward"] == 1.0 and 0.05 < fresh[1]["reward"] < 1.0 and not fresh[2]["valid"]
    stats = {}
    out = arms.rft_dataset(scored, rows, max_per_drug=2, stats=stats, min_reward=0.05)
    assert [o["id"] for o in out] == ["d1/a#0", "d1/a#1"]                     # the identity hit first (reward 1.0), then the partial one
    assert stats["rescored"] == 3 and stats["kept"] == 2
    above = arms.rft_dataset(scored, rows, max_per_drug=2, min_reward=fresh[1]["reward"] + 1e-6)
    assert [o["id"] for o in above] == ["d1/a#0"]                             # the bar drops the partial answer
    assert arms.rft_dataset(scored, rows, max_per_drug=2) == arms.rft_dataset([s_hit], rows, max_per_drug=2)    # identity mode unchanged


def test_rft_dense_mode_ignores_length_terminated_samples_and_stale_flags():
    rows = data_rows()
    cut = sample("d1/a", 0, 256, answer=REFS["d1"], finish_reason="length")
    stale = sample("d1/a", 1, 20, answer="not a molecule", valid=True, closed=True, reward=0.9)      # stored flags say success, text says no
    stats = {}
    assert arms.rft_dataset([cut, stale], rows, stats=stats, min_reward=0.05) == []
    assert stats["rescored"] == 1 and stats["stale_kept"] == 1 and stats["kept"] == 0
