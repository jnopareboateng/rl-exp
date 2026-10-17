import json
from pathlib import Path

import pytest

from rlexp import arms, ident, metrics

SYS = "You are a pharmacology assistant."


def prompt(user="a profile", system=SYS):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def row(rid, group, **extra):
    return {"id": rid, "group": group, "prompt": prompt(), "split": "val", **extra}


def sample(rid, idx, text, **extra):
    return {"id": rid, "sample_idx": idx, "split": "val", "text": text, **extra}


@pytest.mark.parametrize("a,b", [("Imatinib Mesylate", "imatinib"), ("Éléctro-Drug (INN)", "electrodrug"),
                                 ("sodium  valproate", "valproate"), ("X-123 hydrochloride monohydrate", "x123")])
def test_normalize_drops_forms_accents_and_punctuation(a, b):
    assert ident.normalize(a) == b


@pytest.mark.parametrize("text,expected", [
    ("\n</think>\n\nImatinib", "Imatinib"),
    ("</think>\n\n**Imatinib mesylate**\nIt is a kinase inhibitor.", "Imatinib mesylate"),
    ("</think>\n\n`imatinib`, a kinase inhibitor", "imatinib"),
    ("</think>\n\nimatinib or nilotinib", "imatinib"),
    ("</think>\n\nA\n</think>\n\nOlaparib.", "Olaparib"),
    ("", ""), (None, ""), ("</think>\n\n   \n", "")])
def test_answer_name_takes_the_first_line_after_the_last_think_close(text, expected):
    assert ident.answer_name(text) == expected


def test_accepted_names_add_the_curated_preferred_name_of_the_same_drug_only():
    curated = [{"name": "Gleevec-like", "pref_name": "IMATINIB MESYLATE"}, {"name": "other", "pref_name": "NOT THIS"}, {"name": "gleevec-like", "pref_name": None}]
    assert ident.accepted_names("Gleevec-like", curated) == {"gleeveclike", "imatinib"}
    assert ident.accepted_names("Unlisted", curated) == {"unlisted"}


def test_score_is_drug_balanced_and_pass8_looks_at_the_first_eight():
    rows = [row("a/0", "alpha"), row("a/1", "alpha"), row("b/0", "beta")]
    hit, miss = "</think>\n\nAlpha", "</think>\n\nWrong"
    samples = ([sample("a/0", i, hit) for i in range(2)] + [sample("a/0", i, miss) for i in range(2, 4)]    # row a/0: 2 of 4
               + [sample("a/1", i, miss) for i in range(4)]                                                  # row a/1: 0 of 4
               + [sample("b/0", i, "</think>\n\n") for i in range(4)])                                      # beta: nothing answered
    out = ident.score(samples, rows, [])
    assert out["drugs"] == 2 and out["rows"] == 3 and out["samples"] == 12
    assert out["name_match@1"] == pytest.approx((0.25 + 0.0) / 2)       # alpha: mean(0.5, 0) = 0.25, beta: 0
    assert out["name_pass@8"] == pytest.approx((0.5 + 0.0) / 2)         # alpha: mean(1, 0), beta: 0
    assert out["answered"] == pytest.approx((1.0 + 0.0) / 2)
    late = [sample("a/0", i, miss) for i in range(8)] + [sample("a/0", 8, hit)]
    assert ident.score(late, rows[:1], [])["name_pass@8"] == 0.0         # a hit at index 8 is outside pass@8


def test_ident_prompt_keeps_the_profile_and_asks_for_the_name():
    r = row("a/0", "alpha")
    p = ident.ident_prompt(r)
    assert arms.system_text(p) == ident.IDENT_SYSTEM and arms.user_text(p) == "a profile"
    assert p.endswith("<|im_start|>assistant\n<think>\n")


def test_cli_write_and_score(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    rows = {s: [row(f"{s}/0", f"{s}drug", split=s)] for s in ("val", "test_r", "test_t")}
    for s, rs in rows.items():
        (data / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    ident.main(["write", "--data", str(data), "--out", str(tmp_path / "out")])
    got = [json.loads(line) for line in (tmp_path / "out" / "val.jsonl").read_text().splitlines()]
    assert got[0]["prompt"] == ident.ident_prompt(rows["val"][0]) and got[0]["group"] == "valdrug"
    assert json.loads((tmp_path / "out" / "val.meta.json").read_text())["control"] == "identification"
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(sample("val/0", 0, "</think>\n\nValdrug")) + "\n")
    cur = tmp_path / "cur.json"
    cur.write_text("[]")
    capsys.readouterr()
    ident.main(["score", "--samples", str(samples), "--data", str(data), "--curated", str(cur)])
    assert json.loads(capsys.readouterr().out)["name_match@1"] == 1.0


# ------------------------------------------------------------------ the common failure rule and the input checks
@pytest.mark.parametrize("text,finish,hit", [
    ("</think>\n\nAlpha", "stop", True), ("</think>\n\nAlpha", None, True),        # no finish_reason counts as finished
    ("</think>\n\nAlpha", "abort", True),                                          # only "length" is a cut
    ("</think>\n\nAlpha", "length", False),                                        # cut by the cap, whatever it says
    ("Alpha", "stop", False), ("Alpha", "length", False),                           # the think block never closed
    ("thinking about Alpha...", "length", False)])
def test_only_a_closed_finished_sample_can_match_or_count_as_answered(text, finish, hit):
    s = sample("a/0", 0, text, **({"finish_reason": finish} if finish else {}))
    out = ident.score([s], [row("a/0", "alpha")], [])
    assert out["name_match@1"] == out["name_pass@8"] == out["answered"] == (1.0 if hit else 0.0)
    assert ident.usable(s) is hit and ident.usable({"text": text, "finish_reason": finish}) is hit


def test_one_correct_unclosed_length_terminated_name_does_not_make_pass_at_8_one():
    # the review's counterexample: a name inside unfinished reasoning counted as a match, and one sample became "pass@8"
    samples = [sample("a/0", 0, "Alpha", finish_reason="length")] + [sample("a/0", i, "</think>\n\nWrong") for i in range(1, 8)]
    out = ident.score(samples, [row("a/0", "alpha")], [])
    assert out["name_pass@8"] == 0.0 and out["name_match@1"] == 0.0 and out["answered"] == pytest.approx(7 / 8)
    assert ident.usable is not None and metrics.finished({"finish_reason": "length"}) is False        # one rule, defined in metrics


def test_score_rejects_repeated_pairs_unknown_ids_repeated_data_ids_and_mixed_splits():
    rows, hit = [row("a/0", "alpha"), row("b/0", "beta")], "</think>\n\nAlpha"
    with pytest.raises(metrics.InputError, match=r"1 repeated \(id, sample_idx\) pair"):          # a repeat used to change the mean
        ident.score([sample("a/0", 0, hit), sample("a/0", 0, hit)], rows, [])
    with pytest.raises(metrics.InputError, match=r"1 unknown sample id\(s\) .*zzz"):             # and an unknown id used to vanish
        ident.score([sample("a/0", 0, hit), sample("zzz/0", 0, hit)], rows, [])
    with pytest.raises(metrics.InputError, match=r"repeat 1 id"):                                 # one data row used to overwrite the other
        ident.score([sample("a/0", 0, hit)], rows + [row("a/0", "other")], [])
    with pytest.raises(metrics.InputError, match="split is not their data row's"):
        ident.score([{**sample("a/0", 0, hit), "split": "test_t"}], rows, [])
    mixed = rows + [row("c/0", "gamma", split="test_t")]
    with pytest.raises(metrics.InputError, match="span the splits"):                               # never pooled across splits
        ident.score([sample("a/0", 0, hit), {**sample("c/0", 0, hit), "split": "test_t"}], mixed, [])
    assert issubclass(metrics.InputError, ValueError)                                             # the CLI catches ValueError
    assert ident.score([sample("a/0", 0, hit), sample("a/0", 1, hit)], rows, [])["samples"] == 2


def test_cli_scores_one_split_at_a_time_and_opens_only_the_files_it_names(tmp_path, capsys, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for s in ("val", "test_r", "test_t"):
        (data / f"{s}.jsonl").write_text(json.dumps(row(f"{s}/0", f"{s}drug", split=s)) + "\n")
    cur = tmp_path / "cur.json"
    cur.write_text("[]")
    opened, real = [], ident._read
    monkeypatch.setattr(ident, "_read", lambda path: opened.append(Path(path).name) or real(path))

    def write(name, *samples):
        path = tmp_path / name
        path.write_text("".join(json.dumps(x) + "\n" for x in samples))
        return str(path)
    val = write("val.samples.jsonl", sample("val/0", 0, "</think>\n\nValdrug"))
    ident.main(["score", "--samples", val, "--data", str(data), "--curated", str(cur)])
    assert sorted(opened) == ["val.jsonl", "val.samples.jsonl"]                                    # not test_r, not test_t
    assert json.loads(capsys.readouterr().out) == {"split": "val", "drugs": 1, "rows": 1, "samples": 1, "name_match@1": 1.0,
                                                    "name_pass@8": 1.0, "answered": 1.0}
    both = write("both.samples.jsonl", sample("val/0", 0, "</think>\n\nValdrug"),
                 {**sample("test_t/0", 0, "</think>\n\nWrong"), "split": "test_t"})
    opened.clear()
    ident.main(["score", "--samples", both, "--data", str(data), "--curated", str(cur)])
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [(x["split"], x["name_match@1"]) for x in lines] == [("test_t", 0.0), ("val", 1.0)]    # one line per split, never pooled
    assert "test_r.jsonl" not in opened
    (data / "test_t.jsonl").unlink()
    with pytest.raises(SystemExit, match=r"name split 'test_t' but .*test_t.jsonl does not exist"):
        ident.main(["score", "--samples", both, "--data", str(data), "--curated", str(cur)])
    no_split = write("no_split.jsonl", {k: v for k, v in sample("val/0", 0, "x").items() if k != "split"})
    with pytest.raises(SystemExit, match="lack split"):
        ident.main(["score", "--samples", no_split, "--data", str(data), "--curated", str(cur)])
    with pytest.raises(SystemExit, match="holds no samples"):
        ident.main(["score", "--samples", write("empty.jsonl"), "--data", str(data), "--curated", str(cur)])
