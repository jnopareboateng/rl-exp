import json

import pytest

from rlexp import arms, ident

SYS = "You are a pharmacology assistant."


def prompt(user="a profile", system=SYS):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def row(rid, group, **extra):
    return {"id": rid, "group": group, "prompt": prompt(), "split": "val", **extra}


def sample(rid, idx, text):
    return {"id": rid, "sample_idx": idx, "split": "val", "text": text}


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
