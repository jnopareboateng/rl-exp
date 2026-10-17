"""docs/prereg-v1.md and the files that the code runs on must name the same arms (review finding P1-1: the comparator set in the candidates file was not the
registered one, and the difference decided the confirmatory contrast).

The protocol carries one machine-readable line, `<!-- frozen: comparator_set=C0,C1,C3,A0n,A2,A4; rl_set=RL-full,RL-contrastive -->`. The comparator set in
results/arm_candidates.json must equal it (a registered comparator may not be dropped nor an unregistered one added); its RL set may list fewer arms than the
line (an RL arm that did not finish) but never one the line does not name. A selection.json, if present, picks its pair from those sets.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"<!--\s*frozen:\s*comparator_set=([^;]*);\s*rl_set=([^>]*?)\s*-->")


def names(field):
    out = [x.strip() for x in field.split(",")]
    if not out or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", x) for x in out) or len(set(out)) != len(out):
        raise ValueError(f"not a list of distinct arm ids: {field!r}")
    return out


def frozen_sets(text):
    """(comparator_set, rl_set) from the protocol's one `<!-- frozen: ... -->` line, or None when the protocol has none yet; ValueError for a repeated or malformed line."""
    lines = [line.strip() for line in text.splitlines() if re.search(r"<!--\s*frozen", line, re.IGNORECASE)]       # any spelling counts as "there is a line"
    if not lines:
        return None
    if len(lines) > 1:
        raise ValueError("the protocol has more than one frozen line")
    match = LINE.fullmatch(lines[0])
    if not match:
        raise ValueError("the frozen line is malformed: expected '<!-- frozen: comparator_set=A,B; rl_set=C,D -->'")
    return names(match.group(1)), names(match.group(2))


def check(text, candidates, selection=None):
    """AssertionError unless the candidates (and the selection) agree with the protocol's frozen line; returns None when the protocol has no such line."""
    frozen = frozen_sets(text)
    if frozen is None:
        return None
    comparators, rl = frozen
    assert len(candidates["comparator_set"]) == len(set(candidates["comparator_set"])) and len(candidates["rl_set"]) == len(set(candidates["rl_set"])), "a set repeats an arm"
    assert set(candidates["comparator_set"]) == set(comparators), (
        f"comparator_set differs from the protocol: not registered {sorted(set(candidates['comparator_set']) - set(comparators))}, "
        f"registered but absent {sorted(set(comparators) - set(candidates['comparator_set']))}")
    assert candidates["rl_set"] and set(candidates["rl_set"]) <= set(rl), f"rl_set names arms the protocol does not: {sorted(set(candidates['rl_set']) - set(rl))}"
    for family in ("comparator_set", "rl_set"):
        assert set(candidates[family]) <= set(candidates["arms"]), f"{family} names arms without candidates"
    if selection is not None:
        assert selection["comparator"] in comparators and selection["rl"] in rl, "the selection picks an arm outside the registered sets"
        assert set(selection["comparator_set"]) <= set(comparators) and set(selection["rl_set"]) <= set(rl), "the selection's sets are not within the registered sets"
    return frozen


# ------------------------------------------------------------------------------------------------ the real files
def test_the_candidates_and_the_selection_agree_with_the_protocols_frozen_line():
    text = (ROOT / "docs" / "prereg-v1.md").read_text(encoding="utf-8")
    if frozen_sets(text) is None:
        pytest.skip("docs/prereg-v1.md has no '<!-- frozen: comparator_set=...; rl_set=... -->' line yet")
    candidates = json.loads((ROOT / "results" / "arm_candidates.json").read_text(encoding="utf-8"))
    selection_path = ROOT / "results" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    assert check(text, candidates, selection) is not None


# ------------------------------------------------------------------------------------------------ the check itself, on synthetic texts
PROTOCOL = "# Pre-registration\n<!-- frozen: comparator_set=C0,C1,C3,A0n,A2,A4; rl_set=RL-full,RL-contrastive -->\nprose\n"


def cand(**kw):
    sets = {"comparator_set": ["C0", "C1", "C3", "A0n", "A2", "A4"], "rl_set": ["RL-full", "RL-contrastive"], **kw}
    return {**sets, "arms": {a: {"candidates": []} for a in {*sets["comparator_set"], *sets["rl_set"]}}}


def test_the_line_is_parsed_and_a_protocol_without_one_is_skipped_not_passed():
    assert frozen_sets(PROTOCOL) == (["C0", "C1", "C3", "A0n", "A2", "A4"], ["RL-full", "RL-contrastive"])
    assert frozen_sets("no line here\n") is None and check("no line here\n", cand()) is None
    assert frozen_sets("<!--frozen:comparator_set=A , B;rl_set=C-->") == (["A", "B"], ["C"])


@pytest.mark.parametrize("text", [PROTOCOL + PROTOCOL, "<!-- Frozen : comparator_set=A; rl_set=B -->", "<!-- frozen: comparator_set=A,B -->", "<!-- frozen: rl_set=C; comparator_set=A -->",
                                  "<!-- frozen: comparator_set=A,A; rl_set=C -->", "<!-- frozen: comparator_set=; rl_set=C -->", "<!-- frozen: comparator_set=A,B; rl_set=C D -->"])
def test_a_repeated_or_malformed_line_is_an_error(text):
    with pytest.raises(ValueError):
        frozen_sets(text)


def test_candidates_equal_to_the_registered_sets_pass_and_fewer_rl_arms_are_allowed():
    assert check(PROTOCOL, cand())
    assert check(PROTOCOL, cand(rl_set=["RL-full"]))                                    # RL-contrastive did not finish
    assert check(PROTOCOL, cand(comparator_set=["A4", "A2", "A0n", "C3", "C1", "C0"]))   # order is free


@pytest.mark.parametrize("change,message", [
    ({"comparator_set": ["C0", "C1", "A0n", "A2", "A4"]}, "registered but absent \\['C3'\\]"),
    ({"comparator_set": ["C0", "C1", "C3", "A0n", "A2", "A4", "A1"]}, "not registered \\['A1'\\]"),               # an unregistered comparator, the review's P1-1
    ({"comparator_set": ["C0", "C1", "C3", "A0n", "A2", "A1"]}, "not registered \\['A1'\\].*registered but absent \\['A4'\\]"),
    ({"rl_set": ["RL-full", "RL-validity"]}, "rl_set names arms the protocol does not: \\['RL-validity'\\]"),
    ({"rl_set": []}, "rl_set names arms|rl_set"),
    ({"rl_set": ["RL-full", "RL-full"]}, "repeats"),
    ({"comparator_set": ["C0", "C0", "C1", "C3", "A0n", "A2", "A4"]}, "repeats"),
])
def test_a_comparator_the_protocol_does_not_register_or_a_registered_one_that_is_missing_fails(change, message):
    with pytest.raises(AssertionError, match=message):
        check(PROTOCOL, cand(**change))


def test_a_set_may_not_name_an_arm_without_candidates():
    broken = cand()
    broken["arms"].pop("A2")
    with pytest.raises(AssertionError, match="without candidates"):
        check(PROTOCOL, broken)


def test_the_selection_must_pick_from_the_registered_sets():
    selection = {"comparator": "C3", "rl": "RL-full", "comparator_set": ["C0", "C3"], "rl_set": ["RL-full"]}
    assert check(PROTOCOL, cand(), selection)
    for change in ({"comparator": "A1"}, {"rl": "RL-validity"}, {"comparator_set": ["C0", "A1"]}, {"rl_set": ["RL-validity"]}):
        with pytest.raises(AssertionError, match="registered"):
            check(PROTOCOL, cand(), {**selection, **change})
