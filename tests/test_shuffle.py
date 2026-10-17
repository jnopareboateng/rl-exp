"""rlexp.shuffle: a row-level derangement within task_type x prompt-length tercile strata (no prompt text of the real data is used)."""
import itertools
import json
import random
from pathlib import Path

import pytest

from rlexp import arms, metrics, shuffle

IBU = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def row(rid, group, cluster, task, length, split="val", **extra):
    """A data row whose prompt is unique (it starts with its row id) and has `length` characters."""
    return {"id": rid, "group": group, "cluster_id": cluster, "task_type": task, "prompt": (f"<{rid}>" + "x" * length)[:length],
            "ref_smiles": IBU, "ref_conn": "CONN", "ref_scaffold": "SCAF", "split": split, "n_prompt_tokens": length // 4, **extra}


def dataset(seed=0, n=48):
    """Three task types, drugs of 1 to 3 rows, components of 1 to 3 drugs, lengths spread over a wide range."""
    rng = random.Random(seed)
    rows, k = [], 0
    while len(rows) < n:
        drug, comp, task = f"drug{k:02d}", k // 2, rng.choice("ABC")
        for v in range(rng.randint(1, 3)):
            rows.append(row(f"{drug}/{v}", drug, comp, rng.choice("ABC") if v else task, rng.randint(200, 6000)))
        k += 1
    return rows


def tercile_of(rows):
    """The tercile (by rank of length within the task type, ties by id) of every row id, computed independently of the module."""
    out = {}
    for t in {r["task_type"] for r in rows}:
        mine = sorted((r for r in rows if r["task_type"] == t), key=lambda r: (len(r["prompt"]), r["id"]))
        out.update({r["id"]: rank * 3 // len(mine) for rank, r in enumerate(mine)})
    return out


# ------------------------------------------------------------------------------------------------------------- the assignment
def test_hungarian_is_a_minimum_cost_perfect_assignment():
    rng = random.Random(1)
    for n in range(1, 7):
        for _ in range(25):
            cost = [[rng.randrange(0, 30) for _ in range(n)] for _ in range(n)]
            got = shuffle.hungarian(cost)
            assert sorted(got) == list(range(n))
            best = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
            assert sum(cost[i][got[i]] for i in range(n)) == best


# --------------------------------------------------------------------------------------------------------------- the invariants
def test_each_row_gets_a_donor_of_its_task_type_from_another_drug_and_component_in_its_length_tercile():
    rows = dataset()
    new, report = shuffle.shuffle_rows(rows, 0)
    by_id, tercile = {r["id"]: r for r in rows}, tercile_of(rows)
    assert len(new) == len(rows) and report["fallbacks"] == [] and report["balance"]["same_component_rows"] == 0
    for old, now in zip(rows, new):
        donor = by_id[report["row_mapping"][old["id"]]]
        assert now["prompt"] == donor["prompt"] != old["prompt"]
        assert donor["task_type"] == old["task_type"] and donor["group"] != old["group"] and donor["cluster_id"] != old["cluster_id"]
        assert tercile[donor["id"]] == tercile[old["id"]]
    b = report["balance"]
    assert (b["same_task_type_share"], b["same_tercile_share"], b["same_drug_rows"], b["max_rows_per_donor"]) == (1.0, 1.0, 0, 1)
    shuffle.check(rows, new, report)


def test_a_prompt_is_shown_once_and_the_row_mapping_pins_which_one():
    rows = dataset(3)
    new, report = shuffle.shuffle_rows(rows, 4)
    mapping = report["row_mapping"]
    assert sorted(mapping) == sorted(r["id"] for r in rows) and sorted(mapping.values()) == sorted(mapping)     # a bijection of the rows
    shown = [r["prompt"] for r in new]
    assert len(set(shown)) == len(shown) == len(rows)                       # no donor prompt twice (every prompt of the fixture is unique)
    for t in {r["task_type"] for r in rows}:
        assert sorted(r["prompt"] for r in rows if r["task_type"] == t) == sorted(n["prompt"] for n in new if n["task_type"] == t)
    # the drug-level mapping lists the donor drugs of every row of the drug, none of them the drug itself
    for drug, donors in report["mapping"].items():
        assert drug not in donors and donors == sorted({mapping[r["id"]].split("/")[0] for r in rows if r["group"] == drug})
    # forced case: one pair of drugs per task type, so the only valid mapping is the swap inside the pair
    pairs = [row("a/0", "a", 1, "A", 1000), row("b/0", "b", 2, "A", 3000), row("c/0", "c", 3, "B", 500), row("d/0", "d", 4, "B", 900)]
    swapped, rep = shuffle.shuffle_rows(pairs, 0)
    assert rep["row_mapping"] == {"a/0": "b/0", "b/0": "a/0", "c/0": "d/0", "d/0": "c/0"}
    assert [r["prompt"] for r in swapped] == [pairs[1]["prompt"], pairs[0]["prompt"], pairs[3]["prompt"], pairs[2]["prompt"]]
    assert rep["mapping"] == {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}


def test_only_the_prompt_changes_the_input_order_stays_and_the_scorer_runs_on_the_output():
    rows = dataset(5, 30)
    new, _ = shuffle.shuffle_rows(rows, 1)
    assert [r["id"] for r in new] == [r["id"] for r in rows]
    assert [{k: v for k, v in r.items() if k != "prompt"} for r in new] == [{k: v for k, v in r.items() if k != "prompt"} for r in rows]
    own = [{"id": r["id"], "sample_idx": 0, "text": f"t</think>\n{IBU}", "finish_reason": "stop", "n_tokens": 5} for r in rows]
    scored = metrics.score_samples(own, new)                                # the references are those of the original rows
    assert scored == metrics.score_samples(own, rows) and all(s["valid"] and s["conn_match"] for s in scored)


def test_seeded_and_deterministic_and_another_seed_gives_another_valid_mapping():
    rows = dataset(2)
    a, b = shuffle.shuffle_rows(rows, 3), shuffle.shuffle_rows(rows, 3)
    assert a == b
    other = shuffle.shuffle_rows(rows, 4)
    assert other[1]["row_mapping"] != a[1]["row_mapping"] and other[1]["balance"]["same_task_type_share"] == 1.0
    assert len({tuple(shuffle.shuffle_rows(rows, s)[1]["row_mapping"].items()) for s in range(8)}) >= 6


# --------------------------------------------------------------------------------------------------- feasibility and fallbacks
def test_a_component_holding_half_the_rows_is_deranged_without_overlap_on_every_seed():
    # the review's hard case: 6 of 12 drugs in one component (random search left an overlap in 22 of 200 seeds)
    rows = [row(f"d{k:02d}/0", f"d{k:02d}", 0 if k < 6 else k, "A", 1000 + 10 * k) for k in range(12)]
    for seed in range(200):
        new, report = shuffle.shuffle_rows(rows, seed)
        assert report["balance"]["same_component_rows"] == 0 and report["balance"]["same_drug_rows"] == 0, seed
        by_id = {r["id"]: r for r in rows}
        assert all(by_id[d]["cluster_id"] != by_id[o]["cluster_id"] for o, d in report["row_mapping"].items()), seed


def test_a_dominant_component_leaves_the_smallest_possible_overlap_and_says_so():
    rows = [row("a/0", "a", 1, "A", 1000), row("b/0", "b", 1, "A", 1100), row("c/0", "c", 1, "A", 1200), row("d/0", "d", 2, "A", 1300)]
    new, report = shuffle.shuffle_rows(rows, 0)
    assert report["balance"]["same_component_rows"] == 2 * 3 - 4 == 2        # 3 of 4 rows in one component: at least 2m - n must share it
    assert report["balance"]["same_drug_rows"] == 0
    [fallback] = report["fallbacks"]
    assert (fallback["task_type"], fallback["level"], fallback["component_overlap_rows"]) == ("A", "task_type", 2)


def test_a_tercile_without_a_valid_donor_is_merged_with_its_neighbour_and_then_the_length_bin_is_dropped():
    t1 = [row(f"s/{i}", "s", 1, "A", 100 + i) for i in range(3)]             # the shortest tercile is three rows of one drug
    t1 += [row("p/0", "p", 2, "A", 200), row("q/0", "q", 3, "A", 210), row("r/0", "r", 4, "A", 220)]
    t1 += [row("u/0", "u", 5, "A", 300), row("v/0", "v", 6, "A", 310), row("w/0", "w", 7, "A", 320)]
    t2 = [row("s2/0", "s2", 11, "B", 100), row("s2/1", "s2", 11, "B", 101), row("s2/2", "s2", 11, "B", 150),     # merged group still fails
          row("t/0", "t", 12, "B", 160), row("u2/0", "u2", 13, "B", 400), row("v2/0", "v2", 14, "B", 410)]
    new, report = shuffle.shuffle_rows(t1 + t2, 0)
    a = next(s for s in report["strata"] if s["task_type"] == "A")
    b = next(s for s in report["strata"] if s["task_type"] == "B")
    assert a["terciles"] == [3, 3, 3] and [(g["bins"], g["rows"], g["level"]) for g in a["groups"]] == [([0, 1], 6, "merged"), ([2], 3, "tercile")]
    assert b["terciles"] == [2, 2, 2] and [(g["bins"], g["rows"], g["level"]) for g in b["groups"]] == [([0, 1, 2], 6, "task_type")]
    assert [(f["task_type"], f["level"]) for f in report["fallbacks"]] == [("A", "merged"), ("B", "task_type")]
    assert report["balance"]["same_drug_rows"] == 0 and report["balance"]["same_task_type_share"] == 1.0
    by_id = {r["id"]: r for r in t1 + t2}
    tercile = tercile_of(t1 + t2)
    for o, d in report["row_mapping"].items():                                # the third tercile of A kept its own length bin
        if tercile[o] == 2 and by_id[o]["task_type"] == "A":
            assert tercile[d] == 2


@pytest.mark.parametrize("rows,match", [
    ([row("a/0", "a", 1, "A", 100)], "task type 'A'"),                                                           # a single row
    ([row("a/0", "a", 1, "A", 100), row("a/1", "a", 1, "A", 200), row("b/0", "b", 2, "A", 300)], "task type 'A'"),   # a drug owns most rows
    ([row("a/0", "a", 1, "A", 100), row("b/0", "b", 2, "A", 200), row("c/0", "c", 3, "B", 300)], "task type 'B'"),    # one type is a single row
])
def test_a_task_type_that_cannot_be_shown_another_drug_raises(rows, match):
    with pytest.raises(ValueError, match=match):
        shuffle.shuffle_rows(rows, 0)


def test_a_null_cluster_id_is_a_component_of_its_own_drug_and_bad_inputs_raise():
    rows = [row("a/0", "a", None, "A", 100), row("b/0", "b", None, "A", 200)]
    new, report = shuffle.shuffle_rows(rows, 0)
    assert report["balance"]["same_component_rows"] == 0 and report["row_mapping"] == {"a/0": "b/0", "b/0": "a/0"}
    no_id = [{k: v for k, v in r.items() if k != "cluster_id"} for r in rows]
    assert shuffle.shuffle_rows(no_id, 0)[1]["balance"]["same_component_rows"] == 0
    with pytest.raises(ValueError, match="two components"):
        shuffle.shuffle_rows([row("a/0", "a", 1, "A", 100), row("a/1", "a", 2, "A", 150), row("b/0", "b", 3, "A", 200)], 0)
    with pytest.raises(ValueError, match="repeat an id"):
        shuffle.shuffle_rows(rows + rows[:1], 0)
    with pytest.raises(ValueError, match="lacks task_type"):
        shuffle.shuffle_rows([{k: v for k, v in r.items() if k != "task_type"} for r in rows], 0)


def test_random_datasets_either_satisfy_every_invariant_or_are_provably_impossible():
    for seed in range(60):
        rows = dataset(seed, random.Random(seed).randint(6, 40))
        impossible = any(n < 2 or max(c for c in __import__("collections").Counter(r["group"] for r in rows if r["task_type"] == t).values()) * 2 > n
                         for t in {r["task_type"] for r in rows} for n in [sum(r["task_type"] == t for r in rows)])
        if impossible:
            with pytest.raises(ValueError):
                shuffle.shuffle_rows(rows, seed)
            continue
        new, report = shuffle.shuffle_rows(rows, seed)
        shuffle.check(rows, new, report)
        for t in {r["task_type"] for r in rows}:
            mine = [r for r in rows if r["task_type"] == t]
            comps = __import__("collections").Counter(r["cluster_id"] for r in mine)
            m, n = max(comps.values()), len(mine)
            overlap = sum(g["component_overlap_rows"] for s in report["strata"] if s["task_type"] == t for g in s["groups"])
            assert overlap >= max(0, 2 * m - n) and (overlap == 0 if 2 * m <= n else True), (seed, t)


def _crafted(mapping, rows):
    """A shuffle written by hand from a row mapping, with a report that claims every balance number is fine."""
    by_id = {r["id"]: r for r in rows}
    new = [{**r, "prompt": by_id[mapping[r["id"]]]["prompt"]} for r in rows]
    report = {"row_mapping": mapping, "balance": {"same_task_type_share": 1.0, "same_drug_rows": 0, "max_rows_per_donor": 1}}
    return new, report


def test_check_refuses_a_shuffle_that_breaks_exactly_one_invariant():
    rows = [row("a/0", "a", 1, "A", 100), row("b/0", "b", 2, "B", 100), row("c/0", "c", 3, "A", 200), row("d/0", "d", 4, "B", 200),
            row("a/1", "a", 1, "A", 300), row("e/0", "e", 5, "A", 400)]
    good = {"a/0": "c/0", "c/0": "a/0", "b/0": "d/0", "d/0": "b/0", "a/1": "e/0", "e/0": "a/1"}
    new, report = _crafted(good, rows)
    shuffle.check(rows, new, report)                                          # the hand-made baseline is valid
    cross = {**good, "a/0": "b/0", "b/0": "a/0", "c/0": "d/0", "d/0": "c/0"}   # a bijection of other drugs, but across task types
    with pytest.raises(ValueError, match="another task type"):
        shuffle.check(rows, *_crafted(cross, rows))
    own = {**good, "a/0": "a/1", "a/1": "a/0", "c/0": "e/0", "e/0": "c/0"}    # a row shown a prompt of its own drug, same task type
    with pytest.raises(ValueError, match="its own drug"):
        shuffle.check(rows, *_crafted(own, rows))
    shared = {**good, "c/0": "b/0"}                                            # two rows with one donor, one donor unused
    with pytest.raises(ValueError, match="bijection"):
        shuffle.check(rows, *_crafted(shared, rows))
    wrong_prompt = [dict(r) for r in new]
    wrong_prompt[0]["prompt"] = rows[0]["prompt"]                              # the mapping says c/0, the prompt is the row's own
    with pytest.raises(ValueError, match="prompt of its donor"):
        shuffle.check(rows, wrong_prompt, report)
    changed = [dict(r) for r in new]
    changed[1]["ref_smiles"] = "CCC"
    with pytest.raises(ValueError, match="other than the prompt"):
        shuffle.check(rows, changed, report)
    reordered = list(reversed(new))
    with pytest.raises(ValueError, match="bijection"):
        shuffle.check(rows, reordered, report)
    liar = {**report, "balance": {"same_task_type_share": 0.9, "same_drug_rows": 0, "max_rows_per_donor": 1}}
    with pytest.raises(ValueError, match="contradict"):
        shuffle.check(rows, new, liar)


# ------------------------------------------------------------------------------------------------------------------- balance
def test_balance_statistics_are_the_hand_computed_ones():
    # four rows in two task types with two drugs each: the only valid mapping swaps inside the type
    rows = [row("a/0", "a", 1, "A", 100), row("b/0", "b", 2, "A", 300), row("c/0", "c", 3, "B", 500), row("d/0", "d", 4, "B", 900)]
    _, report = shuffle.shuffle_rows(rows, 0)
    b = report["balance"]
    assert b["mean_abs_length_gap"] == (200 + 200 + 400 + 400) / 4 == 300.0
    assert b["baseline_gap_any_other_drug"] == round((200 + 400 + 800 + 200 + 600 + 400) / 6, 1) == 433.3
    assert b["baseline_gap_same_task_type"] == (200 + 400) / 2 == 300.0
    assert b["length_correlation"] == round(150000 / 350000, 4) == 0.4286
    assert (b["same_task_type_share"], b["same_drug_rows"], b["same_component_rows"], b["max_rows_per_donor"]) == (1.0, 0, 0, 1)
    assert b["same_tercile_share"] == 0.0                                    # two-row types are merged: the donor is in the other tercile
    assert [(f["task_type"], f["level"]) for f in report["fallbacks"]] == [("A", "task_type"), ("B", "task_type")]


def test_the_stratified_shuffle_matches_length_far_better_than_a_swap_of_whole_prompts():
    rows = dataset(7, 48)
    _, report = shuffle.shuffle_rows(rows, 0)
    b = report["balance"]
    assert b["mean_abs_length_gap"] < 0.6 * min(b["baseline_gap_same_task_type"], b["baseline_gap_any_other_drug"])
    assert b["length_correlation"] > 0.7


# ---------------------------------------------------------------------------------------------------------------------- the CLI
def test_cli_writes_only_the_requested_split_with_a_meta_file_and_prints_the_balance_without_prompt_text(tmp_path, capsys, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rows = dataset(9, 36)
    for split in ("val", "test_r", "test_t"):
        (data / f"{split}.jsonl").write_text("".join(json.dumps({**r, "split": split}) + "\n" for r in rows))
    read = []
    original = arms._read
    monkeypatch.setattr(arms, "_read", lambda path: read.append(Path(path).name) or original(path))
    shuffle.main(["write", "--data", str(data), "--out", str(tmp_path / "out"), "--splits", "val"])
    assert read == ["val.jsonl"] and sorted(p.name for p in (tmp_path / "out").iterdir()) == ["val.jsonl", "val.meta.json"]
    meta = json.loads((tmp_path / "out" / "val.meta.json").read_text())
    assert meta["control"] == "shuffled_profile" and meta["rows"] == len(rows) and meta["seed"] == 0
    assert set(meta) >= {"strata", "fallbacks", "balance", "mapping", "row_mapping", "design"}
    assert meta["balance"]["same_task_type_share"] == 1.0 and all(s["terciles"] and s["groups"] for s in meta["strata"])
    shuffle.main(["write", "--data", str(data), "--out", str(tmp_path / "default")])     # no --splits: Val only, the Test files stay closed
    assert read[1:] == ["val.jsonl"] and sorted(p.name for p in (tmp_path / "default").iterdir()) == ["val.jsonl", "val.meta.json"]
    capsys.readouterr()
    shuffle.main(["write", "--data", str(data), "--out", str(tmp_path / "out"), "--splits", "val"])     # the run whose output is checked below
    printed, text = capsys.readouterr().out, json.dumps(meta)
    for r in rows[:5]:
        assert r["prompt"][:12] not in text and r["prompt"][:12] not in printed       # no prompt text anywhere
    assert "same task type 1.000" in printed and "mean length gap" in printed and printed.startswith(f"val.jsonl: {len(rows)} rows")
    written = [json.loads(line) for line in (tmp_path / "out" / "val.jsonl").read_text().splitlines()]
    assert [r["id"] for r in written] == [r["id"] for r in rows] and {r["split"] for r in written} == {"val"}
    with pytest.raises(SystemExit, match="error: ValueError"):                             # an unshuffleable split is an error, not a file
        (data / "val.jsonl").write_text(json.dumps({**rows[0], "split": "val"}) + "\n")
        shuffle.main(["write", "--data", str(data), "--out", str(tmp_path / "bad"), "--splits", "val"])
    assert not (tmp_path / "bad").exists()
    # several splits: one that cannot be shuffled stops the call before ANY file is written
    (data / "val.jsonl").write_text("".join(json.dumps({**r, "split": "val"}) + "\n" for r in rows))
    (data / "test_r.jsonl").write_text(json.dumps({**rows[0], "split": "test_r"}) + "\n")
    with pytest.raises(SystemExit, match="error: ValueError"):
        shuffle.main(["write", "--data", str(data), "--out", str(tmp_path / "half"), "--splits", "val,test_r"])
    assert not (tmp_path / "half").exists()
