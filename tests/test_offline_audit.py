"""scripts/offline_audit.py: the extractor audit and the T5 check, on synthetic samples (no data files, no model output)."""
import importlib.util
import json
from pathlib import Path

import pytest

from rlexp import chem

ROOT = Path(__file__).resolve().parents[1]

PARA = "CC(=O)Nc1ccc(O)cc1"                 # paracetamol
ASPIRIN, ASPIRIN2 = "CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1OC(C)=O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBU_FRAGMENT = "c1ccccc1CC(C)C"
REFS = {"drug-para": {"ref_smiles": PARA, "ref_alt_smiles": []}, "drug-caff": {"ref_smiles": CAFFEINE, "ref_alt_smiles": []}}


@pytest.fixture(scope="module")
def oa():
    spec = importlib.util.spec_from_file_location("offline_audit_under_test", ROOT / "scripts" / "offline_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kinds(oa, text):
    return oa.classify(chem.extract_answer(text), oa.last_line_answer(text))


def sample(i, text, sample_idx=0, group="drug-para", finish="stop", split="val", **extra):
    return {"id": i, "group": group, "split": split, "sample_idx": sample_idx, "text": text, "finish_reason": finish, "n_tokens": 7,
            "model": "m", "mode": "think", "temperature": 1.0, "top_p": 1.0, "max_tokens": 256, "seed": 0, **extra}


# ------------------------------------------------------------------------------------------------ reader B
@pytest.mark.parametrize("text", [
    f"reasoning</think>\n{PARA}",
    f"</think>\n`{PARA}`",
    f"</think>\n**{PARA}**",
    f"</think>\n```smiles\n{PARA}\n```",
    f"</think>\n{PARA}\nHope this helps",              # prose after the answer is not a SMILES-only line
    f"</think>\r\n{PARA}\r\n",
    f"</think>\n{PARA}<|im_end|>",
    f"</think>\nCCO\nHello there\n{PARA}\n\n",
    f"</think>\n{PARA}.",                              # a closing full stop, as chem.extract_answer strips it
    f"</think>\n{PARA},",
    f"</think>\n{PARA};",
    f"</think>\n{PARA}:",
    f"</think>\n`{PARA}`.",
    f"</think>\n**{PARA}.**",
])
def test_last_line_answer_reads_the_smiles_only_line(oa, text):
    assert oa.last_line_answer(text) == PARA


def test_reader_b_strips_exactly_what_reader_a_strips_so_a_closing_full_stop_is_no_disagreement(oa):
    assert oa.STRIP == chem._STRIP == "`*.,;:"
    for end in ".,;:":
        assert kinds(oa, f"</think>\n{PARA}{end}") == ("agree",)


def test_last_line_answer_takes_the_last_such_line_and_applies_no_length_guard(oa):
    assert oa.last_line_answer(f"</think>\n{PARA}\nCCO") == "CCO"        # reader A refuses 3-character tokens, B does not
    assert oa.last_line_answer(f"</think>\nCCO\n{PARA}") == PARA


@pytest.mark.parametrize("text", [
    f"{PARA}",                                          # think block not closed
    "</think>\nThe drug is paracetamol.",               # prose
    "</think>\nParacetamol",                            # one word that is not a SMILES
    f"</think>\nSMILES: {PARA}",                        # a label and a SMILES on one line
    f"</think>\n{PARA} is my answer",                   # RDKit reads the first token of this and takes the rest as a name
    "</think>\n[*]CC(=O)O",                              # a dummy atom does not standardise
    "</think>\n",
    None,
    42,
])
def test_last_line_answer_reads_nothing_here(oa, text):
    assert oa.last_line_answer(text) is None


# ------------------------------------------------------------------------------------------------ the kinds of disagreement
def test_classify_each_kind(oa):
    assert kinds(oa, f"</think>\n{PARA}") == ("agree",)
    assert kinds(oa, "no answer here") == ("agree_none",)
    assert kinds(oa, f"</think>\nThe answer is {PARA} for this profile.") == ("a_only",)     # A reads a token inside a sentence
    assert kinds(oa, f"</think>\n{PARA} is my answer") == ("a_only",)                        # ... also one that opens the line
    assert kinds(oa, "</think>\nCCO") == ("b_only", "b_only_guard")                          # A's 8-character guard
    assert kinds(oa, f"</think>\n{PARA}\nsee {IBU_FRAGMENT}") == ("differ",)                  # A reads the last token, B the last line


def test_classify_compares_molecules_not_spellings(oa):
    assert oa.classify(ASPIRIN, ASPIRIN2) == ("agree",)
    assert oa.classify(ASPIRIN, PARA) == ("differ",)
    assert oa.classify("[*]CC(=O)O", None) == ("agree_none",)        # a dummy-atom answer is no molecule for the scorer either
    assert oa.classify(None, None) == ("agree_none",)


# ------------------------------------------------------------------------------------------------ counts and rates
def test_audit_extractor_counts(oa):
    rows = [sample("a", f"</think>\n{PARA}"),                                           # agree
            sample("b", "</think>\nno structure here", sample_idx=1),                   # agree_none
            sample("c", f"</think>\nThe answer is {PARA} for this profile.", sample_idx=2),      # a_only, 18 heavy atoms
            sample("d", f"</think>\nThe answer is {PARA} for this profile.", sample_idx=3, finish="length"),   # a_only, cut
            sample("e", "</think>\nCCO", sample_idx=4),                                 # b_only, 3 heavy atoms: small
            sample("f", f"</think>\n{PARA}\nsee {IBU_FRAGMENT}", sample_idx=5)]         # differ
    c = oa.audit_extractor(rows, min_heavy=8)
    assert (c["samples"], c["finished"]) == (6, 5)
    assert (c["a_reads"], c["b_reads"]) == (4, 3)
    assert {k: c[k] for k in ("agree", "agree_none", "a_only", "b_only", "b_only_guard", "differ")} == \
        {"agree": 1, "agree_none": 1, "a_only": 2, "b_only": 1, "b_only_guard": 1, "differ": 1}
    assert (c["disagree"], c["disagree_finished"]) == (4, 3)             # the cut sample counts in the first only
    assert (c["a_only_small"], c["b_only_small"]) == (0, 1)
    assert c["rate"] == pytest.approx(4 / 6, abs=1e-6) and c["rate_finished"] == pytest.approx(3 / 5, abs=1e-6)
    assert oa.audit_extractor([], 8)["rate"] == 0.0


def test_finished_is_metrics_finished_stop_or_unrecorded_but_not_length_or_abort(oa):
    rows = [sample("a", f"</think>\n{PARA}", 0, finish="stop"), sample("b", f"</think>\n{PARA}", 1, finish="length"), sample("c", f"</think>\n{PARA}", 2, finish="abort"),
            {k: v for k, v in sample("d", f"</think>\n{PARA}", 3).items() if k != "finish_reason"}]
    c = oa.audit_extractor(rows, 8)
    assert c["samples"] == 4 and c["finished"] == 2 and c["agree_finished"] == 2


def test_small_counts_depend_on_the_smallest_reference(oa):
    rows = [sample("e", "</think>\nCCO")]
    assert oa.audit_extractor(rows, min_heavy=3)["b_only_small"] == 0     # ethanol has 3 heavy atoms: not below the smallest reference
    assert oa.audit_extractor(rows, min_heavy=4)["b_only_small"] == 1
    rows = [sample("c", f"</think>\nThe answer is {PARA} for this profile.")]          # paracetamol: 11 heavy atoms
    assert oa.audit_extractor(rows, min_heavy=11)["a_only_small"] == 0
    assert oa.audit_extractor(rows, min_heavy=12)["a_only_small"] == 1


def test_verdict_is_the_stricter_rate_and_needs_more_than_one_percent(oa):
    def v(rate, rate_finished):
        return oa.extractor_verdict({"rate": rate, "rate_finished": rate_finished})["verdict"]

    assert v(0.0, 0.0) == "keep" and v(0.01, 0.01) == "keep"             # exactly 1% is not "more than 1%"
    assert v(0.005, 0.0101) == "switch" and v(0.0101, 0.005) == "switch"


# ------------------------------------------------------------------------------------------------ T5
def t5_rows():
    """Prompt rows of one run, k = 4: p1 spread, p2 no spread, p3 one valid only, p4 two valid and two cut, p5 incomplete."""
    rows = []
    for i, text in enumerate([PARA, ASPIRIN, CAFFEINE, "garbage"]):           # p1: rewards 1.0, ~0.5, low, and an invalid 0
        rows.append(sample("row-p1", f"</think>\n{text}", i))
    rows += [sample("row-p2", f"</think>\n{ASPIRIN}", i) for i in range(4)]       # p2: four identical valid answers
    rows += [sample("row-p3", f"</think>\n{text}", i) for i, text in enumerate([PARA, "garbage", "garbage", "garbage"])]
    rows += [sample("row-p4", f"</think>\n{text}", i, finish=fin) for i, (text, fin) in
             enumerate([(PARA, "stop"), (ASPIRIN, "stop"), (PARA, "length"), (ASPIRIN, "length")])]
    rows += [sample("row-p5", f"</think>\n{PARA}", i) for i in range(3)]          # only 3 of 4 samples
    rows += [sample("row-q1", f"</think>\n{text}", i, group="drug-caff") for i, text in enumerate([CAFFEINE, CAFFEINE, ASPIRIN, ASPIRIN])]
    return rows


def test_t5_run_counts_rows_with_spread_among_valid_finished_samples(oa):
    t = oa.t5_run(t5_rows(), REFS)
    assert t["k"] == 4 and t["rows"] == 5 and t["rows_incomplete"] == 1                # p5 left out
    assert t["rows_ge2_valid"] == 4                                                    # p1, p2, p4, q1 (p3 has one)
    assert t["rows_spread"] == 3                                                       # p1, p4 (1.0 vs ~0.5), q1; not p2
    assert t["share"] == pytest.approx(3 / 5) and t["share_among_ge2_valid"] == pytest.approx(3 / 4) and t["meets_bar"] is True
    assert t["rows_spread_all_samples"] == 4                                           # + p3: invalid samples at reward 0 spread it
    assert t["share_all_samples"] == pytest.approx(4 / 5)


def test_t5_run_matches_the_group_spread_helper_over_the_whole_run(oa):
    from rlexp import reward
    rows = [r for r in t5_rows() if r["id"] != "row-p5"]
    rewards, flags = {}, {}
    for r in rows:
        s = reward.score(r["text"], REFS[r["group"]], finished=r["finish_reason"] != "length")
        rewards.setdefault(r["id"], []).append(s["reward"])
        flags.setdefault(r["id"], []).append(s["valid"])
    assert oa.t5_run(rows, REFS)["share"] == pytest.approx(oa.group_spread(rewards, 0.05, flags))


def test_t5_run_length_cut_samples_are_failures(oa):
    cut = [sample("row-p", f"</think>\n{PARA}", 0, finish="length"), sample("row-p", f"</think>\n{ASPIRIN}", 1, finish="length")]
    done = [sample("row-p", f"</think>\n{ASPIRIN}", 2), sample("row-p", f"</think>\n{ASPIRIN}", 3)]
    t = oa.t5_run(cut + done, REFS)
    assert t["rows_ge2_valid"] == 1 and t["rows_spread"] == 0            # the cut exact match is not a valid answer: two equal ones remain
    assert t["rows_spread_all_samples"] == 1                             # ... but RL sees it as a reward-0 sample next to two valid ones
    t = oa.t5_run([sample("row-p", f"</think>\n{CAFFEINE}", 2), *cut, sample("row-p", f"</think>\n{ASPIRIN}", 3)], REFS)
    assert t["rows_ge2_valid"] == 1 and t["rows_spread"] == 1             # a different valid answer next to the aspirin does spread


def test_t5_k_is_the_most_common_row_size_not_the_longest_row(oa):
    rows = [sample(f"row-p{i}", f"</think>\n{text}", j) for i in range(5) for j, text in enumerate([PARA, ASPIRIN, CAFFEINE, "garbage"])]
    rows += [sample("row-long", f"</think>\n{PARA}", j) for j in range(9)]            # one over-long row (a resumed or duplicated prompt)
    rows += [sample("row-short", f"</think>\n{PARA}", j) for j in range(3)]
    t = oa.t5_run(rows, REFS)
    assert t["k"] == 4 and t["rows"] == 5 and t["rows_incomplete"] == 2               # the five normal rows survive; the long and the short one are counted out
    assert t["rows_ge2_valid"] == 5 and t["rows_spread"] == 5
    tie = [sample("a", f"</think>\n{PARA}", j) for j in range(4)] + [sample("b", f"</think>\n{PARA}", j) for j in range(8)]
    assert oa.t5_run(tie, REFS)["k"] == 8                                             # equal counts: the larger k


def test_t5_run_is_empty_safe(oa):
    t = oa.t5_run([], REFS)
    assert t["rows"] == 0 and t["share"] == 0.0 and t["meets_bar"] is False


# ------------------------------------------------------------------------------------------------ files
def write_run(root, name, rows):
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_refs(path):
    rows = [{"group": g, "ref_smiles": v["ref_smiles"], "ref_alt_smiles": [], "exclude_reason": None, "n_heavy": 11 if g == "drug-para" else 14}
            for g, v in REFS.items()] + [{"group": "dropped", "ref_smiles": ASPIRIN, "exclude_reason": "no_approval_year", "n_heavy": 2}]
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_load_refs_keeps_included_drugs_and_the_smallest_size(oa, tmp_path):
    write_refs(tmp_path / "refs.jsonl")
    refs, smallest = oa.load_refs(tmp_path / "refs.jsonl")
    assert set(refs) == {"drug-para", "drug-caff"} and smallest == 11
    (tmp_path / "bare.jsonl").write_text(json.dumps({"group": "x", "ref_smiles": PARA, "exclude_reason": None}) + "\n")
    assert oa.load_refs(tmp_path / "bare.jsonl")[1] == 0                  # a refs file without n_heavy: nothing counts as small


def test_collect_skips_names_controls_other_splits_and_partial_copies(oa, tmp_path):
    base = [sample(f"row-p{i}", f"</think>\n{PARA}", j) for i in range(3) for j in range(2)]
    write_run(tmp_path, "model-a", base + [sample("t0", f"</think>\n{PARA}", split="train")])
    write_run(tmp_path, "model-a-partial", base[:2])                                  # an exact partial copy of model-a
    write_run(tmp_path, "model-a-t0.3", [{**r, "temperature": 0.3} for r in base])    # same text at another temperature: kept
    write_run(tmp_path, "ident-x", base)
    write_run(tmp_path, "val-control_c", base)
    write_run(tmp_path, "train-only", [sample("t1", "x", split="train")])
    (tmp_path / "empty-run").mkdir()
    runs, hashes, notes = oa.collect(tmp_path)
    assert sorted(runs) == ["model-a", "model-a-t0.3"] and set(hashes) == set(runs)
    assert len(runs["model-a"]) == 6                                                   # the train row is not audited
    assert "model-a" in notes["model-a-partial"] and notes["ident-x"].startswith("the answer is a drug name")
    assert notes["val-control_c"].startswith("CPU control") and notes["train-only"] == "no val samples"
    assert notes["empty-run"] == "no samples.jsonl"


KEEP = ["a-direct", "a1-val", "a2-t0.3", "a2-val", "a4-icl", "b27-direct", "b4-t0.6", "g-9b-name", "name-4b", "nothink-val", "rlfull-s40-t0.6", "think-val", "rl-s40",
        "rlfull-s20-t0.6", "rlfull-s30-t1.0", "rlfull-s40-t1.0", "rlfull-s100-t0.6", "a2-t1.0", "b9-t0.3", "x-seedy", "seed7"]      # near misses of the patterns below stay audited
SKIPPED = {"ident-4b": "the answer is a drug name", "val-control_constant": "CPU control", "val8-control_constant_pool": "CPU control",
           "control_copy_nearest": "CPU control", "x-control_y": "CPU control", "strict-a2-val": "a strict-reader rescoring",
           "a2-shuffle-t1.0": "Val diagnostic family", "r1s40-shuffle-t0.6": "Val diagnostic family", "a2-blank-t1.0": "Val diagnostic family",
           "r1s40-blank-t0.6": "Val diagnostic family", "r1s40": "Val diagnostic family", "r3s20-t0.6": "Val diagnostic family", "a4-t0.6": "Val diagnostic family",
           "a2-t1.0-seed1": "Val run made after", "a2-t1.0-seed2": "Val run made after", "rlfull-s40-t0.6-seed1": "Val run made after",
           "rlfull-s40-t0.6-seed12": "Val run made after", "rlfull-s10-t0.6": "Val run made after", "rlfull-s10-t1.0": "Val run made after",
           "rlfull-s10-t0.3": "Val run made after", "rlfull-s20-t0.3": "Val run made after", "rlfull-s30-t0.3": "Val run made after",
           "rlfull-s40-t0.3": "Val run made after", "rlfull-s100-t0.3": "Val run made after"}


def test_collect_skips_controls_anywhere_strict_rescorings_the_later_families_and_seed_repeats_but_keeps_the_rest(oa, tmp_path):
    for name in [*KEEP, *SKIPPED]:
        write_run(tmp_path, name, [sample(f"row-{name}", f"</think>\n{PARA}  {name}")])           # distinct samples: none is a partial copy of another
    runs, hashes, notes = oa.collect(tmp_path)
    assert sorted(runs) == sorted(KEEP) and set(hashes) == set(runs)
    for name, why in SKIPPED.items():
        assert name not in runs and notes[name].startswith(why), name


def test_collect_drops_the_partial_copy_whatever_the_name_order(oa, tmp_path):
    base = [sample(f"row-p{i}", f"</think>\n{PARA}") for i in range(4)]
    write_run(tmp_path, "a-partial", base[:2])
    write_run(tmp_path, "z-full", base)
    runs, _, notes = oa.collect(tmp_path)
    assert sorted(runs) == ["z-full"] and "z-full" in notes["a-partial"]


def run_main(oa, tmp_path, *extra):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    if not any(runs.iterdir()):
        rows = [sample("row-p1", f"</think>\n{t}", i) for i, t in enumerate([PARA, f"The answer is {PARA} for this profile.", "CCO", ASPIRIN])]
        rows += [sample("row-p2", f"</think>\n{ASPIRIN}", i) for i in range(4)]
        rows += [sample("row-c1", f"</think>\n{CAFFEINE}", i, group="drug-caff") for i in range(4)]
        write_run(runs, "arm-a", rows)
        write_run(runs, "arm-k2", [sample("row-p1", f"</think>\n{PARA}", i) for i in range(2)])      # k = 2: audited, not in T5
        write_refs(tmp_path / "refs.jsonl")
    oa.main(["--runs", str(runs), "--refs", str(tmp_path / "refs.jsonl"),
             "--extractor-out", str(tmp_path / "res" / "extractor_audit.json"), "--t5-out", str(tmp_path / "res" / "t5.json"), *extra])


def test_main_writes_counts_only_and_records_provenance(oa, tmp_path, capsys):
    run_main(oa, tmp_path)
    shown = capsys.readouterr().out
    ext = json.loads((tmp_path / "res" / "extractor_audit.json").read_text())
    t5 = json.loads((tmp_path / "res" / "t5.json").read_text())
    assert set(ext["runs"]) == {"arm-a", "arm-k2"} and ext["split"] == "val" and ext["min_reference_heavy_atoms"] == 11
    assert ext["overall"]["samples"] == 14 and ext["verdict"]["verdict"] == "switch"
    assert set(t5["runs"]) == {"arm-a"} and t5["notes"]["arm-k2"] == "k = 2 < 4" and t5["bar"] == 0.3
    for res in (ext, t5):
        prov = res["provenance"]
        assert "scripts/offline_audit.py --runs" in prov["command"] and "--t5-out" in prov["command"]
        assert set(prov["inputs"]) == {"refs.jsonl", "arm-a/samples.jsonl", "arm-k2/samples.jsonl"} and prov["git"]["script_sha256"]
        assert {"python", "rdkit", "numpy"} <= set(prov["versions"])
    blob = shown + (tmp_path / "res" / "extractor_audit.json").read_text() + (tmp_path / "res" / "t5.json").read_text()
    for text in (PARA, ASPIRIN, CAFFEINE, "The answer is", "row-p1", "row-p2", "row-c1", "drug-caff", "drug-para"):   # no answer, text, id or drug
        assert text not in blob, text


def test_main_refuses_test_splits_and_unknown_drugs(oa, tmp_path):
    with pytest.raises(SystemExit):
        run_main(oa, tmp_path, "--split", "test_t")
    with pytest.raises(SystemExit):
        run_main(oa, tmp_path, "--split", "test_r")
    write_run(tmp_path / "runs2", "arm-b", [sample("row-p1", f"</think>\n{PARA}", group="not-a-drug")])
    with pytest.raises(SystemExit) as e:
        oa.main(["--runs", str(tmp_path / "runs2"), "--refs", str(tmp_path / "refs.jsonl")])
    assert "no included reference" in str(e.value)


# ------------------------------------------------------------------------------------------------ the committed audit
COMMITTED, RUNS = ROOT / "results" / "extractor_audit.json", ROOT / "runs" / "colab-s1"


def test_the_committed_artefacts_pin_29_runs_and_10304_samples():
    """docs/prereg-v1.md section 8 quotes this audit: its population is part of the registered decision."""
    audit = json.loads(COMMITTED.read_text(encoding="utf-8"))
    t5 = json.loads((ROOT / "results" / "t5.json").read_text(encoding="utf-8"))
    assert len(audit["runs"]) == 29 and audit["overall"]["samples"] == 10304 == sum(r["samples"] for r in audit["runs"].values())
    assert set(t5["runs"]) | {n for n in t5["notes"] if n in audit["runs"]} == set(audit["runs"])        # T5 covers the same 29 runs (the k<4 ones as notes)
    assert {n.removesuffix("/samples.jsonl") for n in audit["provenance"]["inputs"] if n.endswith("/samples.jsonl")} == set(audit["runs"])


def test_the_runs_folder_reproduces_the_committed_population_exactly(oa):
    """No later run folder may join (or leave) the audit silently: a folder that SKIP does not name fails here, and then SKIP is extended, or the artefacts are
    regenerated and the numbers quoted in docs/prereg-v1.md section 8 restated."""
    if not RUNS.is_dir():
        pytest.skip("needs the (untracked) runs/colab-s1")
    audit = json.loads(COMMITTED.read_text(encoding="utf-8"))
    runs, hashes, _ = oa.collect(RUNS)
    assert sorted(runs) == sorted(audit["runs"]), (sorted(set(runs) - set(audit["runs"])), sorted(set(audit["runs"]) - set(runs)))
    assert sum(len(rows) for rows in runs.values()) == audit["overall"]["samples"] == 10304
    recorded = {n.removesuffix("/samples.jsonl"): v["sha256"] for n, v in audit["provenance"]["inputs"].items() if n.endswith("/samples.jsonl")}
    assert hashes == recorded
