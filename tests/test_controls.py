"""controls.py on a synthetic dataset and a recording fake scorer (one smoke test uses the real reward.score if present)."""
import inspect
import json
import math
import re

import numpy as np
import pytest

from rlexp import chem, controls, metrics

SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1",
          "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
          "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "CN1CCCC1c1cccnc1", "CCN(CC)CCNC(=O)c1ccc(N)cc1"]
SYSTEM = "You are a pharmacology assistant."


def prompt(user, system=SYSTEM):
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def row(split, group, i, user, smiles, system=SYSTEM):
    return {"id": f"{group}/{i}", "group": group, "task_type": "A", "split": split, "prompt": prompt(user, system),
            "ref_smiles": smiles, "ref_conn": f"conn-{group}", "ref_scaffold": f"scaf-{group}",
            "ref_inchikey": f"key-{group}", "ref_alt_smiles": [], "cluster_id": sum(map(ord, group)),
            "approval_year": 2020, "n_prompt_tokens": 10, "n_completion_tokens": 20}


TOPICS = ["kinase tyrosine phosphorylation oncology", "beta adrenergic heart rate hypertension",
          "bacterial ribosome antibiotic gram positive", "xanthine adenosine receptor stimulant caffeine",
          "cyclooxygenase prostaglandin analgesic fever", "hepatic acetaminophen glucuronidation overdose"]


def dataset():
    """6 train drugs x 2 rows; val 3 drugs, test_r 2, test_t 2. Every eval prompt shares rare words with one train drug."""
    d = {"train": [], "val": [], "test_r": [], "test_t": []}
    for k in range(6):
        d["train"] += [row("train", f"t{k}", i, f"{TOPICS[k]} view {i} plasma half life", SMILES[k]) for i in range(2)]
    d["val"] = [row("val", "v0", 0, "tyrosine kinase oncology marker", SMILES[6]),
                row("val", "v0", 1, "ribosome antibiotic positive infection", SMILES[6]),
                row("val", "v1", 0, "xanthine adenosine stimulant", SMILES[7]),
                row("val", "v2", 0, "completely unknown vocabulary zzqx", SMILES[8])]
    d["test_r"] = [row("test_r", "r0", 0, "prostaglandin cyclooxygenase fever", SMILES[6]),
                   row("test_r", "r1", 0, "beta adrenergic hypertension", SMILES[7])]
    d["test_t"] = [row("test_t", "x0", 0, "acetaminophen glucuronidation", SMILES[8]),
                   row("test_t", "x1", 0, "heart rate beta blocker", SMILES[6])]
    return d


def write_data(path, d):
    path.mkdir(parents=True, exist_ok=True)
    for split, rows in d.items():
        (path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class Recorder:
    """Fake scorer: records what it was given; an answer equal to ref_smiles is an exact hit."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, ref):
        self.calls.append((text, ref))
        ans = text.rsplit("</think>", 1)[1].strip()
        hit = ans == ref["ref_smiles"]
        return {"closed": "</think>" in text, "answer": ans, "valid": True, "tanimoto": float(hit),
                "scaffold_match": hit, "conn_match": hit, "stereo_match": hit, "reward": float(hit)}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ----------------------------------------------------------------------------------- user part, tf-idf
def test_user_part_is_between_the_user_marker_and_the_next_im_end():
    assert controls.user_part(prompt("Hello World", system="SYS")) == "\nHello World"      # literal, newline included
    assert controls.user_part(prompt("A<|im_start|>user B", system="S")) == "\nA<|im_start|>user B"
    with pytest.raises(metrics.InputError, match="user"):
        controls.user_part("<|im_start|>system\nS<|im_end|>\n<|im_start|>assistant\n")
    with pytest.raises(metrics.InputError, match="user"):
        controls.user_part("<|im_start|>user\nnever closed")


def test_tfidf_cosine_matches_hand_computed_values():
    idx, cos = controls.nearest(["a a b", "b c"], ["a b", "c", "zzz"])
    ia = math.log(3 / 2) + 1                      # N = 2 docs; df(a) = df(c) = 1 -> idf ln(3/2) + 1; df(b) = 2 -> idf 1
    d1 = np.array([(1 + math.log(2)) * ia, 1.0, 0.0])           # "a a b": sublinear tf 1 + ln 2 for "a"
    d2 = np.array([0.0, 1.0, ia])                               # "b c"
    q1, q2 = np.array([ia, 1.0, 0.0]), np.array([0.0, 0.0, ia])  # "a b", "c"
    cosine = lambda u, v: u @ v / math.sqrt((u @ u) * (v @ v))
    assert list(idx) == [0, 1, 0]
    assert cos[0] == pytest.approx(cosine(q1, d1)) and cosine(q1, d1) > cosine(q1, d2)
    assert cos[1] == pytest.approx(cosine(q2, d2)) and cos[1] == pytest.approx(ia / math.sqrt(1 + ia * ia))
    assert cos[2] == 0.0                                          # no known token: similarity 0, first row


def test_tfidf_tokenisation_ties_and_identity():
    idx, cos = controls.nearest(["Alpha-Beta, GAMMA!", "same same", "same same"], ["alpha beta gamma", "same same"])
    assert list(idx) == [0, 1] and cos == pytest.approx([1.0, 1.0])     # case and punctuation ignored; tie -> first


# ------------------------------------------------------------------------------------- copy-nearest
def controls_samples(tmp_path, data, name):
    """Run both controls on `data` (split -> rows); return the samples of one control."""
    controls.run(write_data(tmp_path / "data", data), tmp_path / "runs", scorer=Recorder())
    return read(tmp_path / "runs" / name / "samples.jsonl")


def test_copy_nearest_reads_only_the_user_part(tmp_path):
    train = [row("train", "t0", 0, "alpha beta gamma delta", SMILES[0], system="zzz"),
             row("train", "t1", 0, "unrelated words entirely", SMILES[1], system="alpha beta gamma delta " * 5)]
    val = [row("val", "v0", 0, "alpha beta gamma", SMILES[2], system="unrelated words entirely " * 5)]
    data = {"train": train, "val": val, "test_r": [row("test_r", "r0", 0, "alpha", SMILES[3])],
            "test_t": [row("test_t", "x0", 0, "unrelated", SMILES[4])]}
    ans = [r["text"] for r in controls_samples(tmp_path, data, "control_copy_nearest")]
    assert ans == ["</think>\n" + SMILES[0], "</think>\n" + SMILES[0], "</think>\n" + SMILES[1]]


def test_copy_targets_are_train_rows_only(tmp_path):
    d = dataset()
    # an eval prompt that is identical to another eval prompt in a different split: a leaky copier would take its answer
    d["test_r"][0] = row("test_r", "r0", 0, "tyrosine kinase oncology marker", SMILES[6])
    d["test_t"][0] = row("test_t", "x0", 0, "ribosome antibiotic positive infection", SMILES[8])
    got = controls_samples(tmp_path, d, "control_copy_nearest")
    answers = {s["text"].split("\n", 1)[1] for s in got}
    train_refs = {r["ref_smiles"] for r in d["train"]}
    eval_refs = {r["ref_smiles"] for s in ("val", "test_r", "test_t") for r in d[s]}
    assert answers <= train_refs and not answers & (eval_refs - train_refs)
    by_id = {s["id"]: s["text"].split("\n", 1)[1] for s in got}
    assert by_id["v0/0"] == by_id["r0/0"] == SMILES[0]            # kinase / oncology
    assert by_id["v0/1"] == by_id["x0/0"] == SMILES[2]            # ribosome / antibiotic
    assert by_id["v1/0"] == SMILES[3] and by_id["r1/0"] == SMILES[1]


# --------------------------------------------------------------------------------- controls end to end
def test_one_answer_per_prompt_row_scored_through_the_same_path(tmp_path):
    d = dataset()
    data = write_data(tmp_path / "data", d)
    rec = Recorder()
    out = controls.run(data, tmp_path / "runs", scorer=rec)
    ev = [r for s in ("val", "test_r", "test_t") for r in d[s]]
    assert set(out) == {"control_copy_nearest", "control_constant", "control_constant_scaffold"}
    for name in out:
        samples = read(tmp_path / "runs" / name / "samples.jsonl")
        scored = read(tmp_path / "runs" / name / "scored.jsonl")
        meta = json.loads((tmp_path / "runs" / name / "meta.json").read_text())
        assert [s["id"] for s in samples] == [r["id"] for r in ev]            # one per prompt row, val/test_r/test_t order
        assert [s["id"] for s in scored] == [r["id"] for r in ev]
        assert {s["sample_idx"] for s in samples} == {0} and len({s["id"] for s in samples}) == len(ev)
        assert all(s["text"].startswith("</think>\n") and s["model"] == name and s["finish_reason"] == "stop"
                   for s in samples)
        assert [(s["group"], s["split"]) for s in samples] == [(r["group"], r["split"]) for r in ev]
        assert all(set(metrics.SCORE_KEYS) <= set(s) for s in scored)
        assert [s["cluster_id"] for s in scored] == [r["cluster_id"] for r in ev]      # so report() can resample clusters
        assert all(s["ref_alt_smiles"] == [] for s in scored)
        assert meta["model"] == name and meta["n_samples"] == len(ev) and meta["train_drugs"] == 6
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == \
        ["control_constant", "control_constant_scaffold", "control_copy_nearest"]
    assert len(rec.calls) == 3 * len(ev)
    assert all(set(ref) == set(metrics.REF_KEYS + metrics.REF_OPTIONAL) for _, ref in rec.calls)   # never the prompt
    assert not list((tmp_path / "runs").rglob("*.tmp"))


def test_control_scored_files_feed_metrics_report(tmp_path):
    d = dataset()
    d["val"][2] = row("val", "v1", 0, "xanthine adenosine stimulant", SMILES[3])     # v1's reference equals t3's: a copy hit
    controls.run(write_data(tmp_path / "data", d), tmp_path / "runs", scorer=Recorder())
    res = metrics.report(read(tmp_path / "runs" / "control_copy_nearest" / "scored.jsonl"), boot=50)
    val = res["splits"]["val"]
    assert val["bootstrap"]["unit"] == "cluster" and (val["n_drugs"], val["n_rows"]) == (3, 4)
    assert val["metrics"]["exact@1"]["value"] == pytest.approx(1 / 3)                 # v1 hit; v0 and v2 missed
    assert val["metrics"]["valid_closed"]["value"] == 1.0 and val["metrics"]["truncation"]["value"] == 0.0
    for k in ("exact@8", "core@8", "maj@8"):                                           # one sample per row: nothing to take @8 of
        assert val["metrics"][k]["value"] is None and val["metrics"][k]["skipped_rows"] == 4
    const = metrics.report(read(tmp_path / "runs" / "control_constant" / "scored.jsonl"), boot=50)
    assert const["splits"]["test_t"]["metrics"]["exact@1"]["value"] == 0.0


def test_constant_control_is_the_most_central_train_reference_by_bit_tanimoto(tmp_path):
    d = dataset()
    refs = {r["group"]: r["ref_smiles"] for r in d["train"]}
    mols = {g: chem.standardize(s) for g, s in refs.items()}

    def centrality(similarity):
        return {g: np.mean([similarity(mols[g], mols[h]) for h in mols if h != g]) for g in mols}
    bits, counts = centrality(chem.tanimoto_bits), centrality(chem.tanimoto)           # one pair at a time
    best = max(bits, key=bits.get)
    assert max(counts, key=counts.get) != best          # the count-vector winner differs here, so the test tells them apart
    controls.run(write_data(tmp_path / "data", d), tmp_path / "runs", scorer=Recorder())
    texts = {s["text"] for s in read(tmp_path / "runs" / "control_constant" / "samples.jsonl")}
    assert texts == {"</think>\n" + refs[best]}
    meta = json.loads((tmp_path / "runs" / "control_constant" / "meta.json").read_text())
    assert meta["constant_smiles"] == refs[best] and meta["mean_bit_tanimoto_to_train"] == pytest.approx(bits[best])
    assert metrics.best_constant(refs) == (refs[best], pytest.approx(bits[best]))
    flat = lambda doc: " ".join(doc.split())            # docstrings wrap lines
    for doc in (controls.__doc__, metrics.best_constant.__doc__):
        assert "bit Tanimoto" in flat(doc) and "count vectors" in flat(doc)      # says which Tanimoto picks it


def test_metrics_and_controls_use_only_the_public_chem_functions():
    for module in (metrics, controls):
        assert not re.search(r"\bchem\._", inspect.getsource(module)), module.__name__


# ------------------------------------------------------------------------------- constant scaffold
def test_scaffold_constant_picks_the_shortest_reference_of_the_most_common_ring_scaffold():
    benzene = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"]     # three drugs, scaffold c1ccccc1
    refs = {"naproxen": SMILES[1], "ketoprofen": SMILES[2], "caffeine": SMILES[3]}
    refs.update({f"b{i}": s for i, s in enumerate(benzene)})
    smiles, scaffold, n = controls.scaffold_constant(refs)
    assert (smiles, scaffold, n) == ("CC(=O)Nc1ccc(O)cc1", "c1ccccc1", 3)              # paracetamol is the shortest of the three
    # drugs are counted once, however many prompt rows they have: the run uses {group: ref}, so a repeat cannot add a vote
    assert controls.scaffold_constant({**refs, "b0": benzene[0]})[2] == 3


def test_scaffold_constant_ignores_acyclic_references_and_breaks_ties_by_first_seen():
    acyclic = {f"a{i}": "CCCCCCCC" + "C" * i + "O" for i in range(5)}                 # five drugs share the empty scaffold
    ring = {"p": "c1ccccc1CC", "q": "Cc1ccccc1", "r": "C1CCCCC1CC", "s": "CC1CCCCC1"}   # 2 + 2: benzene first, cyclohexane second
    smiles, scaffold, n = controls.scaffold_constant({**acyclic, **ring})
    assert (scaffold, n) == ("c1ccccc1", 2)                       # acyclic never wins, the first scaffold wins a tie
    assert smiles == "Cc1ccccc1" and len(smiles) < len("c1ccccc1CC")      # the shorter of the two benzene references
    assert controls.scaffold_constant({**ring, "p2": "c1ccccc1CC"})[0] == "Cc1ccccc1"   # equal lengths: first seen
    equal = controls.scaffold_constant({"x": "CCc1ccccc1", "y": "CCc1ccccc1".replace("CC", "Cl", 1)[:0] + "OCc1ccccc1"})
    assert equal[0] == "CCc1ccccc1"                                                     # same length, the first in file order


def test_scaffold_constant_errors_are_clear():
    with pytest.raises(metrics.InputError, match="no Train reference has a ring scaffold"):
        controls.scaffold_constant({"a": "CCCCCCCCO", "b": "CCCCCCCCCO"})
    with pytest.raises(metrics.InputError, match="1 reference"):
        controls.scaffold_constant({"a": "CCCCCCCCO", "b": "not a smiles"})


def test_constant_scaffold_control_writes_the_same_layout_and_one_answer_per_row(tmp_path):
    d = dataset()
    controls.run(write_data(tmp_path / "data", d), tmp_path / "runs", scorer=Recorder())
    ev = [r for s in ("val", "test_r", "test_t") for r in d[s]]
    base = tmp_path / "runs" / "control_constant_scaffold"
    samples, scored = read(base / "samples.jsonl"), read(base / "scored.jsonl")
    meta = json.loads((base / "meta.json").read_text())
    assert [s["id"] for s in samples] == [r["id"] for r in ev] == [s["id"] for s in scored]
    assert {s["text"] for s in samples} == {"</think>\n" + SMILES[5]}      # paracetamol: shortest of the 3 benzene drugs
    assert all((s["model"], s["sample_idx"], s["finish_reason"]) == ("control_constant_scaffold", 0, "stop") for s in samples)
    assert all(set(metrics.SCORE_KEYS) <= set(s) for s in scored)
    assert (meta["model"], meta["constant_smiles"], meta["scaffold"], meta["scaffold_drugs"], meta["scaffold_share"]) == \
        ("control_constant_scaffold", SMILES[5], "c1ccccc1", 3, 0.5)                    # 3 of the 6 Train drugs
    assert meta["n_samples"] == len(ev) and meta["train_drugs"] == 6 and not list(base.glob("*.tmp"))


def test_constant_scaffold_run_fails_cleanly_when_train_has_no_ring(tmp_path):
    d = dataset()
    for i, r in enumerate(d["train"]):
        r["ref_smiles"] = "CCCCCCCC" + "C" * (i // 2) + "O"                              # six distinct acyclic references
    with pytest.raises(metrics.InputError, match="ring scaffold"):
        controls.run(write_data(tmp_path / "data", d), tmp_path / "runs", scorer=Recorder())
    assert not (tmp_path / "runs").exists()                                              # nothing half-written


def test_missing_or_malformed_inputs_are_clear_errors(tmp_path):
    d = dataset()
    d.pop("test_t")
    with pytest.raises(metrics.InputError, match="test_t"):
        controls.run(write_data(tmp_path / "a", d), tmp_path / "runs", scorer=Recorder())
    d = dataset()
    d["val"][0]["prompt"] = "no chat markers at all"
    with pytest.raises(metrics.InputError, match="v0/0"):
        controls.run(write_data(tmp_path / "b", d), tmp_path / "runs", scorer=Recorder())
    assert not (tmp_path / "runs").exists()                        # nothing half-written


def test_cli_prints_counts_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(metrics, "_default_scorer", lambda: Recorder())
    controls.main(["--data", str(write_data(tmp_path / "data", dataset())), "--out", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert all(f"{n}: 8 samples" in out for n in ("control_copy_nearest", "control_constant", "control_constant_scaffold"))
    assert "</think>" not in out
    assert (tmp_path / "runs" / "control_constant" / "scored.jsonl").exists()


def test_controls_scored_by_the_real_reward_if_present(tmp_path):
    reward = pytest.importorskip("rlexp.reward")
    if not hasattr(reward, "score"):
        pytest.skip("reward.py v2 (rlexp.reward.score) not written yet")
    d = dataset()
    for rows in d.values():                                        # real conn/scaffold so exact hits can happen
        for r in rows:
            mol = chem.standardize(r["ref_smiles"])
            r["ref_conn"], r["ref_scaffold"] = chem.conn(mol), chem.scaffold(mol)
    controls.run(write_data(tmp_path / "data", d), tmp_path / "runs")
    for name in ("control_copy_nearest", "control_constant", "control_constant_scaffold"):
        scored = read(tmp_path / "runs" / name / "scored.jsonl")
        assert len(scored) == 8 and all(s["closed"] and s["valid"] and s["answer"] for s in scored), name
    # scaffold control: paracetamol (benzene scaffold) is a core hit for exactly the eval drugs with a benzene scaffold
    scored = read(tmp_path / "runs" / "control_constant_scaffold" / "scored.jsonl")
    eval_refs = {r["id"]: r["ref_smiles"] for s in ("val", "test_r", "test_t") for r in d[s]}
    benzene = {i for i, smi in eval_refs.items() if chem.scaffold(chem.standardize(smi)) == "c1ccccc1"}
    assert benzene and benzene != set(eval_refs)                   # the fixture has both kinds of eval drug
    assert {s["id"] for s in scored if s["scaffold_match"]} == benzene and not any(s["conn_match"] for s in scored)
