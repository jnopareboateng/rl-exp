"""Tests for scripts/random_pair_calibration.py, including a mutation pass (6 mutants, all killed)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE = (ROOT / "scripts" / "random_pair_calibration.py").read_text(encoding="utf-8")


def load(source=SOURCE, name="random_pair_calibration"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "random_pair_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    exec(compile(source, str(ROOT / "scripts" / "random_pair_calibration.py"), "exec"), mod.__dict__)
    return mod


rpc = load()

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"


def ref(smiles, split="train"):
    return {"split": split, "ref_smiles": smiles}


def battery(mod):
    from rlexp import chem
    rows = [ref(ASPIRIN), ref(CAFFEINE), ref(IBUPROFEN)]
    mols = mod.structures(rows, {"train"})
    assert len(mols) == 3
    sims = mod.pair_similarities(mols)
    a, c, i = (chem.standardize(s) for s in (ASPIRIN, CAFFEINE, IBUPROFEN))
    expected = sorted([chem.tanimoto(a, c), chem.tanimoto(a, i), chem.tanimoto(c, i)])
    assert sims.size == 3, f"three structures make three pairs, got {sims.size}"
    assert sorted(sims) == pytest.approx(expected), "the pairs must be every unordered pair once, scored by rlexp.chem.tanimoto"
    assert max(sims) < 1.0 and min(sims) >= 0.0

    # a split not asked for contributes nothing
    assert len(mod.structures(rows + [ref(PARACETAMOL, "test_t")], {"train"})) == 3
    assert len(mod.structures(rows + [ref(PARACETAMOL, "test_t")], {"train", "test_t"})) == 4

    # the same structure twice is counted once, so no pair scores 1.0 by duplication
    both = mod.pair_similarities(mod.structures([ref(ASPIRIN), ref(ASPIRIN), ref(CAFFEINE)], {"train"}))
    assert both.size == 1 and both[0] < 1.0, "a duplicated drug must not add a self-pair"

    # the location of a score: the share at or above, and the complementary percentile
    s = mod.summary(np.array([0.1, 0.2, 0.3, 0.4]), at=(0.25,))
    assert s["n_pairs"] == 4 and s["mean"] == pytest.approx(0.25) and s["percentiles"]["50"] == pytest.approx(0.25)
    assert s["at"]["0.2500"]["percentile"] == pytest.approx(50.0)
    assert s["at"]["0.2500"]["share_at_or_above"] == pytest.approx(50.0)
    low = mod.summary(np.array([0.1, 0.2, 0.3, 0.4]), at=(0.0,))["at"]["0.0000"]
    assert low["percentile"] == 0.0 and low["share_at_or_above"] == 100.0
    # a score that a pair reaches exactly counts as at or above it
    exact = mod.summary(np.array([0.1, 0.2, 0.3, 0.4]), at=(0.2,))["at"]["0.2000"]
    assert exact["percentile"] == pytest.approx(25.0) and exact["share_at_or_above"] == pytest.approx(75.0)
    return True


def test_battery_passes_on_the_real_script():
    assert battery(rpc)


def test_a_reference_that_does_not_parse_is_an_error():
    from rlexp import metrics
    with pytest.raises(metrics.InputError):
        rpc.structures([ref("not a molecule"), ref(ASPIRIN)], {"train"})
    with pytest.raises(metrics.InputError):
        rpc.structures([ref(ASPIRIN)], {"train"})            # one structure makes no pair


def test_the_report_holds_no_structure(tmp_path):
    refs = tmp_path / "refs.jsonl"
    refs.write_text("".join(json.dumps(ref(s)) + "\n" for s in (ASPIRIN, CAFFEINE, IBUPROFEN, PARACETAMOL)), encoding="utf-8")
    out, md = tmp_path / "o.json", tmp_path / "o.md"
    rpc.main(["--refs", str(refs), "--splits", "train", "--out", str(out), "--md", str(md)])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_structures"] == 4 and res["n_pairs"] == 6 and res["splits"] == ["train"]
    text = md.read_text(encoding="utf-8") + json.dumps(res)
    assert not any(s in text for s in (ASPIRIN, CAFFEINE, IBUPROFEN, PARACETAMOL))


def test_bad_inputs_exit(tmp_path):
    with pytest.raises(SystemExit):
        rpc.main(["--refs", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "o.json")])
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps(ref(ASPIRIN)) + "\n" + json.dumps(ref(CAFFEINE)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        rpc.main(["--refs", str(refs), "--at", "not-a-number", "--out", str(tmp_path / "o.json")])


# --- the numbers this was pinned to (the four used splits of this repository) -------------------------------------

def test_the_repository_numbers_are_reproduced():
    refs = ROOT / "data" / "processed" / "refs.jsonl"
    if not refs.is_file():
        pytest.skip("data/processed/refs.jsonl is not in this tree")
    from rlexp import metrics
    mols = rpc.structures(metrics.read_jsonl(refs), set(rpc.USED_SPLITS))
    res = rpc.summary(rpc.pair_similarities(mols), at=(0.1949,))
    assert len(mols) == 332 and res["n_pairs"] == 54946
    assert round(res["mean"], 3) == 0.141
    assert [round(res["percentiles"][p], 3) for p in ("50", "75", "90", "99")] == [0.141, 0.180, 0.213, 0.274]
    assert round(res["at"]["0.1949"]["percentile"]) == 83


MUTANTS = (
    ("pairs counted twice", "for j in range(i + 1, len(mols))", "for j in range(len(mols))"),
    ("self-pairs included", "for j in range(i + 1, len(mols))", "for j in range(i, len(mols))"),
    ("duplicates not collapsed", "if key not in seen:", "if True:"),
    ("the split filter is dropped", 'if r.get("split") not in splits or not r.get("ref_smiles"):', 'if not r.get("ref_smiles"):'),
    ("the percentile is inverted", '"percentile": float(100 * (sims < v).mean())', '"percentile": float(100 * (sims > v).mean())'),
    ("the share at or above excludes equality", '"share_at_or_above": float(100 * (sims >= v).mean())', '"share_at_or_above": float(100 * (sims > v).mean())'),
)


@pytest.mark.parametrize("name,before,after", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_is_killed(name, before, after):
    assert SOURCE.count(before) == 1, f"the mutation target {before!r} is not unique in the source"
    mod = load(SOURCE.replace(before, after), name=f"rpc_mutant_{abs(hash(name))}")
    with pytest.raises((AssertionError, ZeroDivisionError, TypeError, KeyError, IndexError)):
        battery(mod)
