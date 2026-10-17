import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("constant_ceiling", ROOT / "scripts" / "constant_ceiling.py")
cc = importlib.util.module_from_spec(spec)
sys.modules["constant_ceiling"] = cc
spec.loader.exec_module(cc)

ASPIRIN, ETHANOL, CAFFEINE = "CC(=O)Oc1ccccc1C(=O)O", "CCO", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_the_oracle_constant_is_the_candidate_closest_to_the_references():
    bc = cc.load_best_constant()
    refs = {"a": ASPIRIN, "b": ASPIRIN}
    out = cc.ceiling(bc, refs, [ETHANOL, ASPIRIN, CAFFEINE], c3_smiles=ETHANOL)
    assert out["oracle"] == pytest.approx(1.0) and out["c3_rank"] == 3 and out["c3"] < out["oracle"] and out["candidates"] == 3 and out["drugs"] == 2
    assert out["top5"][0] == 1.0


def test_c3_missing_from_the_candidates_has_no_rank():
    bc = cc.load_best_constant()
    out = cc.ceiling(bc, {"a": ASPIRIN}, [ASPIRIN], c3_smiles=CAFFEINE)
    assert out["c3_rank"] is None and out["c3"] is None


def test_a_test_split_needs_the_flag(capsys):
    with pytest.raises(SystemExit, match="--allow-test"):
        cc.main(["--splits", "val,test_t"])
