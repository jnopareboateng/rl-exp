import importlib.util
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_bootstrap_bundle.py"


@pytest.fixture(scope="module")
def mb():
    spec = importlib.util.spec_from_file_location("make_bootstrap_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(tmp_path, with_test_file=False):
    for rel in ("gpu/a.py", "gpu/sub/b.sh", "src/rlexp/x.py", "pyproject.toml", "data/processed/train.jsonl", "data/processed/val.jsonl",
                "data/processed/arms/a2_answer_train.jsonl", "data/processed/arms/a2_answer_train.meta.json", "data/processed/arms/a2_answer_val.jsonl",
                "data/processed/arms/a2_answer_val.meta.json", "data/processed/arms/name/val.jsonl", "data/processed/arms/name/test_t.jsonl",
                "data/processed/arms/name/val.meta.json", "data/processed/test_t.jsonl", "gpu/__pycache__/c.pyc"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    if with_test_file:
        (tmp_path / "gpu" / "test_leak.py").write_text("x")


def test_the_bundle_lists_code_and_validation_data_but_never_a_test_file(mb, tmp_path):
    build(tmp_path)
    rel = mb.members(tmp_path)
    assert "gpu/a.py" in rel and "gpu/sub/b.sh" in rel and "src/rlexp/x.py" in rel and "data/processed/arms/name/val.jsonl" in rel
    assert not any("test_" in Path(r).name for r in rel) and "gpu/__pycache__/c.pyc" not in rel and "data/processed/test_t.jsonl" not in rel


def test_a_file_with_a_test_name_inside_gpu_is_refused(mb, tmp_path):
    build(tmp_path, with_test_file=True)
    with pytest.raises(ValueError, match="Test file"):
        mb.members(tmp_path)


def test_main_writes_a_readable_archive(mb, tmp_path, monkeypatch):
    build(tmp_path)
    monkeypatch.setattr(mb, "ROOT", tmp_path)
    monkeypatch.setattr(mb, "members", lambda root=tmp_path: __import__("types").SimpleNamespace() and ["gpu/a.py", "pyproject.toml"])
    mb.main(["--out", str(tmp_path / "out" / "b.tgz")])
    with tarfile.open(tmp_path / "out" / "b.tgz") as tar:
        assert sorted(tar.getnames()) == ["gpu/a.py", "pyproject.toml"]
