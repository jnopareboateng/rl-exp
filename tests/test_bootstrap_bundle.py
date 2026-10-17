"""scripts/make_bootstrap_bundle.py (what may leave the machine) and gpu/bootstrap_colab.sh (what the GPU runtime refuses), on synthetic trees."""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_bootstrap_bundle.py"
BOOT = ROOT / "gpu" / "bootstrap_colab.sh"
TEST_ROWS = "synthetic test split rows\n"


@pytest.fixture(scope="module")
def mb():
    spec = importlib.util.spec_from_file_location("make_bootstrap_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(tmp_path, with_test_file=False, manifest=True):
    for rel in ("gpu/a.py", "gpu/sub/b.sh", "src/rlexp/x.py", "pyproject.toml", "data/processed/train.jsonl", "data/processed/val.jsonl",
                "data/processed/arms/a2_answer_train.jsonl", "data/processed/arms/a2_answer_train.meta.json", "data/processed/arms/a2_answer_val.jsonl",
                "data/processed/arms/a2_answer_val.meta.json", "data/processed/arms/name/val.jsonl", "data/processed/arms/name/test_t.jsonl",
                "data/processed/arms/name/val.meta.json", "data/processed/test_t.jsonl", "gpu/__pycache__/c.pyc"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEST_ROWS if Path(rel).name.startswith("test_") else f"content of {rel}\n")
    if with_test_file:
        (tmp_path / "gpu" / "test_leak.py").write_text("x")
    if manifest:                                            # the manifest of scripts/data_manifest.py: every data file, Test ones included
        data = tmp_path / "data"
        lines = [f"{sha(p.read_text())}  {p.relative_to(data).as_posix()}" for p in sorted(data.rglob("*")) if p.is_file()]
        (data / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def test_the_bundle_lists_code_and_validation_data_but_never_a_test_file(mb, tmp_path):
    build(tmp_path)
    rel = mb.members(tmp_path)
    assert "gpu/a.py" in rel and "gpu/sub/b.sh" in rel and "src/rlexp/x.py" in rel and "data/processed/arms/name/val.jsonl" in rel
    assert "data/processed/val.jsonl" in rel and "data/processed/train.jsonl" in rel          # a manifest entry for them is no reason to refuse them
    assert not any("test_" in Path(r).name for r in rel) and "gpu/__pycache__/c.pyc" not in rel and "data/processed/test_t.jsonl" not in rel


def test_a_file_with_a_test_name_inside_gpu_is_refused(mb, tmp_path):
    build(tmp_path, with_test_file=True)
    with pytest.raises(ValueError, match="Test file"):
        mb.members(tmp_path)


@pytest.mark.parametrize("where", ["data/processed/arms/name/val.jsonl", "gpu/neutral.py", "data/processed/arms/shuffle/val.meta.json"])
def test_a_test_split_under_a_neutral_name_is_refused_by_its_hash(mb, tmp_path, where):
    build(tmp_path)
    leak = tmp_path / where
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_text(TEST_ROWS)                              # byte-identical to data/processed/test_t.jsonl, whatever it is called
    with pytest.raises(ValueError, match="content of a Test file"):
        mb.members(tmp_path)


def test_without_a_manifest_the_name_guard_still_holds_and_a_manifest_without_test_entries_blocks_nothing(mb, tmp_path):
    build(tmp_path, manifest=False)
    assert "gpu/a.py" in mb.members(tmp_path)
    (tmp_path / "data" / "MANIFEST.sha256").write_text(f"{sha('x')}  processed/val.jsonl\n")
    assert mb.leak_hashes(tmp_path) == set()
    (tmp_path / "data" / "MANIFEST.sha256").write_text(f"{sha('x')}  processed/test_r.jsonl\n{sha('y')}  processed/val.jsonl\n")
    assert mb.leak_hashes(tmp_path) == {sha("x")}


def test_a_symlink_is_refused_whether_it_is_the_file_or_a_directory_above_it(mb, tmp_path):
    build(tmp_path)
    (tmp_path / "gpu" / "linked.py").symlink_to(tmp_path / "data" / "processed" / "test_t.jsonl")
    with pytest.raises(ValueError, match="symlink"):
        mb.members(tmp_path)
    (tmp_path / "gpu" / "linked.py").unlink()
    (tmp_path / "gpu" / "dangling.py").symlink_to(tmp_path / "nowhere")                 # a broken link is still a link
    with pytest.raises(ValueError, match="symlink"):
        mb.members(tmp_path)
    (tmp_path / "gpu" / "dangling.py").unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "val.jsonl").write_text("rows")
    shutil.rmtree(tmp_path / "data" / "processed" / "arms" / "name")
    (tmp_path / "data" / "processed" / "arms" / "name").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        mb.members(tmp_path)


def test_gpu_is_limited_to_code_and_text_and_the_rest_is_reported(mb, tmp_path):
    build(tmp_path)
    for name in ("weights.safetensors", "model.bin", "notes.txt", "cfg.json", "README.md", ".hidden", "blob"):
        (tmp_path / "gpu" / name).write_text("x")
    skipped = []
    rel = mb.members(tmp_path, skipped=skipped)
    assert {"gpu/notes.txt", "gpu/cfg.json", "gpu/README.md", "gpu/a.py", "gpu/sub/b.sh"} <= set(rel)
    assert not {"gpu/weights.safetensors", "gpu/model.bin", "gpu/.hidden", "gpu/blob"} & set(rel)
    assert sorted(skipped) == ["gpu/.hidden", "gpu/blob", "gpu/model.bin", "gpu/weights.safetensors"]       # caches are not reported


def test_main_writes_a_readable_archive_and_reports_what_it_skipped(mb, tmp_path, monkeypatch, capsys):
    build(tmp_path)
    monkeypatch.setattr(mb, "ROOT", tmp_path)

    def members(root=tmp_path, skipped=None):
        skipped.append("gpu/model.bin")
        return ["gpu/a.py", "pyproject.toml"]

    monkeypatch.setattr(mb, "members", members)
    mb.main(["--out", str(tmp_path / "out" / "b.tgz")])
    with tarfile.open(tmp_path / "out" / "b.tgz") as tar:
        assert sorted(tar.getnames()) == ["gpu/a.py", "pyproject.toml"]
    assert "skipped 1 gpu/ file(s)" in capsys.readouterr().out


def test_the_real_tree_passes_its_own_guards(mb):
    rel = mb.members(ROOT)                                  # hashes the real train and val files against the real manifest, prints nothing
    assert "gpu/final_eval.py" in rel and not any("test_" in Path(r).name for r in rel)


# ------------------------------------------------------------------------------------------------ gpu/bootstrap_colab.sh
FAKE_UV = """#!/bin/sh
echo "uv $*" >> "$FAKE_LOG"
if [ "$1" = venv ]; then
  mkdir -p "$2/bin"
  printf '#!/bin/sh\\nexec env FAKE_VENV=%s "%s" "$@"\\n' "$2" "$REAL_PY" > "$2/bin/python"
  chmod +x "$2/bin/python"
fi
exit 0
"""
FAKE_TORCH = '''import os
__version__ = "0.0.fake"


def _kind():
    return "TRAIN" if "sft-venv" in os.environ.get("FAKE_VENV", "") else "VLLM"


class cuda:
    @staticmethod
    def is_available():
        return os.environ.get(f"FAKE_CUDA_{_kind()}", "1") == "1"

    @staticmethod
    def device_count():
        return int(os.environ.get(f"FAKE_DEVICES_{_kind()}", "1"))
'''


def boot(tmp_path, script=None, test_file=None, **env):
    """Run gpu/bootstrap_colab.sh in a synthetic tree with a fake nvidia-smi, a fake uv (it only logs and makes the two venvs' pythons) and fake torch, vllm,
    transformers and peft modules whose CUDA state the environment controls. Returns (completed process, uv log text)."""
    tree, fakes, bin_ = tmp_path / "tree", tmp_path / "fakes", tmp_path / "bin"
    (tree / "gpu").mkdir(parents=True)
    (tree / "gpu" / "sample_vllm.py").write_text("")
    (tree / "gpu" / "bootstrap_colab.sh").write_text(script if script is not None else BOOT.read_text())
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 'Fake A100, 40960 MiB, 550.0'\n")
    (bin_ / "uv").write_text(FAKE_UV)
    for tool in ("nvidia-smi", "uv"):
        (bin_ / tool).chmod(0o755)
    for name in ("torch", "vllm", "transformers", "peft"):
        (fakes / name).mkdir(parents=True)
        (fakes / name / "__init__.py").write_text(FAKE_TORCH if name == "torch" else '__version__ = "0.0.fake"\n')
    if test_file:
        (tree / test_file).parent.mkdir(parents=True, exist_ok=True)
        (tree / test_file).write_text("rows")
    log = tmp_path / "uv.log"
    environment = {"PATH": f"{bin_}:/usr/bin:/bin", "FAKE_LOG": str(log), "REAL_PY": sys.executable, "PYTHONPATH": str(fakes), "HOME": str(tmp_path), **env}
    done = subprocess.run(["sh", "gpu/bootstrap_colab.sh"], cwd=tree, env=environment, capture_output=True, text=True, timeout=120)
    return done, log.read_text() if log.exists() else ""


needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")


@needs_sh
def test_bootstrap_succeeds_when_cuda_is_usable_in_both_venvs_and_rdkit_is_pinned(tmp_path):
    done, log = boot(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "vllm venv: torch 0.0.fake cuda True devices 1" in done.stdout and "train venv: torch 0.0.fake cuda True devices 1" in done.stdout
    assert "uv pip install --python .agent-runs/sft-venv/bin/python rdkit==2026.9.1 numpy pytest" in log
    assert "Bootstrap complete" in done.stdout


@needs_sh
@pytest.mark.parametrize("env", [{"FAKE_CUDA_VLLM": "0"}, {"FAKE_DEVICES_VLLM": "0"}, {"FAKE_CUDA_TRAIN": "0"}, {"FAKE_DEVICES_TRAIN": "0"}],
                         ids=["vllm cuda off", "vllm no device", "train cuda off", "train no device"])
def test_bootstrap_exits_nonzero_unless_cuda_and_a_device_exist_in_both_venvs(tmp_path, env):
    done, _ = boot(tmp_path, **env)
    assert done.returncode != 0 and "CUDA is not usable" in done.stderr and "Bootstrap complete" not in done.stdout


@needs_sh
def test_bootstrap_installs_one_pinned_rdkit_and_checks_the_installed_version(tmp_path):
    text = BOOT.read_text()
    assert text.count("2026.9.1") == 1                                         # one place for the pin
    done, log = boot(tmp_path, script=text.replace("RDKIT_VERSION=2026.9.1", "RDKIT_VERSION=2020.3.1"))
    assert "rdkit==2020.3.1" in log                                            # the install follows the variable
    assert done.returncode != 0 and "not the pinned 2020.3.1" in done.stderr   # the fake uv installs nothing: the real 2026.9.1 stays, and that is checked
    assert " rdkit " not in log                                                # never an unpinned rdkit


@needs_sh
@pytest.mark.parametrize("leak", ["data/processed/test_t.jsonl", "data/processed/arms/name/test_r.jsonl", "data/processed/arms/a4_neighbours_test_t.json"])
def test_bootstrap_refuses_a_tree_that_holds_a_test_file_before_installing_anything(tmp_path, leak):
    done, log = boot(tmp_path, test_file=leak)
    assert done.returncode != 0 and "A Test file is in the tree" in done.stderr
    assert log == ""                                                           # uv was never called


@needs_sh
def test_bootstrap_still_refuses_to_run_outside_the_repository_root(tmp_path):
    done, _ = boot(tmp_path, script=BOOT.read_text().replace("gpu/sample_vllm.py", "gpu/nope.py"))
    assert done.returncode != 0 and "repository root" in done.stderr
