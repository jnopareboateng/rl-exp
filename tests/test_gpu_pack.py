"""CPU checks with fake data and engines. These cannot prove GPU behavior.

GPU loading, kernels, memory sizing, CUDA graphs and notebook installs are
UNTESTED here. No test reads the private dataset or relies on scratch files.
"""

import contextlib
import fcntl
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(GPU))
    pack = import_file("gpu_pack_test", GPU / "sample_vllm.py")
    helper = import_file("session1_test", GPU / "session1.py")
    data = tmp_path / "data"
    data.mkdir()
    rows = [dict(id=f"v{i}", group=f"drug-{i // 2}", split="val", prompt=f"fixture-\u03b1{i}<think>\n",
                 n_prompt_tokens=4 + i, cluster_id=i // 2, ref_inchikey="FAKE-KEY",
                 future_field={"accepted": True}) for i in range(6)]
    (data / "val.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    state = dict(calls=[], engines=[], tokenizer_loads=0, cap=(7, 5), devices=2)

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Engine:
        def __init__(self, **kwargs):
            state["engines"].append(kwargs)

        def generate(self, prompts, params, **kwargs):
            if "error" in state:
                raise RuntimeError(state["error"])
            state["calls"].append((prompts, params, kwargs))
            results = []
            for prompt in prompts:
                nothink = bytes(prompt["prompt_token_ids"]).decode().endswith(pack.OFF)
                outputs = [types.SimpleNamespace(
                    index=i, text=("fixture answer" if nothink or i % 2 else "fixture </think> answer"),
                    token_ids=[1] * (i + 1), finish_reason="length" if i % 2 else "stop",
                ) for i in range(params.n)]
                results.append(types.SimpleNamespace(outputs=outputs))
            if state.get("incomplete"):
                results[-1].outputs.pop()
            return results

    def tokenizer(model):
        state["tokenizer_loads"] += 1

        def encode(text, add_special_tokens):
            assert not add_special_tokens
            return list(text.encode())

        return types.SimpleNamespace(encode=encode)

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = Engine, Params
    platform = types.ModuleType("vllm.platforms")
    platform.current_platform = types.SimpleNamespace(
        device_count=lambda: state["devices"], get_device_name=lambda i: "fake GPU",
        get_device_total_memory=lambda i: 15 * 2**30, get_device_capability=lambda i: state["cap"],
        get_device_uuid=lambda i: f"GPU-{i}",
    )
    registry = types.ModuleType("vllm.tokenizers.registry")
    registry.get_tokenizer = tokenizer
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda *args: args
    for name, module in {"vllm": vllm, "vllm.platforms": platform,
                         "vllm.tokenizers.registry": registry, "vllm.lora.request": lora}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(pack, "environment", lambda: {"gpus": []})

    def run(out, *extra):
        argv = ["sample_vllm.py", "--model", "fake-local", "--data", str(data), "--splits", "val",
                "--k", "2", "--max-tokens", "7", "--out", str(out), *map(str, extra)]
        monkeypatch.setattr(sys, "argv", argv)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            pack.main()
        assert "fixture" not in capture.getvalue()
        return capture.getvalue()

    return types.SimpleNamespace(pack=pack, helper=helper, data=data, rows=rows, state=state,
                                 run=run, platform=platform.current_platform, out=tmp_path / "out")


def samples(out):
    return [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]


def meta(out):
    return json.loads((out / "meta.json").read_text())


@pytest.mark.parametrize("mode", ["think", "nothink"])
def test_dry_run_without_site_packages(rig, mode):
    out = rig.out
    result = subprocess.run([sys.executable, "-S", str(GPU / "sample_vllm.py"), "--model", "fake",
                             "--data", str(rig.data), "--splits", "val", "--mode", mode,
                             "--k", "2", "--max-tokens", "7", "--out", str(out),
                             "--prompts-per-batch", "auto", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["prompts"] == 6 and plan["estimated_generated_tokens"] == 84
    assert plan["prompt_tokens"] == sum(4 + i for i in range(6)) + (6 * len(rig.pack.OFF) if mode == "nothink" else 0)
    assert not out.exists()


@pytest.mark.parametrize("mode", ["think", "nothink"])
def test_prompt_and_saved_text_and_extra_fields(rig, mode):
    rig.run(rig.out, "--mode", mode, "--lora", GPU)
    saved = samples(rig.out)
    expected_keys = {"id", "group", "split", "sample_idx", "text", "finish_reason", "n_tokens",
                     "model", "mode", "temperature", "top_p", "max_tokens", "seed"}
    assert len(saved) == 12 and all(set(s) == expected_keys for s in saved)
    assert all(s["text"].startswith(rig.pack.SAVED_OFF) == (mode == "nothink") for s in saved)
    prompts, params, kwargs = rig.state["calls"][0]
    assert prompts[0]["prompt_token_ids"] == list((rig.rows[0]["prompt"] + (rig.pack.OFF if mode == "nothink" else "")).encode())
    assert vars(params) == dict(n=2, temperature=1.0, top_p=1.0, max_tokens=7, seed=0)
    assert kwargs["lora_request"][-1] == str(GPU)
    engine = rig.state["engines"][0]
    assert engine["dtype"] == "float16" and engine["tensor_parallel_size"] == 2
    assert engine["attention_backend"] == "TRITON_ATTN" and engine["language_model_only"]
    assert engine["gpu_memory_utilization"] == 0.8 and engine["enforce_eager"]
    assert engine["max_model_len"] == max(len((r["prompt"] + (rig.pack.OFF if mode == "nothink" else "")).encode()) for r in rig.rows) + 7
    stats = meta(rig.out)
    assert stats["closed_samples"] == (12 if mode == "nothink" else 6)
    assert stats["truncated_samples"] == 6 and stats["per_split"]["val"]["samples"] == 12
    assert all(rig.pack.os.environ[key] == "1" for key in (
        "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK", "HF_HUB_DISABLE_TELEMETRY", "FLASHINFER_NO_DOWNLOAD"))


def test_multi_prompt_writes_are_prompt_atomic(rig, monkeypatch):
    real_fsync = rig.pack.os.fsync
    durable_sizes = []

    def fsync(fd):
        if Path(f"/proc/self/fd/{fd}").resolve().name == "samples.jsonl":
            durable_sizes.append(len(samples(rig.out)))
        return real_fsync(fd)

    monkeypatch.setattr(rig.pack.os, "fsync", fsync)
    rig.run(rig.out, "--prompts-per-batch", 4, "--cuda-graphs")
    assert [len(call[0]) for call in rig.state["calls"]] == [4, 2]
    assert durable_sizes == [0, 2, 4, 6, 8, 10, 12]
    assert [s["sample_idx"] for s in samples(rig.out)] == [0, 1] * 6
    stats = meta(rig.out)
    assert stats["engine"]["max_num_seqs"] == 8
    assert stats["engine"]["max_num_batched_tokens"] >= 8192
    assert stats["engine"]["enforce_eager"] is False
    assert [call["prompts"] for call in stats["calls"]] == [4, 2]
    assert sum(call["generated_tokens"] for call in stats["calls"]) == 18
    assert stats["tokens_per_second"] > 0
    assert "prompts_per_batch" not in stats["configuration"] and "cuda_graphs" not in stats["configuration"]


def test_torn_multi_prompt_call_resumes_only_unfinished_prompts(rig):
    rig.run(rig.out, "--prompts-per-batch", 3)
    original = (rig.out / "samples.jsonl").read_bytes()
    lines = original.splitlines(keepends=True)
    # Kill during the second prompt of the second call, after four whole prompts.
    (rig.out / "samples.jsonl").write_bytes(b"".join(lines[:9]) + lines[9][:12])
    rig.state["calls"].clear()
    rig.run(rig.out, "--prompts-per-batch", 2, "--cuda-graphs")
    assert [len(call[0]) for call in rig.state["calls"]] == [2]
    assert (rig.out / "samples.jsonl").read_bytes() == original
    stats = meta(rig.out)
    assert stats["generated_tokens"] == 6 and stats["total_generated_tokens"] == 18
    assert stats["per_split"]["val"]["completed_prompts"] == 6
    assert stats["per_split"]["val"]["closed_samples"] == 6
    assert {call["invocation"] for call in stats["calls"]} == {1, 2}
    loads = len(rig.state["engines"])
    with (rig.out / "samples.jsonl").open("ab") as stream:
        stream.write(b'{"')
    rig.run(rig.out)
    assert len(rig.state["engines"]) == loads
    assert (rig.out / "samples.jsonl").read_bytes() == original


@pytest.mark.parametrize("changed", [("--mode", "nothink"), ("--k", "3"), ("--seed", "8")])
def test_identity_mismatch_preserves_bytes(rig, changed):
    rig.run(rig.out)
    original = (rig.out / "samples.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="settings changed"):
        rig.run(rig.out, *changed)
    assert (rig.out / "samples.jsonl").read_bytes() == original


def test_input_change_interior_corruption_and_second_writer_fail_closed(rig):
    rig.run(rig.out)
    path = rig.out / "samples.jsonl"
    original = path.read_bytes()
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            rig.run(rig.out)
    assert path.read_bytes() == original
    lines = original.splitlines(keepends=True)
    corrupted = lines[0] + b"bad-json\n" + b"".join(lines[1:])
    path.write_bytes(corrupted)
    with pytest.raises(SystemExit, match="Corrupt samples"):
        rig.run(rig.out)
    assert path.read_bytes() == corrupted
    path.write_bytes(original)
    rig.rows[0]["prompt"] = "changed<think>\n"
    (rig.data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rig.rows))
    with pytest.raises(SystemExit, match="settings changed"):
        rig.run(rig.out)
    assert path.read_bytes() == original


def test_stop_after_seconds_writes_whole_call_and_resumes(rig, monkeypatch):
    tick = iter(range(100))
    monkeypatch.setattr(rig.pack, "time", types.SimpleNamespace(perf_counter=lambda: next(tick)))
    rig.run(rig.out, "--prompts-per-batch", 2, "--stop-after-seconds", 0.1)
    assert len(samples(rig.out)) == 4
    assert meta(rig.out)["status"] == "stopped_time_limit"
    assert meta(rig.out)["completed_prompts"] == 2
    rig.run(rig.out, "--prompts-per-batch", 3)
    assert len(samples(rig.out)) == 12 and meta(rig.out)["status"] == "complete"


def test_seeded_drug_subset_keeps_all_views(rig):
    chosen = set(random.Random(0).sample(sorted({r["group"] for r in rig.rows}), 2))
    assert chosen != {"drug-0", "drug-1"}
    for out in (rig.out, rig.out.with_name("repeat")):
        rig.run(out, "--limit-drugs", 2, "--prompt-subset-seed", 0)
        saved = samples(out)
        assert {s["group"] for s in saved} == chosen
        assert len(saved) == 8
    assert samples(rig.out) == samples(rig.out.with_name("repeat"))
    with pytest.raises(SystemExit, match="settings changed"):
        rig.run(rig.out, "--limit-drugs", 2, "--prompt-subset-seed", 7)
    rig.run(rig.out.with_name("all"), "--limit-drugs", 20)
    assert len(samples(rig.out.with_name("all"))) == 12


@pytest.mark.parametrize("value", ["0", "-1", "bad", "1.5"])
def test_invalid_batch_size(rig, value):
    with pytest.raises(SystemExit):
        rig.run(rig.out, "--prompts-per-batch", value)
    assert not rig.out.exists()


@pytest.mark.parametrize("cap,devices", [((6, 0), 1), ((8, 0), 0)])
def test_unsupported_gpu_fails_before_tokenizer(rig, cap, devices):
    rig.state.update(cap=cap, devices=devices)
    with pytest.raises(SystemExit, match="visible GPUs"):
        rig.run(rig.out)
    assert rig.state["tokenizer_loads"] == 0


def test_incomplete_response_never_commits_partial_prompt(rig):
    rig.state["incomplete"] = True
    with pytest.raises(SystemExit, match="incomplete prompt batch"):
        rig.run(rig.out, "--prompts-per-batch", 3)
    assert len(samples(rig.out)) == 4


@pytest.mark.parametrize("layers,heads,expected_bytes", [(32, 4, 32768), (64, 4, 65536)])
def test_auto_arithmetic_reads_config_and_free_memory(rig, monkeypatch, tmp_path, layers, heads, expected_bytes):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"text_config": dict(
        num_hidden_layers=layers, full_attention_interval=4, num_key_value_heads=heads, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4)}))
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 8 * 2**30}}))
    hub, nvml = types.ModuleType("huggingface_hub"), types.ModuleType("pynvml")
    hub.hf_hub_download = lambda *a: pytest.fail("Local config must not use the network")
    nvml.nvmlInit = nvml.nvmlShutdown = lambda: None
    nvml.nvmlDeviceGetHandleByUUID = lambda uuid: uuid
    nvml.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(free=14 * 2**30)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    p, estimate = rig.pack.auto_batch_size(str(checkpoint), 4, 10240, [{"memory_bytes": 16 * 2**30}], 0.8, rig.platform)
    state = (layers - layers // 4) * (32 * 128 * 128 * 4 + (2 * 16 * 128 + 32 * 128) * 3 * 2)
    budget = max(0, int(16 * 2**30 * 0.8) - 8 * 2**30 - 2**30 - 64 * state)
    assert estimate["kv_bytes_per_token"] == expected_bytes
    assert estimate["recurrent_reserve_bytes_per_gpu"] == 64 * state
    assert estimate["kv_budget_bytes_per_gpu"] == budget
    assert p == max(1, min(16, math.floor(0.7 * budget / (4 * expected_bytes * 10240))))
    _, tiny = rig.pack.auto_batch_size(str(checkpoint), 4, 10240, [{"memory_bytes": 2**30}], 0.8, rig.platform)
    assert tiny["prompts_per_batch"] == 1
    nvml.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(free=90 * 2**30)
    large, _ = rig.pack.auto_batch_size(str(checkpoint), 4, 1, [{"memory_bytes": 80 * 2**30}], 0.8, rig.platform)
    assert large == 16
    nvml.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(free=(14 if handle == "GPU-0" else 12) * 2**30)
    _, two = rig.pack.auto_batch_size(str(checkpoint), 4, 10240, [{"memory_bytes": 16 * 2**30}] * 2, 0.8, rig.platform)
    assert two["kv_bytes_per_token_per_gpu"] == expected_bytes // 2
    assert two["free_bytes_per_gpu"] == [14 * 2**30, 12 * 2**30]
    assert two["kv_budget_bytes_per_gpu"] == max(0, 12 * 2**30 - 4 * 2**30 - 2**30 - 32 * state)
    (checkpoint / "model.safetensors.index.json").unlink()
    with pytest.raises(FileNotFoundError):
        rig.pack.auto_batch_size(str(checkpoint), 4, 10240, [{"memory_bytes": 16 * 2**30}], 0.8, rig.platform)


def test_auto_option_and_scheduler_ceiling(rig, monkeypatch):
    monkeypatch.setattr(rig.pack, "auto_batch_size", lambda *a: (16, {"fixture": True}))
    rig.run(rig.out, "--prompts-per-batch", "auto", "--k", 32)
    stats = meta(rig.out)
    assert stats["prompts_per_batch"] == 16 and stats["auto_batch_estimate"] == {"fixture": True}
    assert stats["engine"]["max_num_seqs"] == 256


def test_runtime_diagnostics_omit_data(rig, monkeypatch, capsys):
    rig.state["error"] = "CUDA out of memory DO_NOT_LOG_DATA"
    monkeypatch.setattr(sys, "argv", ["sample_vllm.py", "--model", "fake", "--data", str(rig.data),
                                     "--splits", "val", "--out", str(rig.out)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(GPU / "sample_vllm.py"), run_name="__main__")
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "memory/KV allocation" in captured.err and "sample_vllm.py:" in captured.err
    assert "DO_NOT_LOG_DATA" not in captured.err and "fixture" not in captured.out + captured.err


def test_session_report_aggregates_only_durable_prompts(rig, monkeypatch, tmp_path):
    scratch = tmp_path / "session"
    scratch.mkdir()
    rig.run(scratch / "think-val", "--prompts-per-batch", 2)
    path = scratch / "think-val/samples.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(samples(scratch / "think-val")[0]) + "\n")
    with path.open("ab") as stream:
        stream.write(b'{"text":"\xe2')
    (scratch / "before.json").write_text(json.dumps({"gpus": [], "python": "fake", "library_versions": {"vllm": "fake"}}))
    (scratch / "steps.tsv").write_text("stack-install\t0\t12\nsmoke-long\t124\t600\n")
    monkeypatch.chdir(tmp_path)
    rig.helper.report(scratch, 0)
    report = json.loads(Path("results/session1_report.json").read_text())
    run = report["runs"]["think-val"]
    assert run["completed_prompts"] == 6 and run["samples"] == 12
    assert run["closed_think_rate"] == 0.5 and run["truncation_rate"] == 0.5
    assert run["mean_generated_tokens"] == 1.5 and run["p90_generated_tokens"] == 2
    assert report["install_timings_s"]["stack-install"] == 12
    assert report["status"] == "partial" and report["runs"]["smoke-long"]["status"] == "failed"
    assert "fixture" not in json.dumps(report) and "text" not in run


def test_session_report_survives_a_malformed_run(rig, monkeypatch, tmp_path):
    broken = tmp_path / "think-val"
    broken.mkdir()
    (broken / "meta.json").write_text('{"status":"running"}')
    monkeypatch.chdir(tmp_path)
    rig.helper.report(tmp_path, 0)
    report = json.loads(Path("results/session1_report.json").read_text())
    assert report["status"] == "partial"
    assert report["runs"]["think-val"]["status"] == "unreadable"
    assert report["runs"]["nothink-val"]["status"] == "not_started"


def test_session_preparation_archives_smoke_and_preserves_probe(rig, monkeypatch, tmp_path):
    source = tmp_path / "data/processed"
    source.mkdir(parents=True)
    for split in ("val", "train"):
        (source / f"{split}.jsonl").write_text("".join(json.dumps(dict(r, split=split)) + "\n" for r in rig.rows))
    scratch = tmp_path / "session"
    scratch.mkdir()
    for name in ("smoke-short", "smoke-long", "think-val"):
        directory = scratch / name
        directory.mkdir()
        (directory / "preserve.txt").write_text("saved data")
    monkeypatch.chdir(tmp_path)
    rig.helper.prepare(scratch)
    assert not (scratch / "smoke-short").exists() and not (scratch / "smoke-long").exists()
    assert len(list((scratch / "archive").glob("*/smoke-*/preserve.txt"))) == 2
    assert (scratch / "think-val/preserve.txt").read_text() == "saved data"


@pytest.mark.parametrize("cap,driver,message", [([6, 0], "580.0", "capability"), ([8, 0], "570.0", "R575")])
def test_session_preflight_rejects_unsupported_runtime(rig, tmp_path, cap, driver, message):
    (tmp_path / "before.json").write_text(json.dumps({"gpus": [dict(
        name="fake", compute_capability=cap, driver_version=driver)]}))
    with pytest.raises(SystemExit, match=message):
        rig.helper.preflight(tmp_path)


def test_posix_session_stops_without_gpu_before_download(tmp_path):
    (tmp_path / "gpu").mkdir()
    for name in ("run_session1.sh", "sample_vllm.py", "session1.py", "env_check.py"):
        shutil.copyfile(GPU / name, tmp_path / "gpu" / name)
    # A controlled inventory tests shell control flow; this is not a GPU test.
    (tmp_path / "gpu/env_check.py").write_text('def environment(): return {"gpus": []}\nif __name__ == "__main__": print(\'{"gpus": []}\')\n')
    old = tmp_path / ".agent-runs/session1"
    old.mkdir(parents=True)
    (old / "after.json").write_text('{"gpus":[{"name":"stale GPU"}]}')
    result = subprocess.run(["sh", "gpu/run_session1.sh"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "No GPU detected" in result.stdout
    assert "START uv-download" not in result.stdout
    lock = tmp_path / ".agent-runs/session1/session.lock"
    assert lock.exists()
    report = json.loads((tmp_path / "results/session1_report.json").read_text())
    assert report["status"] == "failed" and report["gpu_execution"] == "UNTESTED"
    assert report["gpus"] == []
    before = (tmp_path / "results/session1_report.json").read_bytes()
    with lock.open("a+b") as stream:
        # Taking this lock proves the first process released its descriptor.
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refused = subprocess.run(["sh", "gpu/run_session1.sh"], cwd=tmp_path,
                                 capture_output=True, text=True)
        assert refused.returncode == 1 and "Cannot lock this session" in refused.stderr
        assert "START inventory" not in refused.stdout
        assert (tmp_path / "results/session1_report.json").read_bytes() == before


def test_system_text_override_rewrites_only_the_system_turn_and_changes_the_run_identity(rig):
    system, user = "ORIGINAL SYSTEM", "USER TEXT"
    rows = [dict(r, prompt=f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}{r['id']}<|im_end|>\n<|im_start|>assistant\n<think>\n")
            for r in rig.rows]
    (rig.data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rig.run(rig.out, "--system-text", "ANSWER WITH THE SMILES ONLY")
    prompt = bytes(rig.state["calls"][0][0][0]["prompt_token_ids"]).decode()
    assert "ANSWER WITH THE SMILES ONLY" in prompt and system not in prompt and prompt.count("<|im_start|>system") == 1
    assert f"{user}v0<|im_end|>" in prompt and prompt.endswith("<think>\n")
    with pytest.raises(SystemExit, match="settings changed"):
        rig.run(rig.out)                                     # same directory, no override: a different run
    other = rig.out.parent / "other"
    rig.run(other)
    assert meta(other)["configuration"]["data_sha256"] != meta(rig.out)["configuration"]["data_sha256"]
    (rig.data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rig.rows), encoding="utf-8")   # no system turn
    with pytest.raises(SystemExit, match="no system block"):
        rig.run(rig.out.parent / "third", "--system-text", "X")


def test_gpu_memory_utilization_override_reaches_the_engine_and_is_range_checked(rig):
    rig.run(rig.out, "--mode", "think", "--gpu-memory-utilization", "0.93")
    assert rig.state["engines"][0]["gpu_memory_utilization"] == 0.93
    assert meta(rig.out)["engine"]["gpu_memory_utilization"] == 0.93
    for bad in ("0.1", "0.99", "nan"):
        with pytest.raises(SystemExit):
            rig.run(rig.out.with_name("bad"), "--gpu-memory-utilization", bad)
