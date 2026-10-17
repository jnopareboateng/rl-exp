#!/usr/bin/env python3
"""Offline vLLM sampling. All GPU execution, quantization and LoRA are UNTESTED.

Use vLLM 0.31.0. --lora accepts a local PEFT adapter of rank at most 64.
Each prompt is one durable batch of k samples. Resume removes only an incomplete
tail batch and rejects changed inputs/settings. Use one output directory per run.
--prompts-per-batch defaults to 1. auto estimates KV capacity before loading
weights; GPU sizing and CUDA graphs are UNTESTED. Neither changes run identity.
--limit-drugs selects a random subset of sorted drug names with
--prompt-subset-seed (default 0), then keeps all their rows in input order.
--system-text replaces the system-turn text of every prompt before anything else, so it is part of the data hash. The selected rows determine the data hash. A time limit includes startup, but
always permits one call. It stops only after the current call is durable.
--dry-run needs no vLLM, tokenizer, GPU or network. Its token counts are estimates
from n_prompt_tokens (or UTF-8 byte lengths); nothink adds a byte-length allowance.
Only model/tokenizer downloads may use the network. No prompt or sample is logged.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

from env_check import environment

OFF = "\n</think>\n\n"
SAVED_OFF = "</think>\n\n"


def load_rows(args):
    rows, seen = [], set()
    for split in args.splits.split(","):
        if split not in {"train", "val", "test_t", "test_r"}:
            raise SystemExit("Unsupported split")
        with (args.data / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    valid = (isinstance(row["id"], (str, int)) and not isinstance(row["id"], bool)
                             and isinstance(row["group"], str) and row["group"]
                             and row["split"] == split and row["id"] not in seen
                             and isinstance(row["prompt"], str) and row["prompt"].endswith("<think>\n"))
                except (ValueError, KeyError, TypeError):
                    raise SystemExit("Invalid input row") from None
                if not valid:
                    raise SystemExit("Invalid or duplicate input row")
                seen.add(row["id"])
                if getattr(args, "system_text", None) is not None:
                    row["prompt"], n = re.subn(r"(<\|im_start\|>system\n).*?(<\|im_end\|>)", lambda m: m.group(1) + args.system_text + m.group(2),
                                               row["prompt"], count=1, flags=re.S)
                    if n != 1:
                        raise SystemExit("Prompt has no system block to replace")
                rows.append(row)
    groups = sorted({r["group"] for r in rows})
    if args.limit_drugs and args.limit_drugs < len(groups):
        selected = set(random.Random(args.prompt_subset_seed).sample(groups, args.limit_drugs))
        rows = [r for r in rows if r["group"] in selected]
    if not rows:
        raise SystemExit("No prompts selected")
    return rows


def add_counts(counts, samples):
    split = samples[0]["split"]
    entry = counts.setdefault(split, dict(completed_prompts=0, samples=0, generated_tokens=0,
                                         closed_samples=0, truncated_samples=0))
    entry["completed_prompts"] += 1
    entry["samples"] += len(samples)
    entry["generated_tokens"] += sum(s["n_tokens"] for s in samples)
    entry["closed_samples"] += sum("</think>" in s["text"] for s in samples)
    entry["truncated_samples"] += sum(s["finish_reason"] == "length" for s in samples)


def recover(stream, rows, config, k):
    """Check complete prompt batches. Remove a torn final batch under the file lock."""
    stream.seek(0)
    done, batch, start, tokens, counts = set(), [], 0, 0, {}
    by_id = {row["id"]: row for row in rows}
    for line in iter(stream.readline, b""):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("Torn tail")
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if stream.read():
                raise SystemExit("Corrupt samples before end of file") from None
            break
        if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
            raise SystemExit("Samples do not match this run")
        source = by_id.get(row.get("id"))
        if (not source or row["id"] in done or row.get("sample_idx") != len(batch)
                or (batch and row["id"] != batch[0]["id"])
                or any(row.get(key) != value for key, value in config.items())
                or row.get("group") != source["group"] or row.get("split") != source["split"]
                or not isinstance(row.get("text"), str)
                or row.get("finish_reason") not in {"stop", "length", "abort"}
                or not isinstance(row.get("n_tokens"), int) or row["n_tokens"] < 0):
            raise SystemExit("Samples do not match this run")
        batch.append(row)
        if len(batch) == k:
            done.add(row["id"])
            tokens += sum(r["n_tokens"] for r in batch)
            add_counts(counts, batch)
            batch, start = [], stream.tell()
    stream.seek(start)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())
    return done, tokens, counts


def auto_batch_size(model, k, max_len, gpus, utilization, platform):
    """Estimate capacity without creating a CUDA context. GPU behavior is UNTESTED.

    Use checkpoint bytes as a conservative weight bound, including vision/MTP.
    Reserve recurrent/convolution state at the scheduler ceiling and 1 GiB per
    device for workspaces. Use 70% of the remaining KV budget. Graphs and
    allocator overhead still need a real probe.
    """
    from huggingface_hub import hf_hub_download
    import pynvml

    def read_config(name):
        path = Path(model) / name if Path(model).is_dir() else Path(hf_hub_download(model, name))
        return json.loads(path.read_text(encoding="utf-8"))

    config = read_config("config.json")
    text = config.get("text_config", config)
    layers, interval, heads, dim = (text[key] for key in (
        "num_hidden_layers", "full_attention_interval", "num_key_value_heads", "head_dim"))
    if any(not isinstance(n, int) or n < 1 for n in (layers, interval, heads, dim)) or layers < interval:
        raise SystemExit("Invalid KV dimensions; use an explicit --prompts-per-batch")
    full_layers = layers // interval
    tp = len(gpus)
    bytes_per_token = 2 * full_layers * heads * dim * 2
    per_rank_bytes = 2 * full_layers * math.ceil(heads / tp) * dim * 2
    linear_layers = layers - full_layers
    state_bytes = 0
    if linear_layers:
        kh, vh, kd, vd, kernel = (text[key] for key in (
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim"))
        if any(not isinstance(n, int) or n < 1 for n in (kh, vh, kd, vd, kernel)):
            raise SystemExit("Invalid recurrent dimensions; use an explicit --prompts-per-batch")
        state_bytes = linear_layers * (
            math.ceil(vh / tp) * kd * vd * 4
            + (2 * math.ceil(kh / tp) * kd + math.ceil(vh / tp) * vd) * (kernel - 1) * 2)
    recurrent_reserve = state_bytes * min(k * 16, 256)
    weights = read_config("model.safetensors.index.json")["metadata"]["total_size"]
    if not isinstance(weights, int) or weights < 1:
        raise SystemExit("Invalid checkpoint size; use an explicit --prompts-per-batch")
    pynvml.nvmlInit()
    try:
        free = [int(pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByUUID(platform.get_device_uuid(i))).free) for i in range(tp)]
    finally:
        pynvml.nvmlShutdown()
    reserve = 2**30
    budget = max(0, min(min(f, int(g["memory_bytes"] * utilization))
                        - math.ceil(weights / tp) - reserve - recurrent_reserve for f, g in zip(free, gpus)))
    per_sequence = per_rank_bytes * max_len
    p = max(1, min(16, math.floor(0.7 * budget / (k * per_sequence))))
    return p, dict(num_hidden_layers=layers, full_attention_interval=interval,
                   full_attention_layers=full_layers, num_key_value_heads=heads, head_dim=dim,
                   dtype_bytes=2, kv_bytes_per_token=bytes_per_token,
                   kv_bytes_per_token_per_gpu=per_rank_bytes, free_bytes_per_gpu=free,
                   checkpoint_bytes=weights, workspace_reserve_bytes_per_gpu=reserve,
                   recurrent_bytes_per_sequence_per_gpu=state_bytes,
                   recurrent_reserve_bytes_per_gpu=recurrent_reserve,
                   kv_budget_bytes_per_gpu=budget, per_sequence_kv_bytes_at_max_len=per_sequence,
                   max_model_len=max_len, safety_fraction=0.7, prompts_per_batch=p,
                   estimate_status="UNTESTED on GPU")


def write_meta(path, meta):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", default="val,test_t,test_r")
    parser.add_argument("--mode", choices=("think", "nothink"), default="think")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quantization")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--system-text", help="replace the text of the system turn in every prompt (a prompt-variant baseline); part of the run identity")
    parser.add_argument("--limit-drugs", type=int)
    parser.add_argument("--prompt-subset-seed", type=int, default=0)
    parser.add_argument("--prompts-per-batch", default="1", help="positive integer or auto")
    parser.add_argument("--gpu-memory-utilization", type=float, help="override the engine's memory fraction (default 0.85, 0.80 below 20 GiB per GPU); a 27B FP8 checkpoint on a 40 GB card needs about 0.93")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--stop-after-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prompts_per_batch != "auto":
        try:
            args.prompts_per_batch = int(args.prompts_per_batch)
        except ValueError:
            parser.error("--prompts-per-batch must be an integer >= 1 or auto")
        if args.prompts_per_batch < 1:
            parser.error("--prompts-per-batch must be >= 1")
    if args.gpu_memory_utilization is not None and not 0.3 <= args.gpu_memory_utilization <= 0.97:
        parser.error("--gpu-memory-utilization must be between 0.3 and 0.97")
    if (args.k < 1 or args.max_tokens < 1 or not math.isfinite(args.temperature)
            or args.temperature <= 0 or not 0 < args.top_p <= 1
            or args.seed < 0 or args.prompt_subset_seed < 0
            or (args.limit_drugs is not None and args.limit_drugs < 1)
            or (args.stop_after_seconds is not None and
                (not math.isfinite(args.stop_after_seconds) or args.stop_after_seconds <= 0))):
        parser.error("Use positive k, lengths and temperature; 0 < top-p <= 1; seed >= 0")
    rows = load_rows(args)
    if args.dry_run:
        counts = [r.get("n_prompt_tokens") for r in rows]
        counts = [n if isinstance(n, int) and n > 0 else len(r["prompt"].encode())
                  for r, n in zip(rows, counts)]
        counts = [n + (len(OFF.encode()) if args.mode == "nothink" else 0) for n in counts]
        print(json.dumps({"dry_run": True, "prompts": len(rows),
                          "drugs": len({r["group"] for r in rows}), "prompt_tokens": sum(counts),
                          "longest_prompt_tokens": max(counts), "token_counts": "estimated; see docstring",
                          "estimated_generated_tokens": len(rows) * args.k * args.max_tokens,
                          "estimated_max_model_len": max(counts) + args.max_tokens,
                          "prompts_per_batch": args.prompts_per_batch,
                          "prompt_subset_seed": args.prompt_subset_seed,
                          "mode": args.mode, "resume_checked": False, "gpu_execution": "UNTESTED"}))
        return
    config = {key: getattr(args, key) for key in
              ("model", "mode", "temperature", "top_p", "max_tokens", "seed")}
    signature = hashlib.sha256(json.dumps(
        [(r["id"], r["group"], r["split"], r["prompt"]) for r in rows], ensure_ascii=False,
    ).encode()).hexdigest()
    identity = dict(config, k=args.k, data_sha256=signature, quantization=args.quantization,
                    lora=str(args.lora.resolve()) if args.lora else None)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "samples.jsonl").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if (meta and meta.get("configuration") != identity) or (stream.tell() and not meta):
            raise SystemExit("Run settings changed or metadata missing; use a new output directory")
        done, total_tokens, split_counts = recover(stream, rows, config, args.k)
        meta.update(per_split=split_counts, per_split_scope="all durable samples")
        pending = [r for r in rows if r["id"] not in done]
        if not pending:
            meta.update(status="complete", completed_prompts=len(done), total_generated_tokens=total_tokens)
            write_meta(path, meta)
            print(f"prompts={len(done)}/{len(rows)} generated_tokens=0 tokens/s=0.00 (already complete)")
            return
        started = time.perf_counter()
        for key in ("VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK", "HF_HUB_DISABLE_TELEMETRY", "FLASHINFER_NO_DOWNLOAD"):
            os.environ[key] = "1"
        os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform as gpu
        from vllm.tokenizers.registry import get_tokenizer

        gpus = [{"name": gpu.get_device_name(i), "memory_bytes": gpu.get_device_total_memory(i),
                 "compute_capability": list(gpu.get_device_capability(i))}
                for i in range(gpu.device_count())]
        if not gpus or any(g["compute_capability"] < [7, 5] for g in gpus):
            raise SystemExit("vLLM requires visible GPUs with compute capability >= 7.5")
        dtype = "float16" if any(g["compute_capability"] < [8, 0] for g in gpus) else "bfloat16"
        tokenizer = get_tokenizer(args.model)
        prompts = [r["prompt"] + (OFF if args.mode == "nothink" else "") for r in rows]
        token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        engine = {"dtype": dtype, "tensor_parallel_size": len(gpus),
                  "max_model_len": max(map(len, token_ids)) + args.max_tokens,
                  "gpu_memory_utilization": args.gpu_memory_utilization
                  or (0.80 if min(g["memory_bytes"] for g in gpus) < 20 * 2**30 else 0.85),
                  "language_model_only": True, "enforce_eager": not args.cuda_graphs,
                  "enable_prefix_caching": False, "seed": args.seed,
                  "enable_lora": bool(args.lora), "max_lora_rank": 64}
        p, estimate = args.prompts_per_batch, None
        if p == "auto":
            p, estimate = auto_batch_size(args.model, args.k, engine["max_model_len"], gpus,
                                          engine["gpu_memory_utilization"], gpu)
        prefill_cap = 8192 if min(g["memory_bytes"] for g in gpus) < 20 * 2**30 else 32768
        engine.update(max_num_seqs=min(args.k * p, 256), enable_chunked_prefill=True,
                      max_num_batched_tokens=max(8192, min(prefill_cap, p * max(map(len, token_ids))),
                                                 min(args.k * p, 256)))
        if dtype == "float16":
            engine["attention_backend"] = "TRITON_ATTN"
        meta.update(configuration=identity, model=args.model, environment=environment(), gpus=gpus,
                    engine=engine, command_line=sys.argv, timing_scope="last invocation that generated samples",
                    wall_time_s=0, generated_tokens=0, total_generated_tokens=total_tokens,
                    generation_wall_time_s=0, tokens_per_second=0, closed_samples=0, truncated_samples=0,
                    completed_prompts=len(done), prompts_per_batch=p, auto_batch_estimate=estimate,
                    peak_memory_bytes=None, peak_memory_status="unavailable across vLLM workers",
                    closure_scope="saved text; nothink includes the required injected closing prefix",
                    selection=dict(limit_drugs=args.limit_drugs, prompt_subset_seed=args.prompt_subset_seed),
                    stop_after_seconds=args.stop_after_seconds, calls=meta.get("calls", []),
                    invocation=meta.get("invocation", 0) + 1,
                    calls_scope="all completed calls; invocation identifies each process run",
                    status="running")
        write_meta(path, meta)
        llm = LLM(model=args.model, quantization=args.quantization, **engine)
        adapter = None
        if args.lora:
            from vllm.lora.request import LoRARequest
            adapter = LoRARequest("adapter", 1, str(args.lora.resolve()))
        params = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                                max_tokens=args.max_tokens, seed=args.seed)
        pending = [(r, ids) for r, ids in zip(rows, token_ids) if r["id"] not in done]
        for offset in range(0, len(pending), p):
            batch = pending[offset:offset + p]
            tick = time.perf_counter()
            result = llm.generate([{"prompt_token_ids": ids} for _, ids in batch], params,
                                  lora_request=adapter, use_tqdm=False)
            elapsed = time.perf_counter() - tick
            if len(result) != len(batch):
                raise SystemExit("vLLM returned an incomplete prompt batch")
            count = 0
            for (row, _), response in zip(batch, result):
                outputs = sorted(response.outputs, key=lambda x: x.index)
                if [o.index for o in outputs] != list(range(args.k)) or any(o.finish_reason is None for o in outputs):
                    raise SystemExit("vLLM returned an incomplete prompt batch")
                samples = [dict(config, id=row["id"], group=row["group"], split=row["split"],
                                sample_idx=o.index, text=(SAVED_OFF if args.mode == "nothink" else "") + o.text,
                                finish_reason=o.finish_reason, n_tokens=len(o.token_ids)) for o in outputs]
                stream.write("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples).encode())
                stream.flush()
                os.fsync(stream.fileno())
                tokens = sum(s["n_tokens"] for s in samples)
                count += tokens
                done.add(row["id"])
                add_counts(split_counts, samples)
                meta["generated_tokens"] += tokens
                meta["total_generated_tokens"] += tokens
                meta["closed_samples"] += sum("</think>" in s["text"] for s in samples)
                meta["truncated_samples"] += sum(s["finish_reason"] == "length" for s in samples)
            meta["generation_wall_time_s"] += elapsed
            meta["calls"].append(dict(prompts=len(batch), generated_tokens=count, wall_time_s=elapsed,
                                      invocation=meta["invocation"], prompts_per_batch=p,
                                      enforce_eager=engine["enforce_eager"],
                                      tokens_per_second=count / max(elapsed, 1e-9)))
            stopped = (args.stop_after_seconds is not None and
                       time.perf_counter() - started >= args.stop_after_seconds)
            meta.update(wall_time_s=time.perf_counter() - started,
                        tokens_per_second=meta["generated_tokens"] / max(meta["generation_wall_time_s"], 1e-9),
                        completed_prompts=len(done), status=("stopped_time_limit" if stopped else
                                                            "complete" if len(done) == len(rows) else "running"))
            write_meta(path, meta)
            print(f"prompts={len(done)}/{len(rows)} generated_tokens={count} tokens/s={count / max(elapsed, 1e-9):.2f}", flush=True)
            if stopped:
                return


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if os.environ.get("SAMPLER_VERBOSE_ERRORS") == "1":  # opt-in: tracebacks can carry data, so off by default
            traceback.print_exc()
        reason = next((label for word, label in (("memory", "memory/KV allocation"), ("driver", "GPU driver"),
                      ("quant", "quantization"), ("lora", "adapter"), ("dtype", "dtype"))
                       if word in str(error).lower()), "runtime")
        print(f"Sampling failed ({type(error).__name__}): {reason}. Data values are omitted.", file=sys.stderr)
        for frame in traceback.extract_tb(error.__traceback__)[-4:]:
            print(f"  {frame.filename}:{frame.lineno} ({frame.name})", file=sys.stderr)
        sys.exit(1)
