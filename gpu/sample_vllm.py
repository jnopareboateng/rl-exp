#!/usr/bin/env python3
"""Offline vLLM sampling. All GPU execution, quantization and LoRA are UNTESTED.

Use vLLM 0.31.0. --lora accepts a local PEFT adapter of rank at most 64.
Each prompt is one durable batch of k samples. Resume removes only an incomplete
tail batch and rejects changed inputs/settings. Use one output directory per run.
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
import sys
import time
import traceback

from env_check import environment

OFF = "\n</think>\n\n"
SAVED_OFF = "</think>\n\n"


def load_rows(args):
    rows, seen, groups = [], set(), set()
    for split in args.splits.split(","):
        if split not in {"train", "val", "test_t", "test_r"}:
            raise SystemExit("Unsupported split")
        with (args.data / f"{split}.jsonl").open() as stream:
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
                if args.limit_drugs and row["group"] not in groups and len(groups) >= args.limit_drugs:
                    continue
                groups.add(row["group"])
                rows.append(row)
    if not rows:
        raise SystemExit("No prompts selected")
    return rows


def recover(stream, rows, config, k):
    """Check complete prompt batches. Remove a torn final batch under the file lock."""
    stream.seek(0)
    done, batch, start, tokens = set(), [], 0, 0
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
        source = by_id.get(row.get("id"))
        if (not source or row["id"] in done or row.get("sample_idx") != len(batch)
                or (batch and row["id"] != batch[0]["id"])
                or any(row.get(key) != value for key, value in config.items())
                or row.get("group") != source["group"] or row.get("split") != source["split"]
                or not isinstance(row.get("text"), str)
                or not isinstance(row.get("n_tokens"), int) or row["n_tokens"] < 0):
            raise SystemExit("Samples do not match this run")
        batch.append(row)
        if len(batch) == k:
            done.add(row["id"])
            tokens += sum(r["n_tokens"] for r in batch)
            batch, start = [], stream.tell()
    stream.seek(start)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())
    return done, tokens


def write_meta(path, meta):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
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
    parser.add_argument("--limit-drugs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.k < 1 or args.max_tokens < 1 or not math.isfinite(args.temperature)
            or args.temperature <= 0 or not 0 < args.top_p <= 1
            or args.seed < 0 or (args.limit_drugs is not None and args.limit_drugs < 1)):
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
        meta = json.loads(path.read_text()) if path.exists() else {}
        if (meta and meta.get("configuration") != identity) or (stream.tell() and not meta):
            raise SystemExit("Run settings changed or metadata missing; use a new output directory")
        done, total_tokens = recover(stream, rows, config, args.k)
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
                  "gpu_memory_utilization": 0.80 if min(g["memory_bytes"] for g in gpus) < 20 * 2**30 else 0.85,
                  "language_model_only": True, "enforce_eager": True,
                  "enable_prefix_caching": False, "max_num_seqs": args.k,
                  "max_num_batched_tokens": max(2048, args.k), "seed": args.seed,
                  "enable_lora": bool(args.lora), "max_lora_rank": 64}
        if dtype == "float16":
            engine["attention_backend"] = "TRITON_ATTN"
        meta.update(configuration=identity, model=args.model, environment=environment(), gpus=gpus,
                    engine=engine, command_line=sys.argv, timing_scope="last invocation that generated samples",
                    wall_time_s=0, generated_tokens=0, total_generated_tokens=total_tokens,
                    generation_wall_time_s=0, tokens_per_second=0, closed_samples=0, truncated_samples=0,
                    status="running")
        write_meta(path, meta)
        llm = LLM(model=args.model, quantization=args.quantization, **engine)
        adapter = None
        if args.lora:
            from vllm.lora.request import LoRARequest
            adapter = LoRARequest("adapter", 1, str(args.lora.resolve()))
        params = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                                max_tokens=args.max_tokens, seed=args.seed)
        for row, ids in zip(rows, token_ids):
            if row["id"] in done:
                continue
            tick = time.perf_counter()
            result = llm.generate([{"prompt_token_ids": ids}], params, lora_request=adapter, use_tqdm=False)
            elapsed = time.perf_counter() - tick
            if len(result) != 1:
                raise SystemExit("vLLM returned an incomplete prompt batch")
            outputs = sorted(result[0].outputs, key=lambda x: x.index)
            if [o.index for o in outputs] != list(range(args.k)) or any(o.finish_reason is None for o in outputs):
                raise SystemExit("vLLM returned an incomplete prompt batch")
            samples = [dict(config, id=row["id"], group=row["group"], split=row["split"],
                            sample_idx=o.index, text=(SAVED_OFF if args.mode == "nothink" else "") + o.text,
                            finish_reason=o.finish_reason, n_tokens=len(o.token_ids)) for o in outputs]
            stream.write("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples).encode())
            stream.flush()
            os.fsync(stream.fileno())
            count = sum(s["n_tokens"] for s in samples)
            done.add(row["id"])
            meta["generated_tokens"] += count
            meta["total_generated_tokens"] += count
            meta["closed_samples"] += sum("</think>" in s["text"] for s in samples)
            meta["truncated_samples"] += sum(s["finish_reason"] == "length" for s in samples)
            meta["generation_wall_time_s"] += elapsed
            meta.update(wall_time_s=time.perf_counter() - started,
                        tokens_per_second=meta["generated_tokens"] / max(meta["generation_wall_time_s"], 1e-9),
                        completed_prompts=len(done), status="complete" if len(done) == len(rows) else "running")
            write_meta(path, meta)
            print(f"prompts={len(done)}/{len(rows)} generated_tokens={count} tokens/s={count / max(elapsed, 1e-9):.2f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        reason = next((label for word, label in (("memory", "memory/KV allocation"), ("driver", "GPU driver"),
                      ("quant", "quantization"), ("lora", "adapter"), ("dtype", "dtype"))
                       if word in str(error).lower()), "runtime")
        print(f"Sampling failed ({type(error).__name__}): {reason}. Data values are omitted.", file=sys.stderr)
        for frame in traceback.extract_tb(error.__traceback__)[-4:]:
            print(f"  {frame.filename}:{frame.lineno} ({frame.name})", file=sys.stderr)
        sys.exit(1)
