#!/usr/bin/env python3
"""Write results/frozen_arms.json (the spec of docs/prereg-v1.md section 3) from results/selection.json, the adapter hashes and the data manifest, and its sha256 file.

  .venv/bin/python scripts/make_frozen_spec.py [--selection results/selection.json] [--hashes results/adapter_hashes.json] [--manifest data/MANIFEST.sha256]
                                              [--out results/frozen_arms.json]

Each adapter block carries the base "revision" of its hash-file entry (the commit the adapter was trained on; gpu/prepare_final.py passes it to merge_lora.py and checks the adapter directory's
revision.json against it; a hash-file entry without one is a KeyError). Everything the selection decides (temperatures, the RL checkpoint and its run, the comparators) is copied from it; the lead's choices (references, diagnostics, probes) are the constants below, the
Test run list of the protocol. gpu/final_eval.py then refuses the spec if it is not what the selection chose, so a slip here fails loudly instead of being sampled.
"""
import argparse
import hashlib
import json
from pathlib import Path

Q4 = "Qwen/Qwen3.5-4B"
W, C = "adapter_model.safetensors", "adapter_config.json"
PLAIN = ".agent-runs/adapters_plain"      # plain copies of the adapter files on the runtime, outside their run directories: gpu/merge_lora.py treats `<run>/adapters/step-N` as owned by a run and then needs the
A2_ADAPTER = f"{PLAIN}/A2-epoch1"       # run's committed checkpoint state, which is not kept for a finished run; the sha256 of the files is the identity, not the path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_key(run, checkpoint):
    """The adapter_hashes.json entry of an RL candidate: R1-step-40 for rlfull-s40-t0.6, R5-step-60 for rlfull5-s60-t0.3."""
    return f"{'R5' if run.startswith('rlfull5-') else 'R1'}-{checkpoint}"


def build(selection, hashes, manifest_sha256):
    chosen = selection["arms"]
    adapter = lambda path, key: {"path": path, "base": Q4, "revision": hashes[key]["revision"], "sha256": {W: hashes[key][W], C: hashes[key][C]}}
    rl_id = selection["rl"]
    rl = chosen[rl_id]
    a2 = chosen["A2"]
    arms = []

    def arm(id, model, mode, data, system, temperature, k, max_tokens=256, **kw):
        arms.append({"id": id, "model": model, "mode": mode, "data": data, "system": system, "temperature": temperature, "k": k, "max_tokens": max_tokens, **kw})

    for sel_id, label in (("A0n", "4B untrained, no-think"), ("A4", "4B + 3 retrieved examples")):
        s = chosen[sel_id]
        arm(sel_id, Q4, s["spec"]["mode"], s["spec"]["data"], s["spec"]["system"], s["temperature"], 8, s["spec"]["max_tokens"], label=label)
    ad_a2 = adapter(A2_ADAPTER, "A2-epoch1")
    ad_rl = adapter(f"{PLAIN}/{hash_key(rl['run'], rl['checkpoint'])}", hash_key(rl["run"], rl["checkpoint"]))
    for sel_id, ad, s, label in (("A2", ad_a2, a2, "4B answer-only SFT (epoch 1)"), (rl_id, ad_rl, rl, f"4B RL, standard reward ({hash_key(rl['run'], rl['checkpoint'])})")):
        arm(sel_id, f"final_models/{sel_id}", s["spec"]["mode"], s["spec"]["data"], s["spec"]["system"], s["temperature"], 8, s["spec"]["max_tokens"], label=label, adapter=ad)
    for base, ad, s in (("A2", ad_a2, a2), (rl_id, ad_rl, rl)):
        for kind in ("shuffle", "blank"):
            arm(f"{base}-{kind}", f"final_models/{base}", s["spec"]["mode"], kind, "direct", s["temperature"], 8, s["spec"]["max_tokens"], role="diagnostic", of=base, adapter=ad, label=f"{base} with a {kind} profile")
    for kind, max_tokens in (("name", 256), ("ident", 128)):                      # name-given and identification probes of the untrained 4B (the larger models are Val-only references)
        arm(f"{kind}-4B", Q4, "nothink", kind, "dataset", 1.0, 8, max_tokens, role="probe", label=f"4B {kind}")
    return {"frozen": True, "version": "prereg-v1", "splits": ["test_t", "test_r"], "selection_sha256": None, "data_manifest_sha256": manifest_sha256, "arms": arms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    ap.add_argument("--hashes", type=Path, default=Path("results/adapter_hashes.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/MANIFEST.sha256"))
    ap.add_argument("--out", type=Path, default=Path("results/frozen_arms.json"))
    args = ap.parse_args(argv)
    spec = build(json.loads(args.selection.read_text(encoding="utf-8")), json.loads(args.hashes.read_text(encoding="utf-8")), sha256(args.manifest))
    spec["selection_sha256"] = sha256(args.selection)
    args.out.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    digest = sha256(args.out)
    args.out.with_suffix(".sha256").write_text(f"{digest}  {args.out.name}\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(spec['arms'])} arms) sha256 {digest}")


if __name__ == "__main__":
    main()
