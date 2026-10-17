"""Tests for gpu/rl_grpo.py: data and flag validation, the step loop (with a fake policy), the weighted loss and the fp32 head.

CPU tests cannot prove GPU, kernel, generation or memory behaviour: HF `generate` with left padding on the GatedDeltaNet layers, PEFT
adapter loading with is_trainable, memory at 256 sequences and the fp32 head inside generate stay UNTESTED until the GPU gates run.
The loop runs here against FakePolicy, which stands in for rl_grpo.Policy; the torch tests (skipped without torch) use a tiny
hand-built model, compare the gradient with a numpy hand computation, and run the REAL Policy on it (update with clipping and AdamW,
checkpoints with optimizer and RNG state, generation arguments and seeding, a disk resume and the smoke gate). Only transformers.GenerationConfig
and the tokenizer are stand-ins there.
"""
import contextlib
import importlib.util
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from rlexp import arms, chem, grpo, metrics, reward

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu"

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Nc1ccc(O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
          "CN1CCCC1c1cccnc1", "OC(=O)c1ccccc1O", "CCN(CC)CCOC(=O)c1ccc(N)cc1", "CCN(CC)CC(=O)Nc1c(C)cccc1C",
          "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
          "O=C1NC(=O)C(N1)(c1ccccc1)c1ccccc1"]
PLAIN = "CCCC(CC)CCCCO"            # a generic answer: acyclic (no scaffold bonus) and the reference of no drug
SYSTEM_TURN = "<|im_start|>system\nA long dataset system turn that the direct-answer text replaces.<|im_end|>\n"
PROMPT = SYSTEM_TURN + "<|im_start|>user\ncase notes {n}<|im_end|>\n<|im_start|>assistant\n<think>\n"


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("rl_grpo", GPU / "rl_grpo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_grpo"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("rl_grpo", None)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def make_rows(n_drugs, split="train", offset=0):
    """Synthetic processed rows: drug g has 1 + g % 3 views and a real SMILES reference."""
    return [dict(id=f"{split}-{g}-{v}", group=f"{split}-drug-{g:02d}", split=split, prompt=PROMPT.format(n=f"{g}.{v}"),
                 ref_smiles=SMILES[(offset + g) % len(SMILES)], ref_alt_smiles=[])
            for g in range(n_drugs) for v in range(1 + g % 3)]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def make_args(rl, tmp_path, *extra, adapter="none"):
    return rl.config_from_args(["--init-adapter", adapter, "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
                                "--out", str(tmp_path / "out"), *extra])


def loop_args(rl, tmp_path, *extra):
    base = ["--eval-every", "0", "--steps", "5", "--prompts-per-step", "6", "--group-size", "4", "--min-groups", "2", "--seed", "3"]
    return make_args(rl, tmp_path, *base, *extra)


# ---------------------------------------------------------------- the fake policy
def text_for(kind, row):
    if kind == "exact":
        return "\n</think>\n\n" + row["ref_smiles"]
    if kind == "near":
        return "\n</think>\n\n" + SMILES[(SMILES.index(row["ref_smiles"]) + 1) % len(SMILES)]
    if kind == "const":
        return "\n</think>\n\n" + SMILES[0]
    if kind == "plain":
        return "\n</think>\n\n" + PLAIN
    if kind == "bad":
        return "\n</think>\n\nnot a molecule"
    return "CCO is the answer"                                  # "open": no closed think block


def mixed(seed, row, i):
    r = random.Random(f"{seed}:{row['id']}:{i}").random()
    kind = "exact" if r < 0.15 else "near" if r < 0.5 else "bad" if r < 0.8 else "open"
    return kind, "length" if random.Random(f"L{seed}:{row['id']}:{i}").random() < 0.15 else "stop"


class FakePolicy:
    """Stands in for rl_grpo.Policy. A completion is a fake token id that decodes to a fixed text; what a sample says is a function
    of (torch seed, row, index), so a rollout is reproducible exactly like the real one given the weights."""

    def __init__(self, group_size=4, behave=mixed, eval_behave=lambda seed, row, i: ("bad", "stop")):
        self.registry, self.group_size, self.behave, self.eval_behave = [], group_size, behave, eval_behave
        self.calls, self.eval_calls, self.trained, self.saved, self.adapters = [], [], [], [], []

    def decode(self, ids):
        return "".join(self.registry[i - 1000] for i in ids if i >= 1000)

    def make(self, behave, rows, seed, n):
        out = []
        for row in rows:
            group = []
            for i in range(n):
                kind, finish = behave(seed, row, i)
                text = text_for(kind, row)
                if text not in self.registry:
                    self.registry.append(text)
                extra = random.Random(f"{seed}:{row['id']}:{i}:len").randrange(4)
                group.append(dict(ids=[1000 + self.registry.index(text)] + [2] * extra + ([1] if finish == "stop" else []), finish=finish))
            out.append(group)
        return out

    def rollout(self, rows, seed):
        self.calls.append((seed, tuple(r["id"] for r in rows)))
        return self.make(self.behave, rows, seed, self.group_size)

    def evaluate(self, rows, k, seed):
        self.eval_calls.append((seed, tuple(r["id"] for r in rows)))
        return self.make(self.eval_behave, rows, seed, k)

    def train(self, items, lr):
        self.trained.append(dict(items=items, lr=lr))
        return dict(loss=0.25, grad_norm=1.5, micro_batches=1)

    def save(self, state):
        self.saved.append(json.loads(json.dumps(state, allow_nan=False)))      # what a checkpoint holds

    def save_adapter(self, step):
        self.adapters.append(dict(step=step, path=f"adapters/step-{step}", sha256={}))
        return self.adapters[-1]

    mark_initial = reset_peak = lambda self: None
    delta_norm = lambda self: 0.5
    peak_memory = lambda self: 0


def run_loop(rl, args, policy, rows, val_rows=(), state=None, stopped=lambda: False, started=None):
    state = rl.new_state({}, {}) if state is None else state
    return rl.run_steps(args, state, policy, rl.drug_table(rows), list(val_rows), time.monotonic() if started is None else started, stopped,
                        rl.reference_keys(rows))


def resume(rl, saved):
    return rl.restore_state(json.loads(json.dumps(saved)), rl.new_state({}, {}))      # as loaded from disk: no shared lists


# ---------------------------------------------------------------- data and flags
def test_rows_load_with_the_direct_system_prompt(rl, tmp_path):
    rows = make_rows(5)
    loaded = rl.load_rows(write_rows(tmp_path / "train.jsonl", rows), "train")
    assert len(loaded) == len(rows) and all(r["rl_prompt"] == arms.with_system(r["prompt"], arms.DIRECT_SYSTEM) for r in loaded)
    assert arms.DIRECT_SYSTEM in loaded[0]["rl_prompt"] and "dataset system turn" not in loaded[0]["rl_prompt"]
    assert loaded[0]["rl_prompt"].endswith("<|im_start|>assistant\n<think>\n")
    drugs = rl.drug_table(loaded)
    assert [len(d) for d in drugs] == [1, 2, 3, 1, 2] and [d[0]["group"] for d in drugs] == sorted({r["group"] for r in rows})


@pytest.mark.parametrize("damage", [
    lambda r: r.pop("ref_smiles"), lambda r: r.update(ref_smiles=""), lambda r: r.update(ref_smiles="not a molecule ((("),
    lambda r: r.update(ref_smiles="[*]CC(=O)O"), lambda r: r.pop("group"), lambda r: r.update(split="test_t"),
    lambda r: r.update(prompt="no system turn here<|im_start|>assistant\n<think>\n"),
    lambda r: r.update(prompt=PROMPT.format(n=1) + "extra"), lambda r: r.update(ref_alt_smiles=5), lambda r: r.pop("id")])
def test_a_bad_row_is_refused_without_echoing_its_data(rl, tmp_path, damage):
    rows = make_rows(3)
    damage(rows[1])
    with pytest.raises(rl.PackError) as error:
        rl.load_rows(write_rows(tmp_path / "train.jsonl", rows), "train")
    assert "row 2" in str(error.value) or "Row 2" in str(error.value)
    assert all(piece not in str(error.value) for piece in SMILES + ["case notes"])


def test_duplicate_ids_empty_files_and_bad_json_are_refused(rl, tmp_path):
    rows = make_rows(2)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(rl.PackError):
        rl.load_rows(write_rows(tmp_path / "train.jsonl", rows), "train")
    with pytest.raises(rl.PackError):
        rl.load_rows(write_rows(tmp_path / "empty.jsonl", []), "train")
    (tmp_path / "bad.jsonl").write_text("{not json\n")
    with pytest.raises(rl.PackError, match="Invalid JSON"):
        rl.load_rows(tmp_path / "bad.jsonl", "train")


def test_views_of_one_drug_must_share_the_reference_and_the_error_counts_drugs_only(rl, tmp_path):
    rows = make_rows(5)                                                     # drug 1 has 2 views, drug 2 has 3, drug 4 has 2
    assert len(rl.load_rows(write_rows(tmp_path / "train.jsonl", rows), "train")) == len(rows)
    views = lambda rs, g: [r for r in rs if r["group"].endswith(f"{g:02d}")]

    def damaged(change):
        copy = [dict(r) for r in rows]
        change(copy)
        return write_rows(tmp_path / "damaged.jsonl", copy)

    def refused(change, count):
        with pytest.raises(rl.PackError, match=rf"^{count} drug\(s\) have views with different references") as error:
            rl.load_rows(damaged(change), "train")
        assert all(piece not in str(error.value) for piece in SMILES + ["train-drug", "case notes"])      # a count, never a string or a name

    refused(lambda rs: views(rs, 2)[1].update(ref_smiles=SMILES[7]), 1)                                     # another reference in one view
    refused(lambda rs: [v.update(ref_smiles=SMILES[7]) for v in views(rs, 2)[1:]], 1)                       # two views of one drug: still one drug
    refused(lambda rs: views(rs, 2)[2].update(ref_alt_smiles=[SMILES[7]]), 1)                              # other alternates in one view
    refused(lambda rs: (views(rs, 1)[1].update(ref_smiles=SMILES[7]), views(rs, 4)[1].update(ref_alt_smiles=[SMILES[6]])), 2)    # two drugs
    # the same reference spelled the same way is fine: alternates in another order, a lone string for a one-item list, none for []
    ok = damaged(lambda rs: (views(rs, 1)[0].update(ref_alt_smiles=[SMILES[6], SMILES[7]]), views(rs, 1)[1].update(ref_alt_smiles=[SMILES[7], SMILES[6]]),
                             views(rs, 2)[0].update(ref_alt_smiles=[SMILES[5]]), views(rs, 2)[1].update(ref_alt_smiles=SMILES[5]),
                             views(rs, 2)[2].update(ref_alt_smiles=[SMILES[5]]), views(rs, 4)[1].pop("ref_alt_smiles")))
    assert len(rl.load_rows(ok, "train")) == len(rows)
    # different drugs may of course have different references; the same check covers the Val file
    write_rows(tmp_path / "train.jsonl", rows)
    val = make_rows(3, "val", offset=8)
    val[2]["ref_smiles"] = SMILES[0]                                         # the second view of Val drug 1 (rows: 1, 2 and 3 views)
    write_rows(tmp_path / "val.jsonl", val)
    with pytest.raises(rl.PackError, match=r"^1 drug\(s\) have views with different references"):
        rl.load_data(make_args(rl, tmp_path))


@pytest.mark.parametrize("name", ["test_t.jsonl", "test_r.jsonl", "Test_T.JSONL", "my_test_split.jsonl"])
def test_a_test_split_file_is_refused_before_it_is_opened(rl, tmp_path, name):
    with pytest.raises(rl.PackError, match="test split"):
        rl.load_rows(tmp_path / name, "train")                              # the file does not even exist
    with pytest.raises(rl.PackError, match="test split"):
        rl.config_from_args(["--init-adapter", "none", "--train", str(tmp_path / name), "--val", str(tmp_path / "val.jsonl")])
    with pytest.raises(rl.PackError, match="test split"):
        rl.config_from_args(["--init-adapter", "none", "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / name)])


def test_train_and_val_must_be_separate_files_ids_and_drugs(rl, tmp_path):
    with pytest.raises(rl.PackError, match="separate"):
        rl.config_from_args(["--init-adapter", "none", "--train", str(tmp_path / "x.jsonl"), "--val", str(tmp_path / "x.jsonl")])
    train, val = make_rows(4), make_rows(2, "val")
    rl.check_disjoint(train, val)
    with pytest.raises(rl.PackError, match="IDs"):
        rl.check_disjoint(train, [dict(val[0], id=train[0]["id"])])
    with pytest.raises(rl.PackError, match="drug"):
        rl.check_disjoint(train, [dict(val[0], group=train[0]["group"])])


def test_defaults_follow_the_brief(rl, tmp_path):
    a = make_args(rl, tmp_path)
    assert (a.prompts_per_step, a.group_size, a.max_new_tokens, a.temperature, a.lr) == (32, 8, 256, 1.0, 1e-5)
    assert (a.eval_every, a.eval_k, a.save_every, a.patience, a.gen_batch) == (10, 4, 10, 3, 128)
    assert (a.max_tokens_per_batch, a.min_groups, a.seed, a.gate, a.dry_run) == (8192, 16, 0, None, False)
    assert grpo.skip_below(a.min_groups) == 8
    with pytest.raises(SystemExit):
        rl.config_from_args(["--train", "a", "--val", "b"])                  # --init-adapter is required


def test_the_modes_default_to_the_real_reward_and_the_real_profile_and_reject_other_values(rl, tmp_path):
    a = make_args(rl, tmp_path)
    assert (a.reward_mode, a.profile_mode) == ("full", "real")
    assert make_args(rl, tmp_path, "--reward-mode", "validity", "--profile-mode", "none").profile_mode == "none"
    for flags in (["--reward-mode", "dense"], ["--profile-mode", "shuffled"]):
        with pytest.raises(SystemExit):
            make_args(rl, tmp_path, *flags)
    with pytest.raises(rl.PackError, match="reward_mode"):
        rl.validate_config(dict(vars(a), reward_mode="dense"))
    with pytest.raises(rl.PackError, match="profile_mode"):
        rl.validate_config(dict(vars(a), profile_mode="shuffled"))
    with pytest.raises(rl.PackError, match="profile_mode"):                       # a typo must not fall back to the real profile
        rl.load_rows(tmp_path / "train.jsonl", "train", "shuffled")


def test_profile_free_prompts_replace_only_the_user_turn_in_train_and_val(rl, tmp_path):
    train_rows, val_rows = make_rows(4), make_rows(2, "val", offset=8)
    write_rows(tmp_path / "train.jsonl", train_rows)
    write_rows(tmp_path / "val.jsonl", val_rows)
    free = ("<|im_start|>system\n" + arms.DIRECT_SYSTEM + "<|im_end|>\n<|im_start|>user\nDrug: unknown.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n")
    train, val = rl.load_data(make_args(rl, tmp_path, "--profile-mode", "none"))
    assert len(train) == len(train_rows) and len(val) == len(val_rows)
    for row in train + val:                                                   # every Train and every Val prompt, not just Train
        assert row["rl_prompt"] == free and "case notes" not in row["rl_prompt"]
    assert [r["ref_smiles"] for r in train + val] == [r["ref_smiles"] for r in train_rows + val_rows]      # the reward still sees the reference
    assert [r["prompt"] for r in train + val] == [r["prompt"] for r in train_rows + val_rows]            # the stored rows are untouched
    real_train, real_val = rl.load_data(make_args(rl, tmp_path))
    assert all("case notes" in r["rl_prompt"] and "Drug: unknown." not in r["rl_prompt"] for r in real_train + real_val)
    assert all(r["rl_prompt"].endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n") for r in real_train + real_val)
    assert all(arms.DIRECT_SYSTEM in r["rl_prompt"] for r in real_train + real_val)


@pytest.mark.parametrize("flags", [["--steps", "0"], ["--group-size", "1"], ["--prompts-per-step", "0"], ["--lr", "0"], ["--lr", "nan"],
                                   ["--temperature", "-1"], ["--seed", "-1"], ["--eval-k", "0"], ["--min-groups", "0"],
                                   ["--stop-after-seconds", "inf"], ["--gen-batch", "0"], ["--eval-every", "-1"], ["--patience", "0"],
                                   ["--max-tokens-per-batch", "0"], ["--save-every", "0"], ["--max-new-tokens", "0"]])
def test_bad_flags_are_refused(rl, tmp_path, flags):
    with pytest.raises(rl.PackError):
        make_args(rl, tmp_path, *flags)


def test_the_init_adapter_must_be_a_plain_lora_directory(rl, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(rl.PackError, match="lacks"):
        make_args(rl, tmp_path, adapter=str(adapter))
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    config = dict(peft_type="LORA", r=32, lora_alpha=64, lora_dropout=0.0, bias="none")
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    assert make_args(rl, tmp_path, adapter=str(adapter)).init_adapter == str(adapter)
    for damage in (dict(lora_dropout=0.1), dict(use_dora=True), dict(r=0), dict(peft_type="IA3"), dict(bias="all")):
        (adapter / "adapter_config.json").write_text(json.dumps(dict(config, **damage)))
        with pytest.raises(rl.PackError):
            make_args(rl, tmp_path, adapter=str(adapter))


def run_cli(tmp_path, code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)


def test_dry_run_prints_counts_only_and_imports_no_torch(rl, tmp_path):
    write_rows(tmp_path / "train.jsonl", make_rows(8))
    write_rows(tmp_path / "val.jsonl", make_rows(3, "val", offset=8))
    code = (f"import sys; sys.path.insert(0, {str(GPU)!r}); import rl_grpo; "
            f"rl_grpo.main(['--init-adapter', 'none', '--train', {str(tmp_path / 'train.jsonl')!r}, '--val', {str(tmp_path / 'val.jsonl')!r}, "
            f"'--out', {str(tmp_path / 'out')!r}, '--prompts-per-step', '4', '--dry-run']); print('TORCH', 'torch' in sys.modules)")
    done = run_cli(tmp_path, code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "TORCH False"
    plan = json.loads(done.stdout[:done.stdout.rindex("TORCH")])
    assert plan["mode"] == "dry_run" and plan["train"]["rows"] == len(make_rows(8)) and plan["train"]["drugs"] == 8
    assert plan["val"]["rows"] == len(make_rows(3, "val")) and plan["step"]["samples"] == 4 * 8
    assert plan["step"]["skip_below"] == 8 and plan["eval"]["samples"] == len(make_rows(3, "val")) * 4
    assert not (tmp_path / "out").exists()                                     # a dry run writes nothing
    assert all(piece not in done.stdout for piece in SMILES + ["case notes", "dataset system turn"])


def test_the_plan_prints_both_modes_and_whether_the_collapse_rule_applies(rl, tmp_path, capsys):
    write_rows(tmp_path / "train.jsonl", make_rows(8))
    write_rows(tmp_path / "val.jsonl", make_rows(3, "val", offset=8))

    def plan(*flags):
        assert rl.main(["--init-adapter", "none", "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
                        "--out", str(tmp_path / "out"), "--prompts-per-step", "4", "--dry-run", *flags]) == 0
        return json.loads(capsys.readouterr().out)

    real, free = plan(), plan("--profile-mode", "none", "--reward-mode", "validity")
    assert (real["reward_mode"], real["profile_mode"], real["stop_rules"]["collapse_stop"]) == ("full", "real", True)
    assert (free["reward_mode"], free["profile_mode"], free["stop_rules"]["collapse_stop"]) == ("validity", "none", False)
    assert len(set(free["train"]["prompt_tokens"].values())) == 1                                # one fixed prompt: p50 = p90 = max
    assert rl.collapse_stop(rl.config_from_args(["--init-adapter", "none"])) and not rl.collapse_stop(
        rl.config_from_args(["--init-adapter", "none", "--profile-mode", "none"]))
    assert [plan("--min-groups", m)["step"]["skip_below"] for m in ("3", "4", "5", "16")] == [2, 2, 3, 8]      # half, rounded up


def test_the_cli_reports_a_refusal_and_exits_nonzero(rl, tmp_path):
    done = subprocess.run([sys.executable, str(GPU / "rl_grpo.py"), "--init-adapter", "none", "--train", str(tmp_path / "test_t.jsonl"),
                           "--val", str(tmp_path / "val.jsonl"), "--dry-run"], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert done.returncode == 1 and "RL failed (PackError)" in done.stderr and "test split" in done.stderr


@pytest.mark.skipif(not (ROOT / "data/processed/train.jsonl").exists() or not (ROOT / "data/processed/val.jsonl").exists(),
                    reason="the processed rows are not tracked")
def test_the_real_processed_files_load_and_are_disjoint(rl):
    train, val = (rl.load_rows(ROOT / f"data/processed/{name}.jsonl", name) for name in ("train", "val"))
    rl.check_disjoint(train, val)
    assert len(rl.drug_table(train)) > 100 and len(rl.drug_table(val)) > 5
    assert all(arms.DIRECT_SYSTEM in r["rl_prompt"] for r in train + val)


def runner_skeleton(tmp_path):
    """A repo root for gpu/run_rl.sh: the real gpu/ and src/ plus small synthetic data."""
    (tmp_path / "gpu").symlink_to(GPU)
    (tmp_path / "src").symlink_to(ROOT / "src")
    (tmp_path / "data").mkdir()
    write_rows(tmp_path / "data/train.jsonl", make_rows(40))
    write_rows(tmp_path / "data/val.jsonl", make_rows(3, "val", offset=8))
    return {**os.environ, "DRY_RUN": "1", "INIT_ADAPTER": "none", "PLAN_PYTHON": sys.executable,
            "TRAIN_FILE": "data/train.jsonl", "VAL_FILE": "data/val.jsonl"}


def run_runner(tmp_path, env, **over):
    return subprocess.run(["sh", "gpu/run_rl.sh"], cwd=tmp_path, env={**env, **over}, capture_output=True, text=True, timeout=120)


def test_the_runner_passes_the_trainer_its_own_defaults_and_every_override(rl, tmp_path):
    env = runner_skeleton(tmp_path)

    def command(**over):
        done = run_runner(tmp_path, env, **over)
        assert done.returncode == 0, done.stdout + done.stderr
        line = next(l for l in done.stdout.splitlines() if " COMMAND " in l)
        words = shlex.split(line.split(" COMMAND ", 1)[1])
        assert words[0] == "gpu/rl_grpo.py"
        return rl.config_from_args(words[1:]), done

    defaults = rl.config_from_args(["--init-adapter", "none"])
    got, done = command()
    assert {key for key in vars(defaults) if getattr(got, key) != getattr(defaults, key)} == {"train", "val"}   # only the data paths differ
    assert json.loads(done.stdout[done.stdout.index("{"):done.stdout.rindex("}") + 1])["train"]["rows"] == len(make_rows(40))
    tuned, _ = command(STEPS="7", LR="3e-5", GROUP_SIZE="4", MIN_GROUPS="8", GEN_BATCH="16", SEED="5", ALLOW_SLOW_KERNELS="1", GATE="smoke",
                       TEMPERATURE="0.9", EVAL_K="2", PATIENCE="2", STOP_AFTER_SECONDS="99", MAX_NEW_TOKENS="64", PROMPTS_PER_STEP="8",
                       REWARD_MODE="validity", PROFILE_MODE="none")
    assert (tuned.steps, tuned.lr, tuned.group_size, tuned.min_groups, tuned.gen_batch, tuned.seed) == (7, 3e-5, 4, 8, 16, 5)
    assert (tuned.allow_slow_kernels, tuned.gate, tuned.temperature, tuned.eval_k, tuned.patience) == (True, "smoke", 0.9, 2, 2)
    assert (tuned.stop_after_seconds, tuned.max_new_tokens, tuned.prompts_per_step) == (99, 64, 8)
    assert (tuned.reward_mode, tuned.profile_mode) == ("validity", "none")
    assert command(REWARD_MODE="contrastive")[0].reward_mode == "contrastive"


def test_the_runner_stops_early_on_a_bad_gate_missing_data_or_adapter(rl, tmp_path):
    env = runner_skeleton(tmp_path)
    for over, message in (({"GATE": "bogus"}, "GATE must be"), ({"DRY_RUN": "0", "TRAIN_FILE": "missing.jsonl"}, "files are missing"),
                          ({"DRY_RUN": "0", "INIT_ADAPTER": str(tmp_path / "nope")}, "INIT_ADAPTER must be")):
        done = run_runner(tmp_path, env, **over)
        assert done.returncode == 1 and message in done.stderr, (over, done.stderr)
    assert run_runner(tmp_path, env, TRAIN_FILE="data/test_t.jsonl").returncode != 0      # a test split reaches the trainer's refusal


def test_the_docs_name_the_pilot_runs_and_defaults(rl):
    text = (GPU / "RL.md").read_text(encoding="utf-8")
    for needle in ("REWARD_MODE=validity", "REWARD_MODE=contrastive", "OUT=.agent-runs/rl_contrastive", "PROFILE_MODE=none", "INIT_ADAPTER=none", "PROMPTS_PER_STEP=16", "MIN_GROUPS=8", "STEPS=60",
                   "LR=2e-5", "EVAL_EVERY=10", "OUT=.agent-runs/rl_full", "OUT=.agent-runs/rl_validity", "OUT=.agent-runs/rl_noprofile",
                   "OUT=.agent-runs/rl_zero", "Drug: unknown.", "collapse_all", "lift_batch", "train_copy_share",
                   # round 3: the floor, the first-valid endpoint, the objective hash, the gates' thresholds and the known scorer behaviour
                   "max(0, reward_v2", "the floor only adds", "`fv_tanimoto`", "`val_reward`", "`objective_sha256`", "objective changed: use a new output directory",
                   "`trainer_sha256`", "rounded up", "Known scorer behaviour", "leading or trailing `*`", "mean |diff| <= 0.03", "15% of tokens",
                   "within 2% of 1", "Measured, then chosen", "0.015-0.018", "8-11%", "0.994-1.000", "`EVAL_K=8`"):
        assert needle in " ".join(text.split()), needle                              # whatever the line breaks
    assert len(text.splitlines()) <= 100
    assert rl.PROFILE_FREE_USER == "Drug: unknown."
    runner = (GPU / "run_rl.sh").read_text(encoding="utf-8")                        # the defaults stay as they were: the pilot overrides them
    for default in ("PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-32}", "MIN_GROUPS=${MIN_GROUPS:-16}", "STEPS=${STEPS:-100}", "LR=${LR:-1e-5}",
                    "EVAL_EVERY=${EVAL_EVERY:-10}", "REWARD_MODE=${REWARD_MODE:-full}", "PROFILE_MODE=${PROFILE_MODE:-real}"):
        assert default in runner, default


# ---------------------------------------------------------------- identity, state and logs
def test_resume_identity_is_options_data_adapter_dtype_and_core_versions_only(rl, tmp_path, capsys):
    write_rows(tmp_path / "train.jsonl", make_rows(3))
    write_rows(tmp_path / "val.jsonl", make_rows(2, "val"))
    versions = dict(python="3.12", torch="2.10", transformers="5.17", peft="0.21", numpy="2.0")

    def identity(*flags, revision="r1", dtype="bf16", versions=versions, adapter="none"):
        return rl.make_identity(make_args(rl, tmp_path, *flags, adapter=adapter), revision, dtype, versions)

    base = identity()
    assert base == identity()
    for flags in (["--lr", "2e-5"], ["--seed", "1"], ["--group-size", "4"], ["--temperature", "0.9"], ["--prompts-per-step", "8"],
                  ["--max-new-tokens", "128"], ["--eval-k", "2"], ["--eval-every", "5"], ["--patience", "2"], ["--min-groups", "8"],
                  ["--model", "other"], ["--reward-mode", "validity"], ["--reward-mode", "contrastive"], ["--profile-mode", "none"]):
        assert identity(*flags) != base, flags
    # performance and stop settings do not change what the run is
    for flags in (["--steps", "7"], ["--gen-batch", "16"], ["--save-every", "3"], ["--max-tokens-per-batch", "4096"], ["--stop-after-seconds", "5"],
                  ["--out", str(tmp_path / "elsewhere")], ["--allow-slow-kernels"]):
        assert identity(*flags) == base, flags
    assert identity(revision="r2") != base and identity(dtype="fp16") != base
    assert identity(versions=dict(versions, torch="2.11")) != base and identity(versions=dict(versions, peft="0.22")) != base
    assert identity(versions=dict(versions, numpy="2.1", python="3.13")) == base
    (tmp_path / "train.jsonl").write_text("changed\n")
    assert identity() != base
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-1")
    (adapter / "adapter_config.json").write_text(json.dumps(dict(peft_type="LORA", r=32, lora_alpha=64)))
    with_adapter = identity(adapter=str(adapter))
    assert with_adapter != base and with_adapter["init_adapter_sha256"] == rl.adapter_digest(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-2")
    assert identity(adapter=str(adapter)) != with_adapter
    # check_resume: a core mismatch refuses, any other version change only warns
    saved = dict(identity=base, versions=versions)
    rl.check_resume(saved, base, versions)
    with pytest.raises(rl.PackError, match="identity changed"):
        rl.check_resume(saved, identity("--lr", "3e-5"), versions)
    rl.check_resume(saved, base, dict(versions, numpy="2.1"))
    assert "numpy" in capsys.readouterr().out


def test_the_objective_hash_covers_the_scorer_the_update_code_the_system_turn_and_the_prompt_source(rl, tmp_path, monkeypatch):
    base = rl.objective_digest()
    assert base == rl.objective_digest() and len(base) == 64 and set(base) <= set("0123456789abcdef")

    def copy_of(module, suffix=b""):                                          # same file name in another directory: only the content can differ
        directory = tmp_path / f"{module.__name__}-{len(suffix)}"
        directory.mkdir()
        copy = directory / Path(module.__file__).name
        copy.write_bytes(Path(module.__file__).read_bytes() + suffix)
        return str(copy)

    for module in (reward, chem, grpo):                                       # the content of exactly these three files counts, not their place
        with monkeypatch.context() as patch:
            patch.setattr(module, "__file__", copy_of(module))
            assert rl.objective_digest() == base, module.__name__
        with monkeypatch.context() as patch:
            patch.setattr(module, "__file__", copy_of(module, b"\n# an edit that changes no behaviour\n"))
            assert rl.objective_digest() != base, module.__name__
    for module in (metrics, rl.sft, arms):                                    # and no other file does
        with monkeypatch.context() as patch:
            patch.setattr(module, "__file__", copy_of(module, b"\n# an edit\n"))
            assert rl.objective_digest() == base, module.__name__
    with monkeypatch.context() as patch:
        patch.setattr(arms, "DIRECT_SYSTEM", arms.DIRECT_SYSTEM + " ")
        assert rl.objective_digest() != base
    with monkeypatch.context() as patch:
        patch.setattr(rl, "PROFILE_FREE_USER", "Drug: n/a.")
        assert rl.objective_digest() != base

    def with_system_elsewhere(prompt, system):
        return prompt

    def split_prompt_elsewhere(prompt):
        return prompt, "", ""

    def rl_prompt_elsewhere(prompt, profile_mode):
        return prompt

    for owner, name, replacement in ((arms, "with_system", with_system_elsewhere), (arms, "split_prompt", split_prompt_elsewhere),
                                     (rl, "rl_prompt", rl_prompt_elsewhere)):
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, replacement)                           # the prompt transformation: its source counts
            assert rl.objective_digest() != base, name
    assert rl.objective_digest() == base                                      # everything above was undone
    assert rl.rl_prompt(PROMPT.format(n=1), "real") == arms.with_system(PROMPT.format(n=1), arms.DIRECT_SYSTEM)


def test_a_resume_under_another_objective_is_refused_and_a_changed_trainer_only_warns(rl, tmp_path, capsys, monkeypatch):
    write_rows(tmp_path / "train.jsonl", make_rows(3))
    write_rows(tmp_path / "val.jsonl", make_rows(2, "val"))
    versions = dict(python="3.12", torch="2.10", transformers="5.17", peft="0.21")
    trainer = {"rl_grpo.py": "a" * 64, "sft_lora.py": "b" * 64}
    base = rl.make_identity(make_args(rl, tmp_path), "r1", "bf16", versions)
    assert base["objective_sha256"] == rl.objective_digest()
    with monkeypatch.context() as patch:
        patch.setattr(rl, "objective_digest", lambda: "0" * 64)               # the objective is part of the identity itself
        assert rl.make_identity(make_args(rl, tmp_path), "r1", "bf16", versions) != base
    saved = dict(identity=base, versions=versions, trainer_sha256=trainer)
    rl.check_resume(saved, base, versions, trainer)
    assert capsys.readouterr().out == ""                                      # same objective, same trainer: silent
    # a different objective refuses, with its own message and whatever else matches
    with pytest.raises(rl.PackError, match="objective changed: use a new output directory"):
        rl.check_resume(saved, dict(base, objective_sha256="0" * 64), versions, trainer)
    # a checkpoint from before the hash was recorded cannot be shown to match: refused the same way
    with pytest.raises(rl.PackError, match="objective changed"):
        rl.check_resume(dict(saved, identity={k: v for k, v in base.items() if k != "objective_sha256"}), base, versions, trainer)
    with pytest.raises(rl.PackError, match="identity changed"):               # any other difference keeps the generic refusal
        rl.check_resume(saved, dict(base, dtype="fp16"), versions, trainer)
    # the trainer files are recorded, not pinned: a change only warns, naming the file
    rl.check_resume(saved, base, versions, dict(trainer, **{"rl_grpo.py": "c" * 64}))
    out = capsys.readouterr().out
    assert "WARNING" in out and "rl_grpo.py" in out and "sft_lora.py" not in out
    rl.check_resume(dict(saved, trainer_sha256=None), base, versions, trainer)        # a checkpoint without the record warns about both
    assert "rl_grpo.py" in capsys.readouterr().out
    rl.check_resume(saved, base, versions)                                    # no trainer given: nothing to compare
    assert capsys.readouterr().out == ""
    hashes = rl.trainer_hashes()
    assert hashes == {"rl_grpo.py": rl.file_sha256(GPU / "rl_grpo.py"), "sft_lora.py": rl.file_sha256(GPU / "sft_lora.py")}


def test_the_base_revision_comes_from_the_checkpoint_or_the_init_adapter_pin(rl, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert rl.adapter_pin(adapter) is None and rl.adapter_pin(None) is None and rl.base_revision(None, adapter) is None
    (adapter / "revision.json").write_text(json.dumps(dict(revision="abc123")))
    assert rl.adapter_pin(adapter) == "abc123" and rl.base_revision(None, adapter) == "abc123"
    assert rl.base_revision((tmp_path, dict(identity=dict(revision="resumed")), {}), adapter) == "resumed"     # a resume keeps its own
    assert rl.base_revision(None, None) is None and rl.base_revision(None, tmp_path / "plain") is None
    assert rl.KEPT_FILES == ("adapter_model.safetensors", "adapter_config.json", "revision.json")


def test_restore_state_carries_every_loop_field_and_keeps_rule_stops_final(rl):
    fresh = rl.new_state({"a": 1}, {"v": 1})
    saved = dict(fresh, step=3, global_step=3, cursor=17, collapse_streak=2, skipped_updates=1, training_seconds=5.5,
                 first_update_delta_norm=0.5, peak_memory_bytes=9, metrics=[dict(step=s, reward_mean=0.1) for s in (1, 2, 3)],
                 evals=[dict(step=0, val=0.1, train=None)], adapters=[dict(step=2)], status="stopped_time_limit")
    state = rl.restore_state(json.loads(json.dumps(saved)), rl.new_state({"b": 2}, {"v": 2}))
    assert {key: state[key] for key in rl.LOOP_KEYS} == {key: saved[key] for key in rl.LOOP_KEYS}
    assert state["status"] == "running" and state["identity"] == {"b": 2}          # the current run's identity, not the saved one
    for final in grpo.RULE_STOPS:
        assert rl.restore_state(dict(saved, status=final), rl.new_state({}, {}))["status"] == final
    assert rl.restore_state(dict(saved, status="complete"), rl.new_state({}, {}))["status"] == "running"
    with pytest.raises(rl.PackError, match="lacks cursor"):
        rl.restore_state({k: v for k, v in saved.items() if k != "cursor"}, rl.new_state({}, {}))
    for damage in (dict(cursor=-1), dict(cursor=1.5), dict(step=4), dict(step=-1, metrics=[])):
        with pytest.raises(rl.PackError, match="resume cursor"):
            rl.restore_state(dict(saved, **damage), rl.new_state({}, {}))


def test_logs_are_rebuilt_from_the_committed_state(rl, tmp_path):
    state = rl.new_state({}, {})
    state["metrics"] = [dict(step=1, reward_mean=0.1), dict(step=2, reward_mean=0.2)]
    state["evals"] = [dict(step=0, val=0.05, train=None)]
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n')          # a line of a step the checkpoint lacks
    rl.write_logs(tmp_path, state)
    assert [json.loads(l)["step"] for l in (tmp_path / "metrics.jsonl").read_text().splitlines()] == [1, 2]
    assert json.loads((tmp_path / "evals.jsonl").read_text())["val"] == 0.05
    assert json.loads((tmp_path / "summary.json").read_text())["metrics"] == state["metrics"]
    bad = rl.new_state({}, {})
    bad["metrics"] = [dict(step=1, reward_mean=float("nan"))]
    with pytest.raises(ValueError):
        rl.write_logs(tmp_path, bad)                                                        # NaN never reaches a log


# ---------------------------------------------------------------- scoring with a fake decoder
def scoring_fixture():
    row = make_rows(1)[0]
    texts = {10: text_for("exact", row), 11: text_for("near", row), 12: text_for("bad", row), 13: text_for("open", row), 14: "\n</think>\n\n"}
    return row, texts, lambda ids: "".join(texts[i] for i in ids if i in texts)


def test_score_completions_scores_the_decoded_tokens_with_the_reward(rl):
    row, texts, decode = scoring_fixture()
    done = rl.score_completions(decode, [dict(ids=[10, 1], finish="stop"), dict(ids=[11], finish="stop"), dict(ids=[12, 2, 2], finish="stop"),
                                         dict(ids=[13], finish="stop"), dict(ids=[14], finish="stop")], row)
    assert done[0]["reward"] == done[0]["reward_full"] == 1.0 and done[0]["conn_match"] and done[0]["valid"] and done[0]["closed"]
    assert done[0]["key"] == chem.conn(chem.standardize(row["ref_smiles"])) and done[0]["tanimoto"] == pytest.approx(1.0)
    assert done[0]["answer"] == row["ref_smiles"] and not done[0]["truncated"]
    near = reward.score(texts[11], row)
    assert done[1]["reward"] == pytest.approx(near["reward"]) and 0.05 < done[1]["reward"] < 1.0 and done[1]["tanimoto"] == pytest.approx(near["tanimoto"])
    assert done[1]["key"] is not None and done[1]["key"] != done[0]["key"] and done[1]["answer"] == near["answer"]
    assert done[2]["reward"] == 0.0 and not done[2]["valid"] and done[2]["closed"] and done[2]["key"] is None and done[2]["answer"] is None
    assert done[3]["reward"] == 0.0 and not done[3]["closed"] and done[3]["n_tokens"] == 1
    assert done[4]["reward"] == 0.0 and done[4]["closed"] and not done[4]["valid"]
    assert [s["n_tokens"] for s in done] == [2, 1, 3, 1, 1] and done[0]["ids"] == [10, 1]


def test_a_length_terminated_sample_is_a_failure_whatever_its_parsed_answer_says(rl):
    row, texts, decode = scoring_fixture()
    for text_id in (10, 11):                                                   # an exact answer and a near one that never stopped
        assert reward.score(texts[text_id], row)["reward"] > 0.05              # the reward alone would pay for them
        capped = rl.score_completions(decode, [dict(ids=[text_id], finish="length")], row)[0]
        assert capped["truncated"] and capped["reward"] == capped["reward_full"] == 0.0 and not capped["valid"]
        assert capped["tanimoto"] == 0.0 and not capped["conn_match"] and not capped["scaffold_match"]
        assert capped["key"] is None and capped["answer"] is None and capped["closed"] and capped["n_tokens"] == 1
        assert rl.score_completions(decode, [dict(ids=[text_id], finish="length")], row, "validity")[0]["reward"] == 0.0
    assert rl.finished_score(texts[10], row, True)["reward"] == 1.0 and rl.finished_score(texts[10], row, False)["reward"] == 0.0
    assert rl.finished_score(texts[10], row, False)["stereo_match"] is False


def test_validity_mode_trains_on_a_binary_reward_and_keeps_the_full_one(rl):
    row, texts, decode = scoring_fixture()
    completions = [dict(ids=[10], finish="stop"), dict(ids=[11], finish="stop"), dict(ids=[12], finish="stop"), dict(ids=[13], finish="stop"),
                   dict(ids=[11], finish="length"), dict(ids=[10], finish="length")]
    full = rl.score_completions(decode, completions, row)
    binary = rl.score_completions(decode, completions, row, "validity")
    assert [s["reward"] for s in binary] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]            # valid, closed and finished: exact and near alike
    assert [s["reward_full"] for s in binary] == [s["reward"] for s in full] == [s["reward_full"] for s in full]
    assert binary[1]["reward"] == 1.0 > binary[1]["reward_full"] > 0.05 and binary[0]["reward_full"] == 1.0
    assert [s["valid"] for s in binary] == [s["valid"] for s in full] and [s["tanimoto"] for s in binary] == [s["tanimoto"] for s in full]
    with pytest.raises(rl.PackError, match="reward_mode"):
        rl.score_completions(decode, completions, row, "dense")


def test_trim_generated_cuts_at_the_first_end_token(rl):
    assert rl.trim_generated([5, 6, 9, 0, 0], 9, 6) == ([5, 6, 9], "stop")
    assert rl.trim_generated([5, 6, 7, 8], 9, 4) == ([5, 6, 7, 8], "length")
    assert rl.trim_generated([9], 9, 4) == ([9], "stop") and rl.trim_generated([5, 9, 9], 9, 4) == ([5, 9], "stop")


def test_micro_batches_cover_every_sample_once_inside_the_budget(rl):
    lengths = [random.Random(i).randrange(20, 400) for i in range(61)]
    batches = rl.micro_batches(lengths, 1000)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in b) * len(b) <= 1000 for b in batches)
    assert lengths[batches[0][0]] == max(lengths)                              # longest first
    assert rl.micro_batches([5], 5) == [[0]] and rl.micro_batches([], 5) == []
    with pytest.raises(rl.PackError, match="token budget"):
        rl.micro_batches([10, 1001], 1000)


# ---------------------------------------------------------------- the loop
def expected_update(scored):
    """From (row, completions, training rewards) per group: the update's weights by hand. Rewards are centred per group, zero-variance
    groups dropped, one batch std divides the rest, the weight is A / (T_g * G). Returns ([(row id, ids, weight)], G)."""
    centred = [np.array(r) - np.mean(r) for _, _, r in scored if max(r) - min(r) > 1e-9]
    std, kept = np.concatenate(centred).std(), len(centred)
    expected = []
    for row, group, rewards in scored:
        if max(rewards) - min(rewards) <= 1e-9:
            continue
        total = sum(len(c["ids"]) for c in group)
        expected += [(row["id"], tuple(c["ids"]), (r - np.mean(rewards)) / (std + 1e-6) / (total * kept)) for c, r in zip(group, rewards)]
    return expected, kept


def oracle(policy, rows, mode="full", group_size=4):
    """What the update of the policy's first rollout must be, computed from scratch: each sample's reward by the reward module and the
    length rule, centred per group, one batch std, weight A / (T_g * G). Returns (expected (row id, ids, weight) list, scored groups,
    G) where a scored group is (row, completions, full rewards, training rewards)."""
    by_id = {r["id"]: r for r in rows}
    seed, ids = policy.calls[0]
    scored = []
    for row, group in zip((by_id[i] for i in ids), policy.make(policy.behave, [by_id[i] for i in ids], seed, group_size)):
        full, binary = [], []
        for c in group:
            s = reward.score(policy.decode(c["ids"]), row)
            finished = c["finish"] != "length"
            full.append(s["reward"] if finished else 0.0)
            binary.append(1.0 if s["valid"] and s["closed"] and finished else 0.0)
        scored.append((row, group, full, full if mode == "full" else binary))
    expected, kept = expected_update([(row, group, rewards) for row, group, _, rewards in scored])
    return expected, scored, kept


def check_update(policy, expected, kept, group_size=4):
    got = [(row["id"], tuple(ids), w) for row, ids, w in policy.trained[0]["items"]]
    assert len(got) == len(expected) == group_size * kept                       # all samples of retained groups, none of the dropped
    assert sorted((a, b) for a, b, _ in got) == sorted((a, b) for a, b, _ in expected)
    for item in got:
        want = next(w for a, b, w in expected if (a, b) == item[:2])
        assert item[2] == pytest.approx(want, rel=1e-9)
    return got


def test_the_update_uses_every_sample_of_the_retained_groups_with_prompt_level_weights(rl, tmp_path):
    rows = make_rows(12)
    constant = {"train-drug-00", "train-drug-04", "train-drug-07"}           # these groups get one answer in every sample
    policy = FakePolicy(behave=lambda seed, row, i: ("bad", "stop") if row["group"] in constant else mixed(seed, row, i))
    args = loop_args(rl, tmp_path, "--steps", "1", "--prompts-per-step", "12")
    state = run_loop(rl, args, policy, rows)
    assert state["status"] == "complete" and len(policy.trained) == 1
    expected, scored, kept = oracle(policy, rows)
    assert kept <= 12 - len(constant)
    got = check_update(policy, expected, kept)
    assert not {r["id"] for r in rows if r["group"] in constant} & {g[0] for g in got}
    # a length-terminated sample is a failure with reward 0, trained on like any other: the fixture has some that would have paid
    capped = [(row, c) for row, group, _, _ in scored for c in group if c["finish"] == "length"]
    assert capped and any(reward.score(policy.decode(c["ids"]), row)["reward"] > 0 for row, c in capped)
    assert {tuple(c["ids"]) for _, c in capped} & {g[1] for g in got}
    m = state["metrics"][0]
    assert m["groups_kept"] == kept and m["prompts"] == 12 and m["zero_variance_share"] == pytest.approx((12 - kept) / 12)
    assert m["lr"] == pytest.approx(grpo.lr_at(1, args.lr)) and policy.trained[0]["lr"] == m["lr"]
    assert m["truncation"] == pytest.approx(np.mean([c["finish"] == "length" for _, g, _, _ in scored for c in g]))
    assert m["reward_mean"] == pytest.approx(np.mean([r for _, _, f, _ in scored for r in f])) == pytest.approx(m["reward_full_mean"])
    assert 0 < state["first_update_delta_norm"] and m["loss"] == 0.25 and not m["skipped_update"]
    # the diagnostics cover all 12 attempted groups, the 3 dropped ones included: their samples are in the denominators
    full = [r for _, _, f, _ in scored for r in f]
    assert len(full) == 48 and sum(r > 0 for r in full) > 0
    assert m["reward_validity_term"] == pytest.approx(reward.W_VALID * sum(r > 0 for r in full) / 48)      # a reward above 0 is a valid, finished answer
    assert m["reward_identity"] == pytest.approx(sum(r == 1.0 for r in full) / 48)


def test_the_validity_control_trains_on_binary_rewards_and_logs_both(rl, tmp_path):
    rows = make_rows(12)
    policy = FakePolicy()
    args = loop_args(rl, tmp_path, "--steps", "1", "--prompts-per-step", "12", "--reward-mode", "validity")
    state = run_loop(rl, args, policy, rows)
    expected, scored, kept = oracle(policy, rows, "validity")
    check_update(policy, expected, kept)
    m = state["metrics"][0]
    binary = [r for _, _, _, rewards in scored for r in rewards]
    assert set(binary) == {0.0, 1.0} and m["reward_mean"] == pytest.approx(np.mean(binary)) == pytest.approx(m["valid"])
    assert m["reward_full_mean"] == pytest.approx(np.mean([r for _, _, f, _ in scored for r in f])) != pytest.approx(m["reward_mean"])
    assert m["reward_std"] == pytest.approx(np.std(binary)) and m["reward_full_std"] == pytest.approx(np.std([r for _, _, f, _ in scored for r in f]))
    # the same rollouts under the full reward give other weights
    full_policy = FakePolicy()
    run_loop(rl, loop_args(rl, tmp_path, "--steps", "1", "--prompts-per-step", "12"), full_policy, rows)
    assert [w for *_, w in full_policy.trained[0]["items"]] != [w for *_, w in policy.trained[0]["items"]]


def test_a_batch_with_few_surviving_groups_refills_once_and_at_most_doubles(rl, tmp_path):
    rows = make_rows(12)
    flat = {g for g in range(12) if g % 2}                                     # odd drugs carry no signal
    behave = lambda seed, row, i: ("bad", "stop") if int(row["group"][-2:]) in flat else mixed(seed, row, i)
    policy = FakePolicy(behave=behave)
    args = loop_args(rl, tmp_path, "--steps", "1", "--prompts-per-step", "6", "--min-groups", "6")
    state = run_loop(rl, args, policy, rows)
    m = state["metrics"][0]
    assert len(policy.calls) == 2 and m["refilled_prompts"] == len(policy.calls[1][1]) and 0 < m["refilled_prompts"] <= 6
    assert m["prompts"] == 6 + m["refilled_prompts"]
    first = {rows_id.rsplit("-", 1)[0] for rows_id in policy.calls[0][1]}
    second = {rows_id.rsplit("-", 1)[0] for rows_id in policy.calls[1][1]}
    assert not first & second                                                  # the refill draws other drugs
    assert policy.calls[1][0] == grpo.rollout_seed(3, 1, 1) and policy.calls[0][0] == grpo.rollout_seed(3, 1)
    assert m["cursor"] == 6 + m["refilled_prompts"]                            # both draws advanced the schedule
    trained_drugs = {row["group"] for row, _, _ in policy.trained[0]["items"]}
    assert all(int(g[-2:]) not in flat for g in trained_drugs) and len(trained_drugs) == m["groups_kept"]
    # no signal at all: the refill is the whole first batch again, never more
    policy = FakePolicy(behave=lambda seed, row, i: ("bad", "stop"))
    state = run_loop(rl, args, policy, rows)
    assert state["metrics"][0]["prompts"] == 12 and state["metrics"][0]["refilled_prompts"] == 6 and len(policy.calls) == 2


def test_too_few_groups_skip_the_update_but_the_step_still_counts(rl, tmp_path):
    rows = make_rows(12)

    def with_signal(n):
        # only the first n drugs have reward variance; prompts_per_step covers all 12, so a refill has nothing to draw
        return FakePolicy(behave=lambda seed, row, i: mixed(seed, row, i) if int(row["group"][-2:]) < n else ("bad", "stop"))

    args = loop_args(rl, tmp_path, "--steps", "2", "--prompts-per-step", "12", "--min-groups", "6")      # skip below 3
    for signal_groups, trained in ((2, False), (3, True)):
        policy = with_signal(signal_groups)
        state = run_loop(rl, args, policy, rows)
        kept = [m["groups_kept"] for m in state["metrics"]]
        assert state["step"] == 2 and state["status"] == "complete"
        assert all(k <= signal_groups for k in kept)
        for m, k in zip(state["metrics"], kept):
            assert m["skipped_update"] == (k < 3) and (m["loss"] is None) == (k < 3)
    # a group with no variance anywhere: nothing trains, nothing is NaN, the schedule still moves
    policy = FakePolicy(behave=lambda seed, row, i: ("bad", "stop"))
    state = run_loop(rl, args, policy, rows)
    assert policy.trained == [] and state["skipped_updates"] == 2 and state["first_update_delta_norm"] is None
    assert [m["advantage_std"] for m in state["metrics"]] == [None, None] and state["cursor"] == 24
    json.dumps(state, allow_nan=False)


def test_an_odd_min_groups_skips_below_half_rounded_up(rl, tmp_path):
    rows = make_rows(12)

    def with_signal(n):
        # exactly n drugs have reward variance (alternating exact and bad answers); prompts_per_step covers all 12, so no refill has anything to draw
        return FakePolicy(behave=lambda seed, row, i: (("exact" if i % 2 else "bad"), "stop") if int(row["group"][-2:]) < n else ("bad", "stop"))

    args = loop_args(rl, tmp_path, "--steps", "2", "--prompts-per-step", "12", "--min-groups", "3")          # half of 3, rounded up, is 2
    for retained, trained in ((1, False), (2, True), (3, True)):
        policy = with_signal(retained)
        state = run_loop(rl, args, policy, rows)
        assert [m["groups_kept"] for m in state["metrics"]] == [retained] * 2
        assert [m["skipped_update"] for m in state["metrics"]] == [not trained] * 2 and len(policy.trained) == 2 * trained
        assert state["skipped_updates"] == (0 if trained else 2)
    args = loop_args(rl, tmp_path, "--steps", "1", "--prompts-per-step", "12", "--min-groups", "5")          # 3
    for retained, trained in ((2, False), (3, True)):
        policy = with_signal(retained)
        run_loop(rl, args, policy, rows)
        assert bool(policy.trained) is trained, retained


def test_the_learning_rate_warms_up_over_five_steps(rl, tmp_path):
    policy = FakePolicy()
    args = loop_args(rl, tmp_path, "--steps", "7", "--lr", "1e-4")
    run_loop(rl, args, policy, make_rows(12))
    assert [t["lr"] for t in policy.trained] == pytest.approx([2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 1e-4, 1e-4])


def test_a_resumed_run_draws_the_same_prompts_and_seeds_as_an_uninterrupted_one(rl, tmp_path):
    rows = make_rows(12)
    full = FakePolicy()
    uninterrupted = run_loop(rl, loop_args(rl, tmp_path, "--steps", "6", "--prompts-per-step", "5"), full, rows)
    first = FakePolicy()
    run_loop(rl, loop_args(rl, tmp_path, "--steps", "3", "--prompts-per-step", "5"), first, rows)
    checkpoint = first.saved[-1]                                               # the JSON a checkpoint holds
    assert checkpoint["step"] == 3 and checkpoint["cursor"] > 0 and len(checkpoint["metrics"]) == 3
    second = FakePolicy()
    done = run_loop(rl, loop_args(rl, tmp_path, "--steps", "6", "--prompts-per-step", "5"), second, rows, state=resume(rl, checkpoint))
    assert done["step"] == 6 and done["status"] == "complete"
    assert first.calls + second.calls == full.calls                           # same drugs, views and torch seeds, call by call
    assert second.calls[0][0] == grpo.rollout_seed(3, 4)                      # the step number carries over
    assert [m["schedule_hash"] for m in done["metrics"]] == [m["schedule_hash"] for m in uninterrupted["metrics"]]
    assert done["metrics"][:3] == checkpoint["metrics"] and done["cursor"] == uninterrupted["cursor"]
    # the same updates, too: items and learning rates match step by step
    assert [t["lr"] for t in first.trained + second.trained] == [t["lr"] for t in full.trained]
    flat = lambda p: [[(r["id"], p.decode(i), len(i), round(w, 12)) for r, i, w in t["items"]] for t in p.trained]    # texts: ids are per fake policy
    assert flat(first) + flat(second) == flat(full)


def test_a_time_limit_or_a_signal_stops_with_a_resumable_checkpoint(rl, tmp_path):
    rows = make_rows(12)
    args = loop_args(rl, tmp_path, "--steps", "4")
    policy = FakePolicy()
    state = run_loop(rl, args, policy, rows, started=-1e9)
    assert state["status"] == "stopped_time_limit" and state["step"] == 0 and policy.calls == []
    assert policy.saved[-1]["status"] == "stopped_time_limit" and (tmp_path / "out" / "summary.json").exists()
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2                                                  # False before steps 1 and 2, then True
    policy = FakePolicy()
    state = run_loop(rl, args, policy, rows, stopped=stop_after_two)
    assert state["status"] == "stopped_signal" and 0 < state["step"] < 4 and policy.saved[-1]["step"] == state["step"]
    rest = FakePolicy()
    done = run_loop(rl, args, rest, rows, state=resume(rl, policy.saved[-1]))
    assert done["status"] == "complete" and done["step"] == 4


def test_collapse_truncation_and_patience_stop_the_run_and_stay_final(rl, tmp_path):
    rows = make_rows(12)
    # collapse: every group answers the same molecule, so one modal answer over all the groups
    policy = FakePolicy(behave=lambda seed, row, i: ("const", "stop"))
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "9"), policy, rows)
    assert state["status"] == "stopped_collapse" and state["step"] == 3 and state["collapse_streak"] == 3
    first = state["metrics"][0]
    assert first["collapse_score"] == pytest.approx(1 / first["prompts"]) and first["prompts"] == 12   # no signal: the refill doubled the batch
    assert first["collapse_all"] == pytest.approx(1 / 12) and first["top_answer_share"] == 1.0           # one answer everywhere ...
    assert abs(first["lift_batch"]) < 1e-12 and first["modal_pairwise_tanimoto"] == pytest.approx(1.0)  # ... carries no profile information
    assert first["train_copy_share"] == 1.0                                       # and it is the reference of Train drug 0
    # truncation: 40% of the samples hit the length cap
    policy = FakePolicy(behave=lambda seed, row, i: (mixed(seed, row, i)[0], "length" if i < 2 else "stop"))
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "9"), policy, rows)
    assert state["status"] == "stopped_truncation" and state["step"] == 1 and state["metrics"][0]["truncation"] == pytest.approx(0.5)
    # patience: Val is flat while the train reward rises with the step (the fake policy learns from the torch seed)
    def learner(seed, row, i):
        step = seed - 3 * 1_000_003
        return ("exact" if random.Random(f"{seed}:{row['id']}:{i}:p").random() < min(0.95, 0.1 * step) else "bad"), "stop"
    policy = FakePolicy(behave=learner, eval_behave=lambda seed, row, i: ("bad", "stop"))
    args = loop_args(rl, tmp_path, "--steps", "12", "--eval-every", "2", "--patience", "3")
    state = run_loop(rl, args, policy, rows, val_rows=make_rows(3, "val", offset=8))
    assert state["status"] == "stopped_patience" and state["step"] == 6 and [e["step"] for e in state["evals"]] == [0, 2, 4, 6]
    assert state["evals"][-1]["train"] > state["evals"][1]["train"] and policy.saved[-1]["status"] == "stopped_patience"
    # a rule stop is final: resuming does no work
    again = FakePolicy()
    done = run_loop(rl, args, again, rows, val_rows=make_rows(3, "val", offset=8), state=resume(rl, policy.saved[-1]))
    assert done["status"] == "stopped_patience" and again.calls == [] and again.saved == []


def test_the_stop_rule_reads_collapse_score_not_collapse_all(rl, tmp_path):
    rows = make_rows(12)
    even = lambda row: int(row["group"][-2:]) % 2 == 0
    # even drugs answer with a molecule of their own half of the time, odd drugs never produce a valid answer
    policy = FakePolicy(behave=lambda seed, row, i: (("near" if i % 2 == 0 else "bad") if even(row) else "bad", "stop"))
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "5", "--prompts-per-step", "12"), policy, rows)
    assert state["status"] == "complete" and state["collapse_streak"] == 0
    for m in state["metrics"]:
        assert m["collapse_score"] == pytest.approx(1.0) and m["collapse_all"] == pytest.approx(0.5)      # 6 distinct modal answers: 6/6 and 6/12


def test_without_the_profile_a_collapsed_policy_is_not_stopped_but_the_alarm_is_logged(rl, tmp_path):
    rows = make_rows(12)
    policy = FakePolicy(behave=lambda seed, row, i: ("const", "stop"))
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "5", "--profile-mode", "none"), policy, rows)
    assert state["status"] == "complete" and state["step"] == 5 and state["collapse_streak"] == 5      # a constant answer is the optimum here
    assert all(m["collapse_score"] < 0.7 for m in state["metrics"])
    # the other rules still apply
    capped = FakePolicy(behave=lambda seed, row, i: (mixed(seed, row, i)[0], "length" if i < 2 else "stop"))
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "5", "--profile-mode", "none"), capped, rows)
    assert state["status"] == "stopped_truncation" and state["step"] == 1


def test_evals_are_baselined_drug_balanced_and_keep_their_adapters(rl, tmp_path):
    rows, val = make_rows(12), make_rows(4, "val", offset=8)              # val drugs have 1, 2, 3 and 1 rows
    good = lambda seed, row, i: ("exact" if row["group"].endswith(("00", "02")) else "bad", "stop")
    policy = FakePolicy(eval_behave=good)
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "3", "--eval-every", "2"), policy, rows, val_rows=val)
    assert [e["step"] for e in state["evals"]] == [0, 2, 3]                  # the baseline, every 2 steps, and the final step
    assert [a["step"] for a in state["adapters"]] == [2, 3] and [e["adapter"] for e in state["evals"]] == [None, "adapters/step-2", "adapters/step-3"]
    # drug-balanced: two of the four drugs are perfect, whatever their number of rows
    assert [e["val"] for e in state["evals"]] == pytest.approx([0.5, 0.5, 0.5]) and state["evals"][0]["metrics"]["drugs"] == 4
    assert state["evals"][0]["metrics"]["exact"] == pytest.approx(0.5) and state["evals"][0]["metrics"]["samples"] == len(val) * 4
    assert state["evals"][0]["train"] is None
    window = lambda lo, hi: np.mean([m["reward_mean"] for m in state["metrics"] if lo < m["step"] <= hi])
    assert state["evals"][1]["train"] == pytest.approx(window(0, 2)) and state["evals"][2]["train"] == pytest.approx(window(2, 3))
    assert [c[0] for c in policy.eval_calls] == [grpo.eval_seed(3, s) for s in (0, 2, 3)]
    assert {c[1] for c in policy.eval_calls} == {tuple(r["id"] for r in val)}   # every Val row, every time
    assert len((tmp_path / "out" / "evals.jsonl").read_text().splitlines()) == 3
    # evals never touch the schedule: the same run without them draws the same prompts
    plain = FakePolicy()
    run_loop(rl, loop_args(rl, tmp_path, "--steps", "3"), plain, rows)
    assert plain.calls == policy.calls
    assert not plain.eval_calls and plain.adapters == []                       # --eval-every 0 turns the evals off


ans = lambda smiles: "\n</think>\n\n" + smiles
tan = lambda a, b: chem.tanimoto(chem.standardize(a), chem.standardize(b))
conn_of = lambda smiles: chem.conn(chem.standardize(smiles))


def make_groups(rl, specs, reward_mode="full"):
    """Score hand-made samples: specs is [(row, [(text, finish), ...]), ...]; every sample is one fake token that decodes to its text."""
    registry = []

    def token(text):
        if text not in registry:
            registry.append(text)
        return registry.index(text)

    groups = []
    for row, samples in specs:
        completions = [dict(ids=[token(text)], finish=finish) for text, finish in samples]
        groups.append(dict(row=row, samples=rl.score_completions(lambda ids: "".join(registry[i] for i in ids), completions, row, reward_mode)))
    return groups


def test_batch_diagnostics_on_a_hand_made_batch(rl):
    S = SMILES
    rows = [r for r in make_rows(4) if r["id"].endswith("-0")]                  # four drugs whose references are S[0..3]
    bad = ans("not a molecule")
    groups = make_groups(rl, [
        (rows[0], [(ans(S[0]), "stop"), (ans(S[0]), "stop"), (ans(S[4]), "stop"), (bad, "stop")]),
        (rows[1], [(ans(S[4]), "stop"), (ans(S[4]), "stop"), (ans(S[5]), "stop"), (ans(S[1]), "length")]),   # the capped exact answer is no answer
        (rows[2], [(bad, "stop")] * 4),
        (rows[3], [(ans(S[5]), "stop"), (ans(S[1]), "stop"), (ans(S[1]), "stop"), (ans(S[0]), "stop")])])
    got = rl.batch_diagnostics(groups, rl.reference_keys(rows))
    answers = [[S[0], S[0], S[4]], [S[4], S[4], S[5]], [], [S[5], S[1], S[1], S[0]]]       # the valid, finished answers per group
    flat = [a for g in answers for a in g]
    assert len(flat) == 10
    assert got["top_answer_share"] == pytest.approx(3 / 10)                                 # S0 and S4 three times each
    assert got["distinct_answer_share"] == pytest.approx(4 / 10)
    assert got["train_copy_share"] == pytest.approx(5 / 10)                                 # S0 three times and S1 twice are Train references
    assert got["collapse_all"] == pytest.approx(3 / 4)                                      # modal answers S0, S4, S1 over four groups
    assert grpo.collapse_score([[s["key"] for s in g["samples"]] for g in groups]) == pytest.approx(1.0)       # ... but 3 of 3 with an answer
    assert got["modal_pairwise_tanimoto"] == pytest.approx((tan(S[0], S[4]) + tan(S[0], S[1]) + tan(S[4], S[1])) / 3)
    refs = [S[0], S[1], S[2], S[3]]
    lifts = []
    for d, valid in enumerate(answers):
        total = 0.0
        for a in valid:                                                                     # invalid and capped samples add 0 to the mean
            sims = [tan(a, r) for r in refs]
            total += sims[d] - (sum(sims) - sims[d]) / 3
        lifts.append(total / 4)
    assert got["lift_batch"] == pytest.approx(sum(lifts) / 4)
    scaffold = lambda a, r: bool(chem.scaffold(chem.standardize(a))) and chem.scaffold(chem.standardize(a)) == chem.scaffold(chem.standardize(r))
    assert got["reward_validity_term"] == pytest.approx(reward.W_VALID * 10 / 16)
    assert got["reward_tanimoto_term"] == pytest.approx(reward.W_TANI * sum(tan(a, refs[d]) for d, g in enumerate(answers) for a in g) / 16)
    assert got["reward_scaffold_term"] == pytest.approx(reward.W_SCAF * sum(scaffold(a, refs[d]) for d, g in enumerate(answers) for a in g) / 16)
    assert got["reward_identity"] == pytest.approx(2 / 16)
    assert set(got) == {"top_answer_share", "distinct_answer_share", "modal_pairwise_tanimoto", "train_copy_share", "collapse_all", "lift_batch",
                        "reward_validity_term", "reward_tanimoto_term", "reward_scaffold_term", "reward_identity"}


def test_a_constant_answer_has_zero_batch_lift_and_the_edge_cases_are_none(rl):
    rows = [r for r in make_rows(6) if r["id"].endswith("-0")]
    constant = rl.batch_diagnostics(make_groups(rl, [(row, [(ans(SMILES[0]), "stop")] * 4) for row in rows]), rl.reference_keys(rows))
    assert abs(constant["lift_batch"]) < 1e-12 and constant["top_answer_share"] == 1.0 and constant["modal_pairwise_tanimoto"] == pytest.approx(1.0)
    assert constant["distinct_answer_share"] == pytest.approx(1 / 24) and constant["collapse_all"] == pytest.approx(1 / 6)
    assert constant["train_copy_share"] == 1.0
    # no valid answer anywhere: shares have nothing to divide, nothing is None-poisoned that should not be
    nothing = rl.batch_diagnostics(make_groups(rl, [(row, [(ans("not a molecule"), "stop")] * 4) for row in rows]), rl.reference_keys(rows))
    assert nothing["top_answer_share"] is None and nothing["distinct_answer_share"] is None and nothing["train_copy_share"] is None
    assert nothing["modal_pairwise_tanimoto"] is None and nothing["collapse_all"] == 0.0 and nothing["lift_batch"] == 0.0
    assert nothing["reward_validity_term"] == 0.0 and nothing["reward_identity"] == 0.0
    # one group: there are no other drugs and no pair of modal answers
    single = rl.batch_diagnostics(make_groups(rl, [(rows[0], [(ans(SMILES[0]), "stop")] * 4)]), rl.reference_keys(rows))
    assert single["lift_batch"] is None and single["modal_pairwise_tanimoto"] is None and single["top_answer_share"] == 1.0
    # a policy that answers each drug with its own reference has a clearly positive lift
    own = rl.batch_diagnostics(make_groups(rl, [(row, [(ans(row["ref_smiles"]), "stop")] * 4) for row in rows]), rl.reference_keys(rows))
    assert own["lift_batch"] > 0.3 and own["reward_identity"] == 1.0 and own["collapse_all"] == 1.0


def test_reference_keys_are_connectivity_keys_of_references_and_alternates(rl):
    rows = [dict(ref_smiles=SMILES[0], ref_alt_smiles=[SMILES[1]]), dict(ref_smiles=SMILES[2], ref_alt_smiles=SMILES[3]),
            dict(ref_smiles=SMILES[0])]
    keys = rl.reference_keys(rows)
    assert keys == frozenset(conn_of(s) for s in SMILES[:4]) and len(keys) == 4
    assert rl.reference_keys([]) == frozenset() and conn_of(SMILES[5]) not in keys


def test_the_length_rule_and_the_lift_each_have_one_seam_for_metrics_v3(rl, tmp_path, monkeypatch):
    row, texts, decode = scoring_fixture()
    monkeypatch.setattr(rl, "finished_score", lambda text, ref, finished: dict(reward.score(text, ref), reward=7.0))
    assert [s["reward"] for s in rl.score_completions(decode, [dict(ids=[10], finish="stop"), dict(ids=[11], finish="length")], row)] == [7.0, 7.0]
    monkeypatch.setattr(rl, "lift_of", lambda samples: 123.0)
    rows = [r for r in make_rows(3) if r["id"].endswith("-0")]
    assert rl.batch_diagnostics(make_groups(rl, [(r, [(ans(r["ref_smiles"]), "stop")]) for r in rows]), frozenset())["lift_batch"] == 123.0
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "1", "--eval-every", "1"), FakePolicy(), make_rows(12), val_rows=make_rows(3, "val", offset=8))
    assert [e["metrics"]["lift"] for e in state["evals"]] == [123.0, 123.0]


def test_the_val_eval_logs_lift_validity_valid_only_similarity_and_both_rewards(rl, tmp_path):
    rows, val = make_rows(12), make_rows(4, "val", offset=8)                       # Val drugs: references S[8..11], with 1, 2, 3, 1 rows
    pattern = [("exact", "stop"), ("near", "stop"), ("bad", "stop"), ("exact", "length")]
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "1", "--eval-every", "1"), FakePolicy(eval_behave=lambda seed, row, i: pattern[i]),
                     rows, val_rows=val)
    drugs = sorted({r["group"] for r in val})
    ref = {r["group"]: r["ref_smiles"] for r in val}
    near = {d: SMILES[(SMILES.index(ref[d]) + 1) % len(SMILES)] for d in drugs}
    scored = {d: reward.score(ans(near[d]), next(r for r in val if r["group"] == d)) for d in drugs}
    refs = [ref[d] for d in drugs]
    lifts = []
    for i, d in enumerate(drugs):
        total = 0.0
        for a in (ref[d], near[d]):                                                # the bad sample and the capped exact one count 0
            sims = [tan(a, r) for r in refs]
            total += sims[i] - (sum(sims) - sims[i]) / 3
        lifts.append(total / 4)
    for entry in state["evals"]:
        m = entry["metrics"]
        assert m["lift"] == pytest.approx(sum(lifts) / 4) and m["validity"] == pytest.approx(0.5) and m["truncation"] == pytest.approx(0.25)
        assert m["tanimoto"] == pytest.approx(sum(1 + scored[d]["tanimoto"] for d in drugs) / 16)            # every attempt, invalid = 0
        assert m["tanimoto_valid"] == pytest.approx(sum(1 + scored[d]["tanimoto"] for d in drugs) / 8)      # valid finished samples only
        assert m["reward"] == pytest.approx(sum(1 + scored[d]["reward"] for d in drugs) / 16) == pytest.approx(m["reward_full"])
        assert m["exact"] == pytest.approx(0.25)
        assert entry["val"] == m["fv_tanimoto"] != pytest.approx(m["reward"]) and entry["val_reward"] == m["reward"]    # val is the endpoint now
    # the validity control trains on a different reward, but Val also reports the full one
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "1", "--eval-every", "1", "--reward-mode", "validity"),
                     FakePolicy(eval_behave=lambda seed, row, i: pattern[i]), rows, val_rows=val)
    m = state["evals"][0]["metrics"]
    assert m["reward"] == pytest.approx(0.5) and m["reward_full"] == pytest.approx(sum(1 + scored[d]["reward"] for d in drugs) / 16)
    assert state["evals"][0]["val_reward"] == pytest.approx(0.5) and m["lift"] == pytest.approx(sum(lifts) / 4)
    assert state["evals"][0]["val"] == m["fv_tanimoto"] == pytest.approx(1.0)       # the first sample is an exact answer in every row


def test_the_val_eval_reports_the_first_valid_policy_of_metrics_v3_and_val_is_its_tanimoto(rl, tmp_path, capsys):
    rows, val = make_rows(12), make_rows(4, "val", offset=8)               # Val drugs 00-03 with 1, 2, 3, 1 rows; references S[8..11]
    patterns = {"00": [("bad", "stop"), ("open", "stop"), ("exact", "length"), ("near", "stop")],       # the first usable sample is the fourth
                "01": [("exact", "stop"), ("bad", "stop"), ("bad", "stop"), ("bad", "stop")],            # the first
                "02": [("bad", "stop"), ("open", "stop"), ("exact", "length"), ("bad", "stop")],         # none: unparsed, unclosed, capped
                "03": [("bad", "stop"), ("const", "stop"), ("exact", "stop"), ("bad", "stop")]}          # the second, not the better third
    behave = lambda seed, row, i: patterns[row["group"][-2:]][i]
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "1", "--eval-every", "1"), FakePolicy(eval_behave=behave), rows, val_rows=val)
    drugs = sorted({r["group"] for r in val})
    row_of = {r["group"]: r for r in val}
    panel = [row_of[d]["ref_smiles"] for d in drugs]
    answers = {0: SMILES[(SMILES.index(panel[0]) + 1) % len(SMILES)], 1: panel[1], 3: SMILES[0]}       # the first usable answer of each drug
    drawn = {0: 4, 1: 1, 2: 4, 3: 2}
    want = dict(fv_tanimoto=[], fv_lift=[], fv_rate=[], fv_samples=[])
    for d in range(4):
        if d in answers:
            sims = [tan(answers[d], r) for r in panel]
            want["fv_tanimoto"].append(reward.score(ans(answers[d]), row_of[drugs[d]])["tanimoto"])
            want["fv_lift"].append(sims[d] - (sum(sims) - sims[d]) / 3)
        else:
            want["fv_tanimoto"].append(0.0)
            want["fv_lift"].append(0.0)
        want["fv_rate"].append(float(d in answers))
        want["fv_samples"].append(float(drawn[d]))
    out = capsys.readouterr().out
    for entry in state["evals"]:
        for key, values in want.items():
            assert entry["metrics"][key] == pytest.approx(np.mean(values)), key
        assert entry["val"] == entry["metrics"]["fv_tanimoto"] and entry["val_reward"] == entry["metrics"]["reward"]
    assert np.mean(want["fv_rate"]) == 0.75 and np.mean(want["fv_samples"]) == pytest.approx(11 / 4)       # the fixture: 3 of 4 drugs answer, 11 draws
    printed = [json.loads(line)["eval"] for line in out.splitlines() if line.startswith('{"eval"')]
    logged = [json.loads(line) for line in (tmp_path / "out" / "evals.jsonl").read_text().splitlines()]
    assert len(printed) == len(logged) == len(state["evals"])
    for entry in printed + logged:
        assert {"fv_tanimoto", "fv_lift", "fv_samples", "fv_rate"} <= set(entry["metrics"]) and "val_reward" in entry      # in both places


def test_the_first_valid_policy_equals_metrics_v3_on_the_same_samples(rl):
    val = make_rows(6, "val", offset=6)                                     # 6 drugs with 1, 2, 3, 1, 2, 3 rows; references S[6..11]
    rng = random.Random(7)
    kinds = ["exact", "near", "const", "plain", "bad", "open"]
    specs = [(row, [(text_for(rng.choice(kinds), row), rng.choice(["stop", "stop", "length"])) for _ in range(8)]) for row in val[:-1]]
    specs.append((val[-1], [(text_for("bad", val[-1]), "stop")] * 8))        # one row with no usable sample
    groups = make_groups(rl, specs)
    samples = [dict(s, drug=g["row"]["group"], ref=g["row"]["ref_smiles"], prompt=g["row"]["id"]) for g in groups for s in g["samples"]]
    refs = {}
    for row in val:
        refs.setdefault(row["group"], row["ref_smiles"])
    scored = []
    for g in groups:                                                        # the rows metrics.py scores, lift from its definition
        row = g["row"]
        for j, s in enumerate(g["samples"]):
            ok = s["valid"] and s["closed"] and not s["truncated"]
            sims = {d: tan(s["answer"], r) for d, r in refs.items()} if ok else {}
            lift = sims[row["group"]] - np.mean([v for d, v in sims.items() if d != row["group"]]) if ok else 0.0
            scored.append(dict(id=row["id"], group=row["group"], split="val", sample_idx=j, finish_reason="length" if s["truncated"] else "stop",
                               n_tokens=s["n_tokens"], closed=s["closed"], answer=s["answer"], valid=s["valid"], tanimoto=s["tanimoto"],
                               scaffold_match=s["scaffold_match"], conn_match=s["conn_match"], stereo_match=False, reward=s["reward_full"], lift=lift))
    drug_values = metrics.aggregate(scored)["V"]
    ours = grpo.first_valid_policy(samples, rl.similarity)
    for name in grpo.FV_FIELDS:
        assert ours[name] == pytest.approx(float(np.nanmean(drug_values[:, metrics.METRICS.index(name)])), abs=1e-9), name
    assert 0 < ours["fv_rate"] < 1 and ours["fv_samples"] > 1 and ours["fv_lift"] != 0       # the fixture exercises every branch
    # the first usable sample is not the best one, in the pin as in the policy
    best = max(s["tanimoto"] for g in groups for s in g["samples"] if grpo.usable(s))
    assert ours["fv_tanimoto"] < best


def test_patience_follows_val_fv_tanimoto_not_the_val_mean_reward(rl, tmp_path):
    rows, val = make_rows(12), make_rows(3, "val", offset=8)

    def learner(seed, row, i):                                              # the train reward rises with the step
        step = seed - 3 * 1_000_003
        return ("exact" if random.Random(f"{seed}:{row['id']}:{i}:p").random() < min(0.95, 0.1 * step) else "bad"), "stop"

    def val_behave(seed, row, i):                                           # Val: the first sample never changes, later ones improve
        step = seed - 3 * 1_000_003 - (2 << 40)
        return ("near", "stop") if i == 0 else (("exact" if i <= step // 2 else "bad"), "stop")

    policy = FakePolicy(behave=learner, eval_behave=val_behave)
    args = loop_args(rl, tmp_path, "--steps", "12", "--eval-every", "2", "--patience", "3")
    state = run_loop(rl, args, policy, rows, val_rows=val)
    rewards = [e["val_reward"] for e in state["evals"]]
    assert rewards == sorted(rewards) and rewards[-1] > rewards[0] + 0.1                       # the Val mean reward rises at every eval ...
    assert len({e["val"] for e in state["evals"]}) == 1                                      # ... while fv_tanimoto is flat
    assert state["status"] == "stopped_patience" and state["step"] == 6 and [e["step"] for e in state["evals"]] == [0, 2, 4, 6]
    assert state["evals"][-1]["train"] > state["evals"][1]["train"]


def test_publish_adapter_never_loses_a_copy_and_keeps_the_revision_pin(rl, tmp_path, monkeypatch):
    root = tmp_path / "run"
    target = root / "adapters/step-2"

    def writer(weights):
        def write(path):
            (path / "adapter_model.safetensors").write_bytes(weights)
            (path / "adapter_config.json").write_text("{}")
            (path / "README.md").write_text("model card")                           # what save_pretrained also writes
        return write

    listing = lambda: sorted(p.name for p in (root / "adapters").iterdir())
    first = rl.publish_adapter(root, 2, writer(b"old"), "rev1")
    assert sorted(f.name for f in target.iterdir()) == ["adapter_config.json", "adapter_model.safetensors", "revision.json"]
    assert json.loads((target / "revision.json").read_text()) == {"revision": "rev1"} and listing() == ["step-2"]
    assert first == dict(step=2, path="adapters/step-2", sha256={n: rl.file_sha256(target / n) for n in rl.KEPT_FILES})
    inode = target.stat().st_ino
    assert rl.publish_adapter(root, 2, writer(b"old"), "rev1") == first and target.stat().st_ino == inode and listing() == ["step-2"]   # a no-op
    second = rl.publish_adapter(root, 2, writer(b"new"), "rev1")                    # the leftover of a killed run is replaced
    assert second != first and (target / "adapter_model.safetensors").read_bytes() == b"new" and listing() == ["step-2"]
    real_replace = os.replace

    def failing(source, destination):
        if Path(destination).name == "step-2":
            raise OSError("the rename failed")
        return real_replace(source, destination)

    monkeypatch.setattr(rl.os, "replace", failing)
    with pytest.raises(OSError, match="rename failed"):
        rl.publish_adapter(root, 2, writer(b"third"), "rev1")
    monkeypatch.undo()
    assert (target / "adapter_model.safetensors").read_bytes() == b"new" and listing() == ["step-2"]       # the old copy survived the failed swap
    assert {n: rl.file_sha256(target / n) for n in rl.KEPT_FILES} == second["sha256"]
    rl.sft.commit_checkpoint(root, dict(global_step=2, adapters=[second]), lambda path: None)             # now the step is committed
    with pytest.raises(rl.PackError, match="committed"):
        rl.publish_adapter(root, 2, writer(b"fourth"), "rev1")
    assert (target / "adapter_model.safetensors").read_bytes() == b"new"
    assert rl.publish_adapter(root, 2, writer(b"new"), "rev1") == second                                  # an identical repeat is still fine


def flat_samples(groups):
    """What rl_grpo.collect hands to the batch rewards: every sample knows its drug and that drug's reference."""
    return [dict(s, drug=g["row"]["group"], ref=g["row"]["ref_smiles"]) for g in groups for s in g["samples"]]


def test_the_bulk_similarity_equals_the_scorers_tanimoto(rl):
    for answer in (SMILES[0], SMILES[4], SMILES[8], PLAIN):
        bulk = rl.similarities(answer, SMILES)
        assert bulk == [chem.tanimoto(chem.standardize(answer), chem.standardize(r)) for r in SMILES]      # exactly the scorer's number
        assert [rl.similarity(answer, r) for r in SMILES] == bulk and [rl.similarity(r, answer) for r in SMILES] == bulk
    assert rl.similarities("not a molecule", SMILES) == [0.0] * len(SMILES) and rl.similarities(SMILES[0], ["((", SMILES[0], "((", SMILES[1]]) \
        == [0.0, 1.0, 0.0, tan(SMILES[0], SMILES[1])]
    assert rl.similarity(SMILES[0], "(((") == 0.0 and rl.similarities(SMILES[0], []) == []


def test_a_constant_answer_earns_the_validity_term_under_the_contrastive_reward(rl):
    rows = [r for r in make_rows(8) if r["id"].endswith("-0")]
    for invalid in (0, 1, 2):                                                       # the same share of invalid samples for every drug
        specs = [(row, [(ans(PLAIN), "stop")] * (4 - invalid) + [(ans("not a molecule"), "stop")] * invalid) for row in rows]
        samples = flat_samples(make_groups(rl, specs, "contrastive"))
        assert all(s["reward"] is None for s in samples)                            # it needs the batch: not set by scoring one prompt
        rl.apply_contrast(samples)
        assert all(not s["scaffold_match"] and not s["conn_match"] for s in samples)         # the fixture: no bonus, no identity
        validity = sum(s["answer"] is not None for s in samples) / len(samples)
        mean = lambda key: sum(s[key] for s in samples) / len(samples)
        assert mean("reward") == pytest.approx(reward.W_VALID * validity, abs=1e-12)        # 0.05 times validity, as the validity term alone
        assert mean("reward_full") - mean("reward") > 0.02 * validity                         # the generic similarity the full reward pays
        assert all(s["reward"] == 0.0 for s in samples if s["answer"] is None)
        assert all(s["reward_full"] == pytest.approx(0.05 + 0.6 * tan(PLAIN, s["ref"])) for s in samples if s["answer"])


def test_a_constant_answer_with_a_scaffold_bonus_earns_more_than_the_validity_term(rl):
    rows = [r for r in make_rows(8) if r["id"].endswith("-0")]                      # references S[0..7]: six of the eight share the benzene scaffold
    constant = "Cc1ccccc1"
    samples = flat_samples(make_groups(rl, [(row, [(ans(constant), "stop")] * 4) for row in rows], "contrastive"))
    rl.apply_contrast(samples)
    refs = [row["ref_smiles"] for row in rows]
    scaffold = lambda smiles: chem.scaffold(chem.standardize(smiles))
    shared = [bool(scaffold(constant)) and scaffold(constant) == scaffold(r) for r in refs]
    assert sum(shared) == 6 and not any(s["conn_match"] for s in samples) and [s["scaffold_match"] for s in samples[::4]] == shared
    t = [tan(constant, r) for r in refs]
    raw = [samples[4 * d]["reward_full"] - reward.W_TANI * np.mean([t[j] for j in range(8) if j != d]) for d in range(8)]
    # before the floor the similarity terms cancel and the validity term and the scaffold bonus are what is left: 0.05 + 0.3 * 6/8, not 0.05
    assert np.mean(raw) == pytest.approx(reward.W_VALID + reward.W_SCAF * 6 / 8, abs=1e-12) and np.mean(raw) > 5 * reward.W_VALID
    got = [s["reward"] for s in samples]
    assert np.mean(got) == pytest.approx(np.mean([max(0.0, r) for r in raw])) and np.mean(got) >= np.mean(raw) - 1e-12      # the floor only adds
    assert all(g == pytest.approx(max(0.0, raw[d])) for d in range(8) for g in got[4 * d:4 * d + 4]) and min(got) >= 0.0


def test_an_answer_specific_to_its_drug_scores_higher_than_the_same_molecule_given_to_another_drug(rl):
    S = SMILES
    rows = [r for r in make_rows(4) if r["id"].endswith("-0")]                      # references S[0..3]: aspirin, ibuprofen, paracetamol, caffeine
    molecule = S[10]                                                                # naproxen resembles ibuprofen
    assert tan(molecule, S[1]) > max(tan(molecule, S[i]) for i in (0, 2, 3)) + 0.1     # the fixture: it resembles ibuprofen's reference most

    def reward_of_molecule(drug):
        specs = [(row, [(ans(molecule if i == drug else PLAIN), "stop")] * 2) for i, row in enumerate(rows)]
        samples = flat_samples(make_groups(rl, specs, "contrastive"))
        rl.apply_contrast(samples)
        mine = [s for s in samples if s["drug"] == rows[drug]["group"]]
        assert {s["answer"] for s in mine} == {molecule} and len({s["reward"] for s in mine}) == 1
        return mine[0]["reward"], mine[0]["reward_full"]

    specific, specific_full = reward_of_molecule(1)                                 # given to ibuprofen
    other, other_full = reward_of_molecule(2)                                       # the same molecule given to paracetamol
    others = lambda drug: sum(tan(molecule, S[j]) for j in range(4) if j != drug) / 3
    assert specific == pytest.approx(reward.score(ans(molecule), rows[1])["reward"] - reward.W_TANI * others(1)) and specific_full > other_full
    assert other == pytest.approx(reward.score(ans(molecule), rows[2])["reward"] - reward.W_TANI * others(2))
    assert specific > other and specific > reward.W_VALID > other                     # profile-specific beats generic, generic is the validity term


def test_the_contrastive_reward_needs_two_drugs_and_does_not_touch_unusable_samples(rl):
    rows = [r for r in make_rows(3) if r["id"].endswith("-0")]
    single = flat_samples(make_groups(rl, [(rows[0], [(ans(PLAIN), "stop")])], "contrastive"))
    with pytest.raises(rl.PackError, match="at least two drugs"):
        rl.apply_contrast(single)
    # a capped sample whose text holds an exact answer, an unclosed one and an invalid one: all stay at 0, the usable ones move
    specs = [(row, [(ans(PLAIN), "stop"), (ans(row["ref_smiles"]), "length"), ("CCO is the answer", "stop"), (ans("not a molecule"), "stop")]) for row in rows]
    samples = flat_samples(make_groups(rl, specs, "contrastive"))
    rl.apply_contrast(samples)
    for s in samples:
        assert s["reward"] == 0.0 if s["answer"] is None else s["reward"] != s["reward_full"]
    assert [s["answer"] is not None for s in samples[:4]] == [True, False, False, False]
    with pytest.raises(rl.PackError, match="prompts_per_step"):
        rl.config_from_args(["--init-adapter", "none", "--reward-mode", "contrastive", "--prompts-per-step", "1"])
    assert rl.config_from_args(["--init-adapter", "none", "--reward-mode", "contrastive", "--prompts-per-step", "2"]).reward_mode == "contrastive"


def contrastive_oracle(policy, rows, group_size=4, stale=False, floor=True):
    """Expected update when the training reward is the contrastive one, from scratch over EVERY group attempted in the step (all
    rollout calls). stale=True gives each sample only the drugs of its own rollout call as "others": what a missed recompute after a
    refill would leave. floor=False leaves the negative rewards unclipped (what the floor changes). Returns (expected, kept, scored)
    with scored = (row, completions, contrastive rewards, full rewards)."""
    by_id = {r["id"]: r for r in rows}
    groups, call_of = [], []
    for call, (seed, ids) in enumerate(policy.calls):
        for row, group in zip((by_id[i] for i in ids), policy.make(policy.behave, [by_id[i] for i in ids], seed, group_size)):
            groups.append((row, group))
            call_of.append(call)
    scored = []
    for d, (row, group) in enumerate(groups):
        pool = [j for j in range(len(groups)) if j != d and (not stale or call_of[j] == call_of[d])]
        contrastive, full = [], []
        for c in group:
            s = reward.score(policy.decode(c["ids"]), row)
            finished = c["finish"] != "length"
            full.append(s["reward"] if finished else 0.0)
            if s["valid"] and s["closed"] and finished:
                others = np.mean([tan(s["answer"], groups[j][0]["ref_smiles"]) for j in pool])
                raw = s["reward"] - reward.W_TANI * others
                contrastive.append(max(0.0, raw) if floor else raw)
            else:
                contrastive.append(0.0)
        scored.append((row, group, contrastive, full))
    expected, kept = expected_update([(row, group, rewards) for row, group, rewards, _ in scored])
    return expected, kept, scored


def test_the_contrastive_rewards_are_recomputed_over_the_merged_batch_after_a_refill(rl, tmp_path):
    rows = make_rows(12)
    silent = {g for g in range(12) if g % 2}                                          # odd drugs never answer validly: no signal, so a refill
    policy = FakePolicy(behave=lambda seed, row, i: ("bad", "stop") if int(row["group"][-2:]) in silent else mixed(seed, row, i))
    args = loop_args(rl, tmp_path, "--steps", "1", "--prompts-per-step", "6", "--min-groups", "6", "--reward-mode", "contrastive")
    state = run_loop(rl, args, policy, rows)
    m = state["metrics"][0]
    assert len(policy.calls) == 2 and 0 < m["refilled_prompts"] <= 6 and m["prompts"] == 6 + m["refilled_prompts"]
    expected, kept, scored = contrastive_oracle(policy, rows)
    check_update(policy, expected, kept)                                              # every sample saw the same others: the merged batch
    stale, _, _ = contrastive_oracle(policy, rows, stale=True)
    assert max(abs(a[2] - b[2]) for a, b in zip(sorted(expected), sorted(stale))) > 1e-4         # the fixture: a stale recompute would differ
    contrastive = [r for _, _, rewards, _ in scored for r in rewards]
    full = [r for _, _, _, f in scored for r in f]
    assert m["reward_mean"] == pytest.approx(np.mean(contrastive)) and m["reward_full_mean"] == pytest.approx(np.mean(full))
    assert m["reward_mean"] != pytest.approx(m["reward_full_mean"]) and min(contrastive) == 0.0 < max(contrastive)    # nothing below a failure
    assert m["reward_std"] == pytest.approx(np.std(contrastive))
    # the floor is in the update: usable answers that sit closer to the other drugs than to their own would otherwise be negative
    raw = [r for _, _, rewards, _ in contrastive_oracle(policy, rows, floor=False)[2] for r in rewards]
    assert min(raw) < 0 and np.mean(raw) < np.mean(contrastive)
    unfloored, _, _ = contrastive_oracle(policy, rows, floor=False)
    assert max(abs(a[2] - b[2]) for a, b in zip(sorted(expected), sorted(unfloored))) > 1e-4         # a missing floor would change the weights


def test_the_val_eval_scores_the_contrastive_reward_against_the_other_val_drugs(rl, tmp_path):
    rows, val = make_rows(12), make_rows(4, "val", offset=8)
    pattern = [("exact", "stop"), ("near", "stop"), ("plain", "stop"), ("bad", "stop")]
    state = run_loop(rl, loop_args(rl, tmp_path, "--steps", "1", "--eval-every", "1", "--reward-mode", "contrastive"),
                     FakePolicy(eval_behave=lambda seed, row, i: pattern[i]), rows, val_rows=val)
    drugs = sorted({r["group"] for r in val})
    refs = {r["group"]: r["ref_smiles"] for r in val}
    row_of = {r["group"]: r for r in val}
    per_drug, per_drug_full, raw = [], [], []
    for d in drugs:
        near = SMILES[(SMILES.index(refs[d]) + 1) % len(SMILES)]
        contrastive, full = [], []
        for answer in (refs[d], near, PLAIN):
            f = reward.score(ans(answer), row_of[d])["reward"]
            others = np.mean([tan(answer, refs[o]) for o in drugs if o != d])
            raw.append(f - reward.W_TANI * others)
            contrastive.append(max(0.0, raw[-1]))                                   # floored: nothing below a failure
            full.append(f)
        per_drug.append(sum(contrastive) / 4)                                       # the invalid fourth sample counts 0
        per_drug_full.append(sum(full) / 4)
    assert min(raw) < 0                                                             # the fixture: the floor bites here
    for entry in state["evals"]:
        assert entry["metrics"]["reward"] == pytest.approx(np.mean(per_drug)) == pytest.approx(entry["val_reward"])
        assert entry["val"] == entry["metrics"]["fv_tanimoto"] and entry["val"] != pytest.approx(entry["val_reward"])
        assert entry["metrics"]["reward_full"] == pytest.approx(np.mean(per_drug_full)) and entry["metrics"]["reward"] != pytest.approx(entry["metrics"]["reward_full"])


# ---------------------------------------------------------------- the gates' pass/fail logic
def test_parity_statistics_and_the_thresholds_chosen_from_the_accepted_a100_runs(rl):
    stats = rl.parity_stats([[-1.0, -2.0, -3.0, -4.0]], [[-1.0, -2.1, -3.0, -3.9]])
    assert stats["tokens"] == 4 and stats["mean_abs"] == pytest.approx(0.05) and stats["max_abs"] == pytest.approx(0.1)
    assert stats["share_above_0_05"] == pytest.approx(0.5) and stats["mean_ratio"] == pytest.approx(np.mean(np.exp([0, -0.1, 0, 0.1])))
    with pytest.raises(rl.PackError):
        rl.parity_stats([[]], [[]])
    good = dict(mean_abs=0.03, share_above_0_05=0.15, mean_ratio=1.015)
    assert rl.parity_ok(good) and not rl.parity_ok(dict(good, mean_abs=0.031)) and not rl.parity_ok(dict(good, share_above_0_05=0.151))
    assert not rl.parity_ok(dict(good, mean_ratio=0.97)) and not rl.parity_ok(dict(good, mean_ratio=1.03)) and rl.parity_ok(dict(good, mean_ratio=0.99))
    # the thresholds were chosen from the accepted A100 runs: mean |diff| 0.015-0.018, 8-11% above 0.05, mean ratio 0.994-1.000. All pass
    for measured in (dict(mean_abs=0.015, share_above_0_05=0.08, mean_ratio=0.994), dict(mean_abs=0.018, share_above_0_05=0.11, mean_ratio=1.000),
                     dict(mean_abs=0.018, share_above_0_05=0.08, mean_ratio=0.994)):
        assert rl.parity_ok(measured)


class ParityPolicy:
    """Scripted for the parity gate: 20 tokens per completion with sampling log-probs of -1.0; the training forward is `shift` higher on
    the first `count` tokens of every completion."""
    sampling = dict(do_sample=True, temperature=1.0, top_k=0, eos_token_id=[0])
    model = types.SimpleNamespace(get_base_model=lambda: types.SimpleNamespace(
        generation_config=types.SimpleNamespace(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=None, repetition_penalty=1.0)))

    def __init__(self, count, shift):
        self.count, self.shift = count, shift

    def prompt_ids(self, row):
        return [1] * (3 + len(row["id"]))

    def generate(self, prompts, n, seed, logprobs=False):
        return [[dict(ids=list(range(1, 21)), finish="stop", logp=[-1.0] * 20) for _ in range(n)] for _ in prompts]

    def token_logprobs(self, pairs):
        return [[-1.0 + (self.shift if k < self.count else 0.0) for k in range(20)] for _ in pairs]


def test_the_parity_gate_exits_zero_exactly_when_every_case_is_inside_the_thresholds(rl, tmp_path, monkeypatch, capsys):
    write_rows(tmp_path / "train.jsonl", make_rows(8))
    write_rows(tmp_path / "val.jsonl", make_rows(3, "val", offset=8))
    gate = lambda policy: (monkeypatch.setattr(rl, "load_policy", lambda args, existing=None: (policy, {})),
                           rl.main(["--init-adapter", "none", "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
                                    "--out", str(tmp_path / "out"), "--prompts-per-step", "4", "--gate", "parity"]))[1]
    # 2 of 20 tokens above 0.05 (10%), mean |diff| 0.006: inside the thresholds chosen from the A100 runs (the old 5% share would fail it)
    assert gate(ParityPolicy(count=2, shift=0.06)) == 0
    out = capsys.readouterr().out
    assert "PARITY PASS" in out and "FAIL" not in out and out.count('"case"') == 4                    # batch, mixed lengths, its shortest prompt, alone
    # 4 of 20 (20%) above 0.05 with a small mean |diff| and a mean ratio inside 2%: only the share fails, and the exit status says so
    assert gate(ParityPolicy(count=4, shift=0.06)) == 1
    out = capsys.readouterr().out
    assert "PARITY FAIL" in out and "PARITY PASS" not in out and "WARN" not in out
    assert gate(ParityPolicy(count=20, shift=0.04)) == 1                                              # mean |diff| 0.04 > 0.03
    assert gate(ParityPolicy(count=0, shift=0.5)) == 0                                                # no token differs: a perfect match
    capsys.readouterr()


def smoke_state(**over):
    metrics = [dict(step=s, schedule_hash=f"h{s}", cursor=8 * s, rollout_seed=1000 + s, rollout_hash=f"r{s}", loss=0.5) for s in (1, 2, 3)]
    return dict(dict(status="complete", step=3, metrics=metrics, first_update_delta_norm=0.01), **over)


def test_the_smoke_gate_passes_only_when_the_resumed_run_matches_the_uninterrupted_one(rl):
    def verdict(full=None, resumed=None):
        checks, info = rl.smoke_checks(full or smoke_state(), resumed or smoke_state())
        return {name for name, passed in checks.items() if not passed}, info

    assert verdict()[0] == set() and all(v is True for k, v in verdict()[1].items() if "bitwise" in k)
    broken = lambda i, **kv: smoke_state(metrics=[dict(m, **(kv if m["step"] == i else {})) for m in smoke_state()["metrics"]])
    assert verdict(resumed=broken(3, schedule_hash="other"))[0] == {"step 3 schedule equal"}
    assert verdict(resumed=broken(3, cursor=99))[0] == {"step 3 schedule equal"}
    assert verdict(resumed=broken(3, rollout_seed=1))[0] == {"step 3 rollout seed equal"}
    assert verdict(resumed=broken(2, schedule_hash="other"))[0] == {"steps 1-2 identical in both runs (schedule, seed)"}
    failed, info = verdict(resumed=broken(3, rollout_hash="elsewhere"))
    assert failed == set() and not [v for k, v in info.items() if "bitwise" in k][0]      # informational only
    for delta in (0.0, float("nan"), None, float("inf")):
        assert verdict(full=smoke_state(first_update_delta_norm=delta))[0] == {"first update changed the adapter (delta norm > 0 and finite)"}
    assert verdict(full=broken(2, loss=float("nan")))[0] == {"losses finite"}
    assert "resumed from step 2" in verdict(resumed=smoke_state(step=2))[0]
    assert "three steps ran" in verdict(resumed=smoke_state(status="stopped_time_limit"))[0]
    short = smoke_state(metrics=smoke_state()["metrics"][:2], step=2)
    failed, _ = verdict(resumed=short)                                              # a resumed run that lost a step fails, never raises
    assert {"three steps ran", "resumed from step 2", "step 3 schedule equal", "step 3 rollout seed equal"} <= failed


# ---------------------------------------------------------------- torch: the fp32 head, the weighted loss and generation plumbing
def tiny(torch, rl, vocab=7, hidden=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb, mix, head = (torch.randn(vocab, hidden, generator=g), torch.randn(hidden, hidden, generator=g) * 0.7, torch.randn(vocab, hidden, generator=g))

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("emb", emb.clone())
            self.mix = torch.nn.Parameter(mix.clone())
            self.autocast_seen = []

        def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
            self.autocast_seen.append(torch.is_autocast_enabled("cpu"))
            return types.SimpleNamespace(last_hidden_state=torch.tanh(self.emb[input_ids] @ self.mix))

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = rl.fp32_head(head)
            self.config = types.SimpleNamespace(use_cache=False)
            self.generate_calls = []

        def get_base_model(self):
            return self

        def generate(self, input_ids, attention_mask=None, **kwargs):
            """Stands in for HF generate: max_new_tokens tokens drawn from this model's own next-token distribution with the global torch
            RNG (so the seed matters), recording the keyword arguments it was given."""
            self.generate_calls.append(kwargs)
            sequences = input_ids
            for _ in range(kwargs["max_new_tokens"]):
                hidden = self.model(sequences, attention_mask).last_hidden_state[:, -1]
                probs = torch.softmax(self.lm_head(hidden) / kwargs["temperature"], dim=-1)
                sequences = torch.cat([sequences, torch.multinomial(probs, 1)], dim=1)
            return sequences

        def save_pretrained(self, path, safe_serialization=True, save_embedding_layers=False):
            torch.save({name: p.detach() for name, p in self.named_parameters() if p.requires_grad}, Path(path) / "adapter_model.safetensors")
            (Path(path) / "adapter_config.json").write_text("{}")

    return Base(), emb.double().numpy(), mix.double().numpy(), head.double().numpy()


SAMPLES = [([1, 2, 3], [4, 5, 6, 2], 0.5), ([2, 2], [3], -1.25), ([5, 1, 4, 2, 3], [6, 6], 0.75), ([6], [1, 1, 1, 5, 5], 1.5)]


def hand(emb, mix, head, samples, temperature=1.0):
    """The loss -sum(w * log pi) and its gradient with respect to `mix`, in float64 numpy."""
    grad, loss, logps = np.zeros_like(mix), 0.0, []
    for prompt, completion, w in samples:
        seq = prompt + completion
        for t in range(len(prompt) - 1, len(seq) - 1):               # the hidden state at t predicts the token at t + 1
            x = emb[seq[t]]
            h = np.tanh(x @ mix)
            logits = head @ h / temperature
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            y = seq[t + 1]
            logps.append(float(np.log(probs[y])))
            loss += -w * logps[-1]
            dlogits = -w * (np.eye(len(logits))[y] - probs) / temperature
            grad += np.outer(x, (head.T @ dlogits) * (1 - h ** 2))
    return loss, grad, logps


def batches_of(rl, torch, samples, split):
    return [rl.collate([samples[i] for i in part], 0, "cpu", torch) for part in split]


def test_the_fp32_head_stays_fp32_inside_a_bf16_autocast_region(rl, torch):
    weight = torch.randn(11, 6)
    hidden = torch.randn(4, 6).to(torch.bfloat16)
    head = rl.fp32_head(weight)
    want = torch.nn.functional.linear(hidden.float(), weight)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = head(hidden)
        unprotected = torch.nn.functional.linear(hidden.float(), weight)
    assert got.dtype == torch.float32 and torch.equal(got, want)               # bit-identical to a plain fp32 matmul
    assert unprotected.dtype == torch.bfloat16 and not torch.allclose(unprotected.float(), want, rtol=1e-4, atol=0)    # what autocast would do
    assert head.weight.dtype == torch.float32 and not list(head.parameters()) and "weight" not in head.state_dict()
    assert head(hidden.float()).dtype == torch.float32 and head(torch.randn(2, 3, 6).half()).shape == (2, 3, 11)


def test_installing_the_head_copies_the_tied_weight_and_leaves_the_embedding_alone(rl, torch):
    class Lm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(9, 4).to(torch.bfloat16)
            self.lm_head = torch.nn.Linear(4, 9, bias=False).to(torch.bfloat16)
            self.lm_head.weight = self.model.embed_tokens.weight             # tied, as in the 4B

    model = Lm()
    table = model.model.embed_tokens.weight
    before = table.detach().clone()
    head = rl.install_fp32_head(model)
    assert model.lm_head is head and model.model.embed_tokens.weight is table and table.dtype == torch.bfloat16
    assert torch.equal(table, before) and head.weight.dtype == torch.float32 and head.weight.data_ptr() != table.data_ptr()
    assert torch.equal(head.weight, before.float())                            # exactly the bf16 values, widened


def test_completion_logprobs_are_the_next_token_log_softmax_at_the_sampling_temperature(rl, torch):
    model, emb, mix, head = tiny(torch, rl)
    for temperature in (1.0, 2.0):
        batch = rl.collate(SAMPLES, 0, "cpu", torch)
        with torch.no_grad():
            got = rl.completion_logprobs(model, batch, temperature, chunk=3).numpy()
        _, _, want = hand(emb, mix, head, SAMPLES, temperature)
        assert len(got) == sum(len(c) for _, c, _ in SAMPLES)                  # completion positions only, no padding
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert (got < 0).all()
    # collate: only completion positions are supervised and padding is masked
    batch = rl.collate(SAMPLES[:2], 9, "cpu", torch)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 2], [2, 2, 3, 9, 9, 9, 9]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5, 6, 2], [-100, -100, 3, -100, -100, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1] * 7, [1, 1, 1, 0, 0, 0, 0]] and batch["weights"].tolist() == [0.5, -1.25]


def test_the_weighted_gradient_equals_the_hand_computed_gradient_whatever_the_chunking_or_split(rl, torch):
    model, emb, mix, head = tiny(torch, rl)
    want_loss, want_grad, _ = hand(emb, mix, head, SAMPLES)
    splits = [[[0, 1, 2, 3]], [[0, 1], [2, 3]], [[3], [0], [2], [1]], [[2, 0, 3], [1]]]
    for chunk in (1, 3, 5, 256):
        for split in splits:
            model.zero_grad()
            loss = rl.update_gradients(model, batches_of(rl, torch, SAMPLES, split), 1.0, chunk=chunk)
            assert loss == pytest.approx(want_loss, rel=1e-5)
            np.testing.assert_allclose(model.model.mix.grad.double().numpy(), want_grad, rtol=1e-4, atol=1e-6)
    # sign: a positive weight on a token lowers its loss, so the step along -grad raises its log-probability
    positive = [([1, 2], [3], 1.0)]
    model.zero_grad()
    rl.update_gradients(model, batches_of(rl, torch, positive, [[0]]), 1.0)
    before = rl.completion_logprobs(model, rl.collate(positive, 0, "cpu", torch), 1.0).detach().clone()
    with torch.no_grad():
        model.model.mix -= 0.05 * model.model.mix.grad
        after = rl.completion_logprobs(model, rl.collate(positive, 0, "cpu", torch), 1.0)
    assert after.item() > before.item()
    # the sampling temperature is part of the policy being trained
    model, emb, mix, head = tiny(torch, rl)
    rl.update_gradients(model, batches_of(rl, torch, SAMPLES, [[0, 1, 2, 3]]), 2.0)
    np.testing.assert_allclose(model.model.mix.grad.double().numpy(), hand(emb, mix, head, SAMPLES, 2.0)[1], rtol=1e-4, atol=1e-6)


def test_the_autocast_context_wraps_every_micro_batch_forward(rl, torch):
    model, emb, mix, head = tiny(torch, rl)
    ctx = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
    rl.update_gradients(model, batches_of(rl, torch, SAMPLES, [[0, 1], [2, 3]]), 1.0, context=ctx)
    assert model.model.autocast_seen == [True, True]
    # the trunk matmul runs in bf16 here (about 1% off); the fp32 head is what keeps the rest exact
    np.testing.assert_allclose(model.model.mix.grad.double().numpy(), hand(emb, mix, head, SAMPLES)[1], rtol=0.1, atol=0.05)
    model.model.autocast_seen.clear()
    rl.update_gradients(model, batches_of(rl, torch, SAMPLES, [[0, 1, 2, 3]]), 1.0)
    assert model.model.autocast_seen == [False]


def test_a_non_finite_loss_stops_before_backward(rl, torch):
    model, *_ = tiny(torch, rl)
    bad = [([1, 2], [3, 4], float("nan"))]
    with pytest.raises(rl.PackError, match="Non-finite loss"):
        rl.update_gradients(model, batches_of(rl, torch, bad, [[0]]), 1.0)
    assert model.model.mix.grad is None


def test_generate_all_left_pads_sorts_longest_first_and_restores_prompt_order(rl, torch):
    pad, eos, cap = 0, 9, 4
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 1, 2], [3, 3], [5]]
    seen = []

    def call(ids, mask):
        seen.append((ids.tolist(), mask.tolist()))
        rows = []
        for b in range(ids.shape[0]):
            real = [t for t, m in zip(ids[b].tolist(), mask[b].tolist()) if m]
            new = [real[-1] + 1, real[0] + 2, eos] if len(real) % 2 == 0 else [real[-1] + 1] * cap     # even length: stops
            rows.append(ids[b].tolist() + new + [pad] * (cap - len(new)))
        logp = torch.arange(cap, dtype=torch.float32).repeat(ids.shape[0], 1)
        return torch.tensor(rows), logp

    out = rl.generate_all(call, prompts, 2, pad, eos, cap, 4, torch, "cpu")
    assert [len(group) for group in out] == [2] * 5
    expect = [([4, 4, 4, 4], "length"), ([6, 6, 9], "stop"), ([3, 3, 3, 3], "length"), ([4, 5, 9], "stop"), ([6, 6, 6, 6], "length")]
    for group, (ids, finish) in zip(out, expect):
        assert all(c["ids"] == ids and c["finish"] == finish and c["logp"] == [0.0, 1.0, 2.0, 3.0][:len(ids)] for c in group)
    assert [len(ids) for ids, _ in seen] == [4, 4, 2]                          # sub-batches of 4 sequences
    assert [len(seen[0][0][0]), len(seen[1][0][0]), len(seen[2][0][0])] == [5, 2, 1]     # longest prompts first, padded to the longest of each sub-batch
    for ids, mask in seen:
        for row, m in zip(ids, mask):
            assert m == sorted(m) and all(t == pad for t, keep in zip(row, m) if not keep)   # left padding, pad tokens masked
    with pytest.raises(rl.PackError, match="no tokens"):
        rl.generate_all(call, [[]], 1, pad, eos, cap, 4, torch, "cpu")


def test_generate_all_token_budget_caps_rows_by_the_longest_prompt_of_each_sub_batch(rl):
    torch = pytest.importorskip("torch")
    pad, eos, cap = 0, 9, 2
    prompts = [[1] * 6, [2] * 4, [3] * 4, [4] * 2, [5] * 2, [6] * 2, [7] * 1]
    seen = []

    def call(ids, mask):
        seen.append((ids.shape[0], ids.shape[1]))
        return torch.cat([ids, torch.full((ids.shape[0], cap), eos)], dim=1), None

    out = rl.generate_all(call, prompts, 1, pad, eos, cap, 100, torch, "cpu", token_budget=8)
    assert seen == [(1, 6), (2, 4), (4, 2)]                       # 8 // 6 = 1 row, 8 // 4 = 2 rows, 8 // 2 = 4 rows (3 are left, then 1 fits)
    assert [g[0]["finish"] for g in out] == ["stop"] * 7 and [len(g) for g in out] == [1] * 7
    seen.clear()
    rl.generate_all(call, prompts, 1, pad, eos, cap, 3, torch, "cpu", token_budget=10**9)
    assert [rows for rows, _ in seen] == [3, 3, 1]                # the sequence cap still applies
    seen.clear()
    rl.generate_all(call, [[1] * 50], 1, pad, eos, cap, 4, torch, "cpu", token_budget=8)
    assert seen == [(1, 50)]                                      # a prompt longer than the budget still runs alone


# ---------------------------------------------------------------- torch: the real Policy on a tiny model, and real checkpoints on disk
class ByteTokenizer:
    """Stands in for the HF tokenizer: the UTF-8 bytes are the tokens, 250 is <|im_end|> and 251 the padding token."""
    eos_token_id, pad_token_id = 250, 251

    def encode(self, text, add_special_tokens=False):
        return [250] if text == "<|im_end|>" else list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return bytes(i for i in ids if i < 250).decode("utf-8", "replace")


@pytest.fixture
def stub_transformers(monkeypatch):
    """rl_grpo.Policy imports transformers.GenerationConfig; a stand-in keeps these tests free of transformers and peft."""
    module = types.ModuleType("transformers")
    module.GenerationConfig = lambda **fields: types.SimpleNamespace(**fields)
    monkeypatch.setitem(sys.modules, "transformers", module)


def toy_policy(rl, torch, args, vocab=256):
    """The real rl_grpo.Policy over a tiny torch model: its update, AdamW, checkpoints and generate path are the production code."""
    model, emb, mix, head = tiny(torch, rl, vocab=vocab)
    return rl.Policy(model, ByteTokenizer(), args, torch.bfloat16, device="cpu"), emb, mix, head


def toy_weights(torch, path):
    """The tensors the toy's save_pretrained wrote (torch.save into a file named like a safetensors file, which torch.load would send to safetensors)."""
    with Path(path).open("rb") as stream:
        return torch.load(stream, weights_only=True)


def toy_items(policy, scale=1.0):
    """SAMPLES as update items of `policy` (their prompts go in its cache, so no tokenizer is involved)."""
    rows = [dict(id=f"toy-{i}") for i in range(len(SAMPLES))]
    for row, (prompt, _, _) in zip(rows, SAMPLES):
        policy.prompts[row["id"]] = prompt
    return [(row, completion, weight * scale) for row, (_, completion, weight) in zip(rows, SAMPLES)]


def test_policy_train_clips_the_gradient_and_takes_one_adamw_step(rl, torch, stub_transformers, tmp_path):
    policy, emb, mix, head = toy_policy(rl, torch, make_args(rl, tmp_path, "--max-tokens-per-batch", "16"))
    policy.autocast = contextlib.nullcontext                                  # an fp32 trunk, so the numpy hand gradient is exact
    items = toy_items(policy, scale=8.0)
    loss, grad, _ = hand(emb, mix, head, [(policy.prompts[row["id"]], ids, w) for row, ids, w in items])
    norm = float(np.linalg.norm(grad))
    assert norm > 2                                                           # the fixture: the clip at 1.0 bites
    before = policy.params[0].detach().clone()
    result = policy.train(items, 0.01)
    assert result["micro_batches"] == 2 and result["loss"] == pytest.approx(loss, rel=1e-5)
    assert result["grad_norm"] == pytest.approx(norm, rel=1e-4)               # the norm BEFORE clipping is what is reported
    clipped = grad / (norm + 1e-6)                                            # clip_grad_norm_(parameters, 1.0)
    state = policy.optimizer.state[policy.params[0]]
    np.testing.assert_allclose(state["exp_avg"].double().numpy(), 0.1 * clipped, rtol=1e-4, atol=1e-9)           # AdamW saw the clipped gradient
    np.testing.assert_allclose(state["exp_avg_sq"].double().numpy(), 0.001 * clipped ** 2, rtol=1e-3, atol=1e-12)
    # one step: the bias-corrected moments are the gradient itself, so every weight moved by lr * g / (|g| + eps); no weight decay
    after = policy.params[0].detach()
    assert not torch.equal(after, before)
    np.testing.assert_allclose(after.double().numpy(), before.double().numpy() - 0.01 * clipped / (np.abs(clipped) + 1e-8), rtol=1e-4, atol=1e-6)
    assert policy.optimizer.param_groups[0]["lr"] == 0.01 and policy.optimizer.param_groups[0]["weight_decay"] == 0.0
    # a gradient inside the clip is left alone
    small, emb, mix, head = toy_policy(rl, torch, make_args(rl, tmp_path))
    small.autocast = contextlib.nullcontext
    items = toy_items(small, scale=0.05)
    _, grad, _ = hand(emb, mix, head, [(small.prompts[row["id"]], ids, w) for row, ids, w in items])
    assert np.linalg.norm(grad) < 1
    small.train(items, 0.01)
    np.testing.assert_allclose(small.optimizer.state[small.params[0]]["exp_avg"].double().numpy(), 0.1 * grad, rtol=1e-4, atol=1e-9)


def test_policy_checkpoints_restore_the_optimizer_state_and_the_python_numpy_and_torch_rng(rl, torch, stub_transformers, tmp_path):
    args = make_args(rl, tmp_path)
    policy, *_ = toy_policy(rl, torch, args)
    items = toy_items(policy, scale=8.0)
    for _ in range(2):
        policy.train(items, 0.01)                                              # AdamW now holds two steps of moments
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)                                                      # the RNG states a checkpoint must carry
    want = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    policy.save(dict(global_step=2, adapters=[]))
    path, _, _ = rl.sft.load_checkpoint(args.out)
    assert (path / "training.pt").is_file() and (path / "adapter_model.safetensors").is_file()
    fresh, *_ = toy_policy(rl, torch, args)                                    # a new process image: the saved weights, an empty optimizer, other RNGs
    fresh.model.load_state_dict(toy_weights(torch, path / "adapter_model.safetensors"), strict=False)
    random.seed(99)
    np.random.seed(98)
    torch.manual_seed(97)
    assert not fresh.optimizer.state and torch.equal(fresh.params[0], policy.params[0])
    fresh.load_training(path)
    moments = lambda p: p.optimizer.state[p.params[0]]
    for key in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(moments(fresh)[key], moments(policy)[key]), key
    assert float(moments(fresh)["step"]) == 2 and float(moments(fresh)["exp_avg"].abs().sum()) > 0
    assert random.getstate() == want["python"] and torch.equal(torch.get_rng_state(), want["torch"])
    got = np.random.get_state()
    assert got[0] == want["numpy"][0] and np.array_equal(got[1], want["numpy"][1]) and got[2:] == want["numpy"][2:]
    # so the next update is bit-identical; an optimizer restarted from empty moments would not give it
    policy.train(items, 0.01)
    fresh.train(toy_items(fresh, scale=8.0), 0.01)
    assert torch.equal(policy.params[0], fresh.params[0])


def test_policy_generate_passes_the_clean_sampling_config_and_seeds_torch(rl, torch, stub_transformers, tmp_path):
    policy, *_ = toy_policy(rl, torch, make_args(rl, tmp_path, "--temperature", "0.7", "--max-new-tokens", "6", "--gen-batch", "4"))
    base = policy.model.get_base_model()
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    first = policy.generate(prompts, 2, seed=5)
    given = base.generate_calls[0]
    assert given == dict(do_sample=True, temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0, typical_p=1.0, no_repeat_ngram_size=0,
                         epsilon_cutoff=0.0, eta_cutoff=0.0, num_beams=1, use_cache=True, max_new_tokens=6, eos_token_id=[policy.eos],
                         pad_token_id=policy.pad)                              # the effective sampling arguments, every one explicit
    assert policy.eos == 250 and policy.pad == 251
    config = base.generation_config                                           # and the model's own config is replaced by the same clean one
    assert (config.do_sample, config.temperature, config.top_k, config.top_p, config.eos_token_id) == (True, 0.7, 0, 1.0, [250])
    assert [len(group) for group in first] == [2, 2, 2] and all(len(c["ids"]) <= 6 for group in first for c in group)
    # the torch seed is set by every generate call: the same seed gives the same rollouts, another seed does not, and the state before the call is irrelevant
    assert policy.generate(prompts, 2, seed=5) == first and policy.generate(prompts, 2, seed=6) != first
    torch.manual_seed(1234)
    torch.rand(3)
    assert policy.generate(prompts, 2, seed=5) == first
    assert policy.rollout([dict(id="a", rl_prompt="x"), dict(id="b", rl_prompt="yz")], 5) == policy.rollout([dict(id="a", rl_prompt="x"), dict(id="b", rl_prompt="yz")], 5)


def test_policy_snapshot_holds_cpu_copies_of_the_weights_and_the_adamw_moments(rl, torch, stub_transformers, tmp_path):
    policy, *_ = toy_policy(rl, torch, make_args(rl, tmp_path))
    before = policy.snapshot()
    assert before["exp_avg"] == [None] and before["exp_avg_sq"] == [None] and before["step"] == [0.0]
    policy.train(toy_items(policy, scale=8.0), 0.01)
    snap = policy.snapshot()
    state = policy.optimizer.state[policy.params[0]]
    assert torch.equal(snap["parameters"][0], policy.params[0].detach().flatten()) and snap["parameters"][0].device.type == "cpu"
    assert torch.equal(snap["exp_avg"][0], state["exp_avg"].flatten()) and torch.equal(snap["exp_avg_sq"][0], state["exp_avg_sq"].flatten())
    assert snap["step"] == [1.0] and snap["parameters"][0].shape == (25,) and not torch.equal(snap["parameters"][0], before["parameters"][0])
    policy.params[0].data.zero_()
    assert float(snap["parameters"][0].abs().sum()) > 0                         # a copy, not a view


def test_snapshot_distances_are_relative_norms_per_component(rl, torch):
    a = dict(parameters=[torch.tensor([3.0, 4.0])], exp_avg=[torch.tensor([1.0, 0.0])], exp_avg_sq=[None], step=[2.0])
    b = dict(parameters=[torch.tensor([3.0, 4.0])], exp_avg=[torch.tensor([1.0, 2.0])], exp_avg_sq=[torch.tensor([0.0, 2.0])], step=[2.0])
    got = rl.snapshot_distances(a, b)
    assert got == dict(parameters=0.0, exp_avg=pytest.approx(2.0), exp_avg_sq=pytest.approx(2.0), step=0.0)      # a missing tensor reads as zeros
    assert rl.snapshot_distances(a, a) == dict(parameters=0.0, exp_avg=0.0, exp_avg_sq=0.0, step=0.0)
    other = dict(a, parameters=[torch.tensor([3.0, 4.5])], step=[3.0])
    assert rl.snapshot_distances(a, other)["parameters"] == pytest.approx(0.1) and rl.snapshot_distances(a, other)["step"] == 1.0   # 0.5 / ||a|| = 5


def test_the_smoke_state_comparison_passes_inside_the_tolerance_and_is_informational_outside(rl):
    near = dict(parameters=0.0, exp_avg=1e-7, exp_avg_sq=1e-6, step=0.0)                                         # the tolerance itself is inside
    checks, info = rl.smoke_checks(smoke_state(), smoke_state(), near)
    assert [v for k, v in checks.items() if "AdamW" in k] == [True] and not [k for k in info if "AdamW" in k]
    for far in (dict(near, exp_avg_sq=2e-6), dict(near, step=1.0), dict(near, parameters=float("nan"))):
        checks, info = rl.smoke_checks(smoke_state(), smoke_state(), far)
        assert not [k for k in checks if "AdamW" in k] and all(checks.values())                               # never a FAIL: GPU rollouts may differ
        assert [v for k, v in info.items() if "AdamW" in k] == [far]
    checks, info = rl.smoke_checks(smoke_state(), smoke_state())
    assert not [k for k in list(checks) + list(info) if "AdamW" in k]                                         # not measured: neither line
    assert "step 3 rollouts bitwise equal (GPU nondeterminism can break this)" in info                          # the rollout line stays informational


def fixed_registry():
    """Every text a scripted completion can decode to, in a fixed order: the fake token ids (1000 + position) are then the same in every
    DiskPolicy, so a resumed run feeds the model the very tokens an uninterrupted one does."""
    texts = []
    for row in make_rows(12) + make_rows(3, "val", offset=8):
        for kind in ("exact", "near", "const", "plain", "bad", "open"):
            if text_for(kind, row) not in texts:
                texts.append(text_for(kind, row))
    return texts


class DiskPolicy(FakePolicy):
    """FakePolicy's scripted rollouts and scores, with a real rl.Policy over a tiny torch model behind the update, AdamW and the checkpoints:
    Policy.train, Policy.save and Policy.load_training run for real (torch.save, the weights_only load, real optimizer state), so a resume
    restores real weights, moments and RNG states from disk."""

    def __init__(self, rl, torch, **kwargs):
        super().__init__(**kwargs)
        self.rl, self.torch, self.registry = rl, torch, fixed_registry()
        self.real = self.out = self.loaded = self.metric_lines = None
        self.fail = False

    def attach(self, args, existing=None):
        """What load_policy does for a process image: build the model, and on a resume load the adapter weights the checkpoint holds."""
        self.out = args.out
        self.real = toy_policy(self.rl, self.torch, args, vocab=1100)[0]
        if existing:
            self.real.model.load_state_dict(toy_weights(self.torch, existing[0] / "adapter_model.safetensors"), strict=False)
        return self

    def train(self, items, lr):
        super().train(items, lr)
        return self.real.train(items, lr)

    def save(self, state):
        super().save(state)
        self.real.save(state)

    def load_training(self, directory):
        self.loaded = Path(directory)
        self.real.load_training(directory)

    def describe(self):
        return dict(name="fake", memory_bytes=0)

    def mark_initial(self):
        self.real.mark_initial()

    def delta_norm(self):
        return self.real.delta_norm()

    def snapshot(self):
        return self.real.snapshot()

    def rollout(self, rows, seed):
        if self.metric_lines is None and (self.out / "metrics.jsonl").exists():
            self.metric_lines = (self.out / "metrics.jsonl").read_text().splitlines()      # what the log holds when the first step starts
        if self.fail:
            raise RuntimeError("the first step fails")
        return super().rollout(rows, seed)


def disk_rows(rl, tmp_path, rows, val):
    """Write the rows and read them back the way main does, so each carries its rl_prompt (the real Policy tokenizes it)."""
    return [rl.load_rows(write_rows(tmp_path / f"{name}.jsonl", content), name) for name, content in (("train", rows), ("val", val))]


def train_on_disk(rl, torch, tmp_path, *flags, policy=None, rows=None, val=None):
    rows, val = disk_rows(rl, tmp_path, rows or make_rows(12), val or make_rows(3, "val", offset=8))
    policy = policy or DiskPolicy(rl, torch)

    def loader(args, existing=None):
        return policy.attach(args, existing), dict(revision="rev", kernels={}, dtype="bf16", adapter_modules=3, trainable_parameters=7)

    return rl.run_training(loop_args(rl, tmp_path, "--lr", "0.01", *flags), rows, val, time.monotonic(), loader=loader), policy


def test_run_training_commits_a_first_checkpoint_and_resumes_from_disk(rl, torch, stub_transformers, tmp_path):
    boom = DiskPolicy(rl, torch)
    boom.fail = True
    with pytest.raises(RuntimeError):
        train_on_disk(rl, torch, tmp_path, "--steps", "4", policy=boom)
    path, state, _ = rl.sft.load_checkpoint(tmp_path / "out")                  # a failing first step still leaves a resumable run
    assert state["step"] == 0 and state["cursor"] == 0 and state["identity"]["options"]["seed"] == 3
    done, first = train_on_disk(rl, torch, tmp_path, "--steps", "2")
    assert done["step"] == 2 and done["status"] == "complete" and first.loaded == path
    # a step that was logged but never committed must not survive the resume
    with (tmp_path / "out" / "metrics.jsonl").open("a") as stream:
        stream.write('{"step": 3}\n')
    done, second = train_on_disk(rl, torch, tmp_path, "--steps", "4")
    assert [json.loads(line)["step"] for line in second.metric_lines] == [1, 2]
    assert [m["step"] for m in done["metrics"]] == [1, 2, 3, 4] and done["status"] == "complete"
    assert [json.loads(line)["step"] for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3, 4]
    assert second.calls[0][0] == grpo.rollout_seed(3, 3)                       # step 3 follows step 2 on the same schedule
    whole, uninterrupted = train_on_disk(rl, torch, tmp_path, "--out", str(tmp_path / "whole"), "--steps", "4")
    assert first.calls + second.calls == uninterrupted.calls and whole["cursor"] == done["cursor"]   # same drugs, views and seeds
    # the resumed run's weights, AdamW moments and step counts are the uninterrupted run's, bit for bit: the checkpoint round-trips them
    assert rl.snapshot_distances(second.real.snapshot(), uninterrupted.real.snapshot()) == dict(parameters=0.0, exp_avg=0.0, exp_avg_sq=0.0, step=0.0)
    assert uninterrupted.real.snapshot()["step"] == [4.0] and [m["loss"] for m in done["metrics"]] == [m["loss"] for m in whole["metrics"]]
    untrained = toy_policy(rl, torch, loop_args(rl, tmp_path), vocab=1100)[0].snapshot()
    assert rl.snapshot_distances(untrained, uninterrupted.real.snapshot())["parameters"] > 1e-3          # and the updates did move the weights
    # the committed state is the whole story: an option change refuses, a longer run continues
    with pytest.raises(rl.PackError, match="identity changed"):
        train_on_disk(rl, torch, tmp_path, "--steps", "6", "--lr", "2e-5")
    with pytest.raises(rl.PackError, match="identity changed"):
        train_on_disk(rl, torch, tmp_path, "--steps", "6", rows=make_rows(11))        # other Train data
    assert train_on_disk(rl, torch, tmp_path, "--steps", "5", "--gen-batch", "16")[0]["step"] == 5    # performance settings may change


def test_a_resume_without_the_saved_optimizer_state_would_change_the_weights(rl, torch, stub_transformers, tmp_path, monkeypatch):
    """The control for the test above: the equality there is not vacuous. Losing the moments on resume gives other weights."""
    train_on_disk(rl, torch, tmp_path, "--steps", "2")
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "load_state_dict", lambda self, state: None)
        _, broken = train_on_disk(rl, torch, tmp_path, "--steps", "4")
    _, whole = train_on_disk(rl, torch, tmp_path, "--out", str(tmp_path / "whole"), "--steps", "4")
    assert rl.snapshot_distances(broken.real.snapshot(), whole.real.snapshot())["parameters"] > 1e-4


def test_a_rule_stop_in_run_training_is_final_on_disk(rl, torch, stub_transformers, tmp_path):
    done, _ = train_on_disk(rl, torch, tmp_path, "--steps", "9", policy=DiskPolicy(rl, torch, behave=lambda seed, row, i: ("const", "stop")))
    assert done["status"] == "stopped_collapse"
    assert done["metrics"][0]["train_copy_share"] == 1.0         # the constant answer is the reference of a Train drug, not of a Val drug
    again = DiskPolicy(rl, torch)
    done, _ = train_on_disk(rl, torch, tmp_path, "--steps", "9", policy=again)
    assert done["status"] == "stopped_collapse" and again.calls == [] and done["step"] == 3


def test_run_training_refuses_a_changed_objective_and_only_warns_about_a_changed_trainer(rl, torch, stub_transformers, tmp_path, monkeypatch, capsys):
    done, _ = train_on_disk(rl, torch, tmp_path, "--steps", "2")
    real = rl.trainer_hashes()
    assert done["trainer_sha256"] == real and done["identity"]["objective_sha256"] == rl.objective_digest()
    saved = rl.sft.load_checkpoint(tmp_path / "out")[1]
    assert saved["trainer_sha256"] == real and saved["identity"]["objective_sha256"] == rl.objective_digest()       # in the committed state
    with monkeypatch.context() as patch:
        patch.setattr(rl, "objective_digest", lambda: "f" * 64)                 # e.g. the scorer was edited between the two sessions
        with pytest.raises(rl.PackError, match="objective changed: use a new output directory"):
            train_on_disk(rl, torch, tmp_path, "--steps", "4")
    assert rl.sft.load_checkpoint(tmp_path / "out")[1]["step"] == 2             # the refused run wrote nothing
    capsys.readouterr()
    with monkeypatch.context() as patch:
        patch.setattr(rl, "trainer_hashes", lambda: dict(real, **{"rl_grpo.py": "1" * 64}))
        done, _ = train_on_disk(rl, torch, tmp_path, "--steps", "4")
    out = capsys.readouterr().out
    assert done["step"] == 4 and "WARNING: The trainer source changed" in out and "rl_grpo.py" in out and "sft_lora.py" not in out
    assert done["trainer_sha256"]["rl_grpo.py"] == "1" * 64                     # the state records this session's files


def run_smoke_on_toy(rl, torch, tmp_path):
    train, val = disk_rows(rl, tmp_path, make_rows(12), make_rows(3, "val", offset=8))

    def loader(run_args, existing=None):                                       # a new policy for every process image, as the gate rebuilds it
        return (DiskPolicy(rl, torch).attach(run_args, existing),
                dict(revision="rev", kernels={}, dtype="bf16", adapter_modules=3, trainable_parameters=7))

    return rl.run_smoke(loop_args(rl, tmp_path, "--steps", "3", "--lr", "0.01"), train, val, time.monotonic(), loader=loader)


def test_the_smoke_gate_compares_parameters_and_adamw_state_after_the_resumed_update(rl, torch, stub_transformers, tmp_path, capsys, monkeypatch):
    assert run_smoke_on_toy(rl, torch, tmp_path) is True
    out = capsys.readouterr().out
    assert "PASS step 3 parameters and AdamW state equal the uninterrupted run (relative norm of difference <= 1e-06)" in out and "FAIL" not in out
    assert "INFO step 3 rollouts bitwise equal" in out and "True" in next(l for l in out.splitlines() if "rollouts bitwise" in l)
    # lose the optimizer state on resume: the third update differs, and the gate prints the numbers instead of a PASS (never a FAIL)
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", lambda self, state: None)
    assert run_smoke_on_toy(rl, torch, tmp_path) is True
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("INFO step 3 parameters and AdamW state differ"))
    assert "PASS step 3 parameters and AdamW state" not in out and "FAIL" not in out
    numbers = {name: float(value) for name, value in re.findall(r"'(\w+)': ([0-9.e+-]+)", line)}
    assert numbers["parameters"] > 1e-4 and numbers["exp_avg"] > 1e-3 and numbers["exp_avg_sq"] > 1e-3

