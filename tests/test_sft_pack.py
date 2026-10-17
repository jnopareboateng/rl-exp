"""CPU tests cannot prove GPU, kernel or memory behaviour.

Real model loading, numerical loss parity, training, CUDA kernels, bf16/fp16,
optimizer/RNG restoration and merged-model loading in vLLM remain UNTESTED.
These tests need no torch. They use synthetic text and a character tokenizer.
The loss and gradient tests run the real functions on numpy stand-ins for torch.
"""

import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / "gpu"


@pytest.fixture
def pack(monkeypatch):
    monkeypatch.syspath_prepend(str(GPU))
    spec = importlib.util.spec_from_file_location("sft_lora", GPU / "sft_lora.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sft_lora", module)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(map(ord, text))


class MergingTokenizer:
    """Like BPE, it merges two newlines into one token: encode(a + b) can differ from encode(a) + encode(b)."""
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids, i = [], 0
        while i < len(text):
            merged = text[i:i + 2] == "\n\n"
            ids.append(1000 if merged else ord(text[i]))
            i += 2 if merged else 1
        return ids


def fixture_row(index=0, prompt="fixture<think>\n", completion="\n</think>\n\nC1CC1<|im_end|>"):
    return dict(id=f"row-{index}", prompt=prompt, completion=completion)


def make_args(pack, tmp_path, *extra):
    return pack.config_from_args(["--arm", "A2", "--train", str(tmp_path / "train.jsonl"),
                                  "--val", str(tmp_path / "val.jsonl"), *extra])


def test_exact_completion_mask_and_end_marker(pack):
    row, tokenizer = fixture_row(), MergingTokenizer()
    prepared, counts = pack.prepare_rows([row], tokenizer, 1024, "drop")
    item = prepared[0]
    prompt, completion = tokenizer.encode(row["prompt"]), tokenizer.encode(row["completion"])
    # The two newlines at the A2 boundary merge when the joined text is encoded. Training must not do that.
    assert tokenizer.encode(row["prompt"] + row["completion"]) != prompt + completion
    assert item["input_ids"] == prompt + completion
    assert item["labels"][:len(prompt)] == [-100] * len(prompt)
    assert item["labels"][len(prompt):] == completion
    assert item["labels"][-len(pack.END):] == tokenizer.encode(pack.END)
    assert counts["completion_tokens"] == len(completion)


@pytest.mark.parametrize("completion,alone", [("\n</think>\n\nC1CC1", False), ("\n</think>\n\nC1CC1<|im_end|>", True)])
def test_a_completion_must_end_with_the_end_marker_tokens(pack, completion, alone):
    class MarkerOnlyAlone(CharacterTokenizer):
        """The marker is one special id by itself but plain text inside a longer string."""
        def encode(self, text, add_special_tokens=False):
            return [0] if text == pack.END else super().encode(text, add_special_tokens)
    # Without the marker, or with it lost in tokenization, the end of the sequence is not supervised.
    tokenizer = MarkerOnlyAlone() if alone else CharacterTokenizer()
    with pytest.raises(ValueError, match="end marker"):
        pack.prepare_rows([fixture_row(completion=completion)], tokenizer, 1024, "drop")


def test_no_template_or_implicit_special_tokens(pack):
    class BoundaryTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            assert text in (pack.END, row["prompt"], row["completion"])
            return super().encode(text, add_special_tokens)
    row = fixture_row()
    prepared, _ = pack.prepare_rows([row], BoundaryTokenizer(), 1024, "drop")
    assert len(prepared[0]["input_ids"]) == len(row["prompt"] + row["completion"])


def test_clip_preserves_entire_completion_and_drop_counts(pack):
    tokenizer = CharacterTokenizer()
    generation = "<|im_start|>assistant\n<think>\n"
    # The head and the tail of this prompt differ, so clipping the wrong end shows.
    long_prompt = "<|im_start|>system\n" + "s" * 60 + "<|im_end|>\n" + generation
    rows = [fixture_row(0, long_prompt, "a" + pack.END),
            fixture_row(1, "pp", "a" * 80 + pack.END), fixture_row(2, "p", pack.END)]
    prepared, counts = pack.prepare_rows(rows, tokenizer, 60, "clip")
    keep = 60 - 1 - len(pack.END)
    assert counts == dict(rows=3, kept_rows=2, clipped_rows=1, dropped_rows=1,
                         tokens=60 + 1 + len(pack.END), completion_tokens=1 + 2 * len(pack.END),
                         removed_prompt_tokens=len(long_prompt) - keep)
    assert prepared[0]["labels"][-1 - len(pack.END):] == tokenizer.encode("a" + pack.END)
    # The rightmost prompt tokens stay, so the generation prefix survives and the system turn is what is lost.
    assert keep > len(generation)
    assert prepared[0]["input_ids"] == tokenizer.encode(long_prompt[-keep:] + "a" + pack.END)
    assert prepared[0]["input_ids"][keep - len(generation):keep] == tokenizer.encode(generation)
    dropped, drop_counts = pack.prepare_rows(rows, tokenizer, 60, "drop")
    assert len(dropped) == 1 and drop_counts["dropped_rows"] == 2
    assert drop_counts["clipped_rows"] == 0
    assert pack.fit_lengths(2, 22, 24, "clip") == (2, "kept")
    assert pack.fit_lengths(1, 24, 24, "clip") == (0, "dropped")


@pytest.mark.parametrize("seed", [0, 5, 33])
def test_budgets_include_padding_and_effective_row_count(pack, seed):
    lengths = [random.Random(seed + i).randint(2, 64) for i in range(67)]
    plan = pack.epoch_batches(lengths, 100, 9, seed, 0)
    flattened = [i for update in plan for batch in update for i in batch]
    assert sorted(flattened) == list(range(len(lengths)))
    assert all(sum(map(len, update)) == 9 for update in plan[:-1])
    assert sum(map(len, plan[-1])) == len(lengths) % 9
    assert all(max(lengths[i] for i in batch) * len(batch) <= 100 for update in plan for batch in update)
    with pytest.raises(ValueError):
        pack.token_batches([0], [101], 100)


@pytest.mark.parametrize("seed", [0, 7])
def test_updates_are_shuffled_each_epoch_not_shortest_first(pack, seed):
    lengths = list(range(10, 90))
    orders = []
    for epoch in range(3):
        plan = pack.epoch_batches(lengths, 10_000, 10, seed, epoch)
        orders.append([max(lengths[i] for batch in update for i in batch) for update in plan])
    # Rows are sorted by length inside the epoch, so unshuffled updates would run shortest to longest every time.
    assert all(order != sorted(order) for order in orders)
    assert len({tuple(order) for order in orders}) == len(orders)


def test_partial_update_is_not_always_the_longest_rows(pack):
    lengths = list(range(1, 28))
    tails = [set(i for batch in pack.epoch_batches(lengths, 100, 5, 0, epoch)[-1] for i in batch) for epoch in range(6)]
    assert len({tuple(sorted(tail)) for tail in tails}) > 1
    assert any(tail != {25, 26} for tail in tails)


def test_native_precision_uses_capability_not_bf16_emulation(pack):
    assert pack.precision_for_capability((7, 5)) == "float16"
    assert pack.precision_for_capability((8, 0)) == "bfloat16"
    assert pack.precision_for_capability((9, 0)) == "bfloat16"
    with pytest.raises(ValueError):
        pack.precision_for_capability((6, 0))


def test_safe_diagnostics_keep_pack_messages_and_hide_unknown_data(pack, capsys):
    pack.print_failure("SFT", pack.PackError("Resume identity changed"))
    assert "Resume identity changed" in capsys.readouterr().err
    try:
        raise RuntimeError("synthetic private completion must stay hidden")
    except RuntimeError as error:
        pack.print_failure("SFT", error)
    output = capsys.readouterr().err
    assert "synthetic private completion" not in output
    assert "test_sft_pack.py" in output


def test_resume_at_every_optimizer_boundary_and_epoch(pack):
    lengths, batch_size, epochs = [12, 7, 21, 5, 30, 6, 15, 4, 17], 4, 3
    plans = [pack.epoch_batches(lengths, 35, batch_size, 42, epoch) for epoch in range(epochs)]
    expected = [batch for plan in plans for update in plan for batch in update]
    state = dict(epoch=0, next_step=0, global_step=0, history=[])
    prefix = []
    for epoch, plan in enumerate(plans):
        for update in plan:
            # A JSON round trip stands in for loss of all process-local state.
            restored = json.loads(json.dumps(state))
            remaining = []
            for e in range(restored["epoch"], epochs):
                current = pack.epoch_batches(lengths, 35, batch_size, 42, e)
                start = restored["next_step"] if e == restored["epoch"] else 0
                remaining.extend(batch for group in current[start:] for batch in group)
            assert prefix + remaining == expected
            prefix.extend(update)
            state = pack.advance_step(state, len(plan))
        # A kill after the last update must resume validation before the next epoch.
        assert state["epoch"] == epoch and state["next_step"] == len(plan)
        state = pack.finish_epoch(state, len(plan), {"epoch": epoch + 1})
        assert state["next_step"] == 0 and state["epoch"] == epoch + 1
    assert state["global_step"] == sum(map(len, plans))
    assert prefix == expected
    with pytest.raises(ValueError):
        pack.finish_epoch(dict(epoch=0, next_step=0, history=[]), 2, {})


class FakeTensor(np.ndarray):
    """The few torch.Tensor members that completion_loss touches."""
    device = "cpu"

    def float(self):
        return self.astype(np.float32)


class IdentityHeadModel:
    """The hidden state is a one-hot of the input id and the head keeps the largest logit on that id."""
    vocab, sharpness = 5, 5.0

    def get_base_model(self):
        return self

    def model(self, input_ids, attention_mask, use_cache, return_dict):
        return types.SimpleNamespace(last_hidden_state=np.eye(self.vocab)[input_ids].view(FakeTensor))

    def lm_head(self, states):
        return (self.sharpness * states).view(FakeTensor)


def fake_torch():
    def cross_entropy(logits, targets, reduction):
        assert reduction == "sum"
        shifted = logits - logits.max(-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        return -log_probs[np.arange(len(targets)), targets].sum()
    return types.SimpleNamespace(
        float32=np.float32, int64=np.int64, zeros=lambda shape, device, dtype: np.zeros(shape, dtype=dtype),
        tensor=lambda data, device: np.array(data),
        nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=cross_entropy)))


@pytest.mark.parametrize("training", [True, False])
def test_completion_loss_scores_the_next_token_not_a_copy_of_the_input(pack, monkeypatch, training):
    checkpoint = types.ModuleType("torch.utils.checkpoint")
    checkpoint.checkpoint = lambda function, *values, use_reentrant: function(*values)
    monkeypatch.setitem(sys.modules, "torch.utils.checkpoint", checkpoint)
    # Position 1 holds id 1 and must predict 1. Position 2 holds id 1 and must predict 2.
    batch = dict(input_ids=np.array([[3, 1, 1, 2]]), attention_mask=np.ones((1, 4), dtype=int),
                 labels=np.array([[-100, -100, 1, 2]]))
    loss, correct, count = pack.completion_loss(IdentityHeadModel(), batch, fake_torch(), training)
    miss = math.log(math.exp(5) + 4)
    # A hit costs miss - 5 under logits 5 * one-hot; a miss costs the full amount. A copy loss would score both as hits.
    assert count == 2
    assert float(loss) == pytest.approx((miss - 5) + miss, rel=1e-5)
    if not training:
        assert int(correct) == 1


def test_collate_pads_on_the_right_and_neither_supervises_nor_attends_to_padding(pack):
    rows = [dict(input_ids=[5, 6, 7], labels=[-100, 6, 7]),
            dict(input_ids=[8, 9, 10, 11, 12], labels=[-100, -100, 10, 11, 12])]
    batch = pack.collate(rows, 99, "cpu", fake_torch())
    assert batch["input_ids"].tolist() == [[5, 6, 7, 99, 99], [8, 9, 10, 11, 12]]
    assert batch["labels"].tolist() == [[-100, 6, 7, -100, -100], [-100, -100, 10, 11, 12]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


class SummedLoss:
    """A sum of per-token losses theta * x with its gradient (the sum of x) worked out by hand."""
    def __init__(self, value, gradient, sink):
        self.value, self.gradient, self.sink = value, gradient, sink

    def __truediv__(self, divisor):
        return SummedLoss(self.value / divisor, self.gradient / divisor, self.sink)

    def detach(self):
        return self

    def item(self):
        return self.value

    def backward(self):
        self.sink.append(self.gradient)


def test_update_gradient_divides_by_all_completion_tokens_not_each_micro_batch(pack):
    rows = [dict(labels=[-100, 10, 20]), dict(labels=[-100, 30]), dict(labels=[-100, 40, 50, 60])]
    scaler = types.SimpleNamespace(scale=lambda loss: loss)

    def gradient(micro_batches, value=None):
        sink = []
        def forward(batch):
            total = sum(x for row in batch for x in row["labels"] if x != -100)
            return SummedLoss(total if value is None else value, total, sink)
        step_loss, denominator = pack.accumulate_update(rows, micro_batches, forward, scaler)
        return sum(sink), step_loss, denominator
    combined = gradient([[0, 1, 2]])
    assert combined == (pytest.approx(35), 210, 6)
    # Uneven micro-batches must add up to the same gradient as one combined batch.
    assert gradient([[0], [1, 2]]) == combined
    assert gradient([[0, 1], [2]]) == combined
    with pytest.raises(ValueError, match="Non-finite"):
        gradient([[0], [1, 2]], value=float("nan"))


def lora_like_parameters(torch, scaler):
    """A random A and a zero B: weight decay alone cannot move B. The target is far from the output, so the gradient is large."""
    a, b = torch.nn.Parameter(torch.randn(2, 4)), torch.nn.Parameter(torch.zeros(3, 2))
    optimizer = torch.optim.AdamW([a, b], lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    x, y = torch.randn(5, 4), 100 * torch.randn(5, 3)
    scaler.scale(((x @ a.T @ b.T - y) ** 2).sum()).backward()
    return [a, b], optimizer, scheduler


def test_one_real_optimizer_step_moves_a_lora_like_parameter(pack):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    parameters, optimizer, scheduler = lora_like_parameters(torch, scaler)
    assert pack.step_optimizer(parameters, optimizer, scheduler, scaler, torch) is True
    # B starts at zero, so only a learning step can change it: AdamW's first step moves it by about the learning rate.
    assert parameters[1].abs().max().item() == pytest.approx(1e-2, rel=1e-2)
    assert scheduler.last_epoch == 1


def test_gradients_are_clipped_to_norm_one_before_the_step(pack):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    parameters, optimizer, scheduler = lora_like_parameters(torch, scaler)
    assert torch.linalg.vector_norm(parameters[1].grad) > 100
    pack.step_optimizer(parameters, optimizer, scheduler, scaler, torch)
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))
    assert norm.item() == pytest.approx(1.0, rel=1e-3)


def test_fp16_overflow_skips_the_step_for_a_retry_and_any_other_nonfinite_gradient_raises(pack):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    parameters, optimizer, scheduler = lora_like_parameters(torch, scaler)
    parameters[1].grad.fill_(float("inf"))
    assert pack.step_optimizer(parameters, optimizer, scheduler, scaler, torch) is False
    assert scaler.get_scale() == 512.0 and parameters[1].abs().max().item() == 0.0 and scheduler.last_epoch == 0
    plain = torch.amp.GradScaler("cpu", enabled=False)
    parameters, optimizer, scheduler = lora_like_parameters(torch, plain)
    parameters[1].grad.fill_(float("inf"))
    with pytest.raises(ValueError, match="Non-finite gradients"):
        pack.step_optimizer(parameters, optimizer, scheduler, plain, torch)
    assert parameters[1].abs().max().item() == 0.0


def test_best_epoch_is_the_lowest_val_loss_and_a_tie_keeps_the_earlier(pack):
    assert pack.is_best(1.0, None)
    assert pack.is_best(0.9, 1.0)
    assert not pack.is_best(1.1, 1.0)
    assert not pack.is_best(1.0, 1.0)


def test_failed_mid_epoch_checkpoint_keeps_last_commit(pack, tmp_path):
    state = dict(epoch=0, next_step=0, global_step=0, history=[])
    first = pack.commit_checkpoint(tmp_path, state, lambda path: (path / "weights").write_text("first"))
    before = (tmp_path / "checkpoint.json").read_bytes()
    next_state = pack.advance_step(state, 3)
    def fail(path):
        (path / "weights").write_text("partial")
        raise RuntimeError("synthetic failure mid-epoch")
    with pytest.raises(RuntimeError):
        pack.commit_checkpoint(tmp_path, next_state, fail)
    assert (tmp_path / "checkpoint.json").read_bytes() == before
    assert pack.load_checkpoint(tmp_path)[:2] == (first, state)
    assert (first / "weights").read_text() == "first"
    assert not list((tmp_path / "checkpoints").glob(".pending-*"))
    best = pack.commit_checkpoint(tmp_path, next_state, lambda path: (path / "weights").write_text("best"), True)
    last = pack.commit_checkpoint(tmp_path, pack.advance_step(next_state, 3), lambda path: (path / "weights").write_text("last"))
    assert best.exists() and last.exists() and not first.exists()
    assert pack.load_checkpoint(tmp_path)[2]["best"] == str(best.relative_to(tmp_path))


def test_failure_before_pointer_replace_keeps_previous_checkpoint(pack, tmp_path, monkeypatch):
    state = dict(global_step=0)
    previous = pack.commit_checkpoint(tmp_path, state, lambda path: (path / "weights").write_text("old"))
    original = pack.atomic_json
    def fail_pointer(path, value):
        if path.name == "checkpoint.json":
            raise OSError("synthetic pointer failure")
        original(path, value)
    monkeypatch.setattr(pack, "atomic_json", fail_pointer)
    with pytest.raises(OSError):
        pack.commit_checkpoint(tmp_path, dict(global_step=1), lambda path: (path / "weights").write_text("new"))
    assert pack.load_checkpoint(tmp_path)[0] == previous


def make_run(pack, tmp_path, losses=(1.2, 0.9, 1.0), revision="rev-a"):
    """A run directory as the trainer leaves it: each epoch's adapter is published inside that epoch's checkpoint write."""
    root = tmp_path / "run"
    state = dict(epoch=0, global_step=0, best_val_loss=None, epoch_adapters=[], identity=dict(revision=revision))
    for epoch, val_loss in enumerate(losses, 1):
        best = pack.is_best(val_loss, state["best_val_loss"])
        state = {**state, "epoch": epoch, "global_step": 10 * epoch,
                 "best_val_loss": val_loss if best else state["best_val_loss"]}
        def writer(path):
            (path / "adapter_model.safetensors").write_bytes(b"weights %d" % epoch)
            (path / "adapter_config.json").write_text("{}")
            (path / "training.pt").write_bytes(b"optimizer state")
            state["epoch_adapters"] = [*state["epoch_adapters"], pack.commit_epoch_adapter(root, epoch, path, revision)]
        pack.commit_checkpoint(root, state, writer, best)
    return root, state


def adapter_payload(path, weights):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(weights)
    (path / "adapter_config.json").write_text("{}")
    return path


@pytest.fixture
def fake_ml(monkeypatch):
    """Stand-ins for the libraries merge() imports. They stop at the config loader and record the revision it gets."""
    seen = []

    class Stop(Exception):
        pass

    def load_config(name, revision=None, trust_remote_code=False):
        seen.append(revision)
        raise Stop
    ns = types.SimpleNamespace
    libraries = dict(
        torch=ns(), safetensors=ns(safe_open=None),
        peft=ns(PeftConfig=ns(from_pretrained=lambda path: ns(base_model_name_or_path="synthetic/base", revision=None)), PeftModel=None),
        transformers=ns(AutoConfig=ns(from_pretrained=load_config), AutoTokenizer=None,
                        Qwen3_5ForCausalLM=None, Qwen3_5ForConditionalGeneration=None))
    for name, module in libraries.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("USE_HUB_KERNELS", "1")
    return seen, Stop


def test_every_epoch_adapter_is_kept_and_listed_while_the_run_best_is_unchanged(pack, tmp_path):
    import merge_lora
    root, state = make_run(pack, tmp_path)
    adapters = root / "adapters"
    assert sorted(path.name for path in adapters.iterdir()) == ["epoch-1", "epoch-2", "epoch-3"]
    for epoch, entry in enumerate(state["epoch_adapters"], 1):
        directory = adapters / f"epoch-{epoch}"
        assert sorted(path.name for path in directory.iterdir()) == sorted(pack.EPOCH_FILES)    # no optimizer state
        assert (directory / "adapter_model.safetensors").read_bytes() == b"weights %d" % epoch
        assert json.loads((directory / "revision.json").read_text()) == {"revision": "rev-a"}
        assert entry == dict(epoch=epoch, path=f"adapters/epoch-{epoch}",
                             sha256={name: pack.sha256(directory / name) for name in pack.EPOCH_FILES})
        # merge_lora takes such a path, pins its recorded revision and records its epoch.
        assert merge_lora.resolve_adapter(directory) == directory.resolve()
        assert merge_lora.adapter_revision(directory) == "rev-a" and merge_lora.adapter_epoch(directory, None) == epoch
    # Checkpoints keep the latest and the best only. The run's best is still the lowest Val loss.
    latest, saved, manifest = pack.load_checkpoint(root)
    assert len(list((root / "checkpoints").glob("step-*"))) == 2
    assert saved["epoch"] == 3 and saved["epoch_adapters"] == state["epoch_adapters"]
    best_state = json.loads((root / manifest["best"] / "state.json").read_text())
    assert best_state["epoch"] == 2 and merge_lora.resolve_adapter(root) == (root / manifest["best"]).resolve()
    assert merge_lora.adapter_epoch(root / manifest["best"], best_state) == 2


def test_epoch_adapter_publish_is_atomic_and_a_repeat_replaces_only_an_uncommitted_leftover(pack, tmp_path):
    root, source = tmp_path / "run", tmp_path / "payload"
    adapter_payload(source, b"first")
    (source / "adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        pack.commit_epoch_adapter(root, 1, source, "rev-a")
    assert not (root / "adapters/epoch-1").exists() and not list((root / "adapters").glob(".pending-*"))
    adapter_payload(source, b"first")
    first = pack.commit_epoch_adapter(root, 1, source, "rev-a")
    assert pack.commit_epoch_adapter(root, 1, source, "rev-a") == first            # an identical repeat is a no-op
    adapter_payload(source, b"replayed")                                          # a killed save left epoch 1 uncommitted; the replay differs
    again = pack.commit_epoch_adapter(root, 1, source, "rev-a")
    assert (root / "adapters/epoch-1/adapter_model.safetensors").read_bytes() == b"replayed"
    assert again["sha256"]["adapter_model.safetensors"] != first["sha256"]["adapter_model.safetensors"]
    assert [path.name for path in (root / "adapters").iterdir()] == ["epoch-1"]


def test_a_failed_swap_keeps_the_previous_epoch_copy(pack, tmp_path, monkeypatch):
    root, source = tmp_path / "run", adapter_payload(tmp_path / "payload", b"first")
    first = pack.commit_epoch_adapter(root, 1, source, "rev-a")
    adapter_payload(source, b"replayed")
    replace = os.replace

    def failing(src, dst, *args, **kwargs):
        if Path(dst).name == "epoch-1":
            raise OSError("synthetic rename failure")
        return replace(src, dst, *args, **kwargs)
    monkeypatch.setattr(pack.os, "replace", failing)
    with pytest.raises(OSError, match="synthetic"):
        pack.commit_epoch_adapter(root, 1, source, "rev-a")
    published = root / "adapters/epoch-1"
    assert {file.name: pack.sha256(file) for file in published.iterdir()} == first["sha256"]
    assert (published / "adapter_model.safetensors").read_bytes() == b"first"
    assert [path.name for path in (root / "adapters").iterdir()] == ["epoch-1"]


def test_a_committed_epoch_adapter_cannot_be_rewritten(pack, tmp_path):
    root, state = make_run(pack, tmp_path)
    with pytest.raises(ValueError, match="committed"):
        pack.commit_epoch_adapter(root, 2, adapter_payload(tmp_path / "other", b"different"), "rev-a")
    assert (root / "adapters/epoch-2/adapter_model.safetensors").read_bytes() == b"weights 2"
    same = adapter_payload(tmp_path / "same", b"weights 2")
    assert pack.commit_epoch_adapter(root, 2, same, "rev-a") == state["epoch_adapters"][1]


def test_export_takes_only_a_committed_epoch_adapter_that_still_matches_its_hashes(pack, tmp_path):
    import merge_lora
    root, state = make_run(pack, tmp_path)
    # A killed save publishes the epoch before the checkpoint that lists it is committed.
    leftover = adapter_payload(root / "adapters/epoch-4", b"left over")
    (leftover / "revision.json").write_text('{"revision": "rev-a"}')
    with pytest.raises(ValueError, match="not in its run's committed list"):
        merge_lora.resolve_adapter(leftover)
    epoch = root / "adapters/epoch-2"
    (epoch / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from the hashes"):
        merge_lora.resolve_adapter(epoch)
    (epoch / "adapter_model.safetensors").write_bytes(b"weights 2")
    pinned = (epoch / "revision.json").read_bytes()
    (epoch / "revision.json").write_text('{"revision": "rev-b"}')       # the export would silently target another base revision
    with pytest.raises(ValueError, match="differs from the hashes"):
        merge_lora.resolve_adapter(epoch)
    (epoch / "revision.json").write_bytes(pinned)
    assert merge_lora.resolve_adapter(epoch) == epoch.resolve()
    (epoch / "notes.txt").write_text("not part of the commit")
    with pytest.raises(ValueError, match="differs from the hashes"):
        merge_lora.resolve_adapter(epoch)
    (epoch / "notes.txt").unlink()
    # A checkpoint directory is exportable only as the committed latest or best.
    manifest = pack.load_checkpoint(root)[2]
    assert merge_lora.resolve_adapter(root / manifest["latest"]) == (root / manifest["latest"]).resolve()
    stale = adapter_payload(root / "checkpoints/step-00000001-stale", b"pruned long ago")
    with pytest.raises(ValueError, match="committed latest or best"):
        merge_lora.resolve_adapter(stale)


def test_a_trainer_holding_the_run_lock_blocks_every_export_path_inside_the_run(pack, tmp_path, fake_ml):
    import merge_lora
    root, state = make_run(pack, tmp_path)
    epoch, export = root / "adapters/epoch-2", tmp_path / "export"
    inside = [root, epoch, root / pack.load_checkpoint(root)[2]["latest"]]
    with (root / "run.lock").open("a") as trainer:
        fcntl.flock(trainer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in inside:
            with pytest.raises(ValueError, match="trainer holds this run"):
                merge_lora.main(["--base", "synthetic/base", "--adapter", str(path), "--out", str(export)])
    seen, stop = fake_ml
    assert seen == []
    with pytest.raises(stop):                                  # without the trainer the same call reaches the model loader
        merge_lora.main(["--base", "synthetic/base", "--adapter", str(epoch), "--out", str(export)])
    assert seen == ["rev-a"]


def test_merge_passes_the_recorded_base_revision_to_the_loader_and_refuses_another(pack, tmp_path, fake_ml):
    import merge_lora
    root, state = make_run(pack, tmp_path)
    seen, stop = fake_ml
    export = tmp_path / "export"
    for adapter in (root / "adapters/epoch-2", root):              # an epoch adapter (revision.json) and the run's best (state.json)
        with pytest.raises(stop):
            merge_lora.merge("synthetic/base", adapter, export)
        with pytest.raises(stop):
            merge_lora.merge("synthetic/base", adapter, export, revision="rev-a")
    assert seen == ["rev-a"] * 4
    with pytest.raises(ValueError, match="--revision differs"):
        merge_lora.merge("synthetic/base", root / "adapters/epoch-2", export, revision="rev-b")
    with pytest.raises(ValueError, match="--revision differs"):
        merge_lora.merge("synthetic/base", root, export, revision="rev-b")
    assert seen == ["rev-a"] * 4                                  # refused before any model was loaded


def test_an_epoch_adapter_copied_out_of_its_run_needs_a_recorded_or_explicit_revision(pack, tmp_path, fake_ml):
    import merge_lora
    root, state = make_run(pack, tmp_path)
    copy = tmp_path / "elsewhere/epoch-2"
    shutil.copytree(root / "adapters/epoch-2", copy)
    assert merge_lora.adapter_revision(copy) == "rev-a"            # revision.json travels with the copy
    (copy / "revision.json").unlink()
    seen, stop = fake_ml
    with pytest.raises(ValueError, match="records no base revision"):
        merge_lora.merge("synthetic/base", copy, tmp_path / "export")
    with pytest.raises(stop):
        merge_lora.merge("synthetic/base", copy, tmp_path / "export", revision="rev-c")
    assert seen == ["rev-c"]
    # A folder that merely has the name adapters is not a run: no committed list and no run lock apply to it.
    plain = tmp_path / "adapters/epoch-9"
    shutil.copytree(root / "adapters/epoch-2", plain)
    assert merge_lora.resolve_adapter(plain) == plain
    with pytest.raises(stop):
        merge_lora.main(["--base", "synthetic/base", "--adapter", str(plain), "--out", str(tmp_path / "export")])
    assert seen == ["rev-c", "rev-a"] and not (tmp_path / "run.lock").exists()
    # Only this trainer's own directories are checked against the run. Another pack's kept adapter is a plain adapter.
    other = adapter_payload(root / "adapters/step-3", b"another pack")
    assert merge_lora.owning_run(other) is None and merge_lora.resolve_adapter(other) == other
    assert merge_lora.owning_run(root / "adapters/.pending-x") == root.resolve()


def test_export_reuse_requires_the_adapters_base_revision(pack, tmp_path, monkeypatch):
    import merge_lora
    root, state = make_run(pack, tmp_path)
    epoch, output = root / "adapters/epoch-2", tmp_path / "export"
    output.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer_config.json"):
        (output / name).write_bytes(b"synthetic model")

    def write_report(revision):
        pack.atomic_json(output / "merge_report.json", dict(
            base_model="synthetic/base", revision=revision, epoch=2,
            adapter_sha256={name: pack.sha256(epoch / name) for name in ("adapter_model.safetensors", "adapter_config.json")},
            file_sha256={file.name: pack.sha256(file) for file in output.iterdir() if file.name != "merge_report.json"},
            parameters=3, stored_parameters=3, safetensors_bytes=4, files=3))
    monkeypatch.setitem(sys.modules, "torch", None)             # the reuse branch runs before any ML import
    write_report("rev-b")                                       # exported when the default revision had moved on
    with pytest.raises(ValueError, match="different base revision"):
        merge_lora.merge("synthetic/base", epoch, output)
    write_report("rev-a")
    merge_lora.merge("synthetic/base", epoch, output)
    with pytest.raises(ValueError, match="--revision differs"):
        merge_lora.merge("synthetic/base", epoch, output, revision="rev-b")


def test_names_other_packs_import_keep_their_signatures(pack):
    import inspect
    expected = dict(print_failure=["action", "error"], atomic_json=["path", "value"], sync_directory=["path"],
                    commit_checkpoint=["root", "state", "writer", "is_best"], load_checkpoint=["root"],
                    checkpoint_path=["root", "name"], kernel_status=["allow_slow"], precision_for_capability=["capability"],
                    load_text_model=["model_name", "dtype", "revision"], adapter_targets=["model", "linear_type"],
                    runtime_versions=[], sha256=["path"])
    for name, parameters in expected.items():
        assert list(inspect.signature(getattr(pack, name)).parameters) == parameters, name
    assert issubclass(pack.PackError, ValueError)


def test_restart_removes_incomplete_payload_only_under_the_run_lock(pack, tmp_path, monkeypatch):
    args = make_args(pack, tmp_path, "--out", str(tmp_path / "run"))
    for path in (args.train, args.val):
        path.write_text(json.dumps(fixture_row()) + "\n")
    pending = [args.out / "checkpoints/.pending-killed", args.out / "adapters/.pending-killed",
               args.out / "adapters/.pending-replaced-killed"]
    for path in pending:
        path.mkdir(parents=True)
        (path / "partial.bin").write_bytes(b"unfinished")
    kept = args.out / "adapters/epoch-1"
    kept.mkdir()
    seen = []
    monkeypatch.setattr(pack, "run_training", lambda *values: seen.append([path.exists() for path in pending]))
    pack.main(["--arm", "A2", "--train", str(args.train), "--val", str(args.val), "--out", str(args.out)])
    assert seen == [[False, False, False]] and kept.exists()


def test_a_start_while_an_export_or_another_trainer_holds_the_run_lock_says_so(pack, tmp_path, monkeypatch):
    args = make_args(pack, tmp_path, "--out", str(tmp_path / "run"))
    for path in (args.train, args.val):
        path.write_text(json.dumps(fixture_row()) + "\n")
    args.out.mkdir()
    monkeypatch.setattr(pack, "run_training", lambda *values: pytest.fail("trained while the run was locked"))
    with (args.out / "run.lock").open("a") as export:
        fcntl.flock(export, fcntl.LOCK_SH | fcntl.LOCK_NB)         # an export holds the lock shared
        with pytest.raises(ValueError, match="holds this run"):
            pack.main(["--arm", "A2", "--train", str(args.train), "--val", str(args.val), "--out", str(args.out)])


@pytest.mark.parametrize("key,value", [("max_len", 1), ("max_tokens_per_batch", 0), ("batch_size", 0),
                                      ("rank", -1), ("epochs", 0), ("lr", float("nan")),
                                      ("warmup_ratio", 1), ("stop_after_seconds", float("inf")),
                                      ("save_steps", 0), ("seed", -1)])
def test_invalid_config(pack, tmp_path, key, value):
    config = vars(make_args(pack, tmp_path)).copy()
    config[key] = value
    with pytest.raises(ValueError):
        pack.validate_config(config)


@pytest.mark.parametrize("capability,allow_slow,stops", [
    ((8, 0), False, True), ((8, 6), False, True), ((9, 0), False, True), ((8, 0), True, False), ((7, 5), False, False)])
def test_missing_fast_kernel_stops_an_sm80_gpu_unless_allowed(pack, monkeypatch, capsys, capability, allow_slow, stops):
    # A real probe needs CUDA. A missing package takes the same failure path.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(get_device_capability=lambda index: capability)))
    for name in ("fla", "causal_conv1d"):
        monkeypatch.setitem(sys.modules, name, None)
    if stops:
        with pytest.raises(ValueError, match="--allow-slow-kernels"):
            pack.kernel_status(allow_slow)
    else:
        assert pack.kernel_status(allow_slow) == {"fla": "torch fallback", "causal_conv1d": "torch fallback"}
    assert capsys.readouterr().out.count("WARNING") == 2


def test_resume_identity_is_options_data_dtype_and_core_versions_only(pack, tmp_path, capsys):
    args = make_args(pack, tmp_path)
    args.train.write_text("synthetic train\n")
    args.val.write_text("synthetic val\n")
    versions = dict(python="3.12.1", torch="2.10.0", transformers="5.17.0", peft="0.21.2", numpy="2.0.0", triton="3.0.0")
    saved = dict(identity=pack.make_identity(args, "rev", "torch.bfloat16", versions), versions=versions)
    assert set(saved["identity"]["versions"]) == {"torch", "transformers", "peft"}
    assert not args.allow_slow_kernels and "allow_slow_kernels" not in saved["identity"]["options"]

    def resume(arguments=args, revision="rev", dtype="torch.bfloat16", **changed):
        current = {**versions, **changed}
        pack.check_resume(saved, pack.make_identity(arguments, revision, dtype, current), current)
        return capsys.readouterr().out
    assert resume() == ""
    # A different python, numpy or triton only warns and names what changed.
    warning = resume(python="3.12.2", numpy="2.1.0", triton=None)
    assert "WARNING" in warning and "python, numpy, triton" in warning and "torch" not in warning
    assert resume(make_args(pack, tmp_path, "--allow-slow-kernels")) == ""
    for name in ("torch", "transformers", "peft"):
        with pytest.raises(ValueError, match="Resume identity changed"):
            resume(**{name: "9.9.9"})
    with pytest.raises(ValueError, match="Resume identity changed"):
        resume(make_args(pack, tmp_path, "--lr", "0.1"))
    with pytest.raises(ValueError, match="Resume identity changed"):
        resume(revision="other")
    with pytest.raises(ValueError, match="Resume identity changed"):
        resume(dtype="torch.float16")
    args.val.write_text("changed val\n")
    with pytest.raises(ValueError, match="Resume identity changed"):
        resume()


def test_dry_run_cli_needs_only_stdlib(pack, tmp_path):
    row = fixture_row()
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({**row, "id": split}) + "\n")
    # -S excludes site-packages, including torch, transformers and peft.
    result = subprocess.run([sys.executable, "-S", str(GPU / "sft_lora.py"), "--arm", "A2",
                             "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
                             "--dry-run"], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry_run" and plan["max_len"] == 2048 and plan["max_tokens_per_batch"] == 8192
    assert plan["train"]["rows"] == 1 and plan["train"]["estimated_optimizer_steps_per_epoch"] == 1
    assert row["prompt"] not in result.stdout and row["completion"] not in result.stdout
    assert not (tmp_path / "out").exists()
    args = make_args(pack, tmp_path, "--max-len", "16", "--overlength", "clip")
    plan = pack.dry_plan(args, [fixture_row(0, "x" * 100, pack.END), fixture_row(1, "x", "x" * 100 + pack.END)], [row])
    assert plan["train"]["clipped_rows"] == 1 and plan["train"]["dropped_rows"] == 1
    a1 = make_args(pack, tmp_path, "--arm", "A1")
    assert (a1.max_len, a1.max_tokens_per_batch) == (8192, 16384)
    # A default budget never falls below an explicit length cap.
    assert make_args(pack, tmp_path, "--max-len", "20000").max_tokens_per_batch == 20000


def test_row_validation_never_exposes_text_or_reads_test_files(pack, tmp_path):
    path = tmp_path / "train.jsonl"
    row = fixture_row()
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError) as caught:
        pack.read_rows(path, "train")
    assert row["prompt"] not in str(caught.value)
    path.write_text('{"prompt": "synthetic private data", broken json\n')
    with pytest.raises(ValueError) as caught:
        pack.read_rows(path, "train")
    assert "synthetic private" not in str(caught.value)
    with pytest.raises(ValueError):
        pack.read_rows(tmp_path / "test_t.jsonl", "train")


def test_module_scope_covers_all_linears_but_excludes_vision_and_head(pack):
    class Linear:
        pass
    names = ["model.layers.0." + name for name in pack.PROJECTIONS]
    class Model:
        def named_modules(self):
            return [(name, Linear()) for name in [*names, "model.visual.linear", "lm_head"]]
    assert set(pack.adapter_targets(Model(), Linear)) == set(names)


def test_report_uses_durable_state_and_does_not_reuse_old_merge(pack, tmp_path, monkeypatch):
    import sft_report
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "steps.tsv").write_text("train\t0\t2\n")
    run = tmp_path / "run"
    state = dict(global_step=1, identity={"versions": {"python": "fake"}, "options": {"rank": 32, "lr": 2e-4},
                 "revision": "synthetic-revision", "dtype": "bf16", "train_sha256": "fake-train", "val_sha256": "fake-val"},
                 git_revision="fake-git", command=["gpu/sft_lora.py", "--arm", "A2"], history=[dict(epoch=1, val_loss=1.2)],
                 training_seconds=2, status="stopped_time_limit", epoch=1, train={"clipped_rows": 2}, val={},
                 adapter_modules=["synthetic.module"], trainable_parameters=4, tokens=20, completion_tokens=5,
                 gpu={}, kernels={}, peak_memory_bytes=9, best_val_loss=1.2)
    pack.commit_checkpoint(run, state, lambda path: None, True)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "merge_report.json").write_text('{"parameters": 44}')
    destination = tmp_path / "report.json"
    sft_report.report(scratch, "A2", run, merged, destination, 0)
    result = json.loads(destination.read_text())
    assert result["status"] == "partial" and result["merge"] is None
    assert result["training"]["tokens_per_second"] == 10
    assert "identity" not in result["training"]
    assert result["configuration"] == state["identity"] and result["git_revision"] == "fake-git"
    assert result["command"] == state["command"]
    (scratch / "steps.tsv").write_text("train\t0\t2\nmerge-reused\t0\t1\n")
    sft_report.report(scratch, "A2", run, merged, destination, 0)
    result = json.loads(destination.read_text())
    assert result["merge"]["reused"] is True


def test_report_keeps_the_full_runtime_versions_the_identity_no_longer_carries(pack, tmp_path):
    import sft_report
    full = dict(python="3.12.1", torch="2.10.0", transformers="5.17.0", peft="0.21.2", numpy="2.0.0", triton="3.0.0")
    core = {name: full[name] for name in ("torch", "transformers", "peft")}
    for name, recorded, expected in (("new", full, full), ("old", None, core)):      # old states have only identity.versions
        run = tmp_path / name
        state = dict(global_step=1, identity={"versions": core}, history=[], training_seconds=2, status="complete", epoch=1,
                     train={}, val={}, adapter_modules=[], trainable_parameters=4, tokens=20, completion_tokens=5, gpu={},
                     kernels={}, peak_memory_bytes=9, best_val_loss=1.2)
        if recorded:
            state["versions"] = recorded
        pack.commit_checkpoint(run, state, lambda path: None, True)
        destination = tmp_path / f"{name}.json"
        sft_report.report(tmp_path / "scratch", "A2", run, tmp_path / "merged", destination, 0)
        assert json.loads(destination.read_text())["versions"] == expected


def test_resolve_best_adapter_and_reject_path_escape(pack, tmp_path):
    import merge_lora
    root = tmp_path / "run"
    best = pack.commit_checkpoint(root, dict(global_step=1), lambda path: None, True)
    assert merge_lora.resolve_adapter(root) == best
    pack.atomic_json(root / "checkpoint.json", dict(latest="../escape", best="../escape"))
    with pytest.raises(ValueError):
        merge_lora.resolve_adapter(root)


def test_reuse_export_checks_adapter_and_all_files_without_torch(pack, tmp_path, monkeypatch):
    import merge_lora
    adapter, output = tmp_path / "adapter", tmp_path / "export"
    adapter.mkdir()
    output.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
    (adapter / "adapter_config.json").write_text('{"r": 8, "revision": null}')
    for name in ("model.safetensors", "config.json", "tokenizer_config.json"):
        (output / name).write_bytes(b"synthetic model")
    report = dict(base_model="synthetic/base", revision=None,
                  adapter_sha256={name: pack.sha256(adapter / name) for name in ("adapter_model.safetensors", "adapter_config.json")},
                  file_sha256={file.name: pack.sha256(file) for file in output.iterdir()},
                  parameters=3, stored_parameters=3, safetensors_bytes=4, files=3)
    pack.atomic_json(output / "merge_report.json", report)
    assert merge_lora.verify_export(output, adapter, "synthetic/base") == report
    # The reuse branch must execute before any optional ML import.
    monkeypatch.setitem(sys.modules, "torch", None)
    merge_lora.merge("synthetic/base", adapter, output)
    (output / "model.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError):
        merge_lora.verify_export(output, adapter, "synthetic/base")
    (output / "model.safetensors").write_bytes(b"synthetic model")
    (adapter / "adapter_config.json").write_bytes(b"changed alpha")
    with pytest.raises(ValueError):
        merge_lora.verify_export(output, adapter, "synthetic/base")


def test_runner_is_posix_shell():
    subprocess.run(["sh", "-n", str(GPU / "run_sft.sh")], check=True)


def run_runner_dry(tmp_path, **env):
    """Run the shell runner's dry run in a scratch copy. Returns the plan and the trainer's command line."""
    (tmp_path / "gpu").mkdir(parents=True)
    for name in ("run_sft.sh", "sft_lora.py", "merge_lora.py"):
        shutil.copyfile(GPU / name, tmp_path / "gpu" / name)
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({**fixture_row(), "id": split}) + "\n")
    recorder = tmp_path / "python"
    recorder.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{tmp_path}/argv.txt"\nexec "{sys.executable}" "$@"\n')
    recorder.chmod(0o755)
    env = {**os.environ, "DRY_RUN": "1", "BOOTSTRAP_PYTHON": str(recorder), "TRAIN_FILE": "train.jsonl",
           "VAL_FILE": "val.jsonl", **env}
    subprocess.run(["sh", "gpu/run_sft.sh"], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    return json.loads((tmp_path / ".agent-runs/sft-runner/dry-run.log").read_text()), (tmp_path / "argv.txt").read_text().split()


@pytest.mark.skipif(shutil.which("flock") is None, reason="the runner needs flock")
@pytest.mark.parametrize("arm,max_len,budget", [("A1", 8192, 16384), ("A2", 2048, 8192)])
def test_runner_dry_run_defaults_cover_the_longest_row_and_batch_several_rows(pack, tmp_path, arm, max_len, budget):
    plan, _ = run_runner_dry(tmp_path, ARM=arm)
    assert (plan["max_len"], plan["max_tokens_per_batch"]) == (max_len, budget)
    # The shell and the trainer carry separate copies of the defaults. They must agree.
    args = make_args(pack, tmp_path, "--arm", arm)
    assert (args.max_len, args.max_tokens_per_batch) == (max_len, budget)


@pytest.mark.skipif(shutil.which("flock") is None, reason="the runner needs flock")
def test_runner_passes_the_slow_kernel_flag_only_when_asked(tmp_path):
    _, argv = run_runner_dry(tmp_path / "default")
    assert "--allow-slow-kernels" not in argv
    _, argv = run_runner_dry(tmp_path / "allowed", ALLOW_SLOW_KERNELS="1")
    assert "--allow-slow-kernels" in argv


def test_runner_install_failures_say_the_run_stops_without_the_flag():
    lines = [line for line in (GPU / "run_sft.sh").read_text().splitlines() if "install failed" in line]
    assert len(lines) == 2 and all("stops" in line and "--allow-slow-kernels" in line for line in lines)
