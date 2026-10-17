# A1 and A2 LoRA SFT

GPU install, training, kernel execution, numerical loss parity, memory use,
full optimizer/RNG resume, real merge and vLLM loading are **UNTESTED here**.
CPU tests check pure data handling, batch order, atomic checkpoint publication, the
resume identity, the kernel gate and the export rules. They run the loss shift and the
gradient normalisation on numpy stand-ins for torch. Three tests use real CPU torch (a
real AdamW step, the clip to norm 1, fp16 overflow and non-finite gradients). They skip
themselves where torch is absent, so run them under an interpreter that has it.

Run from the repository root. Use local disk for active checkpoints.

```sh
.venv/bin/python gpu/sft_lora.py --arm A2 --dry-run
ARM=A2 STOP_AFTER_SECONDS=3600 sh gpu/run_sft.sh
ARM=A1 STOP_AFTER_SECONDS=7200 sh gpu/run_sft.sh
```

The dry run needs only Python's standard library. It estimates each segment as
`ceil(UTF-8 bytes / 4)` tokens. All its row, drop, clip and batch counts are
estimates. It loads no tokenizer or model and changes no training output.
`DRY_RUN=1 sh gpu/run_sft.sh` runs the same plan without a GPU or installation.

## Training environment and sources

The runner creates or reuses `.agent-runs/sft-venv`. It requires an installed `uv`,
Python 3.12, GNU `timeout`, `flock`, Linux x86_64 and NVIDIA driver R570 or newer.
It does not change `gpu/.venv` or the notebook's base environment.

| Package | Pin | Reason and source |
|---|---|---|
| Python | 3.12 | Present on the measured Colab host. Matches the convolution wheel. |
| torch | 2.10.0+cu128 | [Official CUDA 12.8 wheel](https://download.pytorch.org/whl/cu128/torch/). This training-only version matches the available convolution binary. |
| transformers | 5.17.0 | Same family implementation as the inference pack. Includes the [text-only class](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py) and [checkpoint prefix conversion](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/conversion_mapping.py). |
| peft | 0.21.2 | The stack memo's candidate. [Package metadata](https://pypi.org/pypi/peft/0.21.2/json) permits this torch and Transformers stack. |
| flash-linear-attention, fla-core | 0.5.2 | [Universal Python wheels](https://pypi.org/project/fla-core/0.5.2/) avoid a CUDA extension build. Their Triton kernels still compile at first GPU use. |
| causal-conv1d | 1.7.0 | The [release assets](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0) include `cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`. PyPI has only a source archive. |

These are core pins, not a full dependency lock. The inference environment's
torch 2.13.0 does not have a matching convolution wheel in that release's asset
list. Do not install a torch 2.10 extension into it. The convolution
[setup code](https://github.com/Dao-AILab/causal-conv1d/blob/v1.7.0/setup.py)
selects its wheel by torch minor version, CUDA major version, Python and C++ ABI.
The runner uses the explicit wheel URL. It never builds that extension.
The URL pins the release asset's SHA256:
`8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7`.

The runner logs a failed kernel installation and goes on; the trainer's own check
is the gate. Before importing the model it runs a small forward/backward probe of
`fla` and `causal_conv1d`. On compute capability 8.0 or newer (A100) a missing or
failed probe stops the run with an error that names the kernel. Install the kernels,
or set `ALLOW_SLOW_KERNELS=1` (`--allow-slow-kernels`) to train on the slower torch
implementation. The flag is not part of the resume identity. On Turing the trainer
keeps the warning and the torch path, because this FLA route requires SM80.
Transformers uses
[package-first fallbacks](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/integrations/hub_kernels.py).
Hub kernel replacement is off. Full attention uses torch SDPA. Full-model behaviour
still needs a check.
CUDA 12.8 needs a suitable host driver; see the [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

## Data, loss and batches

The supplied ChatML is used as stored. No chat template, BOS or extra EOS is
added. Prompt and completion are encoded separately without special-token
insertion, then joined. This keeps the exact generation prefix at the boundary.
Only completion labels contribute to loss. The stored end marker is supervised.

The default length cap is 8192 for A1 and 2048 for A2, with a budget of 16384 and
8192 padded tokens per micro-batch. An independent review of the first A2 run found
that the old caps were too small. With `OVERLENGTH=drop`, 4096 silently removed about
a third of the A1 Train rows (the longest traces, 17 of 241 drugs) and 1024 about 10%
of the A2 rows (3 drugs). A budget equal to the cap fit only about 1.5 A2 rows per
forward. The budget must be at least the cap and large enough to batch several rows.
Check the dry-run drop counts (estimates) before lowering a cap. `OVERLENGTH=drop`
removes an overlength row. `OVERLENGTH=clip` keeps the full completion and the
rightmost prompt tokens. It drops the row if the completion leaves no prompt token.
Clipping can remove system instructions and drug facts. Use it only after
checking the aggregate counts. The default avoids truncated targets.

Row IDs must be unique within a file, or the reader refuses it. A3 files
(`python -m rlexp.arms rft`) hold up to two samples per drug by default, often from one
prompt, so their IDs are `<prompt row id>#<sample_idx>`. The prompt and completion are
unchanged and the source row stays in the ID.

Each optimizer update has `BATCH_SIZE` rows, except the final short update.
The short final group is selected at random before length sorting. The full
length groups are shuffled with `SEED + epoch`. Each group is divided into
micro-batches whose padded token count fits `MAX_TOKENS_PER_BATCH`. Padding does
not contribute to loss. Samples remain separate; packing is off. Gradients sum
completion-token losses and divide by all completion tokens in the update.
The cap on tokens is a batch-shape limit, not a promise that GPU memory will fit.

The decoder and output head run through Transformers. The head projects only
supervised positions in chunks of 256 tokens. Checkpoint recomputation avoids
retaining all chunk logits for backward. This limits the 248,320-word vocabulary's
memory cost without a new training library. Numerical parity with the standard
model loss still needs a real model check.

The text-only `Qwen3_5ForCausalLM` class omits the vision tower. LoRA covers every
decoder `Linear`, including all Gated DeltaNet projections, attention and MLP.
It excludes the tied output head and embeddings, following the stack memo's
decoder scope. Actual adapted paths are checked and printed. Rank defaults to
32, alpha to twice rank, and dropout to zero. AdamW uses weight decay 0.01,
cosine decay, warmup ratio 0.05 and gradient clipping at norm 1. Capability SM80
or newer selects bf16. T4 selects fp16 with loss scaling. The decision excludes
[emulated bf16 support](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/__init__.py).
T4 fit and numerical stability still need a separate probe.

## Time limits, checkpoints and merge

`STOP_AFTER_SECONDS` includes training-process startup. The trainer checks it
before each optimizer update and validation batch. It finishes an active update
before saving. A limit can therefore overrun by one update plus checkpoint time.
Validation restarts if interrupted. It never promotes an incomplete validation.
The shell adds a hard training limit of this budget plus 600 seconds and a
30-second kill grace. Installation is outside this budget. Merge has 900 seconds.

Every `SAVE_STEPS` updates and after each epoch, the trainer saves safetensors,
optimizer, cosine scheduler, loss scaler, Python/NumPy/torch/CUDA RNG and the
cursor for the next update. An initial checkpoint also exists before training.
It writes an immutable directory before it atomically replaces `checkpoint.json`.
It keeps the latest and best directories. A failed write leaves the prior
checkpoint intact. A forced kill can replay updates since the last saved step.
Restart removes incomplete payload directories while it holds the run lock.
No partial gradient accumulation is restored. Exact GPU numerical replay remains
UNTESTED. Keep active files on local disk; Drive/FUSE semantics are UNTESTED.

Resume uses the same command and output path. The resume identity is the trainer
options (all of them except the paths, the stop, save and log settings and
`--allow-slow-kernels`), the model revision, the SHA256 of the Train and Val files,
the dtype and the torch, transformers and peft versions. A change in any of these
requires a new output path. Every other runtime version (python, numpy, triton,
safetensors, accelerate, FLA, causal-conv1d) is recorded in the state as `versions`.
A change there, like a changed kernel path, only prints a warning that names the
packages and keeps the saved cursor. This lets a fresh VM resume when it resolves
different transitive versions. The saved
remote model revision is reused. Load only checkpoints produced by this pack:
`training.pt` is trusted local state loaded with Python pickle support.

Each completed epoch records completion-token validation loss and teacher-forced
token accuracy. Accuracy does not measure chemical validity or structure recovery.
Best means the lowest full-epoch Val loss; ties keep the earlier adapter.
`epochs.jsonl` is rebuilt from committed history after resume.

`checkpoint.json` keeps only the latest and the best directory, so a different
selection rule would need a retrain. To avoid that, each epoch end also writes the
adapter alone to `<out>/adapters/epoch-N/`: `adapter_model.safetensors`,
`adapter_config.json` and `revision.json` (the resolved base model revision the run
trained on), with no optimizer state. The copy is made inside the same checkpoint
write: a temporary directory, fsync, then a rename, before the checkpoint that
records the epoch is committed. A kill cannot record an epoch without its adapter.
`summary.json` lists the committed copies as `epoch_adapters`, each with its path and
the SHA256 of all three files. The run root's `best` is unchanged.

A published epoch copy never changes. A repeat of an epoch with identical files does
nothing. A repeat with different files is refused if the epoch is committed. If it is
only the leftover of a killed save, the old directory is renamed aside, the new one
renamed in, and the old one renamed back if that fails, so a failed swap never loses it.
A kill between the two renames loses only an uncommitted leftover, which the replay
writes again. Restart removes the temporary and renamed-aside directories while it
holds the run lock.

To export one, run `python gpu/merge_lora.py --adapter <out>/adapters/epoch-2 --out <dir>`.
Inside a run (`<run>/adapters/epoch-N` or `<run>/checkpoints/step-*`, and this trainer's
unfinished `.pending-*` payloads) the export follows these rules, all checked before
any model loads. Any other directory, including another pack's kept adapters, is a plain
adapter and gets none of them:

- It takes the run's `run.lock` shared and without waiting. While a trainer holds the
  lock, it stops with "A trainer holds this run". Run-root exports follow the same rule.
  The reverse also holds: a trainer start while an export holds the lock stops with
  "Another trainer or an export holds this run".
- An epoch directory must be in the committed `epoch_adapters` of the run, its files
  must match the recorded SHA256 and nothing else may be in it. A leftover from a
  killed save, a changed file or an extra file is refused. A checkpoint directory must
  be the committed latest or best.
- The base revision comes from `revision.json` (an epoch adapter) or `state.json` (a
  checkpoint) and is passed to the loader. `--revision` is optional; if given it must
  equal the recorded one, or the export stops. An epoch adapter copied out of its run
  needs its `revision.json`, or `--revision`, or the export stops.
- An existing export is reused only if it was made at that revision too, besides the
  weight, config and file hashes.

The export report records `epoch` (and, for checkpoints, `best_epoch` and `best_step`).
`revision.json` is used instead of the PEFT config's `revision` field because PEFT is
not installed where these tests run, so that field could not be checked here.

The merge runs on CPU in bf16. It loads the original conditional model and shares
its decoder with the text-only adapter wrapper. After merge, it restores the
decoder to the full model and keeps the original vision weights and config.
It saves canonical conditional-model weight names. It reloads config and checks
parameter counts, safetensors names, shapes, dtype and actual tied-weight keys.
It publishes a new directory only after these checks. An existing export is
reused only if the adapter weights, adapter config, base revision and every saved
file hash match. A changed export fails without replacing it. A merge lock makes
temporary-directory cleanup safe after a kill. A shared training lock protects
the selected adapter from concurrent checkpoint pruning. The same lock rules apply to
epoch and checkpoint directories inside a run (see above).
Use the result as the sampler's `--model`. Actual vLLM loading is UNTESTED.

## Scoring A2

Score the A2 model with `--mode think` and the direct-answer system text. Training
feeds `encode(prompt) + encode(completion)`. The sampler's `--mode nothink` appends
`\n</think>\n\n` to the prompt and encodes the joined string, which gives different
token ids around `</think>` than training saw. For example, two adjacent newlines
can merge into one token when they are encoded together but not when they are
encoded apart. `--mode think` encodes the prompt alone, as training does, and the
trained model writes the closing itself. The system turn is the A2 data's
(`rlexp.arms.DIRECT_SYSTEM`). Use A2-format prompts, or pass `--system-text` with
that text. This follows from the code and the review. It is not yet measured on a
trained model.

## Runner parameters and output

Environment variables: `ARM`, `MODEL`, `TRAIN_FILE`, `VAL_FILE`, `OUT`, `MERGED`,
`TRAIN_VENV`, `MAX_LEN`, `MAX_TOKENS_PER_BATCH`, `BATCH_SIZE`, `OVERLENGTH`, `RANK`,
`LR`, `EPOCHS`, `SEED`, `WARMUP_RATIO`, `SAVE_STEPS`, `LOG_STEPS`,
`STOP_AFTER_SECONDS`, `ALLOW_SLOW_KERNELS` (1 passes `--allow-slow-kernels`),
`BOOTSTRAP_PYTHON`, `UV` and `DRY_RUN`.

The runner writes timestamped step logs and exit codes to
`.agent-runs/sft-runner/`. It writes the aggregate report to
`results/sft_A1_report.json` or `results/sft_A2_report.json`.
`MERGED` is an export parent. Each best checkpoint gets its own model directory
under it. This lets a resumed run export a new best without replacing old weights.
If time expires before a full epoch, it defers merge and reports partial status.
Reports contain no row IDs, prompts, completions or token arrays.
The report also includes configuration, data hashes, model revision, command and
the git revision read by `git rev-parse HEAD`. Git state is never changed.
It marks a verified export reuse as `merge-reused`.
Throughput
counts unpadded input-plus-completion tokens per successful training-step second.
It excludes startup, validation and saving. Step timings show those other costs.

For the first A100 run, check five finite updates, the module list, peak memory,
full validation, adapter effect, forced-kill resume and merged-model vLLM output.
Start with the default token budget. Reduce it together with the length cap if
memory fails. No GPU gate is passed by the CPU tests.
