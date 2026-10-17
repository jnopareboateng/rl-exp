#!/usr/bin/env python3
"""Paper figures F1-F4 and the Val table, from aggregate result files only (CPU; no model runs, no sample text, no molecule strings).

  .venv/bin/python scripts/make_figures.py --split val --out results/figures [--tables] [--allow-test]

  F1  drugs per year of first approval, stacked by split. Counted from data/processed/refs.jsonl (approval_year, split), because
      split_report.json drugs_by_year merges everything before 2015 and 2015-2021 into two bins; the counts are cross-checked against
      those bins and the script stops if they disagree.
  F2  delivered similarity (fv_tanimoto) against generated tokens per answer (fv_tokens): one point per run of results/<split>_baselines.json
  F3  RL training curves from runs/rl/<run>/metrics.jsonl and evals.jsonl (R1, R3, R4; a run that is not on disk is skipped)
  F4  fv_lift against fv_tanimoto, same file as F2
  --tables  write only table_<split>.md and .csv (no figures). Without it the figures and the tables are written.

Every figure is a .png (200 dpi) plus a .csv with exactly the plotted data; captions.json holds one short caption per output. The output
is deterministic: no timestamps, bundled font, no randomness. A Test split (test_r, test_t), or an input path or table that names one
(or the locked evaluation), is refused unless --allow-test is given; that flag is for the locked evaluation after the prereg-v1 tag.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

# the collapse alarm: distinct most-common answers over drugs with a valid answer, below this for 3 steps stops a run
from rlexp.grpo import COLLAPSE_BELOW

SPLIT_NAMES = {"val": "Val", "test_r": "Test-R", "test_t": "Test-T"}
TEST_SPLITS = ("test_r", "test_t")
TEST_PATH = re.compile(r"(?<![a-z])(test[_-][rt]|final|confirmatory)(?![a-z])", re.IGNORECASE)   # a Test split or the locked evaluation (runs/final)
DPI = 200
GREY = "#555555"

# F1: bottom to top. Test-T is on top and in the strong colour: it is the temporal cohort.
STACK = (("train", "Train", "#0072B2"), ("val", "Val", "#CC79A7"), ("test_r", "Test-R", "#009E73"),
         ("buffer", "Buffer (not used)", "#BBBBBB"), ("test_t", "Test-T (temporal cohort)", "#D55E00"))

# F2/F4 families: key -> (legend text, colour, marker). Okabe-Ito colours; every family also has its own marker shape.
FAMILIES = {"control": ("CPU control (no model)", "#333333", "s"), "untrained": ("4B untrained", "#56B4E9", "o"),
            "retrieval": ("4B with retrieved examples", "#CC79A7", "^"), "sft": ("4B fine-tuned (SFT)", "#E69F00", "D"),
            "rl": ("4B after RL", "#D55E00", "P"), "size": ("Larger untrained model", "#0072B2", "v"),
            "name": ("Drug name given (diagnostic)", "#009E73", "h"), "other": ("Other", "#999999", "o")}

# (pattern on the run id, family, arm name in the figures, main arm?). First match wins; edit when a run is added (like val_table.py RUNS).
ARMS = ((r"control_constant_pool", "control", "Best constant answer (C3)", True),
        (r"control_constant_scaffold", "control", "Constant scaffold (C2)", True),
        (r"control_constant", "control", "Constant answer (C0)", True),
        (r"control_copy_nearest", "control", "Nearest-profile copy (C1)", True),
        (r"^rlfull", "rl", "4B after RL", True),
        (r"^(a2|g-a2)", "sft", "4B fine-tuned on answers (A2)", True),
        (r"^a1", "sft", "4B fine-tuned on traces (A1)", True),
        (r"^a4", "retrieval", "4B with retrieved examples (A4)", True),
        (r"^name-4b", "name", "4B told the drug name", False),
        (r"^name-9b", "name", "9B told the drug name", False),
        (r"^name", "name", "Told the drug name", False),
        (r"^b27", "size", "27B untrained", True),
        (r"^b38", "size", "Qwen3.8 27B untrained", True),
        (r"^(b9|g-9b)", "size", "9B untrained", True),
        (r"^(a-direct|b4|g-4b)", "untrained", "4B untrained", True))

# Table columns: key in the table file, header, decimals, signed
TABLE_COLUMNS = (("valid_closed", "Valid answers", 3, False), ("fv_tanimoto", "Delivered similarity", 3, False),
                 ("fv_samples", "Samples per answer", 2, False), ("fv_tokens", "Tokens per answer", 1, False),
                 ("fv_lift", "Delivered lift", 3, True), ("lift", "Lift, all samples", 3, True),
                 ("mean_tanimoto", "Mean similarity, all samples", 3, False), ("core@1", "Core match", 3, False),
                 ("exact@1", "Exact match", 3, False))

F2_FIELDS = ("run", "arm", "family", "decoding", "fv_tokens", "fv_tanimoto", "fv_samples", "prompt_tokens", "labelled")
F4_FIELDS = ("run", "arm", "family", "decoding", "fv_lift", "fv_tanimoto", "fv_samples", "labelled")
F3_FIELDS = ("run", "series", "step", "value", "source")


class Refused(Exception):
    """An input or option this script will not use: a Test split without --allow-test, or a table of another split."""


# ------------------------------------------------------------------------------------------------- guards and readers
def refuse_test_path(path, allow):
    """The last three components of `path` name a Test split or the locked evaluation (runs/final): refuse unless allowed."""
    if not allow and TEST_PATH.search("/".join(Path(path).parts[-3:])):
        raise Refused(f"{path} names a Test split or the locked evaluation; --allow-test is for use after the prereg-v1 tag only")


def finite(value):
    """The value as a float, or None when it is missing, not a number (a bool is not one) or not finite."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def read_jsonl(path):
    """The rows of a JSON-lines file. A malformed LAST line (a file copied while it was being written) is dropped; any other error raises."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def load_table(path, split, allow_test=False):
    """results/<split>_baselines.json as written by scripts/val_table.py. Refused when the path names a Test split without --allow-test, or
    when the file declares another split than the one asked for (a Test table under --split val, or the reverse)."""
    refuse_test_path(path, allow_test)
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("split") != split:
        raise Refused(f"{path} holds split {table.get('split')!r}, not {split!r}")
    for row in table["rows"]:
        if not isinstance(row.get("run"), str):
            raise ValueError(f"{path}: a row has no run name")
    return table


def arm_of(run, label=""):
    """(family, arm name, main arm?) of a run id."""
    for pattern, family, arm, main in ARMS:
        if re.search(pattern, run):
            return family, arm, main
    return "other", label or run, False


def constant_level(rows):
    """The best constant answer (C3): its first-valid similarity, else its mean similarity (equal for a constant); None if absent."""
    for r in rows:
        if re.search(r"control_constant_pool", r["run"]):
            return finite(r.get("fv_tanimoto")) if finite(r.get("fv_tanimoto")) is not None else finite(r.get("mean_tanimoto"))
    return None


# ------------------------------------------------------------------------------------------------- F1 approval years
def year_bin(year, key):
    """Does `year` fall in a split_report.json drugs_by_year bin ('<2015', '2015-2021' or '2022')?"""
    if key.startswith("<"):
        return year < int(key[1:])
    low, _, high = key.partition("-")
    return int(low) <= year <= int(high or low)


def f1_rows(refs, report=None):
    """Tidy counts {year, split, drugs} of single drugs by year of first approval for the splits of STACK (excluded drugs are not drawn):
    the non-empty cells only, by year then stack order. With `report` (split_report.json) every coarse bin must be reproduced."""
    order = [key for key, _, _ in STACK]
    counts = {}
    for r in refs:
        if r["split"] in order:
            year = r.get("approval_year")
            if not isinstance(year, int) or isinstance(year, bool):
                raise ValueError(f"a {r['split']} drug has no approval year")
            counts[(year, r["split"])] = counts.get((year, r["split"]), 0) + 1
    rows = [{"year": y, "split": s, "drugs": n} for (y, s), n in sorted(counts.items(), key=lambda t: (t[0][0], order.index(t[0][1])))]
    for split, entry in (report or {}).get("splits", {}).items():
        for key, expected in entry["drugs_by_year"].items():
            got = sum(r["drugs"] for r in rows if r["split"] == split and year_bin(r["year"], key))
            if split in order and got != expected:
                raise ValueError(f"refs.jsonl and split_report.json disagree for {split} {key}: {got} against {expected}")
    return rows


def split_totals(rows):
    totals = {}
    for r in rows:
        totals[r["split"]] = totals.get(r["split"], 0) + r["drugs"]
    return totals


# ------------------------------------------------------------------------------------------------- F2 and F4 scatter data
def usable(table_rows, *keys):
    """(rows with a finite value for every key, the run names of the rest)."""
    ok = [all(finite(r.get(k)) is not None for k in keys) for r in table_rows]
    return [r for r, good in zip(table_rows, ok) if good], [r["run"] for r, good in zip(table_rows, ok) if not good]


def scatter_rows(table_rows, split, keys, fields):
    """Tidy rows for a run scatter. `labelled` marks the points that get a text label: on Val the best run (fv_tanimoto) of each main arm, that
    is the arm's Val-selected setting; on a Test split no outcome is used and every run of a main arm is labelled (one frozen run per arm)."""
    kept, _ = usable(table_rows, *keys)
    rows = []
    for r in kept:
        family, arm, main = arm_of(r["run"], r.get("label", ""))
        rows.append({"run": r["run"], "arm": arm, "family": family, "decoding": r.get("decoding", ""), "_main": main,
                     **{k: finite(r.get(k)) for k in fields if k not in ("run", "arm", "family", "decoding", "labelled")}})
    best = {}
    for row in rows:
        if row["_main"] and (row["arm"] not in best or row["fv_tanimoto"] > best[row["arm"]]["fv_tanimoto"]):
            best[row["arm"]] = row
    for row in rows:
        row["labelled"] = row["_main"] and (split != "val" or best[row["arm"]] is row)
        del row["_main"]
    return [{k: row[k] for k in fields} for row in rows]


def f2_rows(table_rows, split="val"):
    return scatter_rows(table_rows, split, ("fv_tanimoto", "fv_tokens"), F2_FIELDS)


def f4_rows(table_rows, split="val"):
    return scatter_rows(table_rows, split, ("fv_tanimoto", "fv_lift"), F4_FIELDS)


# ------------------------------------------------------------------------------------------------- F3 RL curves
def moving_average(values, window=5):
    """Trailing mean of up to `window` values ending at each position, ignoring None; None where the window holds none."""
    out = []
    for i in range(len(values)):
        w = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        out.append(sum(w) / len(w) if w else None)
    return out


def load_rl_runs(rl_dir, names, allow_test=False):
    """({name: {"metrics": [...], "evals": [...], "label": str}} for the runs of `names` that have a metrics.jsonl, the names skipped).
    A run whose summary.json says it was evaluated on a Test file is refused."""
    runs, skipped = {}, []
    for name in names:
        folder = Path(rl_dir) / name
        refuse_test_path(folder, allow_test)
        if not (folder / "metrics.jsonl").is_file():
            skipped.append(name)
            continue
        options, command = {}, []
        if (folder / "summary.json").is_file():
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            options, command = summary.get("identity", {}).get("options", {}), summary.get("command") or []
        if "--val" in command and command.index("--val") + 1 < len(command):
            refuse_test_path(command[command.index("--val") + 1], allow_test)
        reward = {"full": "standard reward", "validity": "validity-only reward", "contrastive": "contrastive reward"}.get(options.get("reward_mode"))
        text = ", ".join(t for t in (reward, "profile removed" if options.get("profile_mode") == "none" else None) if t)
        runs[name] = {"metrics": read_jsonl(folder / "metrics.jsonl"),
                      "evals": read_jsonl(folder / "evals.jsonl") if (folder / "evals.jsonl").is_file() else [],
                      "label": f"{name}: {text}" if text else name}
    return runs, skipped


def f3_rows(runs):
    """Tidy rows {run, series, step, value, source} of every plotted curve (None values are left out). A Val evaluation gives the similarity of the
    first valid answer (fv_tanimoto, the endpoint; only runs that sampled 8 per prompt log it) and the mean over all samples (invalid = 0)."""
    rows = []
    for name, run in runs.items():
        for e in run["evals"]:
            m = e.get("metrics", {})
            for series, source in (("val_similarity_first_valid", "fv_tanimoto"), ("val_similarity_all", "tanimoto"),
                                   ("val_validity", "validity" if "validity" in m else "valid_closed")):
                if finite(m.get(source)) is not None:
                    rows.append({"run": name, "series": series, "step": e["step"], "value": finite(m[source]), "source": source})
        steps = [m["step"] for m in run["metrics"]]
        # the training reward is the standard reward of the batch whatever reward the run was trained on (reward_mean is the objective, which
        # differs for the contrastive and validity-only runs and would not compare across runs)
        for series, keys, smooth in (("train_reward", ("reward_full_mean", "reward_mean"), True), ("batch_lift", ("lift_batch",), True),
                                     ("collapse_score", ("collapse_score",), False), ("top_answer_share", ("top_answer_share",), False)):
            values = [next((finite(m.get(k)) for k in keys if finite(m.get(k)) is not None), None) for m in run["metrics"]]
            source = keys[0] if all(finite(m.get(keys[0])) is not None for m in run["metrics"]) else "|".join(keys)
            columns = [(series, values)] + ([(series + "_avg5", moving_average(values))] if smooth else [])
            for label, column in columns:
                rows += [{"run": name, "series": label, "step": s, "value": v, "source": source} for s, v in zip(steps, column) if v is not None]
    return rows


# ------------------------------------------------------------------------------------------------- table
def cell(value, decimals, signed):
    v = finite(value)
    if v is None:
        return "n/a"
    v = round(v, decimals) + 0.0                      # + 0.0 turns -0.0 into 0.0
    return f"{v:+.{decimals}f}" if signed else f"{v:.{decimals}f}"


def table_rows(table):
    """The rows of the Val table: run, label, decoding and the TABLE_COLUMNS values (None where the table has none)."""
    return [{"run": r["run"], "label": r.get("label", r["run"]), "decoding": r.get("decoding", ""),
             **{k: finite(r.get(k)) for k, _, _, _ in TABLE_COLUMNS}} for r in table["rows"]]


def table_md(rows, name, drugs):
    head = "| Arm | Decoding | " + " | ".join(h for _, h, _, _ in TABLE_COLUMNS) + " |"
    lines = [head, "|---|---|" + "---:|" * len(TABLE_COLUMNS)]
    for r in rows:
        values = " | ".join(cell(r[k], d, s) for k, _, d, s in TABLE_COLUMNS)
        lines.append(f"| {r['label'].replace('|', '/')} | {r['decoding'].replace('|', '/')} | {values} |")
    note = (f"{name} split, {drugs} drugs. Similarity is the Tanimoto similarity (Morgan count vectors) to the reference structure. Delivered "
            "similarity and lift are those of the first valid, finished answer among up to eight samples of each prompt (0 if none), with its cost in "
            "samples and generated tokens per answer (prompts are not counted). Lift is the similarity to the drug's own structure minus the mean "
            "similarity to the other drugs' structures. Core match and exact match are the share of samples with the reference's scaffold or "
            "connectivity (core) or its connectivity (exact), averaged over drugs. n/a: fewer than eight samples per prompt, so no first-valid value.")
    return "\n".join(lines) + "\n\n" + note + "\n"


# ------------------------------------------------------------------------------------------------- outputs
def fmt(value):
    """A CSV cell: empty for None, 10 significant digits for a float (below 1e-12 is float noise, the lift of a constant answer, and reads 0)."""
    if isinstance(value, float):
        return f"{0.0 if abs(value) < 1e-12 else value:.10g}"
    return "" if value is None else str(value)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows([fmt(row[k]) for k in fields] for row in rows)


def add_caption(out, key, text):
    """Merge one caption into captions.json (sorted keys, so the file is the same whatever the order of the runs)."""
    path = Path(out) / "captions.json"
    captions = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    captions[key] = text
    path.write_text(json.dumps(captions, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.0, "figure.dpi": DPI, "figure.constrained_layout.use": True, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def save_png(fig, path):
    fig.savefig(path, dpi=DPI, metadata={"Software": None})          # no Software tag, and PNG holds no timestamp: same data, same bytes
    plt_().close(fig)


# ------------------------------------------------------------------------------------------------- drawing
def draw_f1(rows):
    plt = plt_()
    cells = {(r["year"], r["split"]): r["drugs"] for r in rows}
    years = list(range(min(r["year"] for r in rows), max(r["year"] for r in rows) + 1))
    totals = split_totals(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bottom = [0] * len(years)
    for key, text, colour in STACK:
        heights = [cells.get((y, key), 0) for y in years]
        ax.bar(years, heights, bottom=bottom, width=0.8, color=colour, linewidth=0, label=f"{text}, {totals.get(key, 0)}")
        bottom = [b + h for b, h in zip(bottom, heights)]
    cohort = [r["year"] for r in rows if r["split"] == "test_t"]
    if cohort:
        ax.axvspan(min(cohort) - 0.5, max(cohort) + 0.5, color="#D55E00", alpha=0.10, linewidth=0, zorder=0)
    ax.set_ylim(0, max(bottom) * 1.45)
    ax.set_xticks(range(5 * math.ceil(years[0] / 5), years[-1] + 1, 5))
    ax.set_xlabel("Year of first approval")
    ax.set_ylabel("Drugs")
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, handleheight=0.8, labelspacing=0.3)
    return fig


OFFSETS = ((5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"), (-5, -3, "right", "top"), (0, 7, "center", "bottom"),
           (0, -7, "center", "top"), (9, 0, "left", "center"), (-9, 0, "right", "center"), (14, 9, "left", "bottom"), (14, -9, "left", "top"),
           (-14, 9, "right", "bottom"), (-14, -9, "right", "top"), (22, 0, "left", "center"), (-22, 0, "right", "center"),
           (0, 16, "center", "bottom"), (0, -16, "center", "top"), (24, 14, "left", "bottom"), (24, -14, "left", "top"),
           (-24, 14, "right", "bottom"), (-24, -14, "right", "top"))


def place_labels(fig, ax, points, items):
    """Write the text of `items` [(x, y, text, offsets, colour)] beside (x, y): the first of `offsets` (default OFFSETS) whose box stays inside the
    axes and clear of the labels already placed and of every marker (`points`, data coordinates); if none is clear, the one that overlaps least.
    The labels with the most markers around them go first."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw()                                       # settle the layout so the pixel boxes below are final
    renderer, px = fig.canvas.get_renderer(), fig.dpi / 72
    centres = ax.transData.transform(points)
    taken = [Bbox.from_bounds(x - 4.5 * px, y - 4.5 * px, 9 * px, 9 * px) for x, y in centres]

    def crowd(item):
        return -sum(math.dist(ax.transData.transform((item[0], item[1])), c) < 40 * px for c in centres)

    for x, y, text, offsets, colour in sorted(items, key=crowd):
        best = None
        for dx, dy, ha, va in offsets or OFFSETS:
            note = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6, color=colour or "#222222",
                               arrowprops={"arrowstyle": "-", "lw": 0.4, "color": GREY, "shrinkA": 0, "shrinkB": 2} if abs(dx) + abs(dy) > 16 else None)
            note.update_positions(renderer)
            box = Text.get_window_extent(note, renderer)    # the text alone: an annotation's own extent would include its leader line
            inside = ax.bbox.x0 <= box.x0 and box.x1 <= ax.bbox.x1 and ax.bbox.y0 <= box.y0 and box.y1 <= ax.bbox.y1
            hits = sum(box.overlaps(other) for other in taken) + (0 if inside else 100)
            if best is None or hits < best[0]:
                if best is not None:
                    best[1].remove()
                best = (hits, note, box)
            else:
                note.remove()
            if hits == 0:
                break
        taken.append(best[2])


def draw_scatter(rows, xkey, xlabel, ylabel, constant, size, symlog=False, zero_line=False):
    plt = plt_()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=size)
    handles = []
    for family, (text, colour, marker) in FAMILIES.items():
        points = [r for r in rows if r["family"] == family]
        if points:
            ax.scatter([r[xkey] for r in points], [r["fv_tanimoto"] for r in points], s=24, marker=marker, c=colour, edgecolors="black",
                       linewidths=0.4, zorder=3)
            handles.append(Line2D([], [], ls="", marker=marker, mfc=colour, mec="black", mew=0.4, ms=4.5, label=text))
    xs = [r[xkey] for r in rows]
    if symlog:                                             # 0 (a CPU control generates nothing) sits at the left edge, then a log scale
        ax.set_xscale("symlog", linthresh=10, linscale=0.3)
        right = max(max(xs) * 1.5, 30)
        ax.set_xlim(-1.5, right)
        ax.set_xticks([t for t in (0, 10, 100, 1000) if t < right])
        ax.get_xaxis().set_major_formatter(lambda v, _: f"{v:g}")
    else:
        pad = (max(xs) - min(xs)) * 0.08 or 0.01
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
    if zero_line:
        ax.axvline(0, color=GREY, lw=0.5, ls=":", zorder=1)
    ax.set_ylim(max(0, min(r["fv_tanimoto"] for r in rows) - 0.02), max(r["fv_tanimoto"] for r in rows) + 0.03)
    labels = [(r[xkey], r["fv_tanimoto"], r["arm"], None, None) for r in rows if r["labelled"]]
    if constant is not None:                               # the level of the best constant, at the right end of its line, above it or else below
        ax.axhline(constant, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
        labels.append((ax.get_xlim()[1], constant, f"best constant answer, {constant:.3f}", ((-3, 2, "right", "bottom"), (-3, -2, "right", "top")), GREY))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.legend(handles=handles, loc="outside lower center", ncol=4 if size[0] >= 6.5 else 3, frameon=False, handletextpad=0.3, columnspacing=1.2)
    place_labels(fig, ax, [(r[xkey], r["fv_tanimoto"]) for r in rows], labels)
    return fig


def draw_f3(rows, runs, constant):
    plt = plt_()
    from matplotlib.lines import Line2D
    colours = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    similarity, validity, reward, lift, collapse, top = axes[0][0], axes[1][0], axes[0][1], axes[1][1], axes[0][2], axes[1][2]

    def curve(name, series):
        return [(r["step"], r["value"]) for r in rows if r["run"] == name and r["series"] == series]

    def line(ax, points, **style):
        if points:
            ax.plot(*zip(*points), **style)

    for i, name in enumerate(runs):
        colour = colours[i % len(colours)]
        line(similarity, curve(name, "val_similarity_all"), color=colour, marker="o", ms=3, mfc="white", ls="--")
        line(similarity, curve(name, "val_similarity_first_valid"), color=colour, marker="o", ms=3)
        line(validity, curve(name, "val_validity"), color=colour, marker="o", ms=3)
        for ax, series in ((reward, "train_reward"), (lift, "batch_lift")):         # single steps faint, the trailing mean of 5 bold
            line(ax, curve(name, series), color=colour, lw=0.5, alpha=0.3)
            line(ax, curve(name, series + "_avg5"), color=colour, lw=1.3)
        line(collapse, curve(name, "collapse_score"), color=colour)
        line(top, curve(name, "top_answer_share"), color=colour)
        for series in ("val_similarity_all", "val_similarity_first_valid"):
            for start in [p for p in curve(name, series) if p[0] == 0]:               # step 0 is the checkpoint RL starts from (A2)
                similarity.plot(*start, marker="o", ms=6.5, mfc="none", mec="black", mew=0.7, zorder=4)
    starts = [p[1] for name in runs for p in curve(name, "val_similarity_all") if p[0] == 0]
    if starts:
        similarity.annotate("A2 start", (0, min(starts)), xytext=(7, 0), textcoords="offset points", fontsize=6, ha="left", va="center")
    values = [r["value"] for r in rows if r["series"].startswith("val_similarity")]
    if constant is not None:
        similarity.axhline(constant, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
        similarity.text(0.02, constant, f"best constant answer, {constant:.3f}", transform=similarity.get_yaxis_transform(), ha="left", va="bottom",
                        fontsize=6, color=GREY)
        values.append(constant)
    if values:
        similarity.set_ylim(min(values) - 0.01, max(values) + 0.015)
    collapse.axhline(COLLAPSE_BELOW, color=GREY, lw=0.6, ls=":")
    collapse.text(0.98, COLLAPSE_BELOW - 0.02, f"alarm below {COLLAPSE_BELOW:g}", transform=collapse.get_yaxis_transform(), ha="right", va="top",
                  fontsize=6, color=GREY)
    lift.axhline(0, color=GREY, lw=0.5, ls=":")
    validity.set_ylim(0, 1.02)
    collapse.set_ylim(0, 1.05)
    top.set_ylim(bottom=0)
    for ax, text in ((similarity, "Val similarity"), (validity, "Val valid answers (share)"), (reward, "Training reward (5-step mean)"),
                     (lift, "Batch lift (5-step mean)"), (collapse, "Collapse score"), (top, "Share of the most common answer")):
        ax.set_ylabel(text)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax, title in zip(axes[0], ("Held-out Val drugs", "Training batches", "Answer diversity")):
        ax.set_title(title, loc="left")
    handles = [Line2D([], [], color=colours[i % len(colours)], lw=1.3, label=run["label"]) for i, run in enumerate(runs.values())]
    kinds = {r["series"] for r in rows}
    if "val_similarity_first_valid" in kinds:
        handles.append(Line2D([], [], color=GREY, lw=1.0, marker="o", ms=3, label="Val similarity: first valid answer"))
    if "val_similarity_all" in kinds:
        handles.append(Line2D([], [], color=GREY, lw=1.0, ls="--", marker="o", ms=3, mfc="white", label="Val similarity: all samples"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4), frameon=False)
    return fig


# ------------------------------------------------------------------------------------------------- captions
def caption_f1(rows, n_excluded):
    t = split_totals(rows)
    years = sorted({r["year"] for r in rows if r["split"] == "test_t"})
    cohort = f" first approved in {years[0]} to {years[-1]}" if years else ""
    return (f"Drugs by year of first approval, coloured by split ({sum(t.values())} drugs; {n_excluded} excluded drugs are not shown). Test-T, the temporal "
            f"cohort, is every drug{cohort} ({t.get('test_t', 0)} drugs). The buffer holds drugs in the same similarity cluster as a Test-T drug and is "
            "not used.")


def caption_f2(rows, omitted, name, drugs, constant):
    line = f" The dashed line is the best constant answer (C3, {constant:.3f})." if constant is not None else ""
    skipped = f" {len(omitted)} runs with fewer than eight samples per prompt have no first-valid value and are not shown." if omitted else ""
    return (f"One point per run on the {drugs} {name} drugs. Vertical: Tanimoto similarity to the true structure of the first valid answer among up to "
            "eight samples (0 if none is valid). Horizontal: tokens generated per answer, all samples drawn up to that answer; prompts are not counted "
            f"and CPU controls generate none (plotted at 0).{line} Labels mark the best setting of each arm.{skipped} One training seed per arm; with "
            f"{drugs} drugs, differences of a few hundredths are not resolved.")


def caption_f3(runs, rows):
    both = any(r["series"] == "val_similarity_first_valid" for r in rows)
    what = ("as the similarity of the first valid answer (solid line; only runs that sampled eight per prompt log it) and as the mean over all samples "
            "with invalid answers counted as 0 (dashed line)" if both else "as the mean over all samples with invalid answers counted as 0 (these runs "
            "did not log the first-valid metric)")
    return (f"RL pilot runs ({', '.join(runs)}). Left: Val similarity {what}, and the share of valid answers, every 10 steps; circled points are the A2 "
            "checkpoint RL starts from, the dashed grey line is the best constant answer. Middle: training reward (the standard reward, also for runs "
            "trained on another reward) and batch lift, trailing 5-step mean (faint lines: single steps). Right: collapse score (distinct most-common "
            f"answers over drugs with a valid answer; alarm below {COLLAPSE_BELOW:g}) and the share of valid answers equal to the most common one.")


def caption_f4(rows, omitted, name, drugs, constant):
    by = {r["arm"]: r for r in rows}
    c1 = f" The nearest-profile copy (C1) has lift {by['Nearest-profile copy (C1)']['fv_lift']:+.3f}." if "Nearest-profile copy (C1)" in by else ""
    rl = [r for r in rows if r["family"] == "rl"]
    rl_text = (f" The RL checkpoints have lift {min(r['fv_lift'] for r in rl):+.3f} to {max(r['fv_lift'] for r in rl):+.3f} at similarity "
               f"{min(r['fv_tanimoto'] for r in rl):.3f} to {max(r['fv_tanimoto'] for r in rl):.3f}.") if rl else ""
    line = f" The dashed line is the best constant answer (C3, {constant:.3f})." if constant is not None else ""
    skipped = f" {len(omitted)} runs without first-valid values are not shown." if omitted else ""
    return (f"Delivered similarity against lift for each run on the {drugs} {name} drugs. Lift is the first valid answer's similarity to its own drug's "
            "structure minus its mean similarity to the other drugs' structures: it measures association with the drug, not use of the profile, and is 0 "
            f"for a constant answer.{c1}{rl_text}{line}{skipped}")


# ------------------------------------------------------------------------------------------------- main
def make_f1(args, out):
    refs = read_jsonl(args.data / "refs.jsonl")
    report = json.loads((args.data / "split_report.json").read_text(encoding="utf-8"))
    rows = f1_rows(refs, report)
    fig = draw_f1(rows)
    save_png(fig, out / "f1_approval_years.png")
    write_csv(out / "f1_approval_years.csv", rows, ("year", "split", "drugs"))
    add_caption(out, "f1_approval_years", caption_f1(rows, sum(r["split"] == "excluded" for r in refs)))
    return rows


def make_f2(table, args, out):
    name, split = SPLIT_NAMES[args.split], args.split
    rows = f2_rows(table["rows"], split)
    omitted = usable(table["rows"], "fv_tanimoto", "fv_tokens")[1]
    if not rows:
        return rows, omitted
    if any(r["family"] == "other" for r in rows):
        print("F2: runs drawn as Other (not in ARMS): " + ", ".join(r["run"] for r in rows if r["family"] == "other"))
    constant = constant_level(table["rows"])
    if constant is None:
        print("F2 and F4: no best-constant (C3) run in the table, so no constant line is drawn")
    fig = draw_scatter(rows, "fv_tokens", "Generated tokens per answer (log scale; 0 for CPU controls)", "Delivered similarity (first valid answer)",
                       constant, (7.0, 3.5), symlog=True)
    stem = f"f2_similarity_vs_tokens_{split}"
    save_png(fig, out / f"{stem}.png")
    write_csv(out / f"{stem}.csv", rows, F2_FIELDS)
    drugs = max(r.get("drugs") or 0 for r in table["rows"])
    add_caption(out, stem, caption_f2(rows, omitted, name, drugs, constant))
    return rows, omitted


def make_f4(table, args, out):
    name, split = SPLIT_NAMES[args.split], args.split
    rows = f4_rows(table["rows"], split)
    omitted = usable(table["rows"], "fv_tanimoto", "fv_lift")[1]
    if not rows:
        return rows, omitted
    constant = constant_level(table["rows"])
    fig = draw_scatter(rows, "fv_lift", "Lift of the first valid answer (own drug minus other drugs)", "Delivered similarity (first valid answer)",
                       constant, (5.5, 3.5), zero_line=True)
    stem = f"f4_lift_vs_similarity_{split}"
    save_png(fig, out / f"{stem}.png")
    write_csv(out / f"{stem}.csv", rows, F4_FIELDS)
    drugs = max(r.get("drugs") or 0 for r in table["rows"])
    add_caption(out, stem, caption_f4(rows, omitted, name, drugs, constant))
    return rows, omitted


def make_f3(constant, args, out):
    runs, skipped = load_rl_runs(args.rl_dir, [n for n in args.rl_runs.split(",") if n], args.allow_test)
    if not runs:
        return [], skipped
    rows = f3_rows(runs)
    if not rows:                                           # a run that has only just started: nothing to draw yet
        return [], skipped
    fig = draw_f3(rows, runs, constant)
    save_png(fig, out / "f3_rl_training.png")
    write_csv(out / "f3_rl_training.csv", rows, F3_FIELDS)
    add_caption(out, "f3_rl_training", caption_f3(runs, rows))
    return rows, skipped


def make_table(table, args, out):
    rows, name = table_rows(table), SPLIT_NAMES[args.split]
    drugs = max(r.get("drugs") or 0 for r in table["rows"])
    stem = f"table_{args.split}"
    (out / f"{stem}.md").write_text(table_md(rows, name, drugs), encoding="utf-8")
    write_csv(out / f"{stem}.csv", rows, ("run", "label", "decoding", *[k for k, _, _, _ in TABLE_COLUMNS]))
    add_caption(out, stem, f"{name} results of every run: validity, delivered (first-valid) similarity with its cost in samples and tokens per answer, lift, "
                           "mean similarity, core and exact match.")
    return rows


def run(args):
    if args.split in TEST_SPLITS and not args.allow_test:
        raise Refused(f"--split {args.split} is a Test split; --allow-test is for use after the prereg-v1 tag only")
    table_path = args.table or Path("results") / f"{args.split}_baselines.json"
    table = load_table(table_path, args.split, args.allow_test)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    done = []
    if not args.tables:
        refuse_test_path(args.data, args.allow_test)
        make_f1(args, out)
        done.append("F1")
        rows, omitted = make_f2(table, args, out)
        done.append("F2" if rows else "F2 (no run has first-valid similarity and tokens: skipped)")
        if omitted:
            print(f"F2: {len(omitted)} runs without fv_tanimoto or fv_tokens are not drawn: {', '.join(omitted)}")
        rows, skipped = make_f3(constant_level(table["rows"]), args, out)
        done.append("F3" if rows else "F3 (no RL run with logged steps: skipped)")
        if skipped:
            print(f"F3: runs not on disk, skipped: {', '.join(skipped)}")
        rows, omitted = make_f4(table, args, out)
        done.append("F4" if rows else "F4 (no run has first-valid similarity and lift: skipped)")
    make_table(table, args, out)
    done.append("table")
    print(f"wrote to {out}: " + ", ".join(done))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLIT_NAMES), default="val", help="split of the results table behind F2, F4 and the table (default val); F1 and F3 do not depend on it")
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--table", type=Path, help="results table of the split (default results/<split>_baselines.json)")
    ap.add_argument("--data", type=Path, default=Path("data/processed"), help="folder with refs.jsonl and split_report.json (F1)")
    ap.add_argument("--rl-dir", type=Path, default=Path("runs/rl"))
    ap.add_argument("--rl-runs", default="R1,R3,R4", help="RL run folders under --rl-dir, comma separated; absent ones are skipped")
    ap.add_argument("--tables", action="store_true", help="write only the tables, no figures")
    ap.add_argument("--allow-test", action="store_true", help="accept a Test split and Test-named inputs (locked evaluation, after the prereg-v1 tag)")
    args = ap.parse_args(argv)
    try:
        run(args)
    except Refused as e:
        sys.exit(f"refused: {e}")
    except FileNotFoundError as e:
        sys.exit(f"error: missing input {e.filename}")


if __name__ == "__main__":
    main()
