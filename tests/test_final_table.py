"""scripts/final_table.py on synthetic runs: the row schema and the columns of val_table.py, the rules for missing splits, k < 8, blank-profile and identification arms, the
spec's labels and decoding, ordering, the Test refusal, outputs without answers or drug names, determinism; a fake `rlexp.metrics` for the many-arm cases and the REAL
score / report / ident CLIs on a tiny synthetic data directory for one end-to-end case. No real data."""
import importlib.util
import json
import os
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "final_table.py"
SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
          "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "CN1CCCC1c1cccnc1", "OC(=O)CC(O)(CC(O)=O)C(O)=O",
          "NCCc1ccc(O)c(O)c1", "CC(C)CC(N)C(O)=O", "OC(=O)c1ccccc1O"]
CONSTANT_MARKER = "CONSTANT-MOLECULE-MARKER"


@pytest.fixture(scope="module")
def ft():
    spec = importlib.util.spec_from_file_location("final_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------------------------------------- a fake rlexp.metrics / rlexp.ident
def metrics_block(fv=0.2, **over):
    base = {"valid_closed": 0.9, "exact@1": 0.0, "core@1": 0.01, "mean_tanimoto": 0.15, "fv_tanimoto": fv, "fv_samples": 1.5, "fv_tokens": 30.0, "fv_lift": 0.011,
            "lift": 0.022, "tanimoto_valid": 0.2, "core@8": 0.05, "fv_rate": 0.95, "mean_reward": 0.14, "maj@8": 0.5, "truncation": 0.1}
    base.update(over)
    return {k: {"value": v, "lo": None if v is None else v - 0.02, "hi": None if v is None else v + 0.02} for k, v in base.items()}


def split_block(drugs=41, rows=100, fv=0.2, bar=0.176, prompt_tokens=500.0, bar_fv="same", **over):
    block = {"n_drugs": drugs, "n_rows": rows, "n_samples": rows * 8, "prompt_tokens": prompt_tokens, "bootstrap": {"unit": "cluster", "n_units": drugs},
             "metrics": metrics_block(fv, **over), "coverage": {"drugs": drugs, "rows": rows}}
    if bar is not None:
        values = {"fv_tanimoto": bar if bar_fv == "same" else bar_fv, "mean_tanimoto": bar, "mean_reward": 0.15, "lift": 0.0, "exact@1": 0.0, "core@1": 0.0}
        block["constant"] = {"smiles": CONSTANT_MARKER, "mean_bit_tanimoto_to_train": 0.14, "metrics": {k: {"value": v, "lo": 0.0, "hi": 0.1} for k, v in values.items()}}
    return block


class Fake:
    """Stands in for ft.run_cli: `score` writes the scored file, `report` writes the JSON of reports[arm](requested splits), `ident` prints reports[arm]."""

    def __init__(self, ft, monkeypatch, reports, idents=None):
        self.calls, self.reports, self.idents = [], reports, idents or {}
        monkeypatch.setattr(ft, "run_cli", self)

    def __call__(self, module, *args):
        self.calls.append((module, args))
        arg = lambda name: args[args.index(name) + 1]
        if module == "rlexp.ident":
            return "\n".join(json.dumps(r) for r in self.idents[Path(arg("--samples")).parent.name])
        if args[0] == "score":
            Path(arg("--out")).write_text("{}\n")
            return ""
        arm = Path(arg("--scored")).parent.name
        Path(arg("--json")).write_text(json.dumps({"meta": {}, "notes": [], "splits": self.reports[arm](arg("--split").split(","))}))
        return "a text report"

    def of(self, kind, module="rlexp.metrics"):
        return [(m, a) for m, a in self.calls if m == module and a[0] == kind]


def serve(**per_split):
    """A report function: per_split = {split: block}; a requested split the arm lacks gets the empty block of the CLI."""
    return lambda splits: {s: per_split.get(s, {"n_drugs": 0, "n_rows": 0, "n_samples": 0}) for s in splits}


def arm_dir(final, name):
    (final / name).mkdir(parents=True, exist_ok=True)
    (final / name / "samples.jsonl").write_text("{}\n")
    return final / name


def row_cells(md, arm, split="Test-T"):
    """{header: cell} of one arm's row in the table of a split of the markdown."""
    section = md.split(f"## {split}")[1].split("\n## ")[0]
    header = next(x for x in section.splitlines() if x.startswith("| Arm |"))
    row = next(x for x in section.splitlines() if x.startswith(f"| {arm} |"))
    strip = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
    return dict(zip(strip(header), strip(row)))


def go(ft, tmp_path, *extra, splits="test_t,test_r", final=None, md=True, allow=True):
    out = tmp_path / "res" / "table.json"
    argv = ["--final", str(final or tmp_path / "final"), "--splits", splits, "--out", str(out), "--data", str(tmp_path / "data"), "--curated", str(tmp_path / "curated.json"),
            "--boot", "77", *extra]
    if md:
        argv += ["--md", str(tmp_path / "res" / "table.md")]
    if allow:
        argv += ["--allow-test"]
    ft.main(argv)
    return json.loads(out.read_text()), (tmp_path / "res" / "table.md").read_text() if md else None


# ------------------------------------------------------------------------------------------------- rows and columns
def test_a_row_has_the_val_table_columns_and_every_value_comes_from_the_report(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block(fv=0.31), test_r=split_block(30, 80, fv=0.12))})
    res, _ = go(ft, tmp_path)
    t, r = res["rows"]
    assert (t["split"], r["split"]) == ("test_t", "test_r") and (t["drugs"], t["samples"], t["prompt_tokens"]) == (41, 800, 500.0) and (r["drugs"], r["samples"]) == (30, 640)
    for c in ft.COLUMNS:
        assert t[c] == metrics_block(0.31)[c]["value"]                                       # the report's value, untouched
    assert t["fv_tanimoto"] == 0.31 and r["fv_tanimoto"] == 0.12
    assert set(ft.COLUMNS) == set(res["columns"]) and "maj@8" not in t and "truncation" not in t
    for name in ("fv_tanimoto", "mean_tanimoto", "fv_lift", "lift"):
        assert t[f"{name}_ci"] == [t[name] - 0.02, t[name] + 0.02]                             # descriptive component-bootstrap interval
    assert not any(k.endswith("_ci") and k[:-3] not in ("fv_tanimoto", "mean_tanimoto", "fv_lift", "lift") for k in t)
    assert res["splits"] == ["test_t", "test_r"] and res["seed"] == 0 and res["boot"] == 77 and "ORIGINAL prompt" in res["notes"]["prompt_tokens"]


def test_one_report_call_per_arm_names_every_requested_split_in_order_with_the_seed_and_boot(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    fake = Fake(ft, monkeypatch, {"A2": serve(test_t=split_block(), test_r=split_block())})
    go(ft, tmp_path)
    (module, args), = fake.of("report")
    arg = lambda name: args[args.index(name) + 1]
    assert module == "rlexp.metrics" and arg("--split") == "test_t,test_r" and arg("--boot") == "77" and arg("--seed") == "0" and arg("--data") == str(tmp_path / "data")
    assert arg("--scored") == str(tmp_path / "final" / "A2" / "scored.v3.jsonl") and arg("--json") == str(tmp_path / "final" / "A2" / "report.v3.json")


def test_fewer_than_eight_samples_per_row_gives_rows_without_first_valid_values(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "27B-3.8")
    short = split_block(fv=None, fv_samples=None, fv_tokens=None, fv_lift=None, fv_rate=None)
    Fake(ft, monkeypatch, {"27B-3.8": serve(test_t=short)})
    res, md = go(ft, tmp_path, splits="test_t")
    row, = res["rows"]
    assert all(row[k] is None for k in ("fv_tanimoto", "fv_samples", "fv_tokens", "fv_lift", "fv_rate")) and "fv_tanimoto_ci" not in row and "fv_lift_ci" not in row
    assert row["mean_tanimoto"] == 0.15 and row["valid_closed"] == 0.9
    cells = row_cells(md, "27B-3.8")
    assert [cells[h] for h in ("Delivered similarity (95% interval)", "Samples per answer", "Tokens per answer", "Delivered lift")] == ["n/a"] * 4      # and no wrong reason
    assert cells["Mean similarity, all samples"].startswith("0.150") and cells["Valid answers"] == "0.900" and "shared" not in md.split("## Test-T")[1]


# ------------------------------------------------------------------------------------------------- the splits an arm has
def test_an_arm_without_a_requested_split_gets_rows_only_for_the_splits_it_has(ft, tmp_path, monkeypatch):
    final = tmp_path / "final"
    for name in ("27B-3.5", "A2"):
        arm_dir(final, name)
    Fake(ft, monkeypatch, {"27B-3.5": serve(test_t=split_block()), "A2": serve(test_t=split_block(), test_r=split_block(30, 80))})
    res, md = go(ft, tmp_path)
    assert [(r["run"], r["split"]) for r in res["rows"]] == [("A2", "test_t"), ("27B-3.5", "test_t"), ("A2", "test_r")]
    test_r_section = md.split("## Test-R")[1]
    assert "27B-3.5" not in test_r_section and "A2" in test_r_section


def test_the_spec_names_arms_and_splits_that_are_missing_as_gaps(ft, tmp_path, monkeypatch):
    final = tmp_path / "final"
    arm_dir(final, "A2")
    arm_dir(final, "27B-3.5")
    spec = {"frozen": True, "splits": ["test_t", "test_r"], "arms": [
        {"id": "A2", "temperature": 1.0, "k": 8, "mode": "think"},
        {"id": "27B-3.5", "temperature": 1.0, "k": 8, "mode": "nothink", "splits": ["test_t"], "role": "reference"},        # test_t only by design
        {"id": "RL-full", "temperature": 0.6, "k": 8, "mode": "nothink"},                                                    # no samples at all
        {"id": "A2-blank", "temperature": 1.0, "k": 8, "mode": "think", "role": "diagnostic", "data": "blank"}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block()), "27B-3.5": serve(test_t=split_block())})        # A2 lacks test_r although the spec expects it
    res, md = go(ft, tmp_path, "--spec", str(tmp_path / "spec.json"))
    assert res["gaps"] == [{"run": "A2-blank", "split": None}, {"run": "RL-full", "split": None}, {"run": "A2", "split": "test_r"}]
    assert "Planned but not run: A2-blank, RL-full, A2 on Test-R." in md.split("## Test-R")[1]
    assert not any(g["run"] == "27B-3.5" for g in res["gaps"])                       # test_r was never planned for it


def test_no_spec_means_no_gaps(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block())})
    res, md = go(ft, tmp_path)
    assert res["gaps"] == [] and "Planned but not run" not in md


# ------------------------------------------------------------------------------------------------- scoring only when needed
def test_samples_are_scored_only_when_scored_is_missing_or_older(ft, tmp_path, monkeypatch):
    final = tmp_path / "final"
    for name in ("A2", "A4", "A0n"):
        arm_dir(final, name)
    fake = Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in ("A2", "A4", "A0n")})
    for name, scored_time in (("A4", 2000), ("A0n", 500)):                           # fresh, and older than the samples
        (final / name / "scored.v3.jsonl").write_text("{}\n")
        os.utime(final / name / "scored.v3.jsonl", (scored_time, scored_time))
        os.utime(final / name / "samples.jsonl", (1000, 1000))
    go(ft, tmp_path, splits="test_t")
    scored = sorted(Path(a[a.index("--samples") + 1]).parent.name for _, a in fake.of("score"))
    assert scored == ["A0n", "A2"]                                                    # A2: no scored file; A0n: stale; A4 is left alone
    assert len(fake.of("report")) == 3


# ------------------------------------------------------------------------------------------------- blank-profile arms
@pytest.mark.parametrize("run,data", [("A2-blank", None), ("RL-full-blank", None), ("whatever", "blank")])
def test_blank_profile_arms_hide_the_first_valid_columns_in_markdown_and_keep_them_in_json(ft, tmp_path, monkeypatch, run, data):
    arm_dir(tmp_path / "final", run)
    Fake(ft, monkeypatch, {run: serve(test_t=split_block(fv=0.123, mean_tanimoto=0.098, valid_closed=0.77))})
    spec = ["--spec", str(tmp_path / "spec.json")] if data else []
    if data:
        (tmp_path / "spec.json").write_text(json.dumps({"splits": ["test_t"], "arms": [{"id": run, "data": data, "temperature": 1.0, "k": 8, "mode": "think"}]}))
    res, md = go(ft, tmp_path, *spec, splits="test_t")
    row, = res["rows"]
    assert row["shared_draws"] is True and row["fv_tanimoto"] == 0.123 and row["fv_samples"] == 1.5 and row["fv_tokens"] == 30.0 and row["fv_lift"] == 0.011
    cells = row_cells(md, run)
    assert [cells[h] for h in ("Delivered similarity (95% interval)", "Samples per answer", "Tokens per answer", "Delivered lift")] == ["n/a (shared draws)"] * 4
    assert cells["Mean similarity, all samples"].startswith("0.098") and cells["Valid answers"] == "0.770"     # these are printed
    assert "its first valid answer is one shared molecule" in md


def test_other_arms_are_not_marked_as_shared_draws(ft, tmp_path, monkeypatch):
    for name in ("A2", "blanket"):
        arm_dir(tmp_path / "final", name)
    Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in ("A2", "blanket")})
    res, md = go(ft, tmp_path, splits="test_t")
    assert [r["shared_draws"] for r in res["rows"]] == [False, False] and "n/a (shared" not in md


# ------------------------------------------------------------------------------------------------- identification arms
IDENT = {"split": "test_t", "drugs": 41, "rows": 100, "samples": 800, "pass_k": 8, "name_match@1": 0.0641, "name_pass@8": 0.25, "answered": 0.9}


def test_identification_arms_report_name_match_and_pass_k_from_the_ident_scorer_only(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "ident-4B")
    arm_dir(tmp_path / "final", "ident-27B")
    idents = {"ident-4B": [IDENT, {**IDENT, "split": "test_r", "drugs": 30}], "ident-27B": [{**IDENT, "pass_k": 2, "name_pass@2": 0.4, "name_match@1": 0.2}]}
    del idents["ident-27B"][0]["name_pass@8"]
    fake = Fake(ft, monkeypatch, {}, idents)
    res, md = go(ft, tmp_path)
    assert fake.of("score") == [] and fake.of("report") == []                          # a name is not a SMILES: the metrics CLI never sees an identification run
    got = {(r["run"], r["split"]): r for r in res["rows"]}
    assert all(r["kind"] == "identification" for r in res["rows"])
    assert set(got) == {("ident-4B", "test_t"), ("ident-4B", "test_r"), ("ident-27B", "test_t")}
    r = got["ident-4B", "test_t"]
    assert (r["name_match@1"], r["name_pass"], r["pass_k"], r["answered"], r["drugs"], r["samples"]) == (0.0641, 0.25, 8, 0.9, 41, 800)
    assert got["ident-27B", "test_t"]["pass_k"] == 2 and got["ident-27B", "test_t"]["name_pass"] == 0.4          # a run of 2 samples reports pass@2, never a pass@8
    assert "fv_tanimoto" not in r and "mean_tanimoto" not in r
    section = md.split("## Test-T")[1].split("## Test-R")[0]
    assert "Identification" in section and "| ident-4B |" in section and "0.064 | 0.250 | 8 | 0.900 | 41 |" in section and "| ident-27B |" in section
    assert "No arm with generated structures" in section


def test_identification_rows_of_unrequested_splits_are_left_out(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "ident-4B")
    Fake(ft, monkeypatch, {}, {"ident-4B": [IDENT, {**IDENT, "split": "test_r"}]})
    res, _ = go(ft, tmp_path, splits="test_t")
    assert [(r["run"], r["split"]) for r in res["rows"]] == [("ident-4B", "test_t")]


def test_the_ident_scorer_gets_the_curated_names_and_the_data(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "ident-4B")
    fake = Fake(ft, monkeypatch, {}, {"ident-4B": [IDENT]})
    go(ft, tmp_path, splits="test_t")
    (module, args), = fake.calls
    arg = lambda name: args[args.index(name) + 1]
    assert module == "rlexp.ident" and args[0] == "score" and len(fake.of("score", "rlexp.ident")) == 1 and arg("--curated") == str(tmp_path / "curated.json") and arg("--data") == str(tmp_path / "data")


# ------------------------------------------------------------------------------------------------- labels, decoding, order
def test_label_and_decoding_come_from_the_spec_else_the_id(ft, tmp_path, monkeypatch):
    final = tmp_path / "final"
    for name in ("A2", "RL-full", "27B-3.8", "C3", "Other"):
        arm_dir(final, name)
    spec = {"splits": ["test_t", "test_r"], "arms": [
        {"id": "A2", "label": "4B answer-only SFT", "temperature": 1.0, "k": 8, "mode": "think"},
        {"id": "RL-full", "temperature": 0.6, "k": 8, "mode": "nothink"},                                  # no label: the id
        {"id": "27B-3.8", "label": "Qwen3.8-27B", "temperature": 1.0, "k": 2, "mode": "nothink", "role": "reference", "splits": ["test_t"]}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in ("A2", "RL-full", "27B-3.8", "C3", "Other")})
    res, md = go(ft, tmp_path, "--spec", str(tmp_path / "spec.json"), splits="test_t")
    got = {r["run"]: (r["label"], r["decoding"]) for r in res["rows"]}
    assert got["A2"] == ("4B answer-only SFT", "T=1.0, k=8, think mode") and got["RL-full"] == ("RL-full", "T=0.6, k=8, no-think")
    assert got["27B-3.8"] == ("Qwen3.8-27B", "T=1.0, k=2, no-think, Test-T only")
    assert got["C3"] == ("C3", "deterministic, copied to eight samples") and got["Other"] == ("Other", "n/a")        # not in the spec: the id
    assert "| A2: 4B answer-only SFT | 4B arm | T=1.0, k=8, think mode |" in md and "| RL-full | 4B arm |" in md              # id: label, or the id alone when there is no label


def test_a_pipe_in_a_label_or_decoding_does_not_break_the_markdown_table(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    (tmp_path / "spec.json").write_text(json.dumps({"splits": ["test_t"], "arms": [{"id": "A2", "label": "SFT | answers only", "temperature": 1.0, "k": 8, "mode": "think"}]}))
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block())})
    res, md = go(ft, tmp_path, "--spec", str(tmp_path / "spec.json"), splits="test_t")
    assert res["rows"][0]["label"] == "SFT | answers only"                                   # the data keeps the label, the table shows a slash
    cells = row_cells(md, "A2: SFT / answers only")
    assert cells["Arm"] == "A2: SFT / answers only" and cells["Group"] == "4B arm" and cells["Decoding"] == "T=1.0, k=8, think mode" and len(cells) == 12


def test_every_markdown_row_names_the_arm_by_id_and_label_so_a_swapped_label_cannot_rename_it(ft, tmp_path, monkeypatch):
    for name in ("A2", "RL-full", "ident-4B"):
        arm_dir(tmp_path / "final", name)
    spec = {"splits": ["test_t"], "arms": [{"id": "A2", "label": "RL-full", "temperature": 1.0, "k": 8, "mode": "think"},          # A2 wearing RL-full's name
                                            {"id": "RL-full", "label": "A2", "temperature": 0.6, "k": 8, "mode": "think"},
                                            {"id": "ident-4B", "label": "Identification, 4B", "data": "ident", "temperature": 1.0, "k": 2, "mode": "nothink", "role": "probe"}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    ident = [{"split": "test_t", "drugs": 41, "samples": 100, "name_match@1": 0.01, "pass_k": 2, "name_pass@2": 0.02, "answered": 0.9}]
    Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in ("A2", "RL-full")}, {"ident-4B": ident})
    res, md = go(ft, tmp_path, "--spec", str(tmp_path / "spec.json"), splits="test_t")
    assert "| A2: RL-full | 4B arm |" in md and "| RL-full: A2 | 4B arm |" in md and "| ident-4B: Identification, 4B |" in md
    assert [r["label"] for r in res["rows"] if r["run"] == "A2"] == ["RL-full"]                  # the data keeps the label; the table shows both


def test_the_spec_must_be_the_one_the_run_manifest_names_when_there_is_a_run_manifest(ft, tmp_path, monkeypatch):
    import hashlib
    arm_dir(tmp_path / "final", "A2")
    spec = {"splits": ["test_t"], "arms": [{"id": "A2", "label": "A2", "temperature": 1.0, "k": 8, "mode": "think"}]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block())})
    go(ft, tmp_path, "--spec", str(spec_path), splits="test_t")                                      # no run manifest: nothing to compare (a Val directory, a rehearsal)
    manifest = tmp_path / "final" / "manifest.json"
    manifest.write_text(json.dumps({"spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(), "arms": {}}))
    go(ft, tmp_path, "--spec", str(spec_path), splits="test_t")                                      # the spec the run was made under
    spec_path.write_text(json.dumps({**spec, "arms": [{**spec["arms"][0], "label": "Something else"}]}))      # a label edited after the run
    with pytest.raises(SystemExit, match="--spec is not the spec the run in .* was made under"):
        go(ft, tmp_path, "--spec", str(spec_path), splits="test_t")
    manifest.write_text("{")
    with pytest.raises(SystemExit, match="cannot read the run manifest"):
        go(ft, tmp_path, "--spec", str(spec_path), splits="test_t")
    manifest.write_text(json.dumps({"arms": {}}))                                                      # a manifest that names no spec
    with pytest.raises(SystemExit, match="was made under"):
        go(ft, tmp_path, "--spec", str(spec_path), splits="test_t")
    go(ft, tmp_path, splits="test_t")                                                                  # without --spec there is nothing to check


def test_without_a_spec_the_label_is_the_id(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block())})
    res, _ = go(ft, tmp_path, splits="test_t")
    assert (res["rows"][0]["label"], res["rows"][0]["decoding"]) == ("A2", "n/a")


def test_rows_are_ordered_controls_4b_arms_references_diagnostics_with_numbers_read_as_numbers(ft, tmp_path, monkeypatch):
    names = ["name-4B", "A2-blank", "27B-3.5", "9B", "RL-full", "A4", "A2", "A0n", "C3", "C0", "27B-3.8", "RL-full-shuffle", "C10", "C2"]
    for name in names:
        arm_dir(tmp_path / "final", name)
    Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in names})
    res, _ = go(ft, tmp_path, splits="test_t")
    assert [r["run"] for r in res["rows"]] == ["C0", "C2", "C3", "C10", "A0n", "A2", "A4", "RL-full", "9B", "27B-3.5", "27B-3.8", "A2-blank", "name-4B", "RL-full-shuffle"]
    assert [r["family"] for r in res["rows"]] == ["Control"] * 4 + ["4B arm"] * 4 + ["Reference"] * 3 + ["Diagnostic"] * 3


def test_inside_a_family_the_spec_order_comes_first_then_unlisted_arms_by_id(ft, tmp_path, monkeypatch):
    names = ["A0n", "A2", "A4", "RL-full", "name-4B", "name-9B", "name-27B", "name-3.8", "RL-full-shuffle", "A2-shuffle", "C1", "C0"]
    for name in names + ["Zextra", "Aextra"]:
        arm_dir(tmp_path / "final", name)
    listed = ["A0n", "A4", "A2", "RL-full", "A2-shuffle", "RL-full-shuffle", "name-4B", "name-9B", "name-27B", "name-3.8"]           # the prereg run list, not alphabetical
    spec = {"splits": ["test_t"], "arms": [{"id": a, "temperature": 1.0, "k": 8, "mode": "nothink", **({"role": "diagnostic"} if a.startswith("name") or a.endswith("shuffle") else {})} for a in listed]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in [*names, "Zextra", "Aextra"]})
    res, _ = go(ft, tmp_path, "--spec", str(tmp_path / "spec.json"), splits="test_t")
    assert [r["run"] for r in res["rows"]] == ["C0", "C1", "A0n", "A4", "A2", "RL-full", "Aextra", "Zextra", "A2-shuffle", "RL-full-shuffle", "name-4B", "name-9B", "name-27B", "name-3.8"]
    res, _ = go(ft, tmp_path, splits="test_t")                                                       # without the spec: by id
    assert [r["run"] for r in res["rows"] if r["family"] == "Diagnostic"] == ["A2-shuffle", "name-3.8", "name-4B", "name-9B", "name-27B", "RL-full-shuffle"]


def test_the_spec_role_decides_the_family_of_the_arms_it_lists(ft, tmp_path, monkeypatch):
    names = ["A2", "9B", "A2-shuffle", "name-4B", "weird"]
    for name in names:
        arm_dir(tmp_path / "final", name)
    spec = {"splits": ["test_t"], "arms": [{"id": "A2", "temperature": 1.0, "k": 8, "mode": "think"}, {"id": "9B", "role": "reference", "temperature": 1.0, "k": 8, "mode": "nothink"},
                                           {"id": "A2-shuffle", "role": "diagnostic", "temperature": 1.0, "k": 8, "mode": "think"},
                                           {"id": "name-4B", "role": "probe", "temperature": 1.0, "k": 8, "mode": "nothink"},
                                           {"id": "weird", "temperature": 1.0, "k": 8, "mode": "nothink"}]}                 # listed without a role: a selected 4B arm
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    Fake(ft, monkeypatch, {n: serve(test_t=split_block()) for n in names})
    res, _ = go(ft, tmp_path, "--spec", str(tmp_path / "spec.json"), splits="test_t")
    assert {r["run"]: r["family"] for r in res["rows"]} == {"A2": "4B arm", "weird": "4B arm", "9B": "Reference", "A2-shuffle": "Diagnostic", "name-4B": "Diagnostic"}
    assert not any(r["shared_draws"] for r in res["rows"])                                  # a shuffled-profile or name-given run has its own draws per prompt


def test_the_markdown_has_one_table_per_split_in_the_order_asked_with_plain_english_headers(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block(), test_r=split_block(30, 80))})
    _, md = go(ft, tmp_path, splits="test_r,test_t")
    assert md.index("## Test-R") < md.index("## Test-T") and md.count("| Arm | Group | Decoding |") == 2
    for header in ("Valid answers", "Delivered similarity (95% interval)", "Samples per answer", "Tokens per answer", "Delivered lift", "Lift, all samples",
                   "Mean similarity, all samples", "Core match", "Exact match"):
        assert header in md
    assert "fv_" not in md and "tanimoto" not in md.lower().replace("tanimoto similarity", "") and "scored.v3" not in md          # plain English: no code identifiers
    cells = row_cells(md, "A2")
    assert cells["Delivered similarity (95% interval)"] == "0.200 [0.180, 0.220]" and "Prompt tokens" not in cells and "500" not in md      # value and interval; no prompt length
    assert cells["Delivered lift"] == "+0.011" and cells["Lift, all samples"] == "+0.022" and cells["Mean similarity, all samples"] == "0.150"      # signed lifts; no other interval
    assert "Test-T split, 41 drugs" in md.split("## Test-T")[1] and "Test-R split, 30 drugs" in md.split("## Test-R")[1]


# ------------------------------------------------------------------------------------------------- the bar and C3
def test_the_header_holds_the_constant_bar_and_c3_per_split_but_not_the_molecule(ft, tmp_path, monkeypatch):
    for name in ("C3", "A2"):
        arm_dir(tmp_path / "final", name)
    Fake(ft, monkeypatch, {"C3": serve(test_t=split_block(fv=0.187, mean_tanimoto=0.187, bar=0.176), test_r=split_block(30, 80, fv=0.151, mean_tanimoto=0.151, bar=0.141)),
                           "A2": serve(test_t=split_block(fv=0.1, bar=0.176), test_r=split_block(30, 80, bar=0.141))})
    res, md = go(ft, tmp_path)
    assert res["c3"] == {"test_t": {"fv_tanimoto": 0.187, "mean_tanimoto": 0.187}, "test_r": {"fv_tanimoto": 0.151, "mean_tanimoto": 0.151}}
    assert res["constant_bar"]["test_t"] == {"from": "C3", "fv_tanimoto": 0.176, "mean_tanimoto": 0.176, "mean_reward": 0.15, "lift": 0.0, "exact@1": 0.0, "core@1": 0.0}
    assert res["constant_bar"]["test_r"]["fv_tanimoto"] == 0.141
    assert CONSTANT_MARKER not in json.dumps(res) + md                                  # the bar's molecule is never copied
    assert "delivered similarity 0.187" in md.split("## Test-T")[1] and "delivered similarity 0.176" in md.split("## Test-T")[1]


def test_the_bar_skips_a_first_run_whose_bar_has_no_first_valid_value_but_falls_back_to_it(ft, tmp_path, monkeypatch):
    for name in ("C0", "9B"):
        arm_dir(tmp_path / "final", name)
    Fake(ft, monkeypatch, {"C0": serve(test_t=split_block(bar=0.1, bar_fv=None)), "9B": serve(test_t=split_block(bar=0.2))})        # a k < 8 run has no fv on the bar
    res, _ = go(ft, tmp_path, splits="test_t")
    assert res["constant_bar"]["test_t"]["from"] == "9B" and res["constant_bar"]["test_t"]["fv_tanimoto"] == 0.2
    Fake(ft, monkeypatch, {"C0": serve(test_t=split_block(bar=0.1, bar_fv=None)), "9B": serve(test_t=split_block(bar=0.2, bar_fv=None))})
    res, md = go(ft, tmp_path, splits="test_t")
    assert res["constant_bar"]["test_t"]["from"] == "C0" and res["constant_bar"]["test_t"]["fv_tanimoto"] is None and res["constant_bar"]["test_t"]["mean_tanimoto"] == 0.1
    assert "mean similarity 0.100, delivered similarity n/a" in md


def test_the_bar_comes_from_the_first_run_that_has_a_first_valid_value_and_c3_may_be_absent(ft, tmp_path, monkeypatch):
    for name in ("C1", "A2"):
        arm_dir(tmp_path / "final", name)
    Fake(ft, monkeypatch, {"C1": serve(test_t=split_block(bar=None)),
                           "A2": serve(test_t=split_block())})
    res, md = go(ft, tmp_path, splits="test_t")
    assert res["constant_bar"]["test_t"]["from"] == "A2" and res["c3"] == {"test_t": None}
    assert "C3, the reward-optimal constant" not in md
    no_bar = tmp_path / "final2"
    arm_dir(no_bar, "A2")
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block(bar=None))})
    res, md = go(ft, tmp_path, splits="test_t", final=no_bar)
    assert res["constant_bar"] == {"test_t": None} and "Best constant answer" not in md


# ------------------------------------------------------------------------------------------------- refusals and plain errors
@pytest.mark.parametrize("splits", ["test_t", "test_r", "val,test_t", "test_t,test_r"])
def test_a_test_split_needs_allow_test(ft, tmp_path, monkeypatch, splits):
    arm_dir(tmp_path / "final", "A2")
    fake = Fake(ft, monkeypatch, {"A2": serve()})
    with pytest.raises(SystemExit, match="refused: .* is a Test split"):
        go(ft, tmp_path, splits=splits, allow=False)
    assert fake.calls == [] and not (tmp_path / "res" / "table.json").exists()


def test_val_needs_no_flag_and_unknown_splits_are_errors(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    Fake(ft, monkeypatch, {"A2": serve(val=split_block(20, 58))})
    res, md = go(ft, tmp_path, splits="val", allow=False)
    assert res["splits"] == ["val"] and res["rows"][0]["split"] == "val" and "## Val" in md and "Test" not in md
    for bad in ("train", "test", "val,foo", ""):
        with pytest.raises(SystemExit, match="error: splits must come from"):
            go(ft, tmp_path, splits=bad, allow=False)


def test_a_missing_final_directory_or_one_without_samples_is_a_plain_error(ft, tmp_path):
    with pytest.raises(SystemExit, match="error: .*final is not a directory"):
        go(ft, tmp_path)
    (tmp_path / "final" / "A2").mkdir(parents=True)
    with pytest.raises(SystemExit, match="error: no <id>/samples.jsonl"):
        go(ft, tmp_path)
    with pytest.raises(SystemExit, match="error: cannot read the spec"):
        go(ft, tmp_path, "--spec", str(tmp_path / "nope.json"))


def test_a_split_named_twice_is_one_split_and_cells_never_print_a_negative_zero(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    Fake(ft, monkeypatch, {"A2": serve(test_t=split_block())})
    res, md = go(ft, tmp_path, splits="test_t,test_t")
    assert res["splits"] == ["test_t"] and len(res["rows"]) == 1 and md.count("## Test-T") == 1
    assert ft.cell(-1e-9, 3, True) == "+0.000" and ft.cell(-1e-9, 3, False) == "0.000" and ft.cell(0.0004, 3, True) == "+0.000" and ft.cell(-0.0016, 3, True) == "-0.002"
    assert ft.cell(0.00049999, 3, False) == "0.000" and ft.cell(0.12449, 3, False) == "0.124"                  # rounded once, to the printed decimals
    assert [ft.cell(v, 2, False) for v in (None, float("nan"), float("inf"), True, "0.5")] == ["n/a"] * 5 and ft.cell(3, 1, False) == "3.0"


def test_no_rows_on_any_requested_split_is_an_error_and_writes_nothing(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "27B-3.5")
    Fake(ft, monkeypatch, {"27B-3.5": serve(test_t=split_block())})
    with pytest.raises(SystemExit, match="error: no arm has rows on test_r"):
        go(ft, tmp_path, splits="test_r")
    assert not (tmp_path / "res" / "table.json").exists()


def test_a_failing_metrics_call_exits_with_a_cleaned_message_that_names_no_drug(ft, tmp_path, monkeypatch):
    arm_dir(tmp_path / "final", "A2")
    stderr = "error: " + "x" * 400 + " 3 unknown id(s) not in the data files, e.g. ['secretdrugname/0', 'otherdrug/1']"
    monkeypatch.setattr(ft, "subprocess", types.SimpleNamespace(run=lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr=stderr)))
    with pytest.raises(SystemExit) as e:
        go(ft, tmp_path)
    message = str(e.value)
    assert message.startswith("error: rlexp.metrics score failed") and "secretdrugname" not in message and "otherdrug" not in message
    assert not (tmp_path / "res" / "table.json").exists()


def test_run_cli_runs_a_module_with_the_python_of_the_script(ft, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ft, "subprocess", types.SimpleNamespace(run=lambda cmd, **kw: seen.append(cmd) or types.SimpleNamespace(returncode=0, stdout="out", stderr="")))
    assert ft.run_cli("rlexp.ident", "score", "--samples", "s") == "out" and seen == [[ft.sys.executable, "-m", "rlexp.ident", "score", "--samples", "s"]]


# ------------------------------------------------------------------------------------------------- determinism and no data in the output
def test_the_output_is_deterministic_and_has_no_timestamp_or_path_of_the_machine(ft, tmp_path, monkeypatch, capsys):
    for name in ("C3", "A2", "ident-4B"):
        arm_dir(tmp_path / "final", name)
    Fake(ft, monkeypatch, {n: serve(test_t=split_block(), test_r=split_block(30, 80)) for n in ("C3", "A2")}, {"ident-4B": [IDENT]})
    _, md1 = go(ft, tmp_path)
    first = (tmp_path / "res" / "table.json").read_bytes()
    out1 = capsys.readouterr().out
    _, md2 = go(ft, tmp_path)
    assert (tmp_path / "res" / "table.json").read_bytes() == first and md1 == md2 and capsys.readouterr().out == out1
    assert str(tmp_path) not in first.decode() and str(tmp_path) not in md1


def test_no_answer_text_drug_name_or_molecule_reaches_any_output(ft, tmp_path, monkeypatch, capsys):
    final = tmp_path / "final"
    for name in ("A2", "ident-4B"):
        arm = arm_dir(final, name)
        write_jsonl(arm / "samples.jsonl", [{"id": "secretdrug/0", "group": "secretdrug", "split": "test_t", "sample_idx": 0, "text": f"think</think>\n{SMILES[0]}", "finish_reason": "stop"}])
    reports = {"A2": lambda splits: {s: {**split_block(), "answer_marker": SMILES[1]} for s in splits}}
    Fake(ft, monkeypatch, reports, {"ident-4B": [IDENT]})
    res, md = go(ft, tmp_path)
    cap = capsys.readouterr()
    seen = json.dumps(res) + md + cap.out + cap.err
    for marker in (*SMILES, "secretdrug", CONSTANT_MARKER, "think</think>", "answer_marker"):
        assert marker not in seen


# ------------------------------------------------------------------------------------------------- the real CLIs, tiny data
DRUGS = {"train": ["trainone", "traintwo", "trainthree"], "val": ["valone", "valtwo"], "test_t": ["testtone", "testttwo", "testtthree", "testtfour"], "test_r": ["testrone", "testrtwo"]}


def make_world(tmp_path):
    """data/<split>.jsonl with real references (each drug 1 or 2 prompt rows, its own cluster) and the answers of three runs: C3 (a constant, eight copies per row),
    A2 (three unparseable samples, then the exact reference five times) and ident-4B (the drug's name in 2 of 8 samples)."""
    data, k, refs, rows = tmp_path / "data", 0, {}, {}
    data.mkdir()
    for split, groups in DRUGS.items():
        rows[split] = []
        for g in groups:
            refs[g] = SMILES[k]
            rows[split] += [{"id": f"{g}/{i}", "group": g, "task_type": "A", "split": split, "cluster_id": k, "ref_smiles": SMILES[k], "ref_conn": f"conn-{g}",
                             "ref_scaffold": f"scaf-{g}", "ref_alt_smiles": [], "n_prompt_tokens": 100 + k} for i in range(1 + k % 2)]
            k += 1
        write_jsonl(data / f"{split}.jsonl", rows[split])
    (tmp_path / "curated.json").write_text("[]")
    final = tmp_path / "final"

    def samples(name, answer, splits=("test_t", "test_r")):
        sample = lambda r, j: {"id": r["id"], "group": r["group"], "split": r["split"], "sample_idx": j, "text": f"t</think>\n{answer(r, j)}", "finish_reason": "stop",
                               "n_tokens": 7, "model": "m", "mode": "nothink"}
        write_jsonl(arm_dir(final, name) / "samples.jsonl", [sample(r, j) for s in splits for r in rows[s] for j in range(8)])
    samples("C3", lambda r, j: SMILES[11])
    samples("A2", lambda r, j: "no molecule in this text" if j < 3 else r["ref_smiles"])
    samples("9B", lambda r, j: r["ref_smiles"], splits=("test_t",))                                    # a run on test_t only, like a 27B
    samples("ident-4B", lambda r, j: r["group"] if j < 2 else "somethingelse")
    return refs


@pytest.fixture(scope="module")
def real_run(ft, tmp_path_factory):
    """One real run of the script over the tiny world: (directory, JSON, markdown, everything it printed)."""
    import contextlib
    import io
    tmp_path = tmp_path_factory.mktemp("real")
    make_world(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        res, md = go(ft, tmp_path)
    return tmp_path, res, md, out.getvalue() + err.getvalue()


def test_the_real_score_report_and_ident_clis_end_to_end(real_run):
    tmp_path, res, md, printed = real_run
    got = {(r["run"], r["split"]): r for r in res["rows"]}
    assert set(got) == {(a, s) for a in ("C3", "A2", "ident-4B") for s in ("test_t", "test_r")} | {("9B", "test_t")}          # 9B has no test_r rows: none are made up
    assert got["9B", "test_t"]["fv_tanimoto"] == pytest.approx(1.0) and got["9B", "test_t"]["drugs"] == 4 and "9B" not in md.split("## Test-R")[1].split("Identification")[0]
    a2, c3 = got["A2", "test_t"], got["C3", "test_t"]
    assert (a2["drugs"], a2["samples"]) == (4, 6 * 8) and got["A2", "test_r"]["drugs"] == 2 and got["A2", "test_r"]["samples"] == 3 * 8
    # A2: three unparseable samples, then the exact reference five times: the first valid is sample 3 and it is exact
    assert a2["valid_closed"] == pytest.approx(5 / 8) and a2["exact@1"] == pytest.approx(5 / 8) and a2["fv_tanimoto"] == pytest.approx(1.0)
    assert a2["fv_samples"] == pytest.approx(4.0) and a2["fv_rate"] == 1.0 and a2["fv_tokens"] == pytest.approx(28.0) and a2["core@8"] == 1.0
    assert a2["prompt_tokens"] is not None and a2["fv_tanimoto_ci"][0] <= 1.0 <= a2["fv_tanimoto_ci"][1]
    # C3: one constant, always valid, never the reference
    assert c3["valid_closed"] == 1.0 and c3["exact@1"] == 0.0 and c3["fv_samples"] == 1.0 and c3["fv_tanimoto"] == pytest.approx(c3["mean_tanimoto"]) and 0 < c3["fv_tanimoto"] < 1
    assert res["c3"]["test_t"]["fv_tanimoto"] == pytest.approx(c3["fv_tanimoto"]) and res["c3"]["test_r"]["fv_tanimoto"] == pytest.approx(got["C3", "test_r"]["fv_tanimoto"])
    bar = res["constant_bar"]["test_t"]
    assert bar["from"] == "C3" and 0 <= bar["fv_tanimoto"] < 1 and bar["lift"] is not None and "smiles" not in bar          # the first run in table order: a control
    i = got["ident-4B", "test_t"]
    assert (i["kind"], i["drugs"], i["samples"], i["pass_k"]) == ("identification", 4, 6 * 8, 8)
    assert i["name_match@1"] == pytest.approx(0.25) and i["name_pass"] == 1.0 and i["answered"] == 1.0
    # the files an arm leaves behind: the scored file and the report next to its samples; an identification run is not scored as SMILES
    assert (tmp_path / "final" / "A2" / "scored.v3.jsonl").is_file() and (tmp_path / "final" / "A2" / "report.v3.json").is_file()
    assert not (tmp_path / "final" / "ident-4B" / "scored.v3.jsonl").exists()
    for marker in (*SMILES, *[g for gs in DRUGS.values() for g in gs], "no molecule in this text", "somethingelse", "</think>"):
        assert marker not in json.dumps(res) + md + printed


def test_a_second_real_run_scores_nothing_again_and_writes_the_same_bytes(ft, real_run):
    tmp_path, *_ = real_run
    scored = lambda: {p: p.stat().st_mtime_ns for p in (tmp_path / "final").glob("*/scored.v3.jsonl")}
    before, first = scored(), (tmp_path / "res" / "table.json").read_bytes()
    go(ft, tmp_path)
    assert scored() == before and (tmp_path / "res" / "table.json").read_bytes() == first


def test_the_real_cli_path_agrees_with_an_in_process_report(real_run):
    from rlexp import metrics
    tmp_path, res, *_ = real_run
    scored = metrics.read_jsonl(tmp_path / "final" / "A2" / "scored.v3.jsonl", drop=("text",))
    want = metrics.report(scored, splits=["test_t"], boot=77, seed=0, data=metrics.load_data(tmp_path / "data"))["splits"]["test_t"]["metrics"]
    row = next(r for r in res["rows"] if r["run"] == "A2" and r["split"] == "test_t")
    for c in ("valid_closed", "exact@1", "core@1", "mean_tanimoto", "fv_tanimoto", "fv_samples", "fv_tokens", "fv_lift", "lift", "tanimoto_valid", "core@8", "fv_rate", "mean_reward"):
        assert row[c] == want[c]["value"]                                               # same numbers, to the last bit
    assert row["fv_tanimoto_ci"] == [want["fv_tanimoto"]["lo"], want["fv_tanimoto"]["hi"]]
