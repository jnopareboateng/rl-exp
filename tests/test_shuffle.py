import json

import pytest

from rlexp import shuffle


def row(rid, group, cluster, prompt):
    return {"id": rid, "group": group, "cluster_id": cluster, "prompt": prompt, "ref_smiles": "CCO", "split": "val"}


ROWS = [row("a/0", "a", 1, "PA0"), row("a/1", "a", 1, "PA1"), row("b/0", "b", 2, "PB0"), row("c/0", "c", 2, "PC0"),
        row("d/0", "d", 3, "PD0"), row("e/0", "e", 4, "PE0"), row("e/1", "e", 4, "PE1"), row("e/2", "e", 4, "PE2")]


def test_derange_has_no_fixed_point_and_avoids_own_component_when_possible():
    drugs = ["a", "b", "c", "d", "e"]
    clusters = {"a": 1, "b": 2, "c": 2, "d": 3, "e": 4}
    for seed in range(30):
        image = shuffle.derange(drugs, clusters, seed)
        assert sorted(image) == drugs and sorted(image.values()) == drugs
        assert all(image[d] != d for d in drugs)
        assert all(clusters[d] != clusters[image[d]] for d in drugs)
    assert shuffle.derange(drugs, clusters, 3) == shuffle.derange(drugs, clusters, 3)          # seeded
    assert any(shuffle.derange(drugs, clusters, s) != shuffle.derange(drugs, clusters, 0) for s in range(1, 10))


def test_derange_falls_back_when_components_force_overlap_and_rejects_one_drug():
    image = shuffle.derange(["a", "b", "c"], {"a": 1, "b": 1, "c": 1}, 0)                      # one component: only fixed points are avoided
    assert all(image[d] != d for d in image)
    with pytest.raises(ValueError):
        shuffle.derange(["a"], {"a": 1}, 0)


def test_shuffle_rows_keeps_every_field_but_the_prompt_and_preserves_the_prompt_multiset():
    out, image = shuffle.shuffle_rows(ROWS, 0)
    assert [{k: v for k, v in r.items() if k != "prompt"} for r in out] == [{k: v for k, v in r.items() if k != "prompt"} for r in sorted(ROWS, key=lambda r: (r["group"], r["id"]))]
    for r in out:                                                       # the prompt belongs to the drug it was mapped to, never the own drug
        source = r["prompt"][1].lower()
        assert source == image[r["group"]] and source != r["group"]
    assert sorted(r["prompt"] for r in out if r["group"] != "e") and len({r["prompt"] for r in out}) >= 4


def test_views_wrap_when_the_image_has_fewer_rows():
    out, image = shuffle.shuffle_rows(ROWS, 0)
    for r in out:
        if r["group"] == "e":
            target = [x for x in ROWS if x["group"] == image["e"]]
            assert r["prompt"] in {t["prompt"] for t in target}


def test_cli_writes_files_and_mapping_without_prompt_text(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("val", "test_r", "test_t"):
        (data / f"{split}.jsonl").write_text("".join(json.dumps({**r, "split": split}) + "\n" for r in ROWS))
    shuffle.main(["write", "--data", str(data), "--out", str(tmp_path / "out"), "--splits", "val"])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["val.jsonl", "val.meta.json"]
    meta = json.loads((tmp_path / "out" / "val.meta.json").read_text())
    assert meta["control"] == "shuffled_profile" and meta["rows"] == len(ROWS) and set(meta["mapping"]) == {"a", "b", "c", "d", "e"}
    assert "PA0" not in json.dumps(meta) and "PA0" not in capsys.readouterr().out
