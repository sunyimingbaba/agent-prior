"""Check actual candidate selection on independent archived and newly sampled answers."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
STAGE = OUT.parent
REPORTS = STAGE.parent
PREVIOUS = REPORTS / "E3_回填后双收益训练图筛选_20261127"
ARCH = PREVIOUS / "留句核查/独立图库"
sys.path[:0] = [str(ARCH), str(STAGE / "code")]
from 复算 import OLD, COMPARISON, bank_inputs, jsonl, load, read, readout, sha
from check import summarize
from select_pairs import ranked_pair


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def assess(group, seed, block):
    ab = group["items"]
    count = len(ab[0]["other_caption_rr"])
    if count < 2:
        return []
    r2 = np.asarray(group["r2"])
    tied = np.flatnonzero(r2 == r2.max())
    de = np.array([(np.array(x["DE"]) > 0).mean() for x in ab])
    fg = np.array([(np.array(x["FG"]) > 0).mean() for x in ab])
    rr = np.array([x["other_caption_rr"] for x in ab])
    trials = []
    for hidden in range(count):
        i, j = ranked_pair(group, [k for k in range(count) if k != hidden])
        assert i == tied[0]
        picked = i if j is None else j
        assert j is None or r2[picked] < r2[i]
        scores = {}
        for field, base in (("first", [i]), ("tied", tied)):
            scores[field] = dict(held_rr=float(rr[picked, hidden] - rr[base, hidden].mean()),
                                 DE=float(de[picked] - de[base].mean()), FG=float(fg[picked] - fg[base].mean()),
                                 switched=int(j is not None))
        trials.append(dict(seed=seed, block=block, hidden=hidden, base=i, top=tied.tolist(), selected=picked, **scores))
    return trials


def cached_groups(rows, evidence):
    for rel, signature in evidence["inputs"].items():
        assert sha(REPORTS / rel) == signature
    archived = {(g["seed"], g["block"], g["image"]):g for g in jsonl(COMPARISON / "分差核对98逐组.jsonl.gz")}
    bank = bank_inputs()
    for row in rows:
        name = row["image"]
        trials = []
        for seed in "ab":
            path = OLD / "records/real" / seed / "features" / (name + ".pt")
            rawpath = OLD / "records/real" / seed / "raw" / (name + ".pt")
            assert sha(path) == evidence["records"][seed][name]["features"]
            assert sha(rawpath) == evidence["records"][seed][name]["raw"]
            data = load(path)
            assert data["source"] == sha(rawpath) and data["rgb"] == row["rgb_sha256"]
            metrics = readout(row, data, bank)
            for block in (0, 1):
                archive = archived[seed, block, name]
                indices = range(block * 8, block * 8 + 8)
                items = [dict(hard_margins=metrics["hard"][j, :2].tolist(), DE=metrics["hard"][j, 2:].tolist(),
                              FG=metrics["fg"][j].tolist(), other_caption_rr=metrics["rr"][j].tolist()) for j in indices]
                group = dict(items=items, r2=archive["r2"])
                assert np.flatnonzero(np.asarray(group["r2"]) == max(group["r2"])).tolist() == archive["winners_r2"]
                trials.extend(assess(group, seed, block))
        yield dict(image=name, pair=row["pair"], cohort="history98", trials=trials)


def fresh_groups(rows):
    fresh = OUT / "fresh98"
    score_lock = read(fresh / "scoring_complete.json")
    assert len(score_lock["files"]) == 98 and score_lock["threads"] == 4
    assert sha(fresh / "c_complete.json") == score_lock["sampling"]
    proof = read(fresh / "c_complete.json")
    for row in rows:
        name = row["image"]
        p = fresh / "scores" / (name + ".json")
        assert sha(p) == score_lock["files"][name]
        group = read(p)
        assert group["rgb"] == row["rgb_sha256"] and group["source"] == proof["files"][name]
        assert group["eligible"] == proof["eligibility"][name]
        if group["eligible"]:
            assert group["all_readouts_verified"] and group["seed"] == "c"
            yield dict(image=name, pair=row["pair"], cohort="fresh98", trials=assess(group, "c", 0))


def inventory(rows):
    panels = {
        "selected_training_270": PREVIOUS / "sampling/inputs/roster.json",
        "selected_training_pairs": STAGE / "inputs/train_pairs.json",
        "formal_UCM_210": REPORTS / "E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json",
        "formal_RSITMD_448": REPORTS / "E3-2_UCM与RSITMD独立复验_20261124/inputs/rsitmd.json",
        "local_detail_panel_126_including_106_test": REPORTS / "E3_UCM多描述奖励同预算RFPO_20261124/inputs/panel.json",
    }
    rgb = {r["rgb_sha256"] for r in rows}
    assert len(rgb) == len(rows) == 98
    result = {}
    for code, p in panels.items():
        data = read(p)
        if isinstance(data, dict):
            data = data["rows"]
        group = {r["rgb_sha256"] for r in data}
        result[code] = dict(size=len(group), overlap=len(rgb & group), sha256=sha(p))
        assert not rgb & group
    return result


def main(tag):
    torch.set_num_threads(1)
    rows = read(OLD / "inputs/real_roster.json")
    src = ARCH / "输入核验.json"
    evidence = read(src)
    assert sha(STAGE / "code/select_pairs.py") == read(STAGE / "inputs/train_pairs.json")["selection_code"]
    assert set(Counter(r["pair"] for r in rows).values()) == {2}
    boundary = inventory(rows)
    panel = list(cached_groups(rows, evidence) if tag == "history98" else fresh_groups(rows))
    panel = [r for r in panel if r["trials"]]
    dest = OUT / tag
    dest.mkdir(exist_ok=True)
    with gzip.open(dest / "trials.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in panel:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    def results(baseline, seed=None):
        return summarize([dict(row, trials=[t[baseline] for t in row["trials"] if seed is None or t["seed"] == seed])
                          for row in panel])
    res = {mode:results(mode) for mode in ("first","tied")}
    res["by_seed"] = {s:results("first", s) for s in ("ab" if tag == "history98" else "c")}
    res["omitted"] = sorted({r["image"] for r in rows} - {r["image"] for r in panel})
    res["switch_images"] = sum(any(t["selected"] != t["base"] for t in r["trials"]) for r in panel)
    res["source"] = dict(rule=sha(STAGE / "code/select_pairs.py"), protocol=sha(OUT / "口径.md"),
                         script=sha(Path(__file__)), independent_source=sha(src), archive=sha(dest / "trials.jsonl.gz"),
                         separation=boundary)
    save(dest / "summary.json", res)
    print(tag, res["first"], res["tied"]["metrics"], res["switch_images"])


if __name__ == "__main__":
    mode = sys.argv[1]
    assert mode in ("history98", "fresh98")
    main(mode)
