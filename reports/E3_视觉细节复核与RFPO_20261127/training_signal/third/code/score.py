"""Run the original CPU reader and compare strict vs. R2 with all training-visible captions."""
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[2]
POOL = BASE / "aftertool_joint_pool_20261127"
CAL = BASE / "hard_reward_calibration_20261127/code"
sys.path[:0] = [str(POOL / "scoring/code"), str(CAL), str(ROOT.parent)]
from pool_reader import Reader, load, read, save, sha
from reward import redistribute
from replay import replay


def training_fields(data):
    keys = ("rank", "margin", "hard_margins", "other_caption_rr")
    return dict(r2=data["r2"], items=[{k:x[k] for k in keys} for x in data["items"]])


def wait(path):
    while not path.is_file():
        time.sleep(1)


def main():
    sealed = read(ROOT / "inputs/roster_lock.json")
    assert sha(ROOT / "code/score.py") == sealed["scripts"]["score.py"]
    assert sha(POOL/"scoring/inputs/manifest.json") == sealed["reader_inputs"]
    assert sha(POOL/"scoring/code/pool_reader.py") == sealed["reader"]
    assert sha(CAL/"reward.py") == sealed["redistribute"]
    assert sha(ROOT.parent/"replay.py") == sealed["replay"]
    assert sha(ROOT/"inputs/groups.json") == sealed["groups"]
    assert sha(ROOT/"inputs/roster.json") == sealed["roster"]
    reader = Reader()
    assert torch.get_num_threads() == 4
    rows = read(ROOT/"inputs/roster.json")
    roster = read(ROOT/"inputs/groups.json")
    groups = {r["image"]:r for r in roster}
    original = read(POOL/"sampling/a_complete.json")
    assert original["adapter"] == sealed["model"]
    scores, verified = {}, {}
    verified_cohorts = set()
    for row in rows:
        name = row["image"]
        path = ROOT/"raw/e"/(name+".pt")
        wait(path)
        raw = load(path)
        assert raw["query"] == row["orig_query"] == reader.rows[name]["orig_query"]
        assert raw["feedback"] == row["feedback"] == reader.rows[name]["feedback"]
        assert raw["rgb"] == row["rgb_sha256"] == reader.rows[name]["rgb_sha256"]
        assert raw["image"] == name and raw["adapter"] == sealed["model"]
        assert raw["rng"] == sealed["seed_base"] + row["row"]*1000
        prior_raw = POOL/"sampling/raw/a"/(name+".pt")
        assert sha(prior_raw) == original["files"][name]
        ref = load(prior_raw)
        assert ref["first"] == raw["first"] and ref["eligible"] == raw["eligible"]
        if raw["eligible"]:
            assert torch.equal(raw["context"]["prefix"], ref["context"]["prefix"])
            assert raw["context"]["answer"] == ref["context"]["answer"]
            cohort = groups[name]["cohort"]
            if cohort not in verified_cohorts:
                oldscore = POOL / "scores/a" / (name+".json")
                assert training_fields(reader.score(name, ref["hidden"])) == training_fields(read(oldscore))
                verified_cohorts.add(cohort)
                verified[name] = {"raw":sha(prior_raw), "score":sha(oldscore)}
        record = {**groups[name], "rgb_sha256":row["rgb_sha256"], "raw_sha256":sha(path),
                  "first_prefix_matched_batch_a":True, "eligible":raw["eligible"]}
        if raw["eligible"]:
            data = training_fields(reader.score(name,raw["hidden"]))
            record.update(data)
            native = np.array(data["r2"])
            hard = np.array([x["hard_margins"] for x in data["items"]])
            rr = [x["other_caption_rr"] for x in data["items"]]
            top = native == native.max()
            count = (hard>0).sum(1)
            weak = hard.min(1)
            valid = count*top.sum() > count[top].sum()
            updated = redistribute(native,np.where(valid|top,weak,weak.min()),rr)
            record["strict"] = updated.tolist()
            record["replay"] = replay(record)
            assert np.flatnonzero(updated==max(updated)).tolist()==record["replay"]["strict_slots"]
        else:
            record["replay"] = replay(record)
        dst = ROOT/"scores"/(name+".json")
        save(dst,record)
        scores[name] = sha(dst)
        print(len(scores), name, flush=True)
    wait(ROOT/"e_complete.json")
    proof = read(ROOT/"e_complete.json")
    assert set(scores) == set(proof["files"]) and len(scores)==24
    assert all(read(ROOT/"scores"/(n+".json"))["raw_sha256"]==v for n,v in proof["files"].items())
    assert verified_cohorts == {"v122","v148"}
    save(ROOT/"scoring_complete.json", dict(files=scores, sampling=sha(ROOT/"e_complete.json"),
         inputs=sha(ROOT/"inputs/roster_lock.json"), threads=torch.get_num_threads(),
         exact_original_score_checks=verified, code=sha(Path(__file__))))
if __name__=="__main__":
    main()
