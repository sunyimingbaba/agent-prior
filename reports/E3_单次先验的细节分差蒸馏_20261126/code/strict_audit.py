import sys
import numpy as np
from local import ROOT, OLD, read, sha
from exact_reward import exact_rewards
sys.path.insert(0, str(ROOT / "strict_rfpo/code"))
from strict_reward import strict_rewards


def audit():
    stage = ROOT / "strict_rfpo"; lock = read(stage / "inputs/lock.json"); output = {}
    for seed, roster in read(stage / "inputs/train.json").items():
        first = {}
        for arm in ("warmstrict", "coldordinary", "coldstrict"):
            path = stage / "models" / (seed + "_" + arm)
            cp = read(path / "complete.json"); steps = read(path / "steps.json")
            assert cp["initial"] == lock["starts"][seed]["R2" if arm == "warmstrict" else "R0"]
            assert cp["receipt"] == sha(stage / "inputs/lock.json") and cp["steps"] == 24 and cp["logs"] == sha(path / "steps.json")
            assert [(s["image"], s["query"]) for s in steps] == [(r["image"], r["orig_query"]) for r in roster]
            recs = []; first[arm] = steps[0]
            for i, s in enumerate(steps):
                assert s["rng"] == [(91000000 if seed == "a" else 101000000) + i * 1000 + j * 100 for j in range(2)]
                new, frac = strict_rewards(s["scores"]); old, oldfrac = exact_rewards(s["scores"])
                assert np.array_equal(new["ordinary"], old["ordinary"]) and np.array_equal(new["joint"], s["rewards"]["joint"])
                assert frac == s["joint_fraction"]
                top = lambda r: set(np.flatnonzero(r == max(r)))
                recs.append([max(frac) > 0, max(oldfrac) > 0, top(new["joint"]) != top(new["ordinary"]),
                             top(old["joint"]) != top(new["joint"]), not np.array_equal(new["joint"], new["ordinary"])])
            errors = [s["mean_error"] for s in steps]
            output[seed+"_"+arm] = dict(**cp, strict_eligible=sum(r[0] for r in recs), old_eligible=sum(r[1] for r in recs),
                different_top_from_ordinary=sum(r[2] for r in recs), different_top_from_old=sum(r[3] for r in recs),
                different_any_from_ordinary=sum(r[4] for r in recs),
                mean_replay_error=np.mean(errors), max_replay_error=max(s["max_error"] for s in steps),
                sampled_trajectories=sum(len(s["paths"]) for s in steps), greedy_anchors=len(steps))
        historical=read(OLD / "mined_policy" / (seed+"_selected_joint") / "steps.json")[0]
        for a, b in ((historical, first["warmstrict"]), (first["coldordinary"], first["coldstrict"])):
            assert a["paths"] == b["paths"] and a["scores"] == b["scores"]
        for arm in first:
            output[seed+"_"+arm]["first_group_paths_and_scores_matched"] = True
    return output
