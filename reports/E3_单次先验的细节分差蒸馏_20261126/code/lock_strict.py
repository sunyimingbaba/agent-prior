import sys
from fractions import Fraction
import numpy as np
from local import ROOT, OLD, read, sha, dump
from exact_reward import exact_rewards
sys.path.insert(0, str(ROOT / "strict_rfpo/code"))
from strict_reward import strict_rewards


def main():
    stage = ROOT / "strict_rfpo"
    starts = {seed: {label: read(OLD / family / (seed + suffix) / "complete.json")["initial"]
                     for label, family, suffix in (("R0", "exact_policy", "_ordinary"), ("R2", "mined_policy", "_selected_ordinary"))}
              for seed in "ab"}
    audit = []
    for seed in "ab":
        for step in read(OLD / "mined_policy" / (seed + "_selected_joint") / "steps.json"):
            sc = step["scores"]
            old, old_joint = exact_rewards(sc)
            new, new_joint = strict_rewards(sc)
            assert np.array_equal(old["ordinary"], new["ordinary"])
            assert np.array_equal(old["joint"], step["rewards"]["joint"])
            top = lambda r: list(map(int, np.flatnonzero(r == r.max())))
            mean = [sum(Fraction(n, d) for n, d in row) / len(row) for row in sc["rr"]]
            changed = top(old["joint"]) != top(new["joint"])
            audit.append(dict(seed=seed, image=step["image"], step=step["step"], original=bool(max(old_joint)),
                strict=bool(max(new_joint)), changed=changed, former_top=top(old["joint"]), strict_top=top(new["joint"]),
                anchor_mean=float(mean[-1]), former_top_mean=float(mean[top(old["joint"])[0]])))
    assert sum(r["original"] for r in audit) == 26 and sum(r["strict"] for r in audit) == 10
    assert sum(r["changed"] for r in audit) == 16
    dump(stage / "results/historical_reward.json", audit)
    sources = ("mined_train.py", "mined_worker.py", "exact_reward.py")
    implementation = ("train_strict.py", "strict_reward.py", "model_run.py", "evaluate.py", "cross.py")
    dump(stage / "inputs/lock.json", dict(protocol=sha(stage / "实验口径.md"),
         train=sha(stage / "inputs/train.json"), holdout=sha(stage / "inputs/holdout.json"), starts=starts,
         sources={n: sha(OLD / "code" / n) for n in sources},
         implementation={n: sha(stage / "code" / n) for n in implementation},
         historical_lock=sha(OLD / "inputs/mined_lock.json"), cross=sha(stage / "跨库普通检索复核.md")))


if __name__ == "__main__":
    main()
