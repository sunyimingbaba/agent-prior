"""Verify that the chosen trajectory attains both reachable target vectors."""
import torch
from local import ROOT, read, load, dump, feature_path, sha


def main():
    stage = ROOT / "reachable"
    rows = read(ROOT / "inputs/train.json")
    trace = read(ROOT / "inputs/teacher_trace.json")
    summary = []
    for seed, plan in rows.items():
        index = {r["image"]: r for r in plan}
        targets = load(stage / "inputs" / ("targets_" + seed + ".pt"))
        for record in (r for r in trace if r["seed"] == seed):
            name, view = record["image"], record["view"]
            path = feature_path(index[name], seed, view)
            assert sha(path) == record["file"]
            visual = load(path)["visual"][record["winner"]].double()
            target = targets[name, view]
            worst = max(float(torch.max(torch.abs(target[direction].double() @ visual - target[field].double())))
                        for direction, field in (("ordinary_dir", "ordinary_target"), ("hard_dir", "teacher")))
            assert worst < 1e-6
            summary.append(dict(seed=seed, image=name, view=view, candidate=record["winner"], max_abs_error=worst))
    dump(stage / "results/target_check.json", dict(views=len(summary), qualified=sum(r["candidate"] < 16 for r in summary),
         max_abs_error=max(r["max_abs_error"] for r in summary), rows=summary))


if __name__ == "__main__":
    main()
