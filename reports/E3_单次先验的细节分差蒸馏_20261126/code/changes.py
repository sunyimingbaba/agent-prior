"""Trace each change in the three held-out wording judgments on the 126-image panel."""
import numpy as np
from local import ROOT, RFPO, read, dump, calculate
from margin_scores import features


def main():
    rows = read(RFPO / "inputs/panel.json")
    historical = {(r["seed"], r["image"], r["wording"]) for r in read(ROOT / "results/five_cpu.json")}
    arms = ("retained", "parent", "ordinary", "guard", "teacher", "reachable_ordinary", "reachable_teacher",
            "strict_warmstrict", "strict_coldordinary", "strict_coldstrict")
    scores = {}
    for arm in arms:
        scores[arm] = {}
        for seed in "ab":
            values = calculate("primary", features("primary", arm, seed, rows), rows)
            for index, row in enumerate(rows):
                for wording in "CDE":
                    scores[arm][seed, row["image"], wording] = int(values[wording][index])
    changes = {}
    records = []
    for reference in ("retained", "parent"):
        changes[reference] = {}
        for arm in arms:
            if arm == reference:
                continue
            by_seed = {}
            for seed in "ab":
                keys = [key for key in scores[arm] if key[0] == seed]
                delta = np.array([scores[arm][k] - scores[reference][k] for k in keys])
                by_seed[seed] = {"fixed": int((delta > 0).sum()), "lost": int((delta < 0).sum())}
                for (s, n, w), change in zip(keys, delta):
                    if change:
                        records.append(dict(reference=reference, arm=arm, seed=s, image=n, wording=w, change=int(change)))
            changes[reference][arm] = by_seed
    assert {k for k in scores["parent"] if scores["parent"][k] != scores["retained"][k]} == historical
    summary = {}
    for arm in arms:
        rescued = [list(k) for k in sorted(historical) if scores[arm][k] == 1]
        summary[arm] = {"original_five_correct": len(rescued), "details": rescued}
    dump(ROOT / "results/case_changes.json", dict(changes=changes, five=summary, records=records))


if __name__ == "__main__":
    main()
