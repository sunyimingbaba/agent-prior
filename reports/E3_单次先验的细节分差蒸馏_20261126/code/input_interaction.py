"""Compare the change in C/D/E decisions across two inputs on identical pictures and weights."""
import numpy as np
from local import ROOT, FORM, RFPO, read, dump, calculate, summary_by_seed, paired
from margin_scores import features


def main():
    strict = ROOT / "strict_rfpo"
    held = read(ROOT / "inputs/holdout.json")
    primary = read(RFPO / "inputs/panel.json")
    official = read(FORM / "inputs/ucm.json")
    prior = {r["rgb_sha256"]: i for i, r in enumerate(primary)}
    official_ix = {r["rgb_sha256"]: i for i, r in enumerate(official)}
    linked = [r for r in primary if r["rgb_sha256"] in official_ix]
    assert len(linked) == 106
    arms = ("retained", "strict_coldordinary", "strict_coldstrict")
    scores = read(strict / "results/by_image.json")
    idx = {(r["tag"], r["seed"], r["arm"], r["image"]): r["values"]["CDE"] for r in scores if r["tag"].startswith("hold_")}
    result = {}; by_image = []
    for panel, rows in (("held64", held), ("paired106", linked)):
        raw = {}; labels = [r["pair"] for r in rows]
        for arm in arms:
            raw[arm] = {}
            for seed in "ab":
                if panel == "held64":
                    table = {v: np.array([idx["hold_"+v, seed, arm, r["image"]] for r in rows]) for v in ("scene", "first")}
                else:
                    scene = features("primary", arm, seed, primary)[[prior[r["rgb_sha256"]] for r in rows]]
                    first = features("official", arm, seed, official)[[official_ix[r["rgb_sha256"]] for r in rows]]
                    table = {v: calculate("primary", data, rows)["CDE"] for v, data in (("scene", scene), ("first", first))}
                table["first_minus_scene"] = table["first"] - table["scene"]
                raw[arm][seed] = table
                for j, row in enumerate(rows):
                    by_image.append(dict(panel=panel, seed=seed, image=row["image"], arm=arm, scene=table["scene"][j], first=table["first"][j]))
        result[panel] = {"n": len(rows), "pairs": len(set(labels)),
                         "levels": {a: summary_by_seed(r, labels) for a, r in raw.items()},
                         "strict_minus": {r: paired(raw["strict_coldstrict"], raw[r], labels) for r in arms[:-1]}}
    dump(strict / "results/input_comparison.json", result)
    dump(strict / "results/input_comparison_images.json", by_image)


if __name__ == "__main__":
    main()
