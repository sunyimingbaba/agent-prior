"""Probe whether the RFPO advantage changes favor held-out evidence in real samples."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "code"))
from score_candidates import PREV, POOL, redistribute, interval, read, save
from residual import residual_scores


def strict_scores(base, margins, rr, visible):
    tops = base == base.max()
    quality = (margins > 0).sum(axis=1)
    wins = quality * tops.sum() > quality[tops].sum()
    weak = margins.min(axis=1)
    return redistribute(base, np.where(wins | tops, weak, weak.min()), rr, visible)


def advantage(scores):
    return (scores - scores.mean()) / (scores.std(ddof=1) + 1e-8)


def evaluate_group(group, train=False):
    items = group["items"]
    base = np.asarray(group["r2"])
    ab = np.array([item["hard_margins"] for item in items])
    rr = np.array([item["other_caption_rr"] for item in items])
    assert len(items) == len(base) == 8
    if train:
        strict = strict_scores(base, ab, rr, None)
        revised = residual_scores(strict, ab, rr)
        return {"changed": float(np.any(revised != strict))}

    detail = {word: (np.array([item[word] for item in items]) > 0).mean(axis=1)
              for word in ("DE", "FG")}
    cases = []
    for held in range(rr.shape[1]):
        visible = [j for j in range(rr.shape[1]) if j != held]
        strict = strict_scores(base, ab, rr, visible)
        revised = residual_scores(strict, ab, rr[:, visible])
        delta = advantage(revised) - advantage(strict)
        measures = {**detail, "MRR": rr[:, held]}
        cases.append({**{k: float(np.mean(delta * v)) for k, v in measures.items()},
                      "changed": float(np.any(revised != strict))})
    return {k: np.mean([value[k] for value in cases]) for k in cases[0]}


def collect(roster, pools, train=False):
    results = []
    changed_by_seed = {seed: 0 for seed in pools}
    for row in roster:
        measurements = []
        for seed, path in pools.items():
            data = read(path / (row["image"] + ".json"))
            if not data["eligible"] or len(data["items"][0]["other_caption_rr"]) < (1 if train else 2):
                continue
            assert data["rgb"] == row["rgb_sha256"] and data["seed"] == seed
            measures = evaluate_group(data, train)
            changed_by_seed[seed] += int(measures["changed"] > 0)
            measurements.append(measures)
        if measurements:
            results.append(dict(image=row["image"], pair=row["pair"], cohort=row.get("cohort", "external"),
                                metrics={k: np.mean([m[k] for m in measurements]) for k in measurements[0]}))
    return results, changed_by_seed


def main():
    external = PREV / "followup_pool"
    fresh = PREV.parent / "E3_双收益回答对的偏好训练_20261127/validation98/fresh98"
    natural = {s: POOL / "scores" / s for s in "ab"}
    configs = [
        ("194", read(PREV / "inputs/test_panel.json"), natural),
        ("116", read(external / "inputs/rows.json"), {"d": external / "scores"}),
        ("98", read(fresh / "inputs/rows.json"), {"c": fresh / "scores"}),
    ]
    result = {}
    seen = set()
    for name, rows, paths in configs:
        pixels = {r["rgb_sha256"] for r in rows}
        assert seen.isdisjoint(pixels)
        seen |= pixels
        records, counts = collect(rows, paths)
        assert len(records) == int(name)
        result[name] = {**interval(records), "images_changed_by_seed": counts}
        save(ROOT / ("逐图_" + name + ".json"), records)
    training = read(POOL / "scoring/locks/training.json")["rows"]
    assert seen.isdisjoint(r["rgb_sha256"] for r in training)
    records, counts = collect(training, natural, True)
    assert len(records) == 24
    result["train24_changed_by_seed"] = counts
    development, *confirm = [result[n]["metrics"] for n in ("194", "116", "98")]
    result["proceed"] = bool(all(v > 0 for v in counts.values())
                             and all(c["DE"]["mean"] > 0 and c["MRR"]["ci"][0] >= 0
                                     and c["FG"]["mean"] >= 0 for c in confirm)
                             and any(c["DE"]["ci"][0] > 0 for c in confirm)
                             and all(development[k]["mean"] >= 0 for k in ("DE", "MRR")))
    save(ROOT / "留出结果.json", result)
    print(result)


if __name__ == "__main__":
    main()
