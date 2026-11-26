"""Can one archived eight-sample group identify useful training images for a fresh group?"""
from collections import defaultdict
import json
import numpy as np
import torch
from data import PREV, ROOT, best, read, rr, stream
from online_reward import old_choices


def winners(row, bank, visual, coefficients):
    indices = bank["maps"][row["image"]]
    captions = indices["native"]
    alternatives = sorted(set(captions) - {captions[0]})
    if not alternatives:
        return []
    scores = visual[:16] @ bank["native"].T
    negative = ~bank["even"].copy()
    negative[captions] = False
    competitors = scores[:, negative]
    positive = scores[:, alternatives]
    ranks = np.array([[rr(v, neg) for v in vals] for vals, neg in zip(positive, competitors)], dtype=object)
    average = ranks.mean(1)
    top_eight = np.partition(competitors, -8, axis=1)[:, -8:]
    gaps = positive - top_eight.max(1)[:, None]
    base = np.column_stack((average.astype(float), ranks.min(1).astype(float), positive.mean(1), positive.min(1),
                            gaps.mean(1), gaps.min(1), positive.mean(1) - top_eight.mean(1)))
    ids = np.array([indices[k] for k in "ABC"])
    hard = visual[:16] @ bank["hard"][ids.ravel()].T
    margins = hard[:, ::2] - hard[:, 1::2]
    outputs = []
    for start in (0, 8):
        block = slice(start, start+8)
        ranking = average[block]
        packet = {"ref": best(ranking, range(8)), "exact": ranking}
        predictions = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            h = margins[block][:, [i,j]]
            x = np.column_stack((h, h.min(1), h > 0, (h > 0).all(1), base[block]))
            predictions.append((x - x.mean(0)) @ coefficients)
        old = old_choices(predictions[0], packet)
        extra = old_choices(np.mean(predictions, axis=0), packet)
        outputs.append({"old3-old": extra != old, "old3-ordinary": extra != packet["ref"]})
    return outputs


def contrast(mask, target, groups):
    group_ids = sorted(set(groups))
    counts = np.zeros((len(group_ids),4))
    labels = {p:i for i,p in enumerate(group_ids)}
    for pick, value, pair in zip(mask, target, groups):
        counts[labels[pair]] += [value, 1, pick*value, pick]
    weights = np.random.default_rng(261126).multinomial(len(group_ids), np.full(len(group_ids), 1/len(group_ids)), size=20000)
    sample = weights @ counts
    ok = sample[:,3] > 0
    delta = sample[ok,2]/sample[ok,3] - sample[ok,0]/sample[ok,1]
    all_hit, n, selected_hit, k = counts.sum(0)
    return {"all_images_times_directions":int(n), "screened":int(k), "screened_changes":int(selected_hit),
            "all_rate":all_hit/n, "screened_rate":selected_hit/k, "change":selected_hit/k-all_hit/n,
            "interval":np.quantile(delta,[.025,.975]).tolist()}


def main():
    old = read(PREV / "锁定评分参数.json")["model"]
    coefficients = np.array(old["coef"]) / np.array(old["scale_x"])[:, None]
    signals = defaultdict(dict)
    for row, seed, view, bank, visual in stream("dev"):
        if view != "first":
            continue
        z = winners(row, bank, visual, coefficients)
        if z:
            signals[row["image"]][seed] = z
            signals[row["image"]]["pair"] = row["pair"]
    assert all(set(x) == {"a","b","pair"} for x in signals.values())
    out = {"images":len(signals), "pairs":len({x["pair"] for x in signals.values()}), "contrasts":{}}
    for key in ("old3-old", "old3-ordinary"):
        masks, outcomes, ids = [], [], []
        for left, right in (("a","b"), ("b","a")):
            for x in signals.values():
                masks.append(int(x[left][0][key]))
                outcomes.append(int(x[right][1][key]))
                ids.append(x["pair"])
        out["contrasts"][key] = contrast(masks, outcomes, ids)
    (ROOT/"results/开发图预筛复核.json").write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
