"""Diagnose whether existing pairs of answers admit gains that transfer to another wording."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"pair_reward"))
from evaluate import guarded, read, sha, load, write, CALIBRATION, REPORTS


def retrieve_pair(a, b, offset):
    image_correct = np.zeros((8, 8))
    text_correct = np.zeros((8, 8))
    for k in (offset, offset+2):
        image_correct += (a[:, k] > a[:, k+1])[:, None] / 4
        image_correct += (b[:, k] > b[:, k+1])[None, :] / 4
        for difference in (a[:, None, k] - b[None, :, k+1], b[None, :, k] - a[:, None, k+1]):
            text_correct += ((difference > 0) + 0.5*(difference == 0)) / 4
    return image_correct, text_correct


def options(row):
    ranks = row["rr"]
    opts = []
    for h in range(ranks.shape[1]):
        visible = [k for k in range(ranks.shape[1]) if k != h]
        scores = guarded(row["r2"], row["ab"], ranks, visible)
        top = np.flatnonzero(scores == max(scores))
        cover = np.ones(8, dtype=bool)
        for k in visible:
            col = [Fraction.from_float(float(s)) for s in ranks[:, k]]
            threshold = sum(col[i] for i in top)
            cover &= [x * len(top) >= threshold for x in col]
        opts.append((top, cover, ranks[:, h]))
    return opts


def probe(pair):
    left, right = pair
    di, dt = retrieve_pair(left["s"], right["s"], 4)
    fi, ft = retrieve_pair(left["s"], right["s"], 8)
    total = di+dt
    trials = []
    for top_a, cover_a, rr_a in options(left):
        for top_b, cover_b, rr_b in options(right):
            ref = np.ix_(top_a, top_b)
            n = len(top_a) * len(top_b)
            feasible = cover_a[:, None] & cover_b[None, :]
            feasible &= (di * n >= di[ref].sum()) & (dt * n >= dt[ref].sum())
            best = total[feasible].max() if feasible.any() else -1
            if best * n <= total[ref].sum():
                trials.append(np.zeros(6))
                continue
            chosen = feasible & (total == best)
            held = (rr_a[:, None] + rr_b[None, :]) / 2
            trials.append([1., *[v[chosen].mean()-v[ref].mean() for v in (di, dt, fi, ft, held)]])
    return np.mean(trials, axis=0)


def cohort(folder, pairs):
    inputs = folder/"inputs"
    words = load(inputs/"vectors.pt")
    receipts = read(folder/"scoring_complete.json")
    all_pairs = defaultdict(list)
    for row in read(inputs/"rows.json"):
        f = row["image"]
        scorefile = folder/"scores"/(f+".json")
        assert sha(scorefile) == receipts["files"][f]
        data = read(scorefile)
        if not data["eligible"]:
            continue
        featurefile = folder/"readouts"/(f+".pt")
        assert sha(featurefile) == receipts["readouts"][f]
        features = load(featurefile)["visual"].double()
        mat = torch.stack([words[f].double() @ v for v in features]).numpy()
        all_pairs[row["pair"]].append(dict(image=f, side=row["side"], s=mat, words=words[f],
                                           r2=np.array(data["r2"]),
                                           rr=np.array([i["other_caption_rr"] for i in data["items"]]),
                                           ab=np.array([i["hard_margins"] for i in data["items"]])))
    data, order = [], []
    for key, group in sorted(all_pairs.items()):
        if len(group) != 2:
            continue
        group.sort(key=lambda v: v["side"])
        assert torch.equal(group[0]["words"], group[1]["words"].reshape(6,2,-1).flip(1).reshape(12,-1))
        data.append(probe(group))
        order.append(key)
    arr=np.stack(data)
    assert len(arr) == pairs
    keys = ("switched", "DE_I2T", "DE_T2I", "FG_I2T", "FG_T2I", "hidden_MRR")
    draws = np.random.default_rng(281128).multinomial(pairs, np.full(pairs,1/pairs),size=20000)
    ci = np.quantile((draws @ arr)/pairs,[.025,.975],axis=0)
    return dict(pairs=pairs, pairs_ever_switched=int((arr[:,0]>0).sum()),
                metrics={k:dict(mean=float(arr[:,i].mean()),ci=ci[:,i].tolist()) for i,k in enumerate(keys)},
                by_pair={r:dict(zip(keys,v)) for r,v in zip(order,arr)})


if __name__=="__main__":
    torch.set_num_threads(4)
    result = {"98":cohort(REPORTS/"E3_双收益回答对的偏好训练_20261127/validation98/fresh98",49),
              "114":cohort(CALIBRATION/"followup_pool",57)}
    write(ROOT/"结果.json", result)
    print({k:(v["pairs_ever_switched"],v["metrics"]) for k,v in result.items()})
