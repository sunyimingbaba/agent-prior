"""Cross-seed screening by caption gain using development images and frozen R3 rules."""
from collections import defaultdict
import hashlib
import json

import numpy as np
import torch

from data import PREV, ROOT, best, packets, read, rr, stream
from online_reward import old_choices

EPS = 1e-12
GROUPS = ("all", "disagreement", "caption_gain")
METRICS = ("detail", "mrr", "detail_up_and_rank_maintained")


def screen(row, bank, visual, coef):
    """Source group has access to A/B/C and to caption scores only."""
    indices = bank["maps"][row["image"]]
    captions = indices["native"]
    ids = sorted(set(captions) - {captions[0]})
    if len(ids) < 2:
        return None
    fit, test = ~bank["even"].copy(), bank["even"].copy()
    fit[captions] = False
    test[captions] = False
    sim = (visual[:16] @ bank["native"].T)[:8]
    positive = sim[:,ids]
    training_ranks = np.array([[rr(score, neg) for score in vec] for vec, neg in zip(positive, sim[:,fit])], dtype=object)
    held_ranks = np.array([[float(rr(score, neg)) for score in vec] for vec, neg in zip(positive, sim[:,test])])
    top = np.partition(sim[:,fit], -8, axis=1)[:,-8:]
    visible_words = np.array([indices[k] for k in "ABC"])
    hard = (visual[:16] @ bank["hard"][visible_words.ravel()].T)[:8]
    h = hard[:,::2]-hard[:,1::2]

    def choose(visible):
        pos = positive[:,visible]
        ranks = training_ranks[:,visible]
        average = ranks.mean(1)
        gaps = pos - top.max(1)[:,None]
        base = np.column_stack((average.astype(float), ranks.min(1).astype(float), pos.mean(1), pos.min(1),
                                gaps.mean(1), gaps.min(1), pos.mean(1)-top.mean(1)))
        pred = []
        for a,b in ((0,1),(0,2),(1,2)):
            phrase = h[:,[a,b]]
            x = np.column_stack((phrase, phrase.min(1), phrase>0, (phrase>0).all(1), base))
            pred.append((x-x.mean(0))@coef)
        packet = {"ref":best(average,range(8)), "exact":average}
        return old_choices(pred[0],packet), old_choices(np.mean(pred,axis=0),packet)

    cols = np.arange(len(ids))
    old, new = choose(cols)
    gains = []
    for j in cols:
        before, after = choose(cols[cols!=j])
        gains.append(held_ranks[after,j].mean()-held_ranks[before,j].mean())
    return {"disagreement": old != new, "source_caption_gain": float(np.mean(gains))}


def evaluate(row, seed, bank, visual, coef):
    groups = []
    for p in packets([(row,seed,"first",bank,visual)]):
        if p["block"] != 1:
            continue
        pred = p["old"] @ coef
        old = old_choices(pred[0], p)
        new = old_choices(pred.mean(0), p)
        detail = float(p["fine"][new].mean()-p["fine"][old].mean())
        rank = float(p["mrr"][new].mean()-p["mrr"][old].mean())
        groups.append([detail,rank,float(detail>EPS and rank>=-EPS)])
    return np.mean(groups,axis=0)


def summary(rows):
    pairs = sorted({r["pair"] for r in rows})
    pi = {n:i for i,n in enumerate(pairs)}
    weights = np.random.default_rng(261126).multinomial(len(pairs), np.full(len(pairs), 1/len(pairs)), size=20000)
    result = {}
    metrics = {}
    samples = {}
    for group in GROUPS:
        selected = [r for r in rows if r[group]]
        table = np.zeros((len(pairs),2,len(METRICS)+1))
        for r in selected:
            table[pi[r["pair"]],r["direction"]] += [*r["target"],1.]
        by_seed, count = table.sum(0)[:,:-1], table.sum(0)[:,-1]
        boot = np.einsum("bp,pdm->bdm",weights,table)
        valid = (boot[:,:,-1]>0).all(1)
        vals = np.full((len(weights),len(METRICS)),np.nan)
        vals[valid] = (boot[valid,:,:-1]/boot[valid,:,-1:]).mean(1)
        if count.min()>0:
            central = (by_seed/count[:,None]).mean(0)
        else:
            central = None
        result[group] = {"count_by_source_seed":count.astype(int).tolist(), "distinct_images":len({r["image"] for r in selected}),
                         "distinct_pairs":len({r["pair"] for r in selected}),"valid_draws":int(valid.sum()),
                         "means":None if central is None else dict(zip(METRICS,central.tolist())),
                         "intervals":None if not valid.any() else dict(zip(METRICS,np.quantile(vals[valid],[.025,.975],axis=0).T.tolist()))}
        samples[group], metrics[group] = vals, central
    contrasts = {}
    for base in ("all", "disagreement"):
        vals = samples["caption_gain"] - samples[base]
        valid = np.isfinite(vals).all(1)
        delta = metrics["caption_gain"]-metrics[base] if metrics["caption_gain"] is not None and metrics[base] is not None else None
        contrasts["caption_gain_minus_"+base] = {"valid_draws":int(valid.sum()), "means":None if delta is None else dict(zip(METRICS,delta.tolist())),
                "intervals":None if not valid.any() else dict(zip(METRICS, np.quantile(vals[valid],[.025,.975],axis=0).T.tolist()))}
    return result, contrasts


def main():
    model = read(PREV/"锁定评分参数.json")["model"]
    coef = np.array(model["coef"])/np.array(model["scale_x"])[:,None]
    masks, targets, id_pair = {}, {}, {}
    for row,seed,view,bank,visual in stream("dev"):
        if view == "first":
            key = row["image"],seed
            masks[key] = screen(row,bank,visual,coef)
            id_pair[row["image"]] = row["pair"]
    assert len(masks)==408 and len(id_pair)==204
    frozen = hashlib.sha256(json.dumps(sorted(masks.items()),sort_keys=True).encode()).hexdigest()
    for row,seed,view,bank,visual in stream("dev"):
        key = row["image"],seed
        if view == "first" and masks[key] is not None:
            targets[key] = evaluate(row,seed,bank,visual,coef)
    rows = []
    for d,(source,target) in enumerate((("a","b"),("b","a"))):
        for name,pair in id_pair.items():
            m = masks[name,source]
            if m is None:
                continue
            rows.append({"image":name, "pair":pair, "direction":d, **m, "all":True,
                         "caption_gain":m["disagreement"] and m["source_caption_gain"]>EPS,
                         "target":targets[name,target].tolist()})
    levels, contrasts = summary(rows)
    result = {"target_measure":list(METRICS),"version":"two-direction equal weighting on locked old3 vs old",
              "mask_sha256":frozen,"levels":levels,"contrasts":contrasts,"rows":rows}
    (ROOT/"results/配文收益筛图诊断.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(levels,indent=2))


if __name__=="__main__":
    torch.set_num_threads(1)
    main()
