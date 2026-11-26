"""Measure training-signal density and held-caption quality on fresh cold trajectories."""
from collections import defaultdict
import json
import numpy as np
import torch

import backup
from data import ROOT, PREV, read, sha, previous_banks, packets, outcomes
from online_reward import old_choices
from training_opportunities import winners


def sources(roster, receipt):
    banks = previous_banks()
    for seed in "ab":
        folder = ROOT / "cold/records/real" / seed
        manifest = read(folder / "raw/complete.json")
        assert manifest["adapter"] == receipt["starts"][seed]
        hashes = {name:val for path in (folder/"features").glob("complete_*.json")
                  for name,val in read(path)["files"].items()}
        for row in roster:
            name = row["image"]
            f = folder/"features"/(name+".pt")
            raw = folder/"raw"/f.name
            assert sha(raw) == manifest["files"][name] and sha(f) == hashes[name]
            z = torch.load(f, weights_only=True)
            t = torch.load(raw, weights_only=True)
            assert z["source"] == manifest["files"][name] and z["adapter"] == t["adapter"] == receipt["starts"][seed]
            assert t["query"] == row["orig_query"] and t["rgb"] == row["rgb_sha256"] == z["rgb"]
            assert t["feedback"] == row["feedback"] and t["rngs"] == [row["rng_"+seed], row["rng_"+seed]+100]
            b = banks[int(row["tag"] in ("saved", "extra"))]
            yield row, seed, "first", b, z["visual"].double().numpy()


def clusters(roster):
    parent = {r["image"]:r["image"] for r in roster}
    def root(n):
        while n != parent[n]:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    groups = {}
    for row in roster:
        n = row["image"]
        parent[root(n)] = root(row["match"])
        first = groups.setdefault(row["pair"], n)
        parent[root(n)] = root(first)
    return {n:root(n) for n in parent}


def main():
    folder = ROOT/"cold"
    roster = read(folder/"inputs/real_roster.json")
    lock = read(folder/"inputs/real_lock.json")
    assert sha(folder/"inputs/real_roster.json") == lock["roster"]
    assert sha(ROOT/"冷启动训练图预筛复核.md") == lock["protocol"]
    old = read(PREV/"锁定评分参数.json")["model"]
    coef = np.array(old["coef"])/np.array(old["scale_x"])[:,None]
    records = []
    for row,seed,view,bank,visual in sources(roster, lock):
        full = winners(row, bank, visual, coef)
        held = defaultdict(list)
        for p in packets([(row,seed,view,bank,visual)]):
            scores = p["old"] @ coef
            z = outcomes(p, {"old":old_choices(scores[0], p), "old3":old_choices(scores.mean(0),p),
                                      "ordinary":p["ref"]})["outcomes"]
            held[p["block"]].append(z)
        for block, value in enumerate(full):
            record = {"image":row["image"], "seed":seed, "block":block, "status":row["status"],
                      "pair":row["pair"], "match":row["match"], **value}
            for a in ("old", "ordinary"):
                for k in ("detail","mrr"):
                    if held[block]:
                        record[f"held_{k}_old3-{a}"] = float(np.mean([g["old3"][k]-g[a][k] for g in held[block]]))
            records.append(record)
    connected = clusters(roster)
    names = sorted(set(connected.values()))
    idx = {n:i for i,n in enumerate(names)}
    draws = np.random.default_rng(261126).multinomial(len(names),np.full(len(names),1/len(names)),size=20000)
    out = {"images":len(roster),"matched_pairs":lock["pairs"],"clusters":len(names),"groups":len(records),"metrics":{}}
    keys = sorted({k for r in records for k in r if k.startswith("held_") or k in ("old3-old","old3-ordinary")})
    for k in keys:
        table = np.zeros((len(names),2,2))
        by_seed = {}
        for r in records:
            if k not in r:
                continue
            i=idx[connected[r["image"]]]; j=int(r["status"]=="selected")
            table[i,j] += [r[k],1]
        for s in "ab":
            by_seed[s] = {g:np.mean([r[k] for r in records if r["seed"]==s and r["status"]==g and k in r])
                          for g in ("selected","control")}
        sample = np.einsum("bc,cjk->bjk",draws,table)
        good = (sample[:,:,1]>0).all(1)
        diff = np.diff(sample[good,:,0]/sample[good,:,1], axis=1)[:,0]
        totals = table.sum(0)
        rates = totals[:,0]/totals[:,1]
        out["metrics"][k] = dict(control=rates[0], selected=rates[1], diff=rates[1]-rates[0],
                interval=np.quantile(diff,[.025,.975]).tolist(),counts=totals.tolist(),by_seed=by_seed)
    (ROOT/"results/冷启动预筛效果.json").write_text(json.dumps(out,indent=2)+"\n")
    (ROOT/"results/冷启动预筛逐组.json").write_text(json.dumps(records,indent=2)+"\n")
    backup.ROOT = folder
    backup.main()


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
