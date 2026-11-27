"""Evaluate matched graded rewards on archived on-policy candidate groups."""
import ast
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent
OLD = REPORTS / "E3_R3改选收益校准与双批首句验证_20261126"
R2 = REPORTS / "E3-2_UCM回填后回答训练_20261124"
NATIVE = REPORTS / "E3-1_奖励一致性分析_20261112/空间关系候选奖励核验_20261120/native"
LIBRARIES = {
    "v122": REPORTS/"E3_UCM同图偏好对与独立验收_20261124/guard",
    "v148": REPORTS/"E3_UCM多数配文新图复核_20261125",
}
sys.path.insert(0, str(ROOT / "code"))
from detail_reward import graded_bonus


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(p):
    return torch.load(p,map_location="cpu",weights_only=True)


def original_mapping():
    path = R2 / "code/train.py"
    node = next(f for f in ast.parse(path.read_text()).body if getattr(f,"name",None)=="matched_scores")
    scope = {"np":np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),scope)
    return scope["matched_scores"]


def scores_for_seed(rows, seed, source=OLD, libraries=LIBRARIES, detailed=False):
    native = load(NATIVE/"features.pt")
    assert sha(NATIVE/"features.pt") == read(NATIVE/"index.json")["sha256"]
    original_tokens = native["tokens"].tolist()
    native_ids = {tuple(q):i for i,q in enumerate(original_tokens)}
    text = native["features"].double().numpy()
    even = np.array([int(hashlib.sha256(",".join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in original_tokens])
    fresh = load(OLD/"inputs/fresh_text.pt")
    fresh_words = read(OLD/"inputs/独立措辞.json")
    assert fresh["source"] == sha(OLD/"inputs/独立措辞.json")
    word_ids = {w:i for i,w in enumerate(fresh["words"])}
    match = original_mapping()
    base = source/"records/real"/seed
    raw_hashes = read(base/"raw/complete.json")["files"]
    feature_hashes = {n:h for f in (base/"features").glob("complete*.json") for n,h in read(f)["files"].items()}
    for cohort, lib in libraries.items():
        bank = load(lib/"records/text.pt")
        remap = np.array([native_ids[tuple(q)] for q in bank["native_tokens"].tolist()])
        hard = bank["hard"].double().numpy()
        for row in rows:
            if row["cohort"] != cohort:
                continue
            name = row["image"]
            path = base/"features"/(name+".pt")
            assert feature_hashes[name] == sha(path)
            feat = load(path)
            assert feat["rgb"] == row["rgb_sha256"] and feat["source"] == raw_hashes[name]
            v = feat["visual"][:16].double().numpy()
            proxy = np.array(feat["ranks"][:16])
            marks = bank["maps"][name]
            pos = remap[marks["native"]]
            alts = sorted(set(pos)-{pos[0]})
            negative_fit = even.copy()
            negative_test = ~even
            negative_fit[pos] = negative_test[pos] = False
            # Compute equal readout vectors once so that duplicate candidate rows remain exactly tied.
            unique, inv = np.unique(v,axis=0,return_inverse=True)
            sim = (unique @ text.T)[inv]
            margins = sim[:,pos[0]] - sim[:,negative_fit].max(1)
            hard_ids = np.array([marks[k] for k in "ABDE"])
            hs = (unique @ hard[hard_ids.ravel()].T)[inv].reshape(16,4,2)
            hs = hs[:,:,0] - hs[:,:,1]
            outputs = {"DE": (hs[:,2:] > 0).mean(1)}
            if row["attribute"] in fresh_words:
                fg_ids = [word_ids[w] for group in fresh_words[row["attribute"]] for w in group]
                fg = (unique @ fresh["features"][fg_ids].double().numpy().T)[inv].reshape(16,2,2)
                side = row["side"]
                outputs["FG"] = (fg[:,:,side] > fg[:,:,1-side]).mean(1)
            if alts:
                def reciprocal(values, rank):
                    a = (values > rank).sum()
                    b = (values == rank).sum()
                    return (1 / np.arange(a+1,a+b+2)).mean()
                outputs["mrr"] = np.array([np.mean([reciprocal(u[negative_test],u[k]) for k in alts]) for u in sim])
            for block in (0,1):
                sl = slice(block*8,block*8+8)
                r2 = match(margins[sl],1/(1+proxy[sl]))
                bonus = graded_bonus(r2, hs[sl,:2])
                r3 = r2 + bonus
                r2_adv = (r2-r2.mean())/(r2.std(ddof=1)+1e-8)
                r3_adv = (r3-r3.mean())/(r3.std(ddof=1)+1e-8)
                first = r2 == r2.max()
                after = r3 == r3.max()
                numbers = {"new_winner":int(not np.array_equal(first,after)),
                           "nonconstant_bonus":int(np.ptp(bonus)>0),
                           "changed_advantages":int(np.max(np.abs(r3_adv-r2_adv))>1e-8)}
                for k,vector in outputs.items():
                    y = vector[sl]
                    numbers[k+"_r2"] = float(y[first].mean())
                    numbers[k+"_r3"] = float(y[after].mean())
                    numbers[k+"_gain"] = float(y[after].mean()-y[first].mean())
                    numbers[k+"_adv_alignment"] = float(np.mean((r3_adv-r2_adv)*(y-y.mean())))
                record = dict(image=name, pair=row["pair"], cohort=cohort, seed=seed, block=block,
                           winners_r2=np.flatnonzero(first).tolist(), winners_r3=np.flatnonzero(after).tolist(),
                           r2=r2.tolist(), bonus=bonus.tolist(), numbers=numbers)
                if detailed:
                    record["outcomes"] = {k:vals[sl].tolist() for k,vals in outputs.items()}
                yield record


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["cohort"],r["pair"],r["image"]].append(r)
    pairs = sorted({(c,p) for c,p,_ in groups})
    which = {p:i for i,p in enumerate(pairs)}
    weights = np.random.default_rng(271127).multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=20000)
    result = {}
    for k in sorted({k for r in rows for k in r["numbers"]}):
        sums = np.zeros((len(pairs),2))
        for (c,p,_), obs in groups.items():
            assert len(obs)==4
            if k in obs[0]["numbers"]:
                sums[which[c,p]] += [np.mean([x["numbers"][k] for x in obs]),1]
        boot=weights@sums
        means=boot[:,0]/boot[:,1]
        result[k]=dict(mean=sums[:,0].sum()/sums[:,1].sum(),interval=np.quantile(means,[.025,.975]).tolist(),
                       images=int(sums[:,1].sum()),seeds=[np.mean([r["numbers"][k] for r in rows if r["seed"]==s and k in r["numbers"]]) for s in "ab"])
    return {"images":len(groups),"pairs":len(pairs),"groups":len(rows),"metrics":result}


def main():
    pool = read(OLD/"inputs/real_roster.json")
    assert sha(OLD/"inputs/real_roster.json")==read(OLD/"inputs/real_lock.json")["roster"]
    reserved=set()
    for name in ("E3-2_UCM同起点三奖励短训_20261120/inputs/train.json",
                 "E3_UCM多描述奖励同预算RFPO_20261124/inputs/panel.json",
                 "E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json"):
        reserved.update(r["rgb_sha256"] for r in read(REPORTS/name))
    assert not any(r["rgb_sha256"] in reserved for r in pool)
    rows = [r for s in "ab" for r in scores_for_seed(pool,s)]
    assert len(rows)==1080
    with gzip.open(ROOT/"candidates/逐组读数.jsonl.gz","wt") as stream:
        for row in rows:stream.write(json.dumps(row)+"\n")
    out={tag:summary([r for r in rows if tag=="pooled" or r["cohort"]==tag]) for tag in (*LIBRARIES,"pooled")}
    (ROOT/"candidates/两批复核.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
