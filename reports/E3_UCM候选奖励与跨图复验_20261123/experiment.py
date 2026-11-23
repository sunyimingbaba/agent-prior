"""Develop caption-gated hard-negative rewards, then apply the locked rules to disjoint UCM images."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
REPORTS = OUT.parent
EARLY = REPORTS / "E3-1_奖励一致性分析_20261112"
INITIAL = EARLY / "空间关系候选奖励核验_20261120"
CROSS = EARLY / "跨图关系奖励与留出评估_20261120"
FRESH = REPORTS / "E3-2_UCM方向候选覆盖与独立复验_20261120"
ARROW = Path("/Users/botlv/cup-deploy-assets/arrow/ucm")
TOLS = [0, .001, .003, .006, .01]
METRICS = ["direction", "direction_margin", "rr", "native_margin", "all_rr", "rr_first_excluded"]
FILES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    FILES[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(record(path).read_text())


def tensor(path):
    return torch.load(record(path), weights_only=True, map_location="cpu")


def rows(path):
    with record(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def table(name, data):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def load_gallery(split):
    package = tensor(CROSS / split / "features.pt")
    ix = read(CROSS / split / "index.json")
    tok = package["native_tokens"].tolist()
    even = np.array([int(hashlib.sha256(",".join(map(str, t)).encode()).hexdigest()[-1], 16) % 2 == 0 for t in tok])
    if split == "val":
        catalog = tensor(FRESH / "gallery/text.pt")
        assert torch.equal(catalog["native_tokens"], package["native_tokens"])
        assert torch.allclose(catalog["native"], package["native"], rtol=0, atol=2e-5)
        names = ix["names"]
        order = np.array(catalog["caption_order"]).reshape(-1, 5)
        groups = {name: list(dict.fromkeys(map(int, row))) for name, row in zip(names, order, strict=True)}
        for name, p in ix["native_target"].items():
            assert set(groups[name]) == set(p) and groups[name][0] == ix["native_first"][name]
    else:
        ix0 = read(INITIAL / "native/index.json")
        bank0 = tensor(INITIAL / "native/features.pt")
        assert torch.equal(bank0["tokens"], package["native_tokens"])
        assert torch.allclose(bank0["features"], package["native"], rtol=0, atol=2e-5)
        path = ARROW / "ucm_caption_karpathy_label_train.arrow"
        t = pa.ipc.open_file(pa.memory_map(str(record(path)))).read_all()
        groups = {Path(name).name: list(dict.fromkeys(ix0["columns"][c] for c in caps))
                  for name, caps in zip(t["path"].to_pylist(), t["caption"].to_pylist(), strict=True)}
        for name, ps in ix["native_target"].items():
            assert set(groups[name]) == set(ps) and groups[name][0] == ix["native_first"][name]
    return package["native"].double().numpy(), even, groups


def batch_from_rows(dataset, records, visual, held, clusters):
    assert len(records) == len(visual)
    by_image = defaultdict(list)
    for i, row in enumerate(records):
        if row["mode"] == "scene" and int(row["k"]) >= 0:
            by_image[row["image"]].append((int(row["k"]), i))
    for image, positions in by_image.items():
        slots, ids = zip(*sorted(positions))
        assert list(slots) == list(range(8))
        sample = [records[i] for i in ids]
        def column(key):
            return np.array([float(r[key]) for r in sample])
        yield (dataset, image, clusters[image], visual[list(ids)].double().numpy(), column("A"),
               np.stack([column(s) for s in held], axis=1), column("old"))


def datasets(split):
    if split == "train":
        metadata = read(INITIAL / "scoring/rows.json")
        keys = {(r["mode"], r["image"], int(r["k"])): r for r in rows(INITIAL / "第三措辞逐候选.csv")}
        records = [dict(r, C=keys[r["mode"],r["image"],r["k"]]["C"], old=r["reward"]) for r in metadata]
        yield from batch_from_rows("dev32", records, tensor(INITIAL / "scoring/vectors.pt")["visual"],
                                   "BC", {r["image"]: r["pair"] for r in records})
    subdir = CROSS / split
    records = rows(subdir / "candidates.csv")
    persisted = read(subdir / "rows.json")
    assert all((x["image"], x["mode"], int(x["k"])) == (y["image"], y["mode"], y["k"])
               for x,y in zip(records,persisted,strict=True))
    yield from batch_from_rows("dev44" if split == "train" else "val16", records,
                               tensor(subdir / "features.pt")["visual"], "CD",
                               {r["image"]:r["pair"] for r in records})
    if split == "val":
        records = rows(CROSS / "object/candidates.csv")
        yield from batch_from_rows("val11", records, tensor(CROSS / "object/features.pt")["visual"], "CD",
                                   {r["image"]:r["image"] for r in records})
        for rec in read(FRESH / "inputs/fresh.json"):
            image = rec["image"]
            path = FRESH / "outputs/fresh" / (Path(image).stem + ".pt")
            meta = read(path.with_suffix(".json"))
            data = tensor(path)
            assert FILES[str(path)] == meta["bundle"] and meta["phrases"] == rec["A"] + rec["B"]
            idx = [i for i,r in enumerate(meta["logs"]) if r["cue"]=="scene" and r["view"]==0 and r["slot"]>0]
            assert [meta["logs"][i]["slot"] for i in idx] == list(range(1,9))
            vis = data["visual"][0,idx].double().numpy()
            sim = vis @ data["text"].double().numpy().T
            yield "val14",image,image,vis,sim[:,0]-sim[:,1],(sim[:,2]-sim[:,3])[:,None],None


def best(values):
    return np.flatnonzero(values == np.max(values))


def select_close(native, direction, tol):
    if tol == 0:
        return best(native)
    eligible = np.flatnonzero(native >= native.max() - tol)
    return eligible[best(direction[eligible])]


def method_name(n,tol):
    return f"p{n}_tol{tol:g}"


def evaluate(split, thresholds):
    native, even, pos = load_gallery(split)
    output = []
    candidates = []
    photos = set()
    excluded = []
    for dataset, image, cluster, visual, direction, held, old in datasets(split):
        assert image not in photos
        photos.add(image)
        sims = visual @ native.T
        ps = pos[image]
        if len(ps) < 3:
            excluded.append(dict(dataset=dataset,image=image,distinct=len(ps)))
            continue
        neg = np.ones(len(even),dtype=bool)
        neg[ps] = False
        for flip in [False,True]:
            reward_neg = neg & (even != flip)
            eval_neg = neg & ~reward_neg
            reward_penalty = sims[:,reward_neg].max(1)
            evaluation = sims[:,eval_neg]
            positives = sims[:,ps[2:]]
            mrr = 1/(1+(evaluation[:,:,None] > positives[:,None,:]).sum(axis=1))
            best_positive = positives.max(1)
            best_rest = sims[:,ps[1:]].max(1)
            metrics = dict(direction=(held>0).mean(axis=1),direction_margin=held.mean(axis=1),
                           rr=mrr.max(1),all_rr=mrr.mean(1), native_margin=best_positive-evaluation.max(1),
                           rr_first_excluded=1/(1+(evaluation>best_rest[:,None]).sum(axis=1)))
            methods = {"random":np.arange(8)}
            if old is not None:
                methods["old"] = best(old)
            for n in [1,2]:
                native_score = sims[:,ps[:n]].mean(1) - reward_penalty
                if n==1:
                    methods["fixed25"] = best(3*rankdata(native_score)+rankdata(direction))
                for tol in thresholds[n]:
                    methods[method_name(n,tol)] = select_close(native_score,direction,tol)
                for k in range(8):
                    candidates.append(dict(dataset=dataset,image=image,cluster=cluster,swap=int(flip),k=k,
                                           positives=n,score=native_score[k],direction_reward=direction[k],
                                           **{m:values[k] for m,values in metrics.items()}))
            for name, ids in methods.items():
                output.append(dict(dataset=dataset,image=image,cluster=cluster,swap=int(flip),method=name,
                                   k=",".join(map(str,ids)),**{m:float(a[ids].mean()) for m,a in metrics.items()}))
    expected = 76 if split=="train" else 41
    assert len(photos)==expected
    save(split+"_exclusions.json",excluded)
    return output,candidates


def averages(records):
    methods = defaultdict(list)
    for r in records:
        methods[r["method"]].append(r)
    return {name:{m:float(np.mean([r[m] for r in group])) for m in METRICS} for name,group in methods.items()}


def contrast(records, method, ref):
    x = {r["image"]:r for r in records if r["method"]==method}
    y = {r["image"]:r for r in records if r["method"]==ref}
    images = sorted(x.keys() & y.keys())
    by_set = defaultdict(lambda:defaultdict(list))
    diffs = []
    for i,im in enumerate(images):
        by_set[x[im]["dataset"]][x[im]["cluster"]].append(i)
        diffs.append([x[im][m]-y[im][m] for m in METRICS])
    a = np.array(diffs)
    rng=np.random.default_rng(20261123)
    totals=np.zeros((20000,len(METRICS)))
    counts=np.zeros(20000)
    for group in by_set.values():
        clusters=list(group.values())
        draw=rng.integers(len(clusters),size=(20000,len(clusters)))
        totals+=np.stack([a[c].sum(0) for c in clusters])[draw].sum(1)
        counts+=np.array([len(c) for c in clusters])[draw].sum(1)
    cis=np.quantile(totals/counts[:,None],[.0125,.025,.975,.9875],axis=0)
    return dict(n=len(images),**{m:dict(delta=float(a[:,i].mean()),lo95=cis[1,i],hi95=cis[2,i],
                                        lo975=cis[0,i],hi975=cis[3,i]) for i,m in enumerate(METRICS)})


def main(phase):
    lock_path=OUT/"开发锁定.json"
    if phase=="develop":
        result,raw=evaluate("train",{1:TOLS,2:TOLS})
        primary=[r for r in result if r["swap"]==0]
        scores=averages(primary)
        lock={}
        for n in [1,2]:
            base=scores[method_name(n,0)]
            eligible=[t for t in TOLS if t and scores[method_name(n,t)]["rr"]>=base["rr"]-.02]
            lock[n]=max(eligible,key=lambda t:(scores[method_name(n,t)]["direction"],scores[method_name(n,t)]["rr"],-t)) if eligible else 0
        table("开发逐图.csv",result)
        table("开发分项.csv",raw)
        save("开发锁定.json",dict(thresholds=lock, summary=scores,images=sorted({r["image"] for r in result}),
                             code=sha(Path(__file__)),protocol=sha(OUT/"实验口径.md"),inputs=FILES))
        print("Locked",lock)
    else:
        lock=read(lock_path)
        assert lock["code"]==sha(Path(__file__)) and lock["protocol"]==sha(OUT/"实验口径.md")
        thresholds={int(n):sorted({0,t}) for n,t in lock["thresholds"].items()}
        result,raw=evaluate("val",thresholds)
        assert not set(lock["images"]) & {r["image"] for r in result}
        table("验证逐图.csv",result)
        table("验证分项.csv",raw)
        summaries={}
        for swap in [0,1]:
            for dataset in ["all","val16","val11","val14"]:
                group=[r for r in result if r["swap"]==swap and (dataset=="all" or r["dataset"]==dataset)]
                d=dict(means=averages(group),contrasts={})
                for n in [1,2]:
                    current=method_name(n,lock["thresholds"][str(n)])
                    for ref in [method_name(n,0),method_name(1,0),"random","fixed25"]:
                        d["contrasts"][current+" vs "+ref]=contrast(group,current,ref)
                for ref in [method_name(1,0),"random","old"]:
                    if any(r["method"]==ref for r in group):
                        d["contrasts"][method_name(2,0)+" vs "+ref]=contrast(group,method_name(2,0),ref)
                summaries[f"{dataset}_swap{swap}"]=d
        save("验证汇总.json",dict(summary=summaries,inputs=FILES,lock=sha(lock_path)))
        print("Verified 41 disjoint images")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["develop","confirm"])
    main(p.parse_args().phase)
