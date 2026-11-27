"""Held-caption transfer and a locked unseen policy-preference probe."""
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from select_pairs import STAGE, SOURCE, read, sha, ranked_pair


def write(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def aggregate(rows):
    pairs=sorted({(x["cohort"],x["pair"]) for x in rows})
    draws=np.random.default_rng(271127).multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=20000)
    output={}
    for label in ("against_single","against_tie_mean"):
        for k in ("DE","FG","held_rr","switched"):
            totals=np.zeros((len(pairs),2))
            for row in rows:
                values=[v[label][k] for v in row["trials"]]
                totals[pairs.index((row["cohort"],row["pair"]))]+=[np.mean(values),1]
            boot=draws@totals
            output.setdefault(label,{})[k]=dict(mean=float(totals[:,0].sum()/len(rows)),
                             interval=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist())
    return dict(images=len(rows),pairs=len(pairs),trials=sum(len(x["trials"]) for x in rows),effects=output)


def main():
    train=read(STAGE/"inputs/train_pairs.json")["rows"]
    exclude={(x["cohort"],x["pair"]) for x in train}
    roster=sorted(read(SOURCE/"scoring/inputs/rows.json"),key=lambda r:r["rgb_sha256"])
    checksums=read(SOURCE/"scoring/results/b_scores_sha256.json")
    pools={"joint":[], "anchor":[]}
    records=[]
    for r in roster:
        if (r["cohort"],r["pair"]) in exclude or len(r["alt_ids"])<2:
            continue
        name=r["image"]
        path=SOURCE/"scores/b"/(name+".json")
        assert sha(path)==checksums[name]
        g=read(path)
        if not g["eligible"]:
            continue
        rr=np.array([i["other_caption_rr"] for i in g["items"]])
        fine={k:(np.array([x[k] for x in g["items"]])>0).mean(axis=1) for k in ("DE","FG")}
        trials=[]
        columns=range(rr.shape[1])
        raw_path=SOURCE/"sampling/raw/b"/(name+".pt")
        assert sha(raw_path)==g["source"]
        raw=torch.load(raw_path,map_location="cpu",weights_only=True)
        for hidden in columns:
            i,j=ranked_pair(g,[k for k in columns if k!=hidden])
            chosen=i if j is None else j
            if j is not None: assert raw["paths"][i]!=raw["paths"][j]
            t=dict(hidden=hidden,base=i,choice=chosen)
            for label,top in [("against_single",[i]),("against_tie_mean",g["r2_top"])]:
                measures=dict(**fine,held_rr=rr[:,hidden])
                t[label]={k:float(v[chosen]-v[top].mean()) for k,v in measures.items()}
                t[label]["switched"]=int(j is not None)
            trials.append(t)
        records.append(dict(image=name,cohort=r["cohort"],pair=r["pair"],trials=trials))
        for kind in pools:
            i,j=ranked_pair(g,columns,anchor=(kind=="anchor"))
            if j is None:continue
            assert raw["paths"][i]!=raw["paths"][j]
            info={k:r[k] for k in ("image","row","rgb_sha256","pair","cohort","orig_query","feedback","split")}
            info.update(kind=kind,source_seed="b",raw_sha256=g["source"],score_sha256=checksums[name],r2_winner=i,r2_loser=j,
                        r3_winner=i if kind=="anchor" else j,r3_loser=j if kind=="anchor" else i)
            pools[kind].append(info)
    out=STAGE/"validation"
    with gzip.open(out/"逐图留句.jsonl.gz","wt",encoding="utf-8") as f:
        for rec in records:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    summary=aggregate(records)
    write(out/"summary.json",summary)
    want=pools["joint"][:8]
    names={r["image"] for r in want}
    anchors=[r for r in pools["anchor"] if r["image"] not in names][:len(want)]
    n=len(anchors)
    selected=[r for pair in zip(anchors,want[:n]) for r in pair]
    probe=dict(rows=selected,counts={"anchor":n,"joint":n},train_hash=sha(STAGE/"inputs/train_pairs.json"),
               source_scores=sha(SOURCE/"scoring/results/b_scores_sha256.json"),source_raw=sha(SOURCE/"sampling/b_complete.json"))
    lock=STAGE/"inputs/probe_pairs.json"
    write(lock,probe)
    lock.with_suffix(".sha256").write_text(sha(lock)+"\n")
    print(summary["images"],summary["pairs"],summary["trials"],str(summary["effects"]["against_single"]),n,sha(lock))


if __name__=="__main__":
    main()
