"""Can a prior full-trajectory sample point to good cold post-tool training groups?"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
STAGE = OUT.parent
PREV = STAGE.parent / "E3_R2逐候选分值与困难细节加分_20261127"
sys.path.insert(0,str(PREV/"candidates"))
from compare import scores_for_seed, LIBRARIES, OLD, read, load


def screen():
    rows = read(OLD/"inputs/real_roster.json")
    banks = {k:load(p/"records/text.pt") for k,p in LIBRARIES.items()}
    selected = []
    for record in scores_for_seed(rows, "a", detailed=True):
        if record["block"] or "mrr" not in record["outcomes"]:
            continue
        name = record["image"]
        bank = banks[record["cohort"]]
        m = bank["maps"][name]
        text = bank["hard"][[*m["A"],*m["B"]]].double().numpy()
        feat = load(OLD/"records/real/a/features"/(name+".pt"))["visual"][:8].double().numpy()
        unique,inv = np.unique(feat,axis=0,return_inverse=True)
        sim = (unique @ text.T)[inv].reshape(8,2,2)
        correct = (sim[:,:,0] > sim[:,:,1]).mean(axis=1)
        alt = np.asarray(record["outcomes"]["mrr"])
        top = record["winners_r2"]
        slots = np.flatnonzero((correct > correct[top].mean()) & (alt >= alt[top].mean() - 1e-12))
        selected.append(dict(image=name,pair=record["pair"],cohort=record["cohort"],
                             slots=slots.tolist(),selected=bool(len(slots))))
    raw = json.dumps(selected,ensure_ascii=False,indent=2)+"\n"
    (OUT/"旧记录名单.json").write_text(raw)
    print("eligible",len(selected),"selected",sum(x["selected"] for x in selected),"sha256",hashlib.sha256(raw.encode()).hexdigest())


def compare():
    screens = read(OUT/"旧记录名单.json")
    data = []
    for row in screens:
        path = STAGE/"scores/b"/(row["image"]+".json")
        if not path.is_file():
            continue
        b = read(path)
        if b.get("items") and b["items"][0].get("other_caption_rr"):
            data.append((row["cohort"],row["pair"],row["selected"],int(bool(b["opportunity_slots"]))))
    assert data
    pairs = sorted(set((c,p) for c,p,_,_ in data))
    counts = np.zeros((len(pairs),2,2),dtype=int)
    for c,p,selected,outcome in data:
        counts[pairs.index((c,p)),int(selected)] += [outcome,1]
    totals = counts.sum(axis=0)
    draws = np.random.default_rng(112726).multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=20000)
    boots = np.einsum("dp,pki->dki",draws,counts)
    boots = boots[(boots[:,:,1] > 0).all(axis=1)]
    rates = boots[:,:,0] / boots[:,:,1]
    result = dict(counts_unselected_selected=totals.tolist(),rates_unselected_selected=(totals[:,0]/totals[:,1]).tolist(),
                  difference=float(np.diff(totals[:,0]/totals[:,1])[0]),
                  interval=np.quantile(rates[:,1]-rates[:,0],[.025,.975]).tolist(),pairs=len(pairs))
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("screen","compare"))
    torch.set_num_threads(1)
    if parser.parse_args().action=="screen":
        screen()
    else:
        compare()
