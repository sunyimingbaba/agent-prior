"""Independently replay per-image reward selection, bootstrap by UCM scene, and check gallery aggregates."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiment import CROSS, OUT, REPORTS, METRICS, rows, save, sha


def main():
    archive=json.loads((OUT/"额外复验.json").read_text())["groups"]["combined"]
    base_rows=[r for r in rows(OUT/"验证逐图.csv") if r["swap"]=="0"]
    v=[r for r in rows(OUT/"验证分项.csv") if r["swap"]=="0" and r["positives"]=="1"]
    extra=rows(OUT/"额外16图选择.csv")
    ctrl=rows(OUT/"原奖励补齐14图.csv")
    names={r["image"] for r in base_rows}
    assert len(names)==41 and len({r["image"] for r in extra}-names)==16
    known={r["image"]:r for r in base_rows if r["method"]=="old"}
    for im in names:
        a=sorted([r for r in v if r["image"]==im],key=lambda r:int(r["k"]))
        q=np.array([float(x["score"]) for x in a])
        winner=np.flatnonzero(q==q.max())
        ref=next(r for r in base_rows if r["image"]==im and r["method"]=="p1_tol0")
        assert np.array_equal(winner,list(map(int,ref["k"].split(","))))
        if im not in known:
            g=sorted([r for r in ctrl if r["image"]==im],key=lambda r:int(r["k"]))
            scores=np.array([float(z["old"]) for z in g])
            ids=np.flatnonzero(scores==scores.max())
            known[im]={m:np.mean([float(a[k][m]) for k in ids]) for m in METRICS}
    known.update({r["image"]:r for r in extra if r["method"]=="old"})
    selected={r["image"]:r for r in base_rows+extra if r["method"]=="p1_tol0"}
    assert set(known)==set(selected)
    mapping={r["image"]:r["block"] for r in json.loads((REPORTS/"E3-2_UCM同起点三奖励短训_20261120/inputs/validation.json").read_text())}
    clusters={}
    for im in known:
        clusters.setdefault(mapping[im],[]).append([float(selected[im][m])-float(known[im][m]) for m in METRICS])
    g=list(clusters.values())
    sizes=np.array([len(v) for v in g])
    sums=np.array([np.sum(v,axis=0) for v in g])
    draws=np.random.default_rng(20261123).integers(len(g),size=(20000,len(g)))
    boot=sums[draws].sum(1)/sizes[draws].sum(1)[:,None]
    estimates={}
    for i,m in enumerate(METRICS):
        mean=sums[:,i].sum()/57
        assert abs(mean-archive["diff"]["p1_tol0 vs old"][m]["delta"])<1e-12
        estimates[m]=dict(mean=mean,lo=np.quantile(boot[:,i],.025),hi=np.quantile(boot[:,i],.975))
    for fname in ["整库召回.csv","全部57图整库召回.csv"]:
        for r in rows(OUT/fname):
            assert math.isclose(float(r["mR"]),np.mean([float(r[m]) for m in ["i1","i5","i10","t1","t5","t10"]]),abs_tol=1e-12)
    save("校验与场景敏感性.json",dict(n=57,scenes=len(g),differences=estimates,verified_41selections=True,
                                archive_reference_RR=archive["means"]["old"]["rr"]))
    print("57 matched sources validated;",len(g),"UCM scene clusters",estimates["rr"])


if __name__=="__main__":
    main()
