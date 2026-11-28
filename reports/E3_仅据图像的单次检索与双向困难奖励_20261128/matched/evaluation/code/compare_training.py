"""Compare paired R3-minus-R2 differences under the two training-input protocols."""
from collections import defaultdict
from pathlib import Path
import gzip
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parents[1]/"image_only"
sys.path.insert(0,str(OLD/"code"))
spec=importlib.util.spec_from_file_location("common_scores",OLD/"code/score.py")
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
read,sha,dump=score.read,score.sha,score.dump

def differential(root):
    file=root/"results/逐图指标.jsonl.gz"
    assert sha(file)==read(root/"results/complete.json")["files"]["results/逐图指标.jsonl.gz"]
    with gzip.open(file,'rt') as stream:rows=[json.loads(line) for line in stream]
    by={(r["panel"],r["seed"],r["arm"],r["image"]):r for r in rows}
    assert len(by)==len(rows)==(210+164+106)*4
    out={}
    for (panel,s,arm,n),r in by.items():
        if arm!="strict":continue
        base=by[panel,s,"r2",n];assert r["rgb"]==base["rgb"]
        out[panel,s,n]={k:(v-base["metrics"][k]) if v is not None else np.nan for k,v in r["metrics"].items()}
    return out

def main():
    a,b=differential(OLD),differential(ROOT);assert set(a)==set(b)
    g=read(score.FORM/"inputs/ucm.json");ids={r["rgb_sha256"] for r in g}
    f=[r for r in read(score.FINE/"inputs/panel.json") if r["rgb_sha256"] in ids]
    e=read(score.OLD/"extra164/inputs/rows.json")
    panels=dict(official=g,first106=f,extra164=e);out={}
    for p,rows in panels.items():
        names=[r["image"] for r in rows]
        units=[r.get("pair",r["image"]) for r in rows]
        calc=score.extra_stats if p=="extra164" else score.summary_by_seed
        keys=b[p,"a",names[0]].keys();diff={}
        for s in "ab":
            diff[s]={k:np.array([b[p,s,n][k]-a[p,s,n][k] for n in names]) for k in keys}
        out[p]=dict(images=len(rows),units=len(set(units)),interaction=calc(diff,units))
    dump(ROOT/"results/两种训练输入的奖励差异.json",dict(
        contrast="(matched strict − matched R2) − (first-caption-trained strict − first-caption-trained R2), identical image-only evaluation",
        source={r.name:sha(r/"results/逐图指标.jsonl.gz") for r in (OLD,ROOT)},panels=out))

if __name__=="__main__":main()
