"""Evaluate relative-to-reference shifts separately for concordant and opposed preferences."""
from pathlib import Path
import json
import numpy as np

stage=Path(__file__).resolve().parents[1]


def read(f):
    return json.loads(f.read_text())


def indicator(gap):
    return float(gap>0)+0.5*(gap==0)


def summary():
    records=[]
    for seed in "ab":
        base=read(stage/"probe"/f"{seed}_ref.json")
        for arm in ("r2","r3"):
            current=read(stage/"probe"/f"{seed}_{arm}.json")
            assert all(base["signature"][k]==current["signature"][k] for k in ("initial","probe","train","implementation","policy_code"))
            for panel in ("train","probe"):
                old=base["records"][panel]
                new=current["records"][panel]
                assert len(old)==len(new)
                for a,b in zip(old,new):
                    assert all(a[k]==b[k] for k in ("image","slots","target_position","pair","kind"))
                    chosen=a["target_position"]
                    assert [r["tokens"] for r in a["responses"]]==[r["tokens"] for r in b["responses"]]
                    def gap(row):
                        return row["responses"][chosen]["logp"]-row["responses"][1-chosen]["logp"]
                    initial=gap(a)
                    final=gap(b)
                    margin=final-initial
                    records.append(dict(seed=seed,arm=arm,panel=panel,image=a["image"],pair=a["pair"],kind=a["kind"],
                                        target_slot=a["slots"][chosen],initial_gap=initial,final_gap=final,corrected_margin=margin,
                                        corrected_accuracy=indicator(margin),raw_accuracy=indicator(final)))
    result={}
    for panel in ("train","probe"):
        result[panel]={}
        for kind in ("joint","anchor"):
            metrics=("corrected_accuracy","corrected_margin","raw_accuracy","initial_gap","final_gap")
            source=[r for r in records if r["panel"]==panel and r["kind"]==kind]
            names=[r["image"] for r in source if r["seed"]=="a" and r["arm"]=="r2"]
            assert len(names)==len(set(names))
            result[panel][kind]={"images":len(names)}
            for arm in ("r2","r3"):
                x=np.array([[[r[m] for m in metrics] for r in source if r["seed"]==s and r["arm"]==arm] for s in "ab"])
                for s in "ab":
                    assert names==[r["image"] for r in source if r["seed"]==s and r["arm"]==arm]
                result[panel][kind][arm]={m:dict(mean=float(x[:,:,i].mean()),seeds=x[:,:,i].mean(1).tolist()) for i,m in enumerate(metrics)}
    out=stage/"results/回答偏好诊断.json"
    out.write_text(json.dumps(dict(summary=result,rows=records),ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    summary()
