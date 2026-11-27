"""Measure the incremental value of adding visually confirmed C evidence."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/"E3_困难奖励跨图跨措辞校准_20261127"
sys.path.insert(0,str(PREV/"code"))
from panel import read,save,interval,POOL
from reward import redistribute


def evaluate(retrieval,ab,other,c,verified,visible=None):
    base=np.asarray(retrieval)
    hard=np.asarray(ab)
    weak=hard.min(axis=1)
    count=(hard>0).sum(axis=1)
    top=np.flatnonzero(base==base.max())
    gate=count*len(top)>count[top].sum()
    native=np.where(gate|(base==base.max()),weak,min(weak))
    strict=redistribute(base,native,other,visible)
    if np.any(base!=strict):return strict,strict,strict
    admissible=(count*len(top)>=count[top].sum()) & (weak>=weak[top].mean()) & (c>0)
    cand=np.where(admissible|(base==base.max()),c,min(c))
    extra=redistribute(base,cand,other,visible)
    return strict,extra,extra if verified else strict


def study(rows,sources,signals,teacher):
    groups=[]; all_metrics=[]
    for row in rows:
        name=row["image"]
        per=[]
        for seed,folder in sources.items():
            g=read(folder/(name+".json"))
            if not g["eligible"]:continue
            items=g["items"]; rr=np.array([x["other_caption_rr"] for x in items])
            if rr.shape[1]<2:continue
            assert len(signals[seed][name])==8
            c=np.asarray(signals[seed][name])
            ab=np.array([x["hard_margins"] for x in items])
            de,fg=[(np.array([x[k] for x in items])>0).mean(1) for k in ("DE","FG")]
            for h in range(rr.shape[1]):
                strict,extra,new=evaluate(g["r2"],ab,rr,c,teacher[name],[k for k in range(rr.shape[1]) if k!=h])
                m={}
                arms=[("strict",strict),("C",extra),("visual",new),("r2",np.asarray(g["r2"]))]
                for t,v in arms:
                    ix=np.flatnonzero(v==v.max())
                    for k,score in (("DE",de),("FG",fg),("MRR",rr[:,h])):
                        m[t+"_"+k]=score[ix].mean()
                for t,y in [("visual","strict"),("visual","C"),("visual","r2"),("strict","r2"),("C","strict")]:
                    for k in ["DE","FG","MRR"]:m[t+"_minus_"+y+"_"+k]=m[t+"_"+k]-m[y+"_"+k]
                m["visual_changed"]=float(not np.array_equal(new,strict))
                per.append(m)
        if per:
            groups.append(dict(image=name,pair=row['pair'],cohort=row.get('cohort','external'),
                               metrics={k:float(np.mean([r[k] for r in per])) for k in per[0]}))
    return groups,interval(groups)


def main():
    folder=ROOT/"visual/results"
    teacher=read(folder/"verified_truth.json");signals=read(folder/"candidate_margins.json")
    configs={"panel194":(read(PREV/"inputs/test_panel.json"),{s:POOL/"scores"/s for s in "ab"}),
             "panel116":(read(PREV/"followup_pool/inputs/rows.json"),{"d":PREV/"followup_pool/scores"})}
    reports={}
    for label,(rows,sources) in configs.items():
        groups,summary=study(rows,sources,signals,teacher)
        save(ROOT/"results"/(label+"_images.json"),groups)
        reports[label]=summary
    a,b=[reports[k]["metrics"] for k in ["panel194","panel116"]]
    ref="visual_minus_strict_"
    reports["train_visual"]=bool(any(b[ref+k]["ci"][0]>0 for k in ("DE","MRR")) and
              all(x[ref+k]["mean"]>=0 for x in (a,b) for k in ("DE","MRR")) and
              any(b["visual_minus_C_"+k]["mean"]>0 for k in ("DE","MRR")))
    save(ROOT/"results/候选视觉增量.json",reports)
    print(reports["train_visual"],[(n,{k:v for k,v in z['metrics'].items() if 'visual_minus_strict' in k}) for n,z in list(reports.items())[:2]])


if __name__=="__main__":main()
