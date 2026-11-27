"""Cross-check all observed indices and score changes against the sealed production files."""
import gzip
import json
from collections import Counter
import numpy as np
from pairs import HERE,OLD,read,sha,source,save,choices,summarize,held

def main():
    root=HERE.parent
    pairs=read(HERE/"独立A选对.json")["ordered"]
    train_p=root/"inputs/train_pairs.json"
    assert sha(train_p)==train_p.with_suffix(".sha256").read_text().strip()
    actual=read(train_p)["rows"]
    assert len(pairs)==len(actual)==24
    for l,r in zip(pairs,actual):
        assert (l["image"],l["kind"],l["baseline"],l["alternative"],l["preferred"],l["rejected"],l["raw_a"])==(
            r["image"],r["kind"],r["r2_winner"],r["r2_loser"],r["r3_winner"],r["r3_loser"],r["raw_sha256"])
    rows=read(OLD/"scoring/inputs/rows.json"); byname={r["image"]:r for r in rows}
    b=source("b"); obs=held(rows,b,{"ordered":pairs}); names={o["image"] for o in obs}
    with gzip.open(root/"validation/逐图留句.jsonl.gz","rt") as stream:
        compared={o["image"]:o for o in map(json.loads,stream)}
    assert set(compared)==names
    removed={tuple((r["cohort"],r["pair"])) for r in pairs}
    assert removed.isdisjoint(tuple(r["pair"]) for r in obs)
    n=0;tie=[]
    for r in obs:
        name=r["image"];g=b[name];q=compared[name]
        assert len(q["trials"])==len(r["folds"])
        btop=[h for h,x in enumerate(g["r2"]) if x==max(g["r2"])]
        folds=[]
        for f,v in zip(r["folds"],q["trials"]):
            assert (f["hidden"],f["i"],f["chosen"])==(v["hidden"],v["base"],v["choice"])
            transformed=dict(switched=f["switched"])
            for k in ("DE","FG","MRR"):
                key="held_rr" if k=="MRR" else k
                assert np.isclose(f[k],v["against_single"][key],rtol=0,atol=1e-15)
                values=[(z["other_caption_rr"][f["hidden"]] if k=="MRR" else sum(x>0 for x in z[k])/2)
                        for z in g["items"]]
                transformed[k]=values[f["chosen"]]-sum(values[i] for i in btop)/len(btop)
                assert abs(transformed[k]-v["against_tie_mean"][key])<1e-14
            folds.append(transformed);n+=1
        tie.append(dict(image=name,pair=r["pair"],folds=folds))
    metrics=[summarize(obs),summarize(tie)]
    prod=read(root/"validation/summary.json")["effects"]
    for result,ref in zip(metrics,(prod["against_single"],prod["against_tie_mean"])):
        for k,z in result["metrics"].items():
            ex=ref["held_rr" if k=="MRR" else k]
            assert np.allclose([z["value"],*z["interval"]],[ex["mean"],*ex["interval"]],rtol=0,atol=1e-13)
    selected={}
    for kind in ["joint","anchor"]:
        selected[kind]=[]
        for name in sorted(names,key=lambda n:byname[n]["rgb_sha256"]):
            if kind=="anchor" and name in {n for n,_,_ in selected["joint"]}:continue
            i,j=choices(b[name],range(len(byname[name]["alt_ids"])),anchor=kind=="anchor")
            if j is not None:selected[kind].append((name,i,j))
            if len(selected[kind])==8:break
    expected=[p for a,j in zip(selected["anchor"],selected["joint"]) for p in (a,j)]
    probe=read(root/"inputs/probe_pairs.json")["rows"]
    assert [(r["image"],r["r2_winner"],r["r2_loser"]) for r in probe]==expected
    token_counts={}
    for kind in ("joint","anchor"):
        ns=[r for r in pairs if r["kind"]==kind]
        token_counts[kind]={k:sum(r[k] for r in ns) for k in ["preferred_tokens","rejected_tokens",
                                                                "preferred_continuation","rejected_continuation"]}
        token_counts[kind]["preferred_shorter"]=sum(r["preferred_tokens"]<r["rejected_tokens"] for r in ns)
    save(HERE/"候选与训练对交叉核验.json",dict(train_rows=24,train_sha256=sha(train_p),held_images=len(names),
        excluded_training_pairs=len(removed),held_folds=n,held_pairs=len(set(tuple(r["pair"]) for r in obs)),
        held_all_slots_and_scores_match=True,tie_sensitivity_matches=True,probe_rows=len(probe),token_counts=token_counts,
        independent=metrics))
if __name__=="__main__":main()
