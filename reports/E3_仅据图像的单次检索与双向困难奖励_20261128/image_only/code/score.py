"""Score both fixed-input protocols with identical frozen galleries and historical paired readouts."""
from importlib.util import module_from_spec,spec_from_file_location
from collections import Counter,defaultdict
from pathlib import Path
import gzip,hashlib,json,sys,tarfile
import numpy as np
import torch
from prepare import ROOT,REPORTS,OLD,FORM,FINE,QUERY,read,dump,sha
spec=spec_from_file_location("old_210",OLD/"evaluation/code/score.py")
hist=module_from_spec(spec);spec.loader.exec_module(hist)
spec=spec_from_file_location("old_164",OLD/"extra164/code/score.py")
extra=module_from_spec(spec);spec.loader.exec_module(extra)
from score_final import calculate,details,summary_by_seed,paired

def pack(path,folder,hashes):
    if not path.exists():
        tmp=path.with_suffix(".part")
        with tarfile.open(tmp,"w:gz",compresslevel=3) as t:
            for n in sorted(hashes):t.add(folder/n,arcname=n)
        tmp.replace(path)
    check={}
    with tarfile.open(path,"r|gz") as tf:
        for m in tf:
            assert m.isfile() and m.name not in check
            check[m.name]=hashlib.file_digest(tf.extractfile(m),"sha256").hexdigest()
    assert hashes==check
    return dict(sha256=sha(path),files=len(check),bytes=path.stat().st_size)

def inspect(root,tag,panel,rows,lock,server,feedback):
    folder=root/"records"/tag/panel; f=folder/"features.pt";cp=read(folder/"complete.json")
    sig=lock["policies"][tag]["adapter"]
    assert cp["adapter"]==sig and cp["lock"]==sha(root/"inputs/server_lock.json") and cp["query"]==QUERY and cp["features"]==sha(f)
    z=torch.load(f,map_location="cpu",weights_only=True)
    assert z["images"]==[r["image"] for r in rows] and z["adapter"]==sig
    assert set(z["images"])==set(cp["images"]) and z["features"].shape==(len(rows),768)
    hashes={"features.pt":sha(f),"complete.json":sha(folder/"complete.json")}
    tokens=Counter();calls=Counter()
    for i,r in enumerate(rows):
        n=r["image"];p=folder/(n+".pt"); hashes[p.name]=sha(p)
        assert hashes[p.name]==cp["images"][n]
        obs=torch.load(p,map_location="cpu",weights_only=True)
        assert (obs["query"],obs["actual_query"],obs["image"],obs["rgb"],obs["feedback"],obs["lock"],obs["template"],obs["adapter"])==(QUERY,QUERY,n,r["rgb"],r["feedback"],cp["lock"],cp["template"],sig)
        assert obs["actions"] and all(isinstance(t,int) for a in obs["actions"] for t in a)
        assert all(v==feedback[r["rgb"]]["feedback"]==r["feedback"] for v in obs["calls"])
        assert torch.isfinite(obs["prior"]).all() and obs["prior"].shape==(3584,) and torch.equal(obs["v"],z["features"][i])
        tokens[obs["first_input_ids_sha256"]]+=1;calls[len(obs["calls"])]+=1
    tag=tag+"_"+panel
    a=pack(root/"archives"/(tag+".tar.gz"),folder,hashes)
    return z["features"],dict(**a,adapter=sig,first_input_token_hashes=dict(tokens),tool_calls=dict(calls),template=cp["template"],
             torch=cp["torch"],device=cp["device"],cuda=cp["cuda"])
def extra_stats(data,labels):
    groups=sorted(set(labels));idx=np.array([groups.index(x) for x in labels]); n=len(groups)
    weights=np.random.default_rng(271127).multinomial(n,np.full(n,1/n),size=20000)
    a,b=data["a"],data["b"];result={}
    for k in a:
        v=np.stack((a[k],b[k]));x=(v[0]+v[1])/2;ok=np.isfinite(x)
        cnt=np.bincount(idx[ok],minlength=n);total=np.bincount(idx[ok],weights=x[ok],minlength=n)
        ci=np.quantile((weights@total)/(weights@cnt),[.025,.975])
        result[k]=dict(mean=float(np.nanmean(x)),interval=ci.tolist(),seeds=np.nanmean(v,axis=1).tolist(),images=int(ok.sum()))
    return result
def metrics_extra(rs,t,vs):
    m=extra.measures(rs,t,vs);out={k:m[:,i] for i,k in enumerate(("DE","mrr","mR","I2T","T2I"))}
    seq=[j for r in rs for j in r["panel_positive_ids"]];n=len(rs)
    sim=extra.fixed(vs,t["captions"])[:,seq];gold=np.arange(n)[:,None]==np.repeat(np.arange(n),5)
    a,b=extra.recalls(sim,gold),extra.recalls(sim.T,gold.T)
    for k in (1,5,10):
        out["I2T_R"+str(k)]=a[k];out["T2I_R"+str(k)]=b[k].reshape(n,5).mean(1)
    return out
def summarize(value,labels):
    blocks={}
    for panel,models in value.items():
        units=labels[panel]
        delta={s:{k:v-models["r2"][s][k] for k,v in models["strict"][s].items()} for s in "ab"}
        score=extra_stats if panel=="extra164" else summary_by_seed
        blocks[panel]=dict(units=len(set(units)),images=len(units),r2=score(models["r2"],units),strict=score(models["strict"],units),
                           strict_minus_r2=score(delta,units))
    return blocks
def main(root):
    lock=read(root/"inputs/manifest.json");server=read(root/"inputs/server_lock.json")
    baseline=read(ROOT/"inputs/manifest.json")
    assert all(sha(REPORTS/p)==v for p,v in baseline["score_inputs"].items())
    assert lock["query"]==baseline["query"]==QUERY and server["manifest"]==sha(root/"inputs/manifest.json")
    assert sha(root/"feedback_verified.json")==server["feedback"]
    roster=read(root/"inputs/roster.json"); assert roster==read(ROOT/"inputs/roster.json")
    fx=read(root/"feedback_verified.json")["checks"];assert len(fx)==len(roster)==374
    gal=read(FORM/"inputs/ucm.json");ids={r["rgb_sha256"]:i for i,r in enumerate(gal)}
    fine=[r for r in read(FINE/"inputs/panel.json") if r["rgb_sha256"] in ids]
    ext=read(OLD/"extra164/inputs/rows.json")
    labels=dict(official=[r["image"] for r in gal],first106=[r["pair"] for r in fine],extra164=[r["pair"] for r in ext])
    native=torch.load(FORM/"results/ucm_text.pt",weights_only=True);hard=torch.load(FINE/"records/text.pt",weights_only=True)
    et=torch.load(OLD/"extra164/inputs/texts.pt",weights_only=True); cols=native["caption_order"].numpy().reshape(210,5)
    assets={}; values={p:defaultdict(dict) for p in labels}; observations=[]
    for s in "ab":
        for a in ("r2","strict"):
            tag=s+"_"+a
            vec={}
            for p,orig in (("official",gal),("extra",ext)):
                rs=[r for r in roster if r["panel"]==p];assert [r["rgb"] for r in rs]==[r["rgb_sha256"] for r in orig]
                vec[p],assets[tag+"_"+p]=inspect(root,tag,p,rs,lock,server,fx)
            values["official"][a][s]=calculate("official",vec["official"],gal)
            values["first106"][a][s]=details(vec["official"],fine,ids,cols,native,hard)
            values["extra164"][a][s]=metrics_extra(ext,et,vec["extra"])
            for p,rs in (("official",gal),("first106",fine),("extra164",ext)):
                for i,r in enumerate(rs):
                    z={k:float(v[i]) if np.isfinite(v[i]) else None for k,v in values[p][a][s].items()}
                    observations.append(dict(seed=s,arm=a,panel=p,image=r["image"],rgb=r["rgb_sha256"],metrics=z))
    assert len({z["template"] for z in assets.values()})==1
    assert len({k for z in assets.values() for k in z["first_input_token_hashes"]})==1
    result=summarize(values,labels)
    m=result["official"]["strict_minus_r2"]["mR"]; d=result["first106"]["strict_minus_r2"];e=result["extra164"]["strict_minus_r2"]
    gate=dict(de_lower_positive=d["DE"]["interval"][0]>0,official_mR_nonnegative=m["mean"]>=0,
              mrr_nonnegative=d["mrr"]["mean"]>=0)
    result["formal_criteria"]=gate;result["formal_pass"]=all(gate.values())
    result["extra164_preserved"]={k:e[k]["mean"]>=0 for k in ("DE","mR","mrr")}
    result["joint_pass"]=result["formal_pass"] and all(result["extra164_preserved"].values())
    assert all(np.isfinite([m["mean"],d["DE"]["mean"],d["mrr"]["mean"]]))
    dump(root/"archives/index.json",assets);dump(root/"results/单次检索.json",result)
    with gzip.open(root/"results/逐图指标.jsonl.gz","wt",encoding="utf8") as fh:
        for r in observations:fh.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    old=read(OLD/"evaluation/results/单次检索对照.json");oe=read(OLD/"extra164/summary.json")
    previous=dict(official={a:old["official"]["levels"][a] for a in ("r2","strict")},first106={a:old["first106"]["levels"][a] for a in ("r2","strict")},
            extra164={a:oe["levels"]["rfpo_"+a] for a in ("r2","strict")})
    for p in ("official","first106"):previous[p]["strict_minus_r2"]=old[p]["contrasts"]["strict-r2"]
    previous["extra164"]["strict_minus_r2"]=oe["contrasts"]["rfpo_strict"]
    dump(root/"results/原首句输入参照.json",previous)
    files=["inputs/manifest.json","inputs/server_lock.json","inputs/roster.json","feedback_verified.json","archives/index.json",
           "results/单次检索.json","results/原首句输入参照.json","results/逐图指标.jsonl.gz"]
    dump(root/"results/complete.json",dict(images=374,predictions=1496,files={f:sha(root/f) for f in files},scorer=sha(Path(__file__))))
if __name__=="__main__":
    torch.set_num_threads(1)
    main(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT)
