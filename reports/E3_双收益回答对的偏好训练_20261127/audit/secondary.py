"""Replay alternative-caption recalls and independently judged wording from new final features."""
from pathlib import Path
from formal import STAGE,HERE,m
import json
import numpy as np

def main():
    g=m.read(m.FORM/"inputs/ucm.json")
    bank=m.load(m.FORM/"results/ucm_text.pt")
    ids=bank["caption_order"].numpy().reshape(210,5)
    pos=np.flatnonzero(ids.ravel()!=np.repeat(ids[:,0],5))
    owners=pos//5; captions=ids.ravel()[pos]
    den=np.bincount(owners); assert len(pos)==sum(den)==833
    old=STAGE.parent/"E3_R3改选收益校准与双批首句验证_20261126/inputs"
    words=m.load(old/"fresh_text.pt"); mapping=m.read(old/"独立措辞.json")
    assert words["source"]==m.sha(old/"独立措辞.json")
    wi={w:i for i,w in enumerate(words["words"])}
    ids_img={r["rgb_sha256"]:i for i,r in enumerate(g)}
    signature=lambda r:(r["attribute"],r["side"],*r["D"],*r["E"])
    keys={signature(r) for r in m.read(old/"real_roster.json")}
    matched=[r for r in m.read(m.FINE/"inputs/panel.json") if r["rgb_sha256"] in ids_img and signature(r) in keys]
    assert len(matched)==56
    raw={(r["image"],r["seed"],r["arm"]):r for r in m.read(STAGE/"evaluation/results/逐图指标.json")}
    arr={}; fg={}
    for arm in ["r2","r3"]:
        for seed in "ab":
            path=STAGE/"evaluation/records"/(seed+"_"+arm)/"features.pt"
            assert m.sha(path)==m.read(path.with_name("complete.json"))["features"]
            v=m.load(path)["features"].double().numpy();sim=(v@bank["native"].double().numpy().T)[:,captions]
            nf=[];nt=np.zeros((len(g),3))
            for i,s in enumerate(sim):
                z=s.copy(); z[captions==ids[i,0]]=-np.inf
                nf.append(m.hits(z,np.flatnonzero(owners==i)))
            for idx,s in enumerate(sim.T):
                nt[owners[idx]]+=m.hits(s,[owners[idx]])
            counts=np.concatenate([nf,nt],1);arr[arm,seed]=counts
            for i,r in enumerate(g):
                saved=raw[r["image"],seed,arm]["alternatives"]
                assert saved["captions"]==den[i] and np.allclose(saved["hits"],counts[i],rtol=0,atol=1e-13)
            wording=v@words["features"].double().numpy().T
            fg[arm,seed]=np.array([np.mean([wording[ids_img[r["rgb_sha256"]],wi[grp[r["side"]]]]>
                                          wording[ids_img[r["rgb_sha256"]],wi[grp[1-r["side"]]]] for grp in mapping[r["attribute"]]])
                                  for r in matched])
    contrast=np.stack([arr["r3",s]-arr["r2",s] for s in "ab"])
    per=contrast.mean(0)
    def recall_mean(arr, sizes):
        return np.concatenate([arr[...,:3]/210,arr[...,3:]/sizes[...,None]],axis=-1).mean(axis=-1)
    observed=float(recall_mean(per.sum(0),np.asarray(den.sum())))
    rng=np.random.default_rng(20261125)
    draws=[]
    for j in range(40):
        z=rng.integers(210,size=(500,210))
        draws.extend(recall_mean(per[z].sum(1),den[z].sum(1)))
    ci=np.quantile(draws,[.025,.975]).tolist()
    seedvals=[float(recall_mean(group.sum(0),np.asarray(den.sum()))) for group in contrast]
    official=m.read(STAGE/"evaluation/results/单次检索对照.json")["alternatives"]["contrasts"]["r3-r2"]["mR"]
    assert np.allclose([observed,*ci,*seedvals],[official["mean"],*official["interval"],*official["seeds"]],rtol=0,atol=1e-12)
    fg_delta=m.summarize(np.stack([fg["r3",s]-fg["r2",s] for s in "ab"]),[r["pair"] for r in matched])
    orig=m.read(STAGE/"results/单次新措辞复核.json")["r3_minus_r2"]["FG"]
    assert np.allclose([fg_delta["mean"],*fg_delta["interval"]],[orig["mean"],*orig["interval"]],rtol=0,atol=1e-12)
    result=dict(noninput_recall=dict(captions=len(pos),images=len(g),seeds=seedvals,mean=observed,interval=ci),
                wording56=fg_delta,code_sha256=m.sha(Path(__file__)))
    (HERE/"非输入配文与新措辞独立验算.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)
if __name__=="__main__":main()
