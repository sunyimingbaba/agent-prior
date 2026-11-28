"""Score the first-action training trajectories on unused wording with the historical native readout."""
from collections import defaultdict
from pathlib import Path
import sys
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
STAGE=HERE.parents[1]
sys.path.insert(0,str(STAGE/"audit"))
from image_inputs import CONVERT, READOUT, image, tables, read, load, sha
from estimate import top_weights

@torch.inference_mode()
def main():
    pool=STAGE.parent/"E3_回填后双收益训练图筛选_20261127/scoring"
    fields=load(pool/"inputs/vectors.pt")
    assert sha(pool/"inputs/vectors.pt")==read(pool/"inputs/manifest.json")["compiled"]["vectors.pt"]
    rows=read(pool/"locks/training.json")["rows"]
    assert len(rows)==24
    db=tables(); fn=READOUT/"cup_native_trace.pt"
    assert sha(fn)==read(READOUT/"proof.json")["trace"]
    reader=torch.jit.load(str(fn),map_location="cpu").eval()
    grouped=defaultdict(list); checked=0.
    for seed in "ab":
        for arm in ("r2","strict"):
            folder=HERE.parent/"models"/(seed+"_"+arm)
            steps=read(folder/"steps.json")
            assert sha(folder/"steps.json")==read(folder/"complete.json")["scores"]
            for r,z in zip(rows,steps,strict=True):
                assert r["image"]==z["image"]
                tr=folder/"trajectories"/(r["image"]+".pt")
                assert z["trajectory"]==sha(tr)
                h=load(tr)["terminal_hidden"].float()
                px=CONVERT(image(dict(split=r["split"],row=r["row"],image=r["image"],rgb=r["rgb_sha256"]),db))
                unique,index=torch.unique(h,dim=0,return_inverse=True)
                vs=reader(px[None].repeat(len(unique),1,1,1),unique)[index]
                score=vs.double()@fields[r["image"]].double().T
                signed=(score[:,::2]-score[:,1::2]).numpy()
                error=np.max(np.abs(signed[:,:2]-np.array(z["hard"])))
                checked=max(checked,float(error))
                assert error<3e-4 and np.array_equal(signed[:,:2]>0,np.array(z["hard"])>0)
                w2,w3=(top_weights(np.asarray(z[k])) for k in ("r2","r3"))
                measure={}
                for label,block in (("AB",signed[:,:2]),("DE",signed[:,2:4]),("FG",signed[:,4:])):
                    if block.shape[1]:
                        assert block.shape[1]==2
                        v=(block>0).mean(1)
                        measure[label]=[float(w2@v),float(w3@v)]
                grouped[r["image"]].append(measure)
    summaries={}
    for key in ("AB","DE","FG"):
        keep=[r for r in rows if key in grouped[r["image"]][0]]
        if not keep:continue
        pairs=sorted({(r["cohort"], r["pair"]) for r in keep})
        index=[pairs.index((r["cohort"],r["pair"])) for r in keep]
        values=np.array([np.mean([e[key] for e in grouped[r["image"]]],axis=0) for r in keep])
        weights=np.random.default_rng(271127).multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=20000)
        den=np.bincount(index);d=values[:,1]-values[:,0]
        ci=np.quantile((weights@np.bincount(index,d))/(weights@den),[.025,.975])
        summaries[key]=dict(images=len(keep),groups=4*len(keep),pairs=len(pairs),r2=values[:,0].mean(),r3=values[:,1].mean(),delta=d.mean(),interval=ci.tolist())
    import json
    (HERE/"未用于奖励的措辞.json").write_text(json.dumps(dict(summary=summaries,maximum_AB_margin_error=checked),indent=2)+"\n")
    print(summaries)

if __name__=="__main__":
    torch.set_num_threads(3);main()
