"""Independent-image checks of fixed pairwise and on-policy models, with paired uncertainty."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
sys.path.insert(0,str(REPORTS/"E3-2_UCM与RSITMD独立复验_20261124/code"))
from metrics import fixed,recalls


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(rows,texts,vs):
    sims=fixed(vs,texts['captions'])
    hard=fixed(vs,texts['details'])
    idx=np.array([r['panel_de_ids'] for r in rows])
    target=hard[np.arange(len(rows))[:,None],idx]
    de=(target[:,::2]>target[:,1::2]).mean(1)
    mrr=[]; n=len(rows);ordered=[]
    for i,r in enumerate(rows):
        pos=r['panel_positive_ids'];ordered.extend(pos);alts=sorted(set(pos)-{pos[0]})
        mask=np.ones(sims.shape[1],dtype=bool);mask[pos]=False;neg=sims[i,mask]
        mrr.append(np.mean([np.mean(1/np.arange(1+(neg>sims[i,c]).sum(),2+(neg>=sims[i,c]).sum())) for c in alts]) if alts else np.nan)
    assert len(ordered)==5*n
    gold=(np.arange(n)[:,None]==np.repeat(np.arange(n),5)[None,:]);z=sims[:,ordered]
    image=np.stack(list(recalls(z,gold).values())).mean(0)
    text=np.stack(list(recalls(z.T,gold.T).values())).mean(0).reshape(n,5).mean(1)
    return np.stack([de,mrr,(image+text)/2,image,text],axis=1)


def main():
    rows=read(ROOT/'inputs/rows.json');manifest=read(ROOT/'inputs/manifest.json')
    assert sha(ROOT/'inputs/rows.json')==manifest['rows'] and sha(ROOT/'inputs/texts.pt')==manifest['texts']
    texts=torch.load(ROOT/'inputs/texts.pt',weights_only=True)
    ids=sorted({r['pair'] for r in rows})
    assert len(ids)==82
    weights=np.random.default_rng(271127).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=20000)
    def stats(values):
        assert values.shape==(2,164,5)
        pooled=np.nanmean(values,axis=0)
        out={}
        for i,key in enumerate(['DE','MRR','mR','I2T','T2I']):
            counts=np.zeros((len(ids),2))
            for j,r in enumerate(rows):
                if np.isfinite(pooled[j,i]):
                    counts[ids.index(r['pair'])]+=[pooled[j,i],1]
            boots=weights@counts
            out[key]=dict(n=int(counts[:,1].sum()),mean=float(np.nanmean(pooled[:,i])),
                     seeds=np.nanmean(values[:,:,i],axis=1).tolist(),ci=np.quantile(boots[:,0]/boots[:,1],[.025,.975]).tolist())
        return out
    levels={};raw={}
    for arm in ('rfpo_r2','rfpo_strict','dpo_r2','dpo_r3'):
        vals=[]
        for seed in 'ab':
            f=ROOT/'records'/f'{seed}_{arm}.pt';meta=read(f.with_suffix('.json'))
            assert sha(f)==meta['sha256'] and meta['inputs']==manifest['rows']
            g=torch.load(f,map_location='cpu',weights_only=True)
            assert g['adapter']==meta['adapter'] and [r['rgb'] for r in g['traces']]==[r['rgb_sha256'] for r in rows]
            vals.append(measures(rows,texts,g['visual']))
        raw[arm]=np.stack(vals);levels[arm]=stats(raw[arm])
    diff={end:stats(raw[end]-raw[start]) for start,end in [('dpo_r2','dpo_r3'),('rfpo_r2','rfpo_strict')]}
    np.savez_compressed(ROOT/'per_image.npz',**raw)
    (ROOT/'summary.json').write_text(json.dumps(dict(levels=levels,contrasts=diff),indent=2)+'\n')
    print({k:{t:v[t] for t in ['DE','MRR','mR']} for k,v in diff.items()})


if __name__=='__main__':main()
