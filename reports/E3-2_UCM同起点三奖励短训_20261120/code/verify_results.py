"""Recompute all primary endpoints into a separate directory, compare server results, and verify first-state controls."""
import hashlib,tempfile
from pathlib import Path
import numpy as np
import torch
import analysis
from shared import ROOT,read,dump,sha
def main():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        for name in ('inputs','eval'):(d/name).symlink_to(ROOT/name,target_is_directory=True)
        analysis.ROOT=d;analysis.main()
        comparisons={}
        def numeric(x,y):
            if isinstance(x,dict):
                assert x.keys()==y.keys()
                return max((numeric(v,y[k]) for k,v in x.items()),default=0.)
            assert isinstance(x,(float,int)) and isinstance(y,(float,int))
            return abs(x-y)
        for name in ('summary.json','contrasts.json','decision.json','behavior.json'):
            a=d/'results'/name;b=ROOT/'results'/name;maxerr=numeric(read(a),read(b));assert maxerr<1e-12
            comparisons[name]=dict(maxerr=maxerr,local_sha=sha(a),server_sha=sha(b),byte_equal=sha(a)==sha(b))
    data={}
    for arm in analysis.ARMS:
        folder=ROOT/'first_state'/arm;s=read(folder/'summary.json');rows=read(folder/'rows.json')
        assert sha(folder/'vectors.pt')==s['vectors_sha'] and len(rows)==s['native_and_detail_rows']==242
        f=torch.load(folder/'vectors.pt',weights_only=True,map_location='cpu')
        text=torch.load(ROOT/'eval'/arm/'text.pt',weights_only=True,map_location='cpu')
        scores=analysis.fixed(f['visual'][:210],text['native'])[:,text['caption_order']]
        owned=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:]
        mr=np.mean([vals.mean() for sc,gt in [(scores,owned),(scores.T,owned.T)] for vals in analysis.recalls(sc,gt).values()])
        assert abs(s['native']['mR']-mr)<1e-12
        tq=analysis.fixed(f['visual'][210:],text['relation'])
        rel=read(ROOT/'inputs/detail.json');items=[]
        for j,r in enumerate(rel):
            vals=[]
            for w in 'CD':
                p,n=[text['texts'].index(q) for q in r[w]]
                vals.append((tq[2*j,p]>tq[2*j,n]) and (tq[2*j+1,p]<tq[2*j+1,n]))
            items.append(np.mean(vals))
        assert abs(float(np.mean(items))-s['detail_both'])<1e-12
        exact=sum(z['endpoint_matches'] and z['hidden_equal'] for z in rows)
        assert exact==sum(z['endpoint_matches'] for z in rows)
        data[arm]=dict(native_mR=mr,detail_both=s['detail_both'],natural_single_step_exact=exact,
           native_single_step_exact=sum(r['endpoint_matches'] for r in rows[:210]),
           detail_single_step_exact=sum(r['endpoint_matches'] for r in rows[210:]))
    dump(ROOT/'results/local_recompute.json',dict(primary_outputs=comparisons,first_state=data))
if __name__=='__main__':main()
