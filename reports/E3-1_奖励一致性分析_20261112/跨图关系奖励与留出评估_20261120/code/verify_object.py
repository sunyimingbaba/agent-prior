"""CPU audit of the independently sampled two-object cohort and the raw scores used in the report."""
import csv,hashlib
import numpy as np
import torch
from shared import ROOT,read,dump,sha
from pathlib import Path
def main():
    out=ROOT/'object';info=read(out/'summary.json');assert info['feature_sha']==sha(out/'features.pt')
    cases={r['image']:r for r in read(ROOT/'object_inputs/cases.json')}
    main_names={r['image'] for r in read(ROOT/'inputs/selected.json')}
    assert main_names.isdisjoint(cases)
    keep={r['image'] for r in read(ROOT/'object_inputs/pixel_review.json') if r['keep']}
    for p,h in read(ROOT/'object_inputs/lock.json').items():assert sha(ROOT/'object_inputs'/p)==h,p
    f=torch.load(out/'features.pt',map_location='cpu',weights_only=True)
    sims=f['visual'].double().numpy()@f['text'].double().numpy().T
    with (out/'candidates.csv').open(encoding='utf-8-sig') as fp:rows=list(csv.DictReader(fp))
    assert len(rows)==info['rows']==len(keep)*18 and {r['image'] for r in rows}==keep
    errors=[];events=0
    for i,z in enumerate(rows):
        k=int(z['k']);r=cases[z['image']];mode=z['mode']
        p=out/'samples'/mode/Path(z['image']).stem/('greedy.pt' if k==-1 else f'sample{k}.pt')
        h=torch.load(p,weights_only=True,map_location='cpu').contiguous()
        rec=read(p.with_suffix('.json'))
        assert sha(p)==rec['sha256'] and hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==rec['vector_sha256']
        assert rec['query']==r['scene_query'] and rec['rgb_sha256']==r['rgb_sha256'] and rec['mode']==mode and rec['k']==k and rec['seed']==int(z['seed'])
        assert h.shape==(3584,) and torch.isfinite(h).all()
        for w in 'ABCD':
            pos,neg=[f['texts'].index(q) for q in r[w]]
            m=(sims[i,pos]-sims[i,neg])*(1 if mode=='scene' else -1)
            errors.append(abs(float(z[w])-m))
        events+=len(rec['events'])
    worst=float(max(errors));assert worst<1e-12
    dump(ROOT/'object_verification.json',dict(images=len(keep),trajectories=len(rows),recomputed_direction_margins=len(errors),max_abs_error=worst,tool_events=events))
if __name__=='__main__':main()
