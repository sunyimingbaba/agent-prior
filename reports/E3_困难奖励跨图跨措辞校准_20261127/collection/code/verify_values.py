"""Independently reconstruct margins and per-caption ranks from the downloaded projections."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; REPORTS=ROOT.parents[1]
OLD=REPORTS/'E3_回填后双收益训练图筛选_20261127/scoring'
NATIVE=REPORTS/'E3-1_奖励一致性分析_20261112/空间关系候选奖励核验_20261120/native/features.pt'
def read(p):return json.loads(p.read_bytes())
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    torch.set_num_threads(1)
    native=load(NATIVE); text=native['features'].double().numpy()
    even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in native['tokens'].tolist()])
    hard=load(OLD/'inputs/vectors.pt')
    meta={r['image']:r for r in read(OLD/'inputs/rows.json')}
    vals=0;maximum=0;captions=0
    for path in sorted((ROOT/'pool270/scores').glob('*.json')):
        r=read(path); name=r['image']; row=meta[name]
        z=load(ROOT/'pool270/readouts'/(name+'.pt'))['visual'].double().numpy()
        p=ROOT/'pool270/raw'/(name+'.pt')
        assert r['source']==sha(p)
        fit=even.copy();test=~even;fit[row['native_ids']]=test[row['native_ids']]=False
        for v,item in zip(z,r['items'],strict=True):
            sim=text@v
            signed=hard[name].double().numpy()@v
            signed=signed[::2]-signed[1::2]
            measured=np.array(item['hard_margins']+item['DE']+item['FG'])
            delta=max(float(abs(signed-measured).max()), abs(sim[row['native_ids'][0]]-sim[fit].max()-item['margin']))
            maximum=max(maximum,delta);assert delta<1e-12
            for col,rr in zip(row['alt_ids'],item['other_caption_rr'],strict=True):
                a=int((sim[test]>sim[col]).sum()); b=int((sim[test]==sim[col]).sum())
                assert abs(np.mean(1/np.arange(a+1,a+b+2))-rr)<1e-12
                captions+=1
            vals+=1
    assert vals==952
    report=dict(images=238,states=vals,individual_captions=captions,maximum_abs_margin_error=maximum)
    (ROOT/'readout_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__':main()
