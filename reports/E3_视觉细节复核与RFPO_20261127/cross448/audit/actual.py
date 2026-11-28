"""Verify the image-paired transfer change on the two frozen DPO checkpoints."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch
from check import ROOT,FORM,REP,read,load,sha,pooled,by_image


def summary(delta, den):
    metrics=['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','I2T','T2I','mR']
    n=len(den);print(n,den.sum(),flush=True)
    means=delta.mean(0);r=np.random.default_rng(20261125);boot=[]
    for j in range(40):
        pick=r.integers(n,size=(500,n))
        boot.append(pooled(means[pick].sum(1),np.full(500,n),den[pick].sum(1)))
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    point=pooled(means.sum(0),np.array(n),np.array(den.sum()))
    seeds=[pooled(d.sum(0),np.array(n),np.array(den.sum())) for d in delta]
    return {k:dict(mean=v,ci=ci[:,j].tolist(),seeds=[s[j] for s in seeds]) for j,(k,v) in enumerate(zip(metrics,point))}


def main():
    cases=read(FORM/'inputs/rsitmd.json');bank=load(FORM/'results/rsitmd_text.pt')
    locked=read(ROOT/'inputs/manifest.json')
    assert locked['roster']==sha(FORM/'inputs/rsitmd.json') and locked['gallery_text']==sha(FORM/'results/rsitmd_text.pt')
    trace=Path('/Users/botlv/cup-deploy-assets/e3-source-transfer-20261125/readout')
    readout=read(trace/'proof.json')
    assert sha(trace/'cup_native_trace.pt')==readout['trace']==read(ROOT/'inputs/server_lock.json')['reader']
    assert readout['reader']=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
    assert readout['reader'] in (FORM/'code/readouts.py').read_text()
    training=REP/'E3_双收益回答对的偏好训练_20261127';training_cases=read(training/'inputs/train_pairs.json')['rows']
    assert not {r['rgb_sha256'] for r in cases}&{r['rgb_sha256'] for r in training_cases}
    order=[c['image'] for c in cases];results={};data=defaultdict(dict)
    assets=Path('/Users/botlv/cup-deploy-assets/e3-joint-answer-pairs-20261127/models')
    for mode in ('native','alternatives'):
        for tag,proof in locked['policies'].items():
            d=ROOT/'records'/tag
            bundle=load(d/'features.pt');cp=read(d/'complete.json')
            assert sha(d/'features.pt')==cp['features'] and cp['adapter']==bundle['adapter']==proof['adapter']
            assert cp['lock']==sha(ROOT/'inputs/server_lock.json') and bundle['images']==order and read(ROOT/'inputs/server_lock.json')['lock']==sha(ROOT/'inputs/manifest.json')
            assert sha(assets/tag/'adapter/adapter_model.safetensors')==proof['adapter']
            assert sha(training/'models'/tag/'steps.json')==proof['scores']
            assert sha(training/'models'/tag/'complete.json')==proof['train_proof']
            if mode=='native':
                for j,row in enumerate(cases):
                    pf=d/(row['image']+'.pt');assert sha(pf)==cp['images'][row['image']]
                    x=load(pf);assert x['rgb']==row['rgb_sha256'] and x['query']==row['orig_query']
                    assert torch.equal(bundle['features'][j],x['v']) and x['adapter']==proof['adapter']
            v,n,ids=by_image(bundle['features'],bank,mode);data[mode][tag]=v
        delta=np.stack([data[mode][s+'_r3']-data[mode][s+'_r2'] for s in 'ab'])
        assert (len(ids),n.sum())==((446,1739) if mode=='alternatives' else (448,2240))
        stats=summary(delta,n)
        theirs=read(ROOT/'results/双向检索.json')[mode]['r3_minus_r2']
        for k,entry in stats.items():
            record=theirs[k]
            np.testing.assert_allclose([entry['mean'],*entry['ci'],*entry['seeds']],
                [record['mean'],*record['interval'],record['seeds']['a'],record['seeds']['b']],rtol=0,atol=1e-12)
        results[mode]={'images':len(ids),'captions':int(n.sum()),'score':stats}
    obj={'training_unique_pixels':len({r['rgb_sha256'] for r in training_cases}),'overlap_pixels':0,
         'retrieval_checkpoint':{'source':'ucm_main_domain_tool.ckpt','sha256':readout['reader'],'trace':readout['trace']},
         'delta_r3_r2':results}
    (ROOT/'audit/独立跨库差值与区间.json').write_text(json.dumps(obj,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(1);main()
