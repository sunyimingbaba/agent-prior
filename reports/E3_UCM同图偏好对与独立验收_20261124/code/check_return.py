import io,tarfile,json,sys,hashlib
from pathlib import Path
import torch
from dev_metrics import ROOT,HIST,read,sha,dump,load
def main(backup,stages):
    counts={};adapters={};sources={}
    for stage in stages:
        idx=read(ROOT/'archives'/(stage+'_index.json'));count=0
        for fname,sig in idx.items():
            p=ROOT/'archives'/fname;assert sha(p)==sig
            with tarfile.open(p) as tf:
                for v in tf:
                    if not v.name.endswith('/complete.json'):continue
                    tag,sub,_=v.name.split('/');seed,arm=tag.split('_',1);proof=json.load(tf.extractfile(v))
                    vec=load((ROOT/'confirm' if stage=='confirm' else ROOT/'pilot/vectors')/(tag+'_'+sub+'.pt'))
                    assert vec['adapter']==proof['adapter'] and set(vec['images'])==set(proof['files'])
                    for nm,h in zip(vec['images'],vec['sources']):
                        raw=tf.extractfile(f'{tag}/{sub}/{nm}.pt').read();assert hashlib.sha256(raw).hexdigest()==proof['files'][nm]==h
                        z=torch.load(io.BytesIO(raw),weights_only=True)
                        assert z['prior'].shape==(3584,) and z['prior'].dtype==torch.bfloat16 and torch.isfinite(z['prior']).all() and z['adapter']==proof['adapter'] and z['image']==nm
                        count+=1
                    if arm!='start' and tag not in adapters:
                        h=sha(backup/tag/'adapter/adapter_model.safetensors');assert h==proof['adapter']==read(ROOT/'pilot'/tag/'complete.json')['final'];adapters[tag]=h
        counts[stage]=count
    dump(ROOT/'results/return_verification.json',dict(traces=counts,adapters=adapters))
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2:])
