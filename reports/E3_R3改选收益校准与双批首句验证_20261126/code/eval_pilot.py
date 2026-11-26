"""Evaluate one greedy policy output per image for the official primary and gallery panels."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'dual_input_rfpo_20261126/code'))
from model_run import setup,read,dump,sha,savept,table,rgb,policy,reader,greedy,datasets,TRANS

@torch.inference_mode()
def main(seed,arm):
    setup();torch.set_num_threads(2)
    src=ROOT/'models'/(seed+'_'+arm);sig=sha(src/'adapter/adapter_model.safetensors')
    cp=read(src/'complete.json');assert cp['final']==sig and cp['lock']==sha(ROOT/'inputs/pilot/decision.json')
    m,processor=policy(src);native=reader();ds=datasets()
    tables={k:table(k) for k in ('val','test')}
    dest=ROOT/'evaluate'/(seed+'_'+arm)
    for tag in ('official','primary'):
        rows,fb,field=ds[tag];folder=dest/tag;folder.mkdir(exist_ok=True,parents=True);vectors=[]
        for row in rows:
            n=row['image'];path=folder/(n+'.pt');q=row[field]
            if path.exists():
                z=torch.load(path,weights_only=True)
                assert (z['adapter'],z['query'],z['rgb'])==(sig,q,row['rgb_sha256'])
            else:
                image=rgb(tables[row['split']],row)
                h,tokens,calls,_=greedy(m,processor,image,q,fb[n])
                v=native(TRANS(image)[None].cuda(),h.float()[None].cuda())[0].cpu()
                z=dict(image=n,query=q,rgb=row['rgb_sha256'],adapter=sig,actions=tokens,calls=calls,prior=h,v=v)
                savept(path,z)
            vectors.append(z['v'])
        target=dest/(tag+'.pt')
        savept(target,dict(images=[r['image'] for r in rows],features=torch.stack(vectors),adapter=sig))
        dump(folder/'complete.json',dict(features=sha(target),images={r['image']:sha(folder/(r['image']+'.pt')) for r in rows}))
        print(seed,arm,tag,len(rows),flush=True)

if __name__=='__main__':main(*sys.argv[1:])
