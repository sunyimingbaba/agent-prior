"""Apply a training-only scene-to-question map and generate one prior per image."""
import sys
from pathlib import Path
import torch
from remote_common import ROOT,BASE,setup,table,rgb,read,dump,sha,savept
from eval_io import datasets,weights
from prefixes import greedy
from policy import policy

@torch.inference_mode()
def main(seed):
    setup();torch.set_num_threads(1)
    route=read(ROOT/'inputs/route.json')
    assert route['protocol']==sha(ROOT/'单次部署核查口径.md')
    model_path,sig=weights(seed+'_deployed');model,proc=policy(model_path);ds=datasets()
    tables={s:table(s) for s in ('val','test')}
    for sub in ('primary','official'):
        rows,feedback,field=ds[sub];folder=ROOT/'records/route'/seed/sub;folder.mkdir(exist_ok=True,parents=True)
        features={};raw={}
        for i,c in enumerate(rows):
            name=c['image'];f=folder/(name+'.pt');top=feedback[name].split(' (',1)[0]
            entry=route['categories'].get(top);query=c[field]+(' '+entry['question'] if entry else '')
            if f.is_file():
                old=torch.load(f,weights_only=True);assert old['query']==query and old['rgb']==c['rgb_sha256'] and old['adapter']==sig
                continue
            im=rgb(tables[c['split']],c)
            h,actions,calls,_=greedy(model,proc,im,query,feedback[name])
            record=dict(image=name,rgb=c['rgb_sha256'],adapter=sig,top=top,attribute=entry['attribute'] if entry else '',
                query=query,original=c[field],prior=h,rounds=actions,calls=calls,feedback=feedback[name])
            if i==0:
                orig,ids,tools,_=greedy(model,proc,im,c[field],feedback[name])
                record['baseline']=dict(prior=orig,rounds=ids,calls=tools)
            savept(f,record)
            if (i+1)%25==0:print(sub,seed,i+1,len(rows),flush=True)
        dump(folder/'complete.json',dict(adapter=sig,route=sha(ROOT/'inputs/route.json'),files={c['image']:sha(folder/(c['image']+'.pt')) for c in rows}))
if __name__=='__main__':main(sys.argv[1])
