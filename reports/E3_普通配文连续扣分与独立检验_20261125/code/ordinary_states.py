"""Collect training and development single-pass states for the actual ordinary deployment prompt."""
from fresh_remote import ROOT,BASE,table,rgb,read,dump,sha,setup,savept,weights,policy,greedy
import torch
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(1);t=table();src=BASE/'source_transfer_2h_20261125'
    for seed in 'ab':
        p,sig=weights(seed+'_deployed');model,proc=policy(p)
        for tag in ('first','second','third'):
            rs=read(src/'inputs'/(tag+'.json'));folder=ROOT/'records/ordinary'/tag/seed
            folder.mkdir(exist_ok=True,parents=True)
            for j,c in enumerate(rs):
                n=c['image'];out=folder/(n+'.pt')
                if out.exists():
                    z=torch.load(out,weights_only=True);assert z['adapter']==sig and z['rgb']==c['rgb_sha256'] and z['query']==c['orig_query'];continue
                q=c['orig_query'];h,actions,calls,_=greedy(model,proc,rgb(t,c),q,c['feedback'])
                savept(out,dict(image=n,rgb=c['rgb_sha256'],adapter=sig,query=q,prior=h,actions=actions,calls=calls))
                if j%35==0:print(tag,seed,j+1,len(rs),flush=True)
            dump(folder/'complete.json',dict(adapter=sig,files={c['image']:sha(folder/(c['image']+'.pt')) for c in rs}))
        del model,proc;torch.cuda.empty_cache()
if __name__=='__main__':main()
