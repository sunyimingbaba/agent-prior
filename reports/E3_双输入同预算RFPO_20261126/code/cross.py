import sys,torch
from model_run import ROOT,setup,locked,read,dump,sha,savept,rgb,reader,policy,greedy,TRANS
from cross_route import dataset
@torch.inference_mode()
def main(seed,arm):
    assert arm in ('dualplain','dualjoint');lock=locked();setup();torch.set_num_threads(2)
    src=ROOT/'models'/(seed+'_'+arm)
    sig=sha(src/'adapter/adapter_model.safetensors');assert sig==read(src/'complete.json')['final']
    model,proc=policy(src);native=reader();tb,rows,fb=dataset()
    dest=ROOT/'cross'/(seed+'_'+arm);dest.mkdir(parents=True,exist_ok=True);vs=[];files={}
    for r in rows:
        n=r['image'];p=dest/(n+'.pt');q=r['orig_query']
        if p.exists():
            z=torch.load(p,weights_only=True);assert (z['rgb'],z['query'],z['adapter'])==(r['rgb_sha256'],q,sig)
        else:
            im=rgb(tb,r);h,act,calls,_=greedy(model,proc,im,q,fb[r['rgb_sha256']][0])
            v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
            z=dict(image=n,query=q,adapter=sig,rgb=r['rgb_sha256'],prior=h,actions=act,calls=calls,v=v);savept(p,z)
        vs.append(z['v']);files[n]=sha(p)
    f=dest/'features.pt';savept(f,dict(images=[r['image'] for r in rows],features=torch.stack(vs),adapter=sig))
    dump(dest/'complete.json',dict(features=sha(f),images=files,protocol=sha(ROOT/'实验口径.md')));print(seed,arm,len(rows),flush=True)
if __name__=='__main__':main(*sys.argv[1:])
