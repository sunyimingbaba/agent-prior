import sys,time,torch
from env_pair import ROOT,read,dump,sha,savept,rgb,setup
from deploy_pair import datasets
from readouts import load_model
@torch.inference_mode()
def main(tags):
    setup();m=load_model(False);sets=datasets();dest=ROOT/'pilot/vectors';dest.mkdir(exist_ok=True)
    for tag in tags:
        cp=ROOT/'pilot'/tag/'complete.json'
        while not cp.is_file():time.sleep(2)
        for name,(t,rs,ans,qkey) in sets.items():
            folder=ROOT/'pilot/greedy'/tag/name;out=[];sources=[];files=[]
            for c in rs:
                nm=c['image'];p=folder/(nm+'.pt')
                while not p.is_file():time.sleep(1)
                z=torch.load(p,weights_only=True);assert z['image']==nm and z['rgb']==c['rgb_sha256'] and z['query']==c[qkey] and z['adapter']==read(cp)['final']
                v=m.img_embeds(m.preprocess(rgb(t,c))[None],agent_hidden=z['prior'].float()[None])['image_feats'][0]
                assert abs(v.norm().item()-1)<1e-4 and torch.isfinite(v).all()
                out.append(v);sources.append(sha(p));files.append(nm)
            done=folder/'complete.json'
            while not done.is_file():time.sleep(1)
            assert read(done)['files']==dict(zip(files,sources))
            savept(dest/f'{tag}_{name}.pt',dict(images=files,features=torch.stack(out),adapter=read(cp)['final'],sources=sources))
            print(tag,name,len(out),flush=True)
if __name__=='__main__':main(sys.argv[1:])
