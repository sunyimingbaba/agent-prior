"""Additional 52 photographs, using the original full-trajectory sampler and frozen CPU reader."""
import sys,time,importlib.util
from pathlib import Path
import torch
from fresh_remote import ROOT,BASE,setup,table,rgb,read,dump,sha,savept,weights,policy,prompt,greedy

def roster():
    p=ROOT/'inputs/covered_extra.json';assert sha(p)==read(ROOT/'inputs/covered_lock.json')['files']['extra']
    return read(p)
@torch.inference_mode()
def sample(seed):
    entry=BASE/'attribute_questions_20261125/code/run.py'
    spec=importlib.util.spec_from_file_location('source_sampler',entry);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    path,sig=weights(seed+'_deployed');m,proc=policy(path);t=table()
    d=ROOT/'records/covered/extra'/seed/'raw';d.mkdir(parents=True,exist_ok=True)
    rows=roster()
    for j,c in enumerate(rows):
        n=c['image'];f=d/(n+'.pt');q=c['scene_query'];answer=c['feedback']
        if f.exists():
            z=torch.load(f,weights_only=True);assert z['rgb']==c['rgb_sha256'] and z['adapter']==sig;continue
        im=rgb(t,c);h0,actions,calls,_=greedy(m,proc,im,q,answer)
        inp=proc(text=[prompt(proc,q)],images=[im],return_tensors='pt').to('cuda')
        hs=[];paths=[];rngs=[]
        for k in (0,1):
            rng=(51000000 if seed=='a' else 61000000)+c['row']*1000+k*100
            h,p=mod.collect(m,proc,inp,list(range(8)),rng,answer);hs+=h;paths+=p;rngs.append(rng)
        savept(f,dict(image=n,rgb=c['rgb_sha256'],adapter=sig,query=q,feedback=answer,hidden=torch.stack(hs+[h0]),paths=paths,greedy=actions,calls=calls,rngs=rngs))
        if j%15==0:print(seed,j+1,len(rows),flush=True)
    dump(d/'complete.json',dict(adapter=sig,sampler=sha(entry),entry=sha(Path(__file__)),files={c['image']:sha(d/(c['image']+'.pt')) for c in rows}))
@torch.inference_mode()
def project(shard):
    from readouts import load_model,scorer_for
    rows=roster()[shard::2];t=table();m=load_model(False);proxy=scorer_for(load_model(True),BASE/'relation_candidates_20261120/gallery/train.pt')
    for s in 'ab':
        base=ROOT/'records/covered/extra'/s;d=base/'features';d.mkdir(exist_ok=True,parents=True)
        for j,c in enumerate(rows):
            n=c['image'];f=base/'raw'/(n+'.pt');out=d/f.name
            while not f.exists():time.sleep(1)
            if out.exists():assert torch.load(out,weights_only=True)['source']==sha(f);continue
            z=torch.load(f,weights_only=True);assert z['rgb']==c['rgb_sha256'] and z['query']==c['scene_query']
            pix=m.preprocess(rgb(t,c))[None];cache={};vs=[];rs=[]
            for h in z['hidden'].float():
                k=h.numpy().tobytes()
                if k not in cache:cache[k]=(m.img_embeds(pix,agent_hidden=h[None])['image_feats'][0],int(proxy.score(h.tolist(),c['row'])[0]))
                v,r=cache[k];vs.append(v);rs.append(r)
            savept(out,dict(rgb=z['rgb'],adapter=z['adapter'],source=sha(f),visual=torch.stack(vs),ranks=rs))
            if j%10==0:print(s,shard,j+1,len(rows),flush=True)
        dump(d/('complete_'+str(shard)+'.json'),dict(files={c['image']:sha(d/(c['image']+'.pt')) for c in rows}))
if __name__=='__main__':
    setup();torch.set_num_threads(1 if sys.argv[1]=='sample' else 3)
    if sys.argv[1]=='sample':sample(sys.argv[2])
    else:project(int(sys.argv[2]))
