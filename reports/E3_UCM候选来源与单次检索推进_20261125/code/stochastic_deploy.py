"""Two independent one-draw deployments; no best-of-N or reward readout at inference."""
import hashlib
import os
from gpu_turn import priority_training,lease
import sys
import tarfile
import time
import torch
from remote_common import ROOT,setup,table,rgb,read,dump,sha,savept

def check():
    if time.time()>=1795593060:raise SystemExit(75)

def rng_for(seed,sub,rgb_sha,draw):
    return int(hashlib.sha256(f'deploy-one1125|{seed}|{sub}|{rgb_sha}|{draw}'.encode()).hexdigest()[:14],16)

@torch.inference_mode()
def generate(seed):
    from eval_io import weights,datasets
    from sample_sources import module
    from prefixes import greedy
    from policy import policy,prompt
    setup();torch.set_num_threads(1)
    path,sig=weights(seed+'_deployed');m,proc=policy(path);tables={k:table(k) for k in ('val','test')}
    for sub,(rows,feedback,field) in datasets().items():
        if sub not in ('primary','official'):continue
        out=ROOT/'records/stochastic'/seed/sub;out.mkdir(exist_ok=True,parents=True)
        for j,c in enumerate(rows):
            check()
            if priority_training():
                print('Releasing GPU after saved images for an admitted training pair.',flush=True)
                os.execv(sys.executable,[sys.executable,'-u',__file__,seed])
            nm=c['image'];q=c[field];answer=feedback[nm];p=out/(nm+'.pt')
            if p.is_file():
                z=torch.load(p,weights_only=True);assert z['adapter']==sig and z['query']==q;continue
            im=rgb(tables[c['split']],c);ins=proc(text=[prompt(proc,q)],images=[im],return_tensors='pt').to('cuda')
            h0,greedy_ids,calls,_=greedy(m,proc,im,q,answer)
            draws=[]
            for i in range(2):
                rng=rng_for(seed,sub,c['rgb_sha256'],i)
                states,paths=module.collect(m,proc,ins,[0],rng,answer)
                assert len(states)==len(paths)==1
                draws.append(dict(state=states[0],ids=paths[0],rng=rng))
            savept(p,dict(image=nm,query=q,rgb=c['rgb_sha256'],adapter=sig,feedback=answer,greedy=greedy_ids,baseline=h0,draws=draws))
            if (j+1)%30==0:print(seed,sub,j+1,len(rows),flush=True)
        dump(out/'complete.json',dict(adapter=sig,files={c['image']:sha(out/(c['image']+'.pt')) for c in rows}))

@torch.inference_mode()
def project():
    from eval_io import datasets
    from readouts import load_model
    setup();torch.set_num_threads(3);m=load_model(False);tables={k:table(k) for k in ('val','test')}
    ds=datasets()
    for seed in 'ab':
        for sub in ('primary','official'):
            rows,answers,field=ds[sub];out=ROOT/'records/stochastic'/seed/sub
            vec=[];meta=[];proof={};sig=None
            for c in rows:
                p=out/(c['image']+'.pt')
                while not p.is_file():check();time.sleep(1)
                z=torch.load(p,weights_only=True);sig=z['adapter'] if sig is None else sig
                assert z['query']==c[field] and z['adapter']==sig and z['rgb']==c['rgb_sha256'] and z['feedback']==answers[c['image']]
                assert [d['rng'] for d in z['draws']]==[rng_for(seed,sub,z['rgb'],i) for i in range(2)]
                pix=m.preprocess(rgb(tables[c['split']],c))[None];local={}
                def embed(h):
                    key=h.numpy().tobytes() if h.dtype!=torch.bfloat16 else h.float().numpy().tobytes()
                    if key not in local:local[key]=m.img_embeds(pix,agent_hidden=h.float()[None])['image_feats'][0].clone()
                    return local[key]
                vec.append(torch.stack([embed(z['baseline'])]+[embed(d['state']) for d in z['draws']]))
                meta.append(dict(image=c['image'],same=[d['ids']==z['greedy'] for d in z['draws']],
                    rounds=[len(d['ids']) for d in z['draws']],tokens=[sum(map(len,d['ids'])) for d in z['draws']]))
                proof[c['image']]=sha(p)
            while not (out/'complete.json').is_file():check();time.sleep(1)
            assert proof==read(out/'complete.json')['files']
            f=out.parent/(sub+'_features.pt');savept(f,dict(visual=torch.stack(vec),meta=meta,adapter=sig))
            a=out.parent/(sub+'.tar.gz')
            with tarfile.open(a.with_suffix('.part'),'w:gz',compresslevel=3) as tf:
                for c in rows:tf.add(out/(c['image']+'.pt'),arcname=c['image']+'.pt')
                tf.add(out/'complete.json',arcname='complete.json')
            a.with_suffix('.part').replace(a);dump(out.parent/(sub+'_read.json'),dict(features=sha(f),archive=sha(a),adapter=sig))
            print(seed,sub,len(rows),flush=True)

if __name__=='__main__':
    check()
    if sys.argv[1]=='project':
        project()
    else:
        while priority_training():check();time.sleep(3)
        with lease():generate(sys.argv[1])
