"""Use each finished RFPO checkpoint once per picture with the original deployment input."""
import sys,time,tarfile,os
import torch
from remote_common import ROOT,BASE,setup,table,rgb,read,dump,sha,savept
from eval_io import datasets
from prefixes import greedy
from policy import policy

def bundle(tag,sub):return ROOT/'records/learned'/tag/sub

@torch.inference_mode()
def generate(tag):
    setup();torch.set_num_threads(1);arm=ROOT/'pilot'/tag
    proof=read(arm/'complete.json');sig=sha(arm/'adapter/adapter_model.safetensors');assert sig==proof['final']
    m,proc=policy(arm);ds=datasets();tables={s:table(s) for s in ('val','test')}
    for sub in ('primary','official'):
        rows,answers,field=ds[sub];out=bundle(tag,sub);out.mkdir(exist_ok=True,parents=True)
        for c in rows:
            if time.time()>=float(os.environ.get('E3_END_EPOCH','inf')):raise SystemExit(75)
            nm=c['image'];p=out/(nm+'.pt')
            if p.exists():
                old=torch.load(p,weights_only=True);assert old['adapter']==sig and old['query']==c[field];continue
            h,actions,calls,_=greedy(m,proc,rgb(tables[c['split']],c),c[field],answers[nm])
            savept(p,dict(image=nm,rgb=c['rgb_sha256'],adapter=sig,query=c[field],prior=h,actions=actions,calls=calls))
        dump(out/'complete.json',dict(adapter=sig,files={c['image']:sha(out/(c['image']+'.pt')) for c in rows}))
        print(tag,sub,len(rows),flush=True)

@torch.inference_mode()
def project(tags):
    from readouts import load_model
    setup();torch.set_num_threads(3);m=load_model(False)
    ds=datasets();tables={s:table(s) for s in ('val','test')}
    for tag in tags:
        for sub in ('primary','official'):
            rows,answers,field=ds[sub];out=bundle(tag,sub)
            f=out.parent/(sub+'_features.pt');ready=out.parent/(sub+'_read.json')
            if ready.exists():
                assert sha(f)==read(ready)['features'];continue
            fs=[];names=[];proof={};sig=None
            for c in rows:
                nm=c['image'];p=out/(nm+'.pt')
                while not p.exists():
                    if time.time()>=float(os.environ.get('E3_END_EPOCH','inf')):raise SystemExit(75)
                    time.sleep(1)
                z=torch.load(p,weights_only=True);assert z['image']==nm and z['rgb']==c['rgb_sha256'] and z['query']==c[field]
                sig=z['adapter'] if sig is None else sig;assert z['adapter']==sig
                ft=m.img_embeds(m.preprocess(rgb(tables[c['split']],c))[None],agent_hidden=z['prior'].float()[None])['image_feats'][0]
                assert abs(ft.norm().item()-1)<1e-4;fs.append(ft);names.append(nm);proof[nm]=sha(p)
            while not (out/'complete.json').exists():time.sleep(1)
            assert read(out/'complete.json')['files']==proof and read(ROOT/'pilot'/tag/'complete.json')['final']==sig
            savept(f,dict(images=names,features=torch.stack(fs),adapter=sig))
            arc=out.parent/(sub+'.tar.gz')
            with tarfile.open(arc.with_suffix('.part'),'w:gz',compresslevel=3) as tf:
                for c in rows:tf.add(out/(c['image']+'.pt'),arcname=c['image']+'.pt')
                tf.add(out/'complete.json',arcname='complete.json')
            arc.with_suffix('.part').replace(arc);dump(ready,dict(features=sha(f),archive=sha(arc),adapter=sig))
            print(tag,sub,len(rows),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='generate':generate(sys.argv[2])
    else:project(sys.argv[2:])
