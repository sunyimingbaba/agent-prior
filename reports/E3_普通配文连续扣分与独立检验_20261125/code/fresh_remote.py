"""Blind collection and canonical CPU readout; never compute candidate evaluation outcomes here."""
from pathlib import Path
import sys,os,importlib.util,time,hashlib
import torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'source_transfer_2h_20261125/code'))
from remote_common import setup,table,rgb,read,dump,sha,savept
from runtime import ASSETS
from eval_io import weights
from prefixes import greedy
from policy import policy,prompt

def roster(tag):
    p=ROOT/'inputs'/(tag+'.json');assert sha(p)==read(ROOT/'inputs/sampling_lock.json')[tag]
    return read(p)
def folder(tag,seed,kind):
    d=ROOT/'records'/tag/seed/kind;d.mkdir(exist_ok=True,parents=True);return d
@torch.inference_mode()
def feedback():
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    m,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    m=m.float().eval().cuda();cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']));head.load_state_dict(cp['head']);head=head.eval().cuda()
    t=table();out={}
    for tag in ('fresh','extension'):
        for c in roster(tag):
            a,conf=scene_classify_with_conf(rgb(t,c),model=m,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
            out[c['image']]=[a,float(conf)]
    dump(ROOT/'inputs/feedback.json',out)
@torch.inference_mode()
def sample(seed):
    entry=BASE/'attribute_questions_20261125/code/run.py'
    spec=importlib.util.spec_from_file_location('source_sampler',entry);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p,sig=weights(seed+'_deployed');model,proc=policy(p);t=table();feed=read(ROOT/'inputs/feedback.json')
    for tag in ('fresh','extension'):
        d=folder(tag,seed,'raw');rs=roster(tag)
        for j,c in enumerate(rs):
            nm=c['image'];f=d/(nm+'.pt');q=c['scene_query'];ans=feed[nm][0]
            if f.exists():
                rec=torch.load(f,weights_only=True);assert (rec['adapter'],rec['rgb'],rec['query'],rec['feedback'])==(sig,c['rgb_sha256'],q,ans);continue
            im=rgb(t,c);anchor,actions,calls,_=greedy(model,proc,im,q,ans)
            inp=proc(text=[prompt(proc,q)],images=[im],return_tensors='pt').to('cuda')
            hidden=[];paths=[];seeds=[]
            for b in range(2):
                rng=(31000000 if seed=='a' else 41000000)+c['row']*1000+b*100;h,ids=mod.collect(model,proc,inp,list(range(8)),rng,ans)
                hidden+=h;paths+=ids;seeds.append(rng)
            hidden=torch.stack(hidden+[anchor]);assert hidden.shape==(17,3584) and torch.isfinite(hidden).all()
            savept(f,dict(image=nm,rgb=c['rgb_sha256'],adapter=sig,query=q,hidden=hidden,paths=paths,greedy=actions,calls=calls,rngs=seeds,feedback=ans))
            if j%12==0:print(tag,seed,j+1,len(rs),flush=True)
        dump(d/'complete.json',dict(adapter=sig,entry=sha(Path(__file__)),sampler=sha(entry),files={r['image']:sha(d/(r['image']+'.pt')) for r in rs}))
@torch.inference_mode()
def project(shard):
    from readouts import load_model,scorer_for,clip
    t=table();m=load_model(False);proxy=scorer_for(load_model(True),BASE/'relation_candidates_20261120/gallery/train.pt')
    records=ROOT/'records';records.mkdir(exist_ok=True)
    if shard==0:
        import numpy as np
        native=BASE/'relation_candidates_20261120/native';index=read(native/'index.json')
        assert sha(native/'features.pt')==index['sha256'];b=torch.load(native/'features.pt',weights_only=True)
        rows=roster('fresh')+roster('extension');words=sorted({q for r in rows for k in 'ABCDE' for q in r[k]})
        tok=clip.tokenize(words,context_length=77-m.prompt_length);assert len(tok.unique(dim=0))==len(words)
        h=torch.cat([m.txt_embeds(x)['text_feats'] for x in tok.split(12)]);maps={}
        for r in rows:
            ps=[index['columns'][q] for q in r['captions']];assert torch.equal(clip.tokenize(r['captions'],context_length=77-m.prompt_length),b['tokens'][ps])
            maps[r['image']]=dict(native=ps,**{k:[words.index(q) for q in r[k]] for k in 'ABCDE'})
        p=records/'text.pt';savept(p,dict(texts=words,hard=h,tokens=tok,maps=maps,native=b['features'],native_tokens=b['tokens']))
        dump(records/'read_complete.json',dict(text=sha(p),gallery=index['sha256'],reader=sha(ASSETS/'ucm_main_domain_tool.ckpt')))
    for seed in 'ab':
        for tag in ('fresh','extension'):
            rows=roster(tag)[shard::2];src=folder(tag,seed,'raw');dst=folder(tag,seed,'features')
            for j,c in enumerate(rows):
                nm=c['image'];f=src/(nm+'.pt');out=dst/f.name
                while not f.exists():time.sleep(1)
                if out.exists():
                    assert torch.load(out,weights_only=True)['source']==sha(f);continue
                z=torch.load(f,weights_only=True);assert z['rgb']==c['rgb_sha256'] and z['image']==nm
                pix=m.preprocess(rgb(t,c))[None];cache={};vs=[];rs=[]
                for h in z['hidden'].float():
                    key=h.numpy().tobytes()
                    if key not in cache:
                        v=m.img_embeds(pix,agent_hidden=h[None])['image_feats'][0]
                        assert torch.isfinite(v).all() and abs(v.norm().item()-1)<1e-4
                        cache[key]=v,int(proxy.score(h.tolist(),c['row'])[0])
                    v,r=cache[key];vs.append(v);rs.append(r)
                savept(out,dict(rgb=z['rgb'],source=sha(f),adapter=z['adapter'],visual=torch.stack(vs),ranks=rs))
                if j%12==0:print(tag,seed,shard,j+1,len(rows),flush=True)
            dump(dst/('complete_'+str(shard)+'.json'),dict(files={r['image']:sha(dst/(r['image']+'.pt')) for r in rows}))
if __name__=='__main__':
    setup();torch.set_num_threads(2 if sys.argv[1]=='project' else 1)
    if sys.argv[1]=='feedback':feedback()
    elif sys.argv[1]=='sample':sample(sys.argv[2])
    else:project(int(sys.argv[2]))
