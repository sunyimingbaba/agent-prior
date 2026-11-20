"""Frozen downstream readout on two named objects. Paired unit here is an original view and its exact rotation."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata
from shared import ROOT,REBUILD,read,dump,sha,setup,table,rgb
from readouts import load_model,RetrievalScorer,clip
from agent.grpo import reward
from analyze_split import estimate,csvout,selected
def main():
    setup();out=ROOT/'object'; t=table('val')
    lock=read(ROOT/'object_inputs/lock.json')
    for file,h in lock.items():assert sha(ROOT/'object_inputs'/file)==h
    verdict={r['image']:r['keep'] for r in read(ROOT/'object_inputs/pixel_review.json')}
    cases=[r for r in read(ROOT/'object_inputs/cases.json') if verdict[r['image']]]
    frozen=load_model(False);proxy=load_model(True);sc=RetrievalScorer.__new__(RetrievalScorer)
    sc.pl=proxy;sc.device='cuda';sc.cache_path=str(REBUILD/'proxy_gallery_20261113/grpo_val_cache_rebuilt_last50.pt')
    sc.image_features,sc.image_indices=sc._load_or_build_cache();assert sc.image_indices==list(range(len(t)))
    texts=sorted({s for r in cases for w in 'ABCD' for s in r[w]})
    tokens=clip.tokenize(texts,context_length=77-frozen.prompt_length,truncate=False)
    assert len({tuple(z) for z in tokens.tolist()})==len(texts)
    candidates=[];features=[];pairs=[]
    with torch.inference_mode():
        text=torch.cat([frozen.txt_embeds(z.cuda())['text_feats'].cpu() for z in tokens.split(32)])
        for r in cases:
            im=rgb(t,r); picks={};results={}
            e=dict(image=r['image'],relation=r['relation'],scene=r['scene'])
            for mode in ['scene','rotated']:
                x=im if mode=='scene' else im.transpose(Image.Transpose.ROTATE_180)
                img=frozen.preprocess(x)[None].cuda()
                g=[]
                for k in range(-1,8):
                    stem=out/'samples'/mode/Path(r['image']).stem/('greedy.pt' if k<0 else f'sample{k}.pt')
                    rec=read(stem.with_suffix('.json'))
                    assert rec['image']==r['image'] and rec['query']==r['scene_query'] and rec['sha256']==sha(stem)
                    h=torch.load(stem,map_location='cpu',weights_only=True)
                    v=frozen.img_embeds(img,agent_hidden=h[None].cuda())['image_feats'].cpu()
                    features.append(v);sims=(v.double()@text.double().T)[0].numpy()
                    z=dict(image=r['image'],mode=mode,k=k,seed=rec['seed'])
                    for w in 'ABCD':
                        p,n=[texts.index(s) for s in r[w]];z[w]=float((sims[p]-sims[n])*(1 if mode=='scene' else -1))
                    if mode=='scene':
                        rank,rr=sc.score(h.float().tolist(),r['row']);tr='\n'.join(q['decoded'] for q in rec['rounds'])
                        z['rank0']=rank;z['old']=reward.total_reward(tr,rank,lam=1.,mode='soft')
                    g.append(z);candidates.append(z)
                g0={w:np.array([z[w] for z in g]) for w in 'ABCD'};results[mode]=g0
                rules={'AB':selected(rankdata(g0['A'][1:])+rankdata(g0['B'][1:]))+1,'random':np.arange(1,9),'greedy':np.array([0])}
                if mode=='scene':rules['old']=selected(np.array([z['old'] for z in g[1:]]))+1
                picks[mode]=rules
                for key,ids in rules.items():
                    e[mode+'_'+key]=float(np.mean([(g0[w][ids]>0).mean() for w in 'CD']))
            for key in ('random','greedy','AB'):
                v=[]
                for w in 'CD':
                    a=results['scene'][w][picks['scene'][key]];b=results['rotated'][w][picks['rotated'][key]]
                    both=(a>0)&(b>0) if key=='random' else (a[:,None]>0)&(b[None,:]>0)
                    v.append(float(both.mean()))
                e['both_'+key]=float(np.mean(v))
            pairs.append(e)
    csvout(out/'candidates.csv',candidates);csvout(out/'images.csv',pairs)
    torch.save(dict(visual=torch.cat(features),text=text,tokens=tokens,texts=texts),out/'features.pt')
    stats={k:estimate([r[k] for r in pairs]) for k in pairs[0] if k not in ('image','scene','relation')}
    for mode in ['scene','rotated','both']:
        for method in ['random']+(['old'] if mode=='scene' else []):
            stats[mode+'_AB_minus_'+method]=estimate([r[mode+'_AB']-r[mode+'_'+method] for r in pairs])
    dump(out/'summary.json',dict(source_photos=len(pairs),rows=len(candidates),stats=stats,lock_sha=sha(ROOT/'object_inputs/lock.json'),feature_sha=sha(out/'features.pt')))
if __name__=='__main__':main()
