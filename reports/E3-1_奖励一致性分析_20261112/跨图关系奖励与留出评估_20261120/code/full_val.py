"""Substitute only prespecified validation rows into the 210-image native archive and evaluate true raw captions."""
import csv,hashlib,io
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from shared import ROOT,ASSETS,BASE,table,read,dump,sha,setup
from readouts import clip,load_model
from analyze_split import csvout

def hits(scores,positives,k):
    # Expected hit for a uniform ordering within an exact cosine tie at K.
    cutoff=np.partition(scores,-k,axis=1)[:,-k:]
    cutoff=cutoff.min(1)[:,None]
    better=scores>cutoff
    tied=scores==cutoff
    sure=(better & positives).any(axis=1)
    nt=tied.sum(axis=1);good=(tied&positives).sum(axis=1)
    slots=k-better.sum(axis=1)
    miss=np.ones(len(scores))
    for i in range(k):
        active=slots>i
        miss[active]*=np.maximum(nt[active]-good[active]-i,0)/(nt[active]-i)
    return np.where(sure,1.,1-miss).mean()

def main():
    setup(); out=ROOT/'val';assert (out/'summary.json').exists()
    t=table('val'); ix=read(out/'index.json'); rows=read(out/'rows.json')
    assert ix['names']==t['path'].to_pylist()
    f=torch.load(out/'features.pt',weights_only=True,map_location='cpu')
    m=load_model(False)
    cap=t['caption'].to_pylist()
    assert len(cap)==210 and all(len(x)==5 for x in cap)
    raw=[s for row in cap for s in row]
    tok=clip.tokenize(raw,context_length=77-m.prompt_length,truncate=False)
    lookup={tuple(q):i for i,q in enumerate(f['native_tokens'].tolist())}
    order=[lookup[tuple(q)] for q in tok.tolist()]
    c=f['native'][order].double().numpy()
    archive=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val'
    feats=[];files={}
    with torch.inference_mode():
        for i in range(len(t)):
            file=archive/f'{i}.pt';h=torch.load(file,map_location='cpu',weights_only=True).reshape(1,-1).cuda()
            files[str(i)]=sha(file)
            img=Image.open(io.BytesIO(t['image'][i].as_py())).convert('RGB')
            feats.append(m.img_embeds(m.preprocess(img)[None].cuda(),agent_hidden=h)['image_feats'].cpu())
    v=torch.cat(feats)
    torch.save(dict(visual=v,raw_to_column=order),out/'archive_features.pt')
    scores=v.double().numpy()@c.T
    allcand=f['visual'].double().numpy()@c.T
    meta={(r['image'],r['mode'],r['k']):i for i,r in enumerate(rows)}
    with (out/'images.csv').open(encoding='utf-8-sig') as fp: groups=[r for r in csv.DictReader(fp) if r['mode']=='scene']
    cases={r['image']:r for r in read(ROOT/'inputs/retained.json')}
    matches=0; hist=0
    for r in groups:
        idx=cases[r['image']]['row']
        p=ROOT/'samples/original'/Path(r['image']).stem/'greedy.json'
        rec=read(p);assert rec['history']['archive_sha']==files[str(idx)]
        if rec['history']['bit_equal']:
            hist+=1
            assert torch.equal(f['visual'][meta[r['image'],'original',-1]],v[idx]);matches+=1
    real=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:]
    def evaluate(s):
        d={f'{metric}_R{k}':float(hits(vals,gt,k)) for metric,vals,gt in [('I2T',s,real),('T2I',s.T,real.T)] for k in (1,5,10)}
        d['mR']=float(np.mean(list(d.values())));return d
    baseline=evaluate(scores); result=[]; summaries={}
    # Same uniform draw per image and repetition for all selectors; each tied winner equally likely.
    uniforms=np.random.default_rng(20261120).random((64,len(groups)))
    for rule in ['random','old','AB','greedy']:
        vals=[]
        options=[[int(s) for s in r[rule+'_k'].split(',')] for r in groups]
        fixed=all(len(x)==1 for x in options)
        for draw,rand in enumerate(uniforms[:1] if fixed else uniforms):
            s=scores.copy()
            for j,r in enumerate(groups):
                k=options[j][int(rand[j]*len(options[j]))]
                s[cases[r['image']]['row']]=allcand[meta[r['image'],'scene',k]]
            data=evaluate(s); vals.append(data)
            result.append(dict(rule=rule,draw=draw,**data))
        summaries[rule]={key:float(np.mean([v[key] for v in vals])) for key in vals[0]}
    csvout(out/'full_caption_draws.csv',result)
    dump(out/'full_caption.json',dict(images=210,captions=len(raw),replaced=len(groups),draws=64,archive=baseline,selectors=summaries,
         historical_equal=hist,archive_visual_equal_verified=matches,archive_path=str(archive),hashes=files))
if __name__=='__main__':main()
