"""Audit the four training runs and evaluate single-pass priors in the frozen reader."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from analyze import ROOT,read,dump,sha,write_csv,boot
from scipy.stats import rankdata

def ranked_hits(sim,positive):
    # Probability that a best positive is among the first k after uniformly breaking exact ties.
    from math import comb
    res=[]
    for row,mask in zip(sim,positive):
        best=row[mask].max();higher=int((row>best).sum());nt=int((row==best).sum());cp=int((row[mask]==best).sum())
        rec=[]
        for k in (1,5,10):
            take=min(k-higher,nt)
            rec.append(0. if take<=0 else 1. if take>nt-cp else 1-comb(nt-cp,take)/comb(nt,take))
        res.append(rec)
    return np.array(res)

def main(inputs):
    pil=ROOT/'pilot';vdir=pil/'vectors';index=read(vdir/'index.json');torch.set_num_threads(4)
    def load(p):return torch.load(p,weights_only=True,map_location='cpu')
    for f,v in index.items():assert sha(vdir/f)==v
    assert len(index)==12
    proof=read(inputs/'lock.json')
    assert sha(inputs/'retrieval_text.pt')==proof['retrieval_text.pt'] and sha(inputs/'e1_extension.json')==proof['e1_extension.json']
    for f,v in read(inputs/'e1_extension.json')['files'].items():assert sha(inputs/'e1'/f)==v
    g=load(inputs/'retrieval_text.pt');vrows=read(inputs/'validation.json');n=len(vrows);assert n==210
    owner=np.arange(n)[:,None]==np.repeat(np.arange(n),5)[None,:]
    e1=load(inputs/'e1/features.pt');ec=read(inputs/'e1/cases.json');words={s:i for i,s in enumerate(e1['texts'])}
    cf=read(ROOT/'confirm/inputs/panel.json');cb=load(ROOT/'confirm/records/text.pt')
    train=read(pil/'train.json');assert not ({r['image'] for r in train}&{r['image'] for r in cf});assert len(train)==32
    checks={};per=[];vals={};clusters={};levels={};ks=read(ROOT/'inputs/continuous_lock.json')['kappa']
    for seed in ('a','b'):
        for arm in ('R2','R3'):
            dest=pil/(seed+'_'+arm);steps=read(dest/'steps.json');cp=read(dest/'complete.json')
            assert cp['train']==sha(pil/'train.json') and cp['initial']==read(ROOT/'records/provenance.json')['adapters'][seed]
            assert [r['image'] for r in train]==[r['image'] for r in steps] and cp['groups']==32
            for r in steps:
                sc=r['scored'];old=np.array([1/(1+z['rank']) for z in sc]);order=np.array([z['native']+(ks*z['A'] if arm=='R3' else 0.) for z in sc])
                ranks=rankdata(order,method='min').astype(int)-1
                reward=np.array([np.sort(old)[j:j+sum(ranks==j)].mean() for j in ranks])+r['format']
                assert np.allclose(reward,r['reward'],atol=1e-12,rtol=0)
                assert np.allclose((reward-reward.mean())/(reward.std(ddof=1)+1e-8),r['advantages'],atol=1e-9)
            checks[seed+'_'+arm]=dict(updates=cp['updated'],max_group_mean_logp_error=max(r['mean_abs_logp_err'] for r in steps),adapter=cp['final'])
        a,b=[read(pil/(seed+'_'+v)/'steps.json')[0] for v in ('R2','R3')]
        assert all(a[k]==b[k] for k in ('image','ids','third','rng','scored','format'))
        for arm in ('start','R2','R3'):
            tag=seed+'_'+arm;vec=load(vdir/(tag+'_val.pt'));conf=load(vdir/(tag+'_confirm.pt'))
            assert vec['adapter']==conf['adapter']
            if arm!='start':assert vec['adapter']==checks[tag]['adapter']
            assert vec['images']==[r['image'] for r in vrows] and conf['images']==[r['image'] for r in cf]
            x=vec['features'].double();xx=conf['features'].double()
            sim=(x@g['native'].double().T)[:,g['caption_order']].numpy()
            i=ranked_hits(sim,owner);t=ranked_hits(sim.T,owner.T).reshape(n,5,3).mean(1)
            mr=np.column_stack([i,t]).mean(1);metrics={'val_mR':mr,'val_I2T':i.mean(1),'val_T2I':t.mean(1)}
            ids={'val_mR':vec['images'],'val_I2T':vec['images'],'val_T2I':vec['images']}
            for k in ids:clusters[k]=ids[k]
            fine=(x@e1['text_features'].double().T).numpy();idv={s:i for i,s in enumerate(vec['images'])}
            for k in ('attribute','relation'):
                rows=[r for r in ec if r['cohort']=='extension' and ('relation' if r['template']=='shadow_direction' else r['dimension'])==k]
                assert len(rows)==36
                label='val_'+k
                metrics[label]=np.array([float(fine[idv[r['image']],words[r['positive']]]>fine[idv[r['image']],words[r['negative']]]) for r in rows])
                clusters[label]=[r['pair'] for r in rows];ids[label]=[r['image'] for r in rows]
            scores=(xx@cb['hard'].double().T).numpy()
            for k in 'ABCD':
                key='confirm_'+k
                ix=[cb['maps'][c['image']][k] for c in cf]
                metrics[key]=np.array([float(z[p]>z[q]) for z,(p,q) in zip(scores,ix)])
                clusters[key]=[r['pair'] for r in cf];ids[key]=conf['images']
            metrics['confirm_BCD']=np.stack([metrics['confirm_'+k] for k in 'BCD']).mean(0)
            clusters['confirm_BCD']=clusters['confirm_B'];ids['confirm_BCD']=conf['images']
            vals[tag]=metrics;levels[tag]={k:float(v.mean()) for k,v in metrics.items()}
            levels[tag]['six_recalls']=list(np.column_stack([i,t]).mean(0))
            for k,vs in metrics.items():
                for im,c,v in zip(ids[k],clusters[k],vs):per.append(dict(tag=tag,endpoint=k,image=im,cluster=c,score=v))
    diffs={}
    for arm,ctrl in [('R3','R2'),('R3','start'),('R2','start')]:
        diff={}
        for k in vals['a_R2']:
            d=np.array([vals[s+'_'+arm][k]-vals[s+'_'+ctrl][k] for s in ('a','b')])
            diff[k]=dict(paired=boot(100*d.mean(0),clusters[k]),seed_points=(100*d.mean(1)).tolist())
        diffs[arm+'-'+ctrl]=diff
    write_csv(pil/'deployment_by_image.csv',per)
    dump(pil/'deployment.json',dict(levels=levels,differences_pp=diffs,training_checks=checks))
if __name__=='__main__':main(Path(sys.argv[1]))
