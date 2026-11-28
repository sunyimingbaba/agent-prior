"""Frozen A/B-only prior comparison ranker, with separate D/E, F/G and caption holdouts."""
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import sys
import numpy as np
import torch
from prepare import ROOT,PREV,CAL,SOURCE,SIZE,read,dump,sha
spec=spec_from_file_location('old_pair',PREV/'evaluate.py');previous=module_from_spec(spec);spec.loader.exec_module(previous)
from reward import redistribute
ARMS=('r2','strict','prior_contrast')
def probabilities(row):
    base=row['r2'];m=row['ab'];rr=row['rr']
    top=base==base.max();weak=m.min(1);count=(m>0).sum(1)
    cs=row['prior_effect'][:,:2].min(1)
    valid=(count*top.sum()>count[top].sum())&(weak>weak[top].max())&(cs>0.)
    signal=np.where(top|valid,cs,cs.min())
    votes={a:[] for a in ARMS};hidden={a:[] for a in ARMS};records=[]
    for col in range(rr.shape[1]):
        vis=[i for i in range(rr.shape[1]) if i!=col]
        ranks=dict(r2=base,strict=previous.guarded(base,m,rr,vis),prior_contrast=redistribute(base,signal,rr,vis))
        fold={}
        for a,b in ranks.items():
            pick=b==b.max();probs=pick/pick.sum();votes[a].append(probs);hidden[a].append(float(probs@rr[:,col]))
            fold[a]=dict(winners=np.flatnonzero(pick).tolist(),rewards=b.tolist())
        records.append(fold)
    return {a:np.mean(votes[a],axis=0) for a in ARMS},{a:np.mean(hidden[a]) for a in ARMS},records

def score(tag):
    lock=read(ROOT/'inputs/lock.json');assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    assert sha(PREV/'evaluate.py')==lock['paired_source'] and sha(CAL/'code/reward.py')==lock['redistribute']
    batch=read(ROOT/'inputs/groups.json')[tag];old=SOURCE[tag]
    assert all(sha(old/k)==v for k,v in batch['proofs'].items())
    original=read(old/'scoring_complete.json');words=torch.load(old/'inputs/vectors.pt',map_location='cpu',weights_only=True)
    folder=ROOT/'records'/tag;proof=read(folder/'complete.json')
    assert proof['device']=='cpu' and proof['threads']==4 and proof['lock']==sha(ROOT/'inputs/lock.json')
    assert proof['anchors_equal']==proof['new_counterfactual_slots']==2*8*SIZE[tag]
    oldrows={r['image']:r for r in read(PREV/('逐图_'+tag+'.json'))}
    def load(p):return torch.load(p,map_location='cpu',weights_only=True)
    records=[];levels={a:[] for a in ARMS}
    for pair,rs in batch['pairs'].items():
        path=folder/(pair+'.pt');assert sha(path)==proof['files'][pair];data=load(path)
        assert data['images']==[r['image'] for r in rs] and data['rgb']==[r['rgb_sha256'] for r in rs]
        visual=data['visual'];assert visual.shape==(2,2,8,768)
        sides=[]
        for i,r in enumerate(rs):
            n=r['image'];p=old/'scores'/(n+'.json');f=old/'readouts'/(n+'.pt')
            assert sha(p)==original['files'][n] and sha(f)==data['native_sources'][i]==original['readouts'][n]
            fact=read(p);assert torch.equal(visual[i,i],load(f)['visual'])
            w=words[n].double();assert w.shape==(12,768)
            ow=words[rs[1-i]['image']].double();assert torch.equal(w[::2],ow[1::2]) and torch.equal(w[1::2],ow[::2])
            sim=torch.stack([w@h.double() for h in visual.reshape(-1,768)]).reshape(2,2,8,12).numpy()
            marg=sim[:,:,:,::2]-sim[:,:,:,1::2]
            fixed_pixel=marg[:,i]-marg[:,1-i].mean(1)[:,None,:]
            native=marg[i,i];stored=np.array([z['hard_margins']+z['DE']+z['FG'] for z in fact['items']])
            np.testing.assert_allclose(native,stored,atol=1e-12,rtol=0)
            g=dict(row=r,text_scores=sim[i,i],ab=stored[:,:2],r2=np.array(fact['r2']),rr=np.array([z['other_caption_rr'] for z in fact['items']]),
                   prior_effect=fixed_pixel.mean(0),pixels=fixed_pixel)
            assert g['rr'].shape[0]==8 and g['rr'].shape[1]>=2
            sides.append(g)
        selected=[];held=[];folds=[]
        for s in sides:
            x,y,z=probabilities(s);selected.append(x);held.append(y);folds.append(z)
        for i,g in enumerate(sides):
            scores={}
            for a in ARMS:
                p=selected[i][a]
                scores[a]=previous.row_score(g,sides[1-i],p,selected[1-i][a],held[i][a])
                for label,sl in (('AB',slice(0,2)),('DE',slice(2,4)),('FG',slice(4,6))):
                    scores[a][label+'_prior_margin']=float(p@g['prior_effect'][:,sl].mean(1))
                    scores[a][label+'_prior_positive']=float(p@(g['prior_effect'][:,sl]>0).mean(1))
            prev=oldrows[g['row']['image']]
            for a in ('r2','strict'):
                np.testing.assert_array_equal(selected[i][a],prev['winners'][a])
                for k,b in prev['scores'][a].items():np.testing.assert_allclose(scores[a][k],b,atol=1e-14,rtol=0)
            records.append(dict(image=g['row']['image'],rgb=g['row']['rgb_sha256'],pair=pair,side=i,
                 winners={a:x.tolist() for a,x in selected[i].items()},folds=folds[i],scores=scores,
                 AB_candidate_effect=g['prior_effect'][:,:2].tolist(),AB_candidate_effect_own_pixels=g['pixels'][i,:,:2].tolist(),
                 AB_candidate_effect_partner_pixels=g['pixels'][1-i,:,:2].tolist()))
            for a in ARMS:levels[a].append(scores[a])
    count=SIZE[tag];k=list(levels['r2'][0])
    vals={a:np.array([[s[t] for t in k] for s in z]).reshape(count,2,-1).mean(1) for a,z in levels.items()}
    weights=np.random.default_rng(281128).multinomial(count,np.full(count,1/count),size=20000)
    def est(v):
        ci=np.quantile(weights@v/count,[.025,.975],axis=0)
        return {b:dict(mean=float(v[:,j].mean()),interval=ci[:,j].tolist()) for j,b in enumerate(k)}
    return records,dict(images=len(records),pairs=count,heldout_metrics=k,
        levels={a:est(v) for a,v in vals.items()},contrasts={m:est(vals['prior_contrast']-vals[m]) for m in ('r2','strict')},
        changes_vs_strict=sum(r['winners']['strict']!=r['winners']['prior_contrast'] for r in records),
        changes_vs_r2=sum(r['winners']['r2']!=r['winners']['prior_contrast'] for r in records))

def main():
    results={}
    for tag in ('98','114'):
        x,r=score(tag);dump(ROOT/'results'/('逐图_'+tag+'.json'),x);results[tag]=r
    dev,con=[results[t]['contrasts']['strict'] for t in ('98','114')]
    gd=dict(dev_DE_positive=dev['DE_I2T']['mean']>0,confirm_DE_lower_positive=con['DE_I2T']['interval'][0]>0,
         both_MRR_and_T2I_preserved=all(v[k]['mean']>=0 for v in (dev,con) for k in ('MRR','DE_T2I')),
         both_better_detail_and_preserved_MRR_than_R2=all(results[t]['contrasts']['r2']['DE_I2T']['mean']>0 and results[t]['contrasts']['r2']['MRR']['mean']>=0 for t in ('98','114')))
    results['gate']=gd;results['advance']=all(gd.values());dump(ROOT/'results/先验贡献检验.json',results)
    print(results['gate'],flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);main()
