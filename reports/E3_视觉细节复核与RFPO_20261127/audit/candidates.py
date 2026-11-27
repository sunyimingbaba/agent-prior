"""Recompute RFPO winners from grounded per-candidate observations on both collections."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
from phrases import ROOT, REP, FOR, POOL, read, load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redistribute(base,ok,top,signal):
    if not np.any(ok):
        return base.copy()
    ids=np.flatnonzero(ok | np.isin(np.arange(8),top))
    ranks=ids[np.argsort(signal[ids],kind='stable')]
    vals=np.sort(base[ids]);out=base.copy()
    for value in set(signal[ranks]):
        positions=signal[ranks]==value
        out[ranks[positions]]=vals[positions].mean()
    return out


def native_and_bonus(items):
    margins=np.array([z['margin'] for z in items])
    proxy=np.array([1/(1+z['rank']) for z in items]); hard=np.array([z['hard_margins'] for z in items])
    base=np.empty(8)
    order=np.argsort(margins,kind='stable')
    for margin in margins:
        slots=margins==margin;base[slots]=np.sort(proxy)[slots[order]].mean()
    weak=np.maximum(hard.min(1),0)
    ranks=np.array([(np.count_nonzero(weak<v)+0.5*(np.count_nonzero(weak==v)-1)) for v in weak])
    bonus=np.zeros(8) if np.ptp(ranks)==0 else ((ranks-ranks.min())/np.ptp(ranks))*(np.ptp(base)/7)
    return base,bonus


def visual_vote(row):
    p=ROOT/'visual/results/visual'/(row['image']+'.json');z=read(p)
    assert z['rgb']==row['rgb_sha256'] and z['side']==row['side']
    sign=1-2*row['side']
    a,b=[o['ab'] for o in z['observations']]
    return sign*(a[0]-a[1])>0 and sign*(b[1]-b[0])>0


def collect(visual=False,external_only=False):
    margins={}
    if visual:
        if external_only:
            margins['d']=read(ROOT/'visual/results/external_c.json')
        else:
            margins=read(ROOT/'visual/results/candidate_margins.json')
        cv=load(ROOT/'visual/inputs/c_vectors.pt')
    pools={'dev':read(FOR/'inputs/test_panel.json'),'external':read(FOR/'followup_pool/inputs/rows.json')}
    archive=POOL/'scoring/results/candidate_scores.jsonl.gz'
    assert digest(archive)==read(POOL/'scoring/results/score_archive.json')['sha256']
    with gzip.open(archive,'rt') as f:saved=[json.loads(line) for line in f]
    external=FOR/'followup_pool'; proofs=read(external/'scoring_complete.json')
    for panel,rows in pools.items():
        if external_only and panel!='external':continue
        for r in rows:
            n=r['image']
            for seed in ('ab' if panel=='dev' else 'd'):
                if seed=='d':
                    sp=external/'scores'/(n+'.json')
                    assert digest(sp)==proofs['files'][n]
                    raw=external/'raw/d'/(n+'.pt')
                else:
                    sp=POOL/'scores'/seed/(n+'.json')
                    raw=POOL/'sampling/raw'/seed/(n+'.pt')
                g=read(sp)
                if seed!='d':
                    assert any(x['seed']==seed and x['image']==n and x['items']==g.get('items') for x in saved)
                assert g['source']==digest(raw) and g['rgb']==r['rgb_sha256'] and g['seed']==seed
                if not g['eligible'] or len(r['alt_ids'])<2:continue
                if visual and seed=='d':
                    pf=external/'readouts'/(n+'.pt')
                    assert digest(pf)==proofs['readouts'][n]
                    projected=load(pf)['visual'].double()
                    scores=projected@cv[n].double().T
                    np.testing.assert_allclose((scores[:,0]-scores[:,1]).numpy(),margins[seed][n],atol=1e-12,rtol=0)
                elif visual:
                    docs=read(ROOT/'visual/results/candidates'/seed/(n+'.json'))
                    assert docs['historical_score']==digest(sp) and docs['source']==g['source']
                    assert docs['native_vectors']==digest(ROOT/'visual/inputs/c_vectors.pt') and docs['max_ABDEFG_error']<1e-10
                    assert margins[seed][n]==docs['margin_true']
                items=g['items'];base,bonus=native_and_bonus(items)
                assert np.array_equal(base,g['r2']) and np.allclose(bonus,g['bonus'],atol=1e-15,rtol=0)
                top=np.flatnonzero(base==base.max());assert top.tolist()==g['r2_top']
                bonus_top=np.flatnonzero((base+bonus)==(base+bonus).max());assert bonus_top.tolist()==g['r3_top']
                hard=np.array([x['hard_margins'] for x in items])
                count=(hard>0).sum(1);weak=hard.min(1)
                rr=np.array([x['other_caption_rr'] for x in items]);assert rr.shape==(8,len(r['alt_ids']))
                guard=np.array([[Fraction.from_float(float(v))*len(top)>=sum(Fraction.from_float(float(u)) for u in rr[top,k])
                                 for k,v in enumerate(row)] for row in rr])
                scores={k:(np.asarray([x[k] for x in items])>0).mean(1) for k in ('DE','FG') if items[0][k] is not None}
                metrics=defaultdict(list)
                for held in range(rr.shape[1]):
                    permitted=guard[:,np.arange(rr.shape[1])!=held].all(1)&(base<base.max())
                    effective=permitted & (count*len(top)>count[top].sum()) & (weak>max(weak[top]))
                    strict=redistribute(base,effective,top,weak)
                    arms={'r2':base,'bonus':np.array(g['r3']),'strict':strict}
                    if visual:
                        cs=np.array(margins[seed][n])
                        assert cs.shape==(8,) and np.isfinite(cs).all()
                        secondary=permitted & (count*len(top)>=count[top].sum()) & (weak>=weak[top].mean()) & (cs>0) & (cs>max(cs[top]))
                        no_guard=redistribute(base,secondary,top,cs) if np.array_equal(base,strict) else strict
                        arms['c']=no_guard;arms['visual']=no_guard if visual_vote(r) else strict
                    for a,v in arms.items():
                        z=np.flatnonzero(v==v.max())
                        for k,values in {**scores,'mrr':rr[:,held]}.items():
                            metrics[a,k].append(float(values[z].mean()))
                        metrics[a,'pick'].append(float(set(z)!=set(top)))
                        metrics[a,'feedback'].append(float(np.any(v!=base)))
                yield panel,seed,r,{k:np.mean(v) for k,v in metrics.items()}


def summarize(rows,visual):
    panels={}
    for name in ('dev','external'):
        chosen=[(s,r,m) for p,s,r,m in rows if p==name]
        if not chosen:continue
        keys=sorted({(r['cohort'],r['pair']) for s,r,m in chosen})
        inds={u:i for i,u in enumerate(keys)}
        rng=np.random.default_rng(271127)
        weights=rng.multinomial(len(keys), np.full(len(keys),1/len(keys)),size=20000)
        arms=['r2','bonus','strict']+(['c','visual'] if visual else [])
        refs=[('strict','r2'),('strict','bonus')]+([('visual','strict'),('c','strict'),('visual','c')] if visual else [])
        groups=defaultdict(list)
        for s,r,m in chosen:groups[r['image']].append((s,r,m))
        out={}
        for a,b in [(a,None) for a in arms]+refs:
            metrics={}
            for k in ['DE','FG','mrr','pick','feedback']:
                t=np.zeros((len(keys),2));ss=defaultdict(list)
                for entries in groups.values():
                    vals=[]
                    for s,r,m in entries:
                        if (a,k) not in m:continue
                        v=m[a,k]-(m[b,k] if b else 0);vals.append(v);ss[s].append(v)
                    if vals:t[inds[r['cohort'],r['pair']]] += [np.mean(vals),1]
                total=t.sum(0)
                if total[1]==0:continue
                z=weights@t; valid=z[:,1]>0
                metrics[k]={'images':int(total[1]),'mean':total[0]/total[1],'ci':np.quantile(z[valid,0]/z[valid,1],[.025,.975]).tolist(),
                            'seed_means':{s:float(np.mean(v)) for s,v in ss.items()}}
            out[a if b is None else a+'-'+b]=metrics
        panels[name]={'images':len(groups),'pairs':len(keys),'counts':len(chosen),'results':out}
    return panels


def main(visual,external_only=False):
    rows=list(collect(visual,external_only));out=summarize(rows,visual)
    assert (external_only or out['dev']['images']==194) and out['external']['images']==116
    dest='视觉附加逐图独立核验' if visual else '严格保护候选独立核验'
    if external_only:dest='外部116图'+dest
    (ROOT/'results'/(dest+'.json')).write_text(json.dumps(out,indent=2)+'\n')
    with gzip.open(ROOT/'results'/(dest+'_逐图.jsonl.gz'),'wt') as f:
        for p,s,r,m in rows:
            row={'panel':p,'seed':s,'image':r['image'],'pair':r['pair'],'metrics':{'_'.join(k):v for k,v in m.items()}}
            f.write(json.dumps(row)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--visual',action='store_true');ap.add_argument('--external',action='store_true');args=ap.parse_args();main(args.visual,args.external)
