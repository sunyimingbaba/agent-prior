"""CPU float64 recomputation. Ties are integrated exactly and the pair is the resampling unit."""
import sys,itertools,csv
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from shared import ROOT,read,dump,sha
def csvout(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def selected(x):
    return np.flatnonzero(x==x.max())
def estimate(groups):
    x=np.array(groups,float);rng=np.random.default_rng(20261120)
    ci=np.quantile(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1),[.025,.975])
    return dict(mean=float(x.mean()),lo=float(ci[0]),hi=float(ci[1]),pairs=len(x))
def summary(rows,fn):
    by={}
    for r in rows:by.setdefault(r['pair'],[]).append(fn(r))
    assert all(len(v)==2 for v in by.values())
    return estimate([np.mean(v) for v in by.values()])
def main(split):
    out=ROOT/split; link=read(out/'binding.json')
    for p,h in link['files'].items():assert sha(out/p)==h,p
    ix=read(out/'index.json'); data=read(out/'rows.json')
    f=torch.load(out/'features.pt',map_location='cpu',weights_only=True)
    cases={r['image']:r for r in read(ROOT/'inputs/retained.json') if r['split']==split}
    for key in ['text','visual','native','proxy','proxy_gallery']:
        a=f[key];assert torch.isfinite(a).all() and (a.norm(dim=-1)-1).abs().max()<1e-5
    def products(key):return f['visual'].double().numpy()@f[key].double().numpy().T
    rel=products('text');nat=products('native')
    pr=f['proxy'].double().numpy()@f['proxy_gallery'].double().numpy().T
    rank_ties=0
    for i,z in enumerate(data):
        r=cases[z['image']];rot=z['mode']=='rotated'; assert ix['names'][r['row']]==r['image']
        for w in 'ABCD':
            pos,neg=[ix['texts'].index(q) for q in r[w]]
            z[w]=float((rel[i,pos]-rel[i,neg])*(-1 if rot else 1))
        if not rot:
            ranks=pr[z['proxy_i']];rr=int((ranks>ranks[r['row']]).sum())
            if rr!=z['rank0']:
                assert abs(rr-z['rank0'])<=(abs(ranks-ranks[r['row']])<1e-6).sum()-1
                rank_ties+=1
            positives=ix['native_target'][r['image']]
            negative=np.ones(len(nat[i]),bool);negative[positives]=False
            for tag,p in [('RR',positives),('RR_other',[j for j in positives if j!=ix['native_first'][r['image']]])]:
                if p:z[tag]=1/(1+int((nat[i,negative]>nat[i,p].max()).sum()))
    csvout(out/'candidates.csv',data)
    items=[];chosen={}
    for (mode,name),group in itertools.groupby(sorted(data,key=lambda r:(r['mode'],r['image'],r['k'])),key=lambda r:(r['mode'],r['image'])):
        g=list(group);assert [r['k'] for r in g]==list(range(-1,8))
        c=cases[name]
        cols={key:np.array([z[key] for z in g]) for key in ('A','B','C','D','RR','RR_other','old') if key in g[0]}
        rules={'AB':selected((rankdata(cols['A'][1:])+rankdata(cols['B'][1:]))/2)+1,'random':np.arange(1,9),'greedy':np.array([0])}
        if 'old' in cols:rules['old']=selected(cols['old'][1:])+1
        e=dict(mode=mode,image=name,pair=c['pair'],scene=c['scene'],contrast=c['contrast'],side=c['side']);items.append(e)
        for rule,ids in rules.items():
            chosen[(mode,name,rule)]={w:cols[w][ids] for w in 'CD'}
            e[rule+'_k']=','.join(str(id-1) for id in ids)
            for w in 'CD':e[rule+'_'+w]=float((cols[w][ids]>0).mean());e[rule+'_'+w+'_margin']=float(cols[w][ids].mean())
            e[rule+'_mean']=(e[rule+'_C']+e[rule+'_D'])/2
            for metric in ('RR','RR_other'):
                if metric in cols:e[rule+'_'+metric]=float(cols[metric][ids].mean())
    csvout(out/'images.csv',items); totals={};pairouts=[]
    for mode in sorted({r['mode'] for r in items}):
        group=[r for r in items if r['mode']==mode]
        rules=['AB','random','greedy']+(['old'] if mode!='rotated' else [])
        metrics=['C','D','mean']+(['RR'] if mode!='rotated' else [])
        outm={};totals[mode]=outm
        for rule in rules:
            for metric in metrics:outm[rule+'_'+metric]=summary(group,lambda r:r[rule+'_'+metric])
        for rule in ['random']+(['old'] if mode!='rotated' else []):
            for metric in metrics:outm['AB_minus_'+rule+'_'+metric]=summary(group,lambda r:r['AB_'+metric]-r[rule+'_'+metric])
        for pair in sorted({r['pair'] for r in group}):
            xs=sorted([r for r in group if r['pair']==pair],key=lambda r:r['side']);assert len(xs)==2
            entry=dict(mode=mode,pair=pair,scene=xs[0]['scene'])
            for ru in rules:
                for w in 'CD':
                    l,h=[chosen[(mode,z['image'],ru)][w] for z in xs]
                    entry[ru+'_'+w+'_assignment']=float(np.mean(l[:,None]+h[None,:]>0))
                    entry[ru+'_'+w+'_both']=float(np.mean((l[:,None]>0)&(h[None,:]>0)))
            pairouts.append(entry)
        for ru in rules:
            for suff in ('assignment','both'):
                outm[ru+'_'+suff]=estimate([(r[ru+'_C_'+suff]+r[ru+'_D_'+suff])/2 for r in pairouts if r['mode']==mode])
    csvout(out/'pairs.csv',pairouts);dump(out/'summary.json',totals)
    if split=='train':
        scene=totals['scene']
        conditions={key:scene[key]['mean']>0 for key in ['AB_minus_random_mean','AB_minus_old_mean']}
        for key in ['AB_minus_random_C','AB_minus_random_D']:conditions[key]=scene[key]['mean']>=0
        conditions['native_guard']=scene['AB_minus_random_RR']['mean']>=-.02
        dump(out/'decision.json',dict(pass_gate=bool(all(conditions.values())),conditions=conditions,
            summary_sha=sha(out/'summary.json'),protocol_sha=sha(ROOT/'inputs/protocol.json')))
    else:
        inter=[]
        for e in [z for z in items if z['mode']=='rotated']:
            line=dict(pair=e['pair'],image=e['image'])
            for ru in ['AB','random','greedy']:
                for w in 'CD':
                    x=chosen[('scene',e['image'],ru)][w]; y=chosen[('rotated',e['image'],ru)][w]
                    # Marginals are averaged across ties. For the random selector, match the same draw across pixels.
                    if ru=='random':
                        both=(x>0)&(y>0); responsive=x+y>0
                    else:
                        both=(x[:,None]>0)&(y[None,:]>0); responsive=x[:,None]+y[None,:]>0
                    line[ru+'_'+w+'_both']=float(both.mean());line[ru+'_'+w+'_responsive']=float(responsive.mean())
            inter.append(line)
        csvout(out/'rotation.csv',inter)
        dump(out/'rotation.json',{ru+'_'+key:summary(inter,lambda r:(r[ru+'_C_'+key]+r[ru+'_D_'+key])/2)
                                 for ru in ['AB','random','greedy'] for key in ['both','responsive']})
    print(split,len(data),len(cases),'rank rounding differences',rank_ties,flush=True)
if __name__=='__main__':main(sys.argv[1])
