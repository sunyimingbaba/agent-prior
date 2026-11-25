"""Add a disjoint, frozen training-image gallery to the ordinary caption reward floor."""
from collections import defaultdict
import sys
import numpy as np
from rosters import ROOT,REP,PRIOR,read,sha,dump
from gallery import source,one_set,choose
from report_scores import load,bank
from gate import K,wins,rr_matrix,rank_value
from run_stage import boot

class TwoWay:
    def __init__(self,seed,cases):
        external=PRIOR/'guard'
        p=REP/'E3-2_UCM有效图组定向短训_20261124/pilot/vectors'
        path=p/(seed+'_start_fresh.pt')
        assert sha(path)==read(p/'index.json')[path.name]
        z=load(path);rows=read(external/'inputs/panel.json');b=bank(external)
        assert z['images']==[r['image'] for r in rows] and len(rows)==122
        names={r['image'] for r in rows};rgb={r['rgb_sha256'] for r in rows}
        for c,other,_,_ in cases:
            assert c['image'] not in names and c.get('rgb',c.get('rgb_sha256')) not in rgb
            assert other['hashes']==b['hashes']
        self.native=b['native']
        self.sim=z['features'].double().numpy()@self.native.T
        self.true_ids=[set(b['maps'][r['image']]['native']) for r in rows]
        self.cache={};self.adapter=z['adapter']
    def text_negatives(self,idx):
        if idx not in self.cache:
            mask=[idx not in other for other in self.true_ids]
            self.cache[idx]=np.sort(self.sim[mask,idx])
            assert len(self.cache[idx])>0
        return self.cache[idx]
    def __call__(self,b,c,visual,ranks,refs,sign):
        ix=b['maps'][c['image']];ns=visual@b['native'].T
        mask=sign.copy();mask[ix['native']]=False;left=ns[:,mask]
        i2t=rr_matrix(ns[:,refs],left);t2i=[]
        for idx in refs:
            neg=self.text_negatives(idx);s=ns[:,idx]
            low=np.searchsorted(neg,s,side='left');high=np.searchsorted(neg,s,side='right')
            t2i.append([rank_value(len(neg)-int(j),int(j-i)) for i,j in zip(low,high)])
        quality=(i2t+np.asarray(t2i,dtype=object).T).sum(1)/4
        margin=ns[:,refs[0]]-left.max(1);r2=wins(margin,ranks)
        gate=quality>=quality[r2].sum()/int(r2.sum())
        h=visual@b['hard'].T;d=h[:,ix['A'][0]]-h[:,ix['A'][1]]
        dense=np.unique(margin+K*d,return_inverse=True)[1]
        return visual[wins(dense+17*gate,ranks)]

def main(tag):
    by_seed=defaultdict(list);letters=None
    for c,s,b,v,q,letters in source(tag):by_seed[s].append((c,b,v,q))
    results={};per=[];ref=read(ROOT/'results'/('gallery_'+tag+'.json'))
    for split,swap in [('primary',False),('reversed',True)]:
        selected=[]
        for s,cases in by_seed.items():
            teacher=TwoWay(s,cases)
            before,n=one_set(cases,swap,choose,letters);after,n2=one_set(cases,swap,teacher,letters);assert n==n2
            assert n==ref[split]['coverage'][s]
            for a,b in zip(before,after):
                assert all(a[k]==b[k] for k in ('image','pair','held','refs'))
                selected.append(dict(seed=s,pair=a['pair'],image=a['image'],old=a,new=b))
        base={};labels=[r['pair'] for r in selected]
        for arm in ('scene','mixed','question'):
            metrics={}
            def measure(r,rule):
                z=np.array(r[rule][arm])
                return {'mR':z.mean(),'I2T':z[:3].mean(),'T2I':z[3:].mean(),'detail':r[rule]['detail'][arm]}
            for key in ('mR','I2T','T2I','detail'):
                old,new=[np.array([measure(r,name)[key] for r in selected]) for name in ('old','new')]
                if key!='detail':assert abs(old.mean()-ref[split]['metrics'][arm][key]['mean'])<1e-12
                metrics[key]={k:boot(v,np.ones(len(v)),labels) for k,v in [('old',old),('new',new),('change',new-old)]}
                metrics[key]['seeds']={s:np.mean((new-old)[[r['seed']==s for r in selected]]) for s in 'ab'}
            base[arm]=metrics
        results[split]=base;per.extend([dict(split=split,**r) for r in selected])
    dump(ROOT/'results'/('two_way_'+tag+'.json'),results)
    dump(ROOT/'records'/('two_way_'+tag+'_images.json'),per)
if __name__=='__main__':main(sys.argv[1])
