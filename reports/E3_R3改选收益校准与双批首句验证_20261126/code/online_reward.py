"""CPU-compatible eight-trajectory winner rule using only training-visible labels."""
import json
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'inputs/pilot'
sys.path.insert(0,str(PILOT))
from pilot_rules import best,old_choices,choose,decide,unit,stats_matrix,gains_features


def rr(value,neg):
    start=int((neg>value).sum())+1
    n=int((neg==value).sum())+1
    return sum(Fraction(1,k) for k in range(start,start+n))/n


class Scorer:
    def __init__(self,seed):
        self.cfg=json.loads((PILOT/'reward.json').read_text())
        self.pairs=json.loads((PILOT/'pair_by_name.json').read_text())
        self.idx=json.loads((PILOT/'positives.json').read_text())
        self.bank=dict(np.load(PILOT/'text.npz'))
        self.proto=dict(np.load(PILOT/(seed+'_prototypes.npz')))
        self.trees=dict(np.load(PILOT/'trees.npz'))
        self.old=np.array(self.cfg['old'])

    def tree_prediction(self,x):
        x=x.astype(np.float32)
        out=np.zeros((len(x),4))
        slots=np.arange(len(x))
        for n in range(self.cfg['tree_count']):
            stem=str(n)+'_'
            node=np.zeros(len(x),dtype=int)
            left,right,feature,thresh=[self.trees[stem+k] for k in ('left','right','feature','threshold')]
            while True:
                alive=left[node]!=-1
                if not alive.any():break
                j=slots[alive];ids=node[j]
                node[j]=np.where(x[j,feature[ids]]<=thresh[ids],left[ids],right[ids])
            out+=self.trees[stem+'v'][node]
        return out/self.cfg['tree_count']

    def run(self,name,v,held=None):
        nat=self.bank['native'].astype(float,copy=False)
        ids=self.idx[name];alts=sorted(set(ids)-{ids[0]})
        if held is not None:alts.pop(held)
        negmask=self.bank['odd'].copy();negmask[ids]=False
        sim=v@nat.T;neg=sim[:,negmask];pos=sim[:,alts]
        exact=np.array([[rr(x,pool) for x in row] for row,pool in zip(pos,neg)],dtype=object)
        avg=exact.mean(1);ref=best(avg,range(8))
        vals=exact.astype(float)
        eight=np.partition(neg,-8,axis=1)[:,-8:];maximum=eight.max(1)
        gaps=pos-maximum[:,None]
        base=np.column_stack((avg.astype(float),vals.min(1),pos.mean(1),pos.min(1),gaps.mean(1),gaps.min(1),pos.mean(1)-eight.mean(1)))
        hard=v@self.bank[name].astype(float).reshape(6,-1).T
        h=hard[:,::2]-hard[:,1::2]
        sortedby=np.argsort(-(nat[alts]@nat[negmask].T),axis=1,kind='stable')[:,:8]
        near=pos-neg[:,sortedby].mean(2)
        first=sim[:,ids[0]]
        first_input=np.column_stack((first,first-maximum,[float(rr(c,q)) for c,q in zip(first,neg)]))
        spread=float(np.mean(nat[alts]@nat[ids[0]]))
        old=[];new=[];old_input=[]
        for a,b in ((0,1),(0,2),(1,2)):
            phrase=h[:,[a,b]]
            x=np.column_stack((phrase[:,0],phrase[:,1],phrase.min(1),phrase[:,0]>0,phrase[:,1]>0,(phrase>0).all(1),base))
            old_input.append(x-x.mean(0))
            old.append(old_input[-1]@self.old)
            x=gains_features(phrase,ref,pos,vals,gaps,first_input,near,spread,base)
            if self.cfg['switch']['family']=='trees':
                new.append(self.tree_prediction(x))
            else:
                m=self.cfg['ridge'];new.append(((x-m['mean'])/m['scale'])@m['coef'])
        old_ab=old[0];old3=np.mean(old,axis=0)
        packet=dict(ref=ref,exact=avg)
        old_top=old_choices(old_ab,packet);triple=old_choices(old3,packet)
        switch=self.cfg['switch']
        chosen=choose(packet,np.mean(new,axis=0),triple,switch['weight'],switch['mode'])
        local=self.cfg['neighbor'];k=local['k']
        textq=unit(nat[[ids[0]]+alts].mean(0))
        base_local=np.column_stack((old_input[0][:,6:],stats_matrix(v@nat[[ids[0]]].T,neg),
            stats_matrix(pos,neg),stats_matrix((v@textq)[:,None],neg),(gaps>0).mean(1),gaps.min(1)))
        if k:
            available=np.flatnonzero(self.proto['pairs']!=self.pairs[name]);texts=self.proto['text'][available]
            ri=self.proto['image'][available]@unit(v.mean(0));rt=texts@textq
            ordering=np.argsort(-(ri/max(np.std(ri),1e-12)+rt/max(np.std(rt),1e-12)),kind='stable')[:k]
            actual=available[ordering];spans=self.proto['spans']
            support=np.concatenate([self.proto['captions'][spans[j]:spans[j+1]] for j in actual])
            center=unit(texts[ordering].mean(0))
            proxy=np.column_stack((stats_matrix(v@support.T,neg),stats_matrix(v@texts[ordering].T,neg),
                                   stats_matrix((v@center)[:,None],neg),(v@unit(textq+center))[:,None]))
            base_local=np.column_stack((base_local,proxy))
        delta=base_local-base_local[ref].mean(0)
        local_features=np.column_stack((delta,delta*(1-avg[ref].astype(float).mean())))
        estimate=local_features@np.array(local['coefficients'][str(k)])
        neighbor=decide(packet,old3,estimate,local['weight'],local['mode'])
        return dict(ordinary=ref,old=old_top,old3=triple,switch=chosen,neighbor=neighbor)
