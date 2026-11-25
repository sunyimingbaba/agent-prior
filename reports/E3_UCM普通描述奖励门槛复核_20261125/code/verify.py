"""Validate all saved folds and independently rebuild decisions and rank-pair comparisons for one pair per panel."""
import gzip
import json
import sys
from fractions import Fraction
import numpy as np
from scipy.stats import rankdata
from panels import ROOT,OLD,read,sha,dump,question_panels,transfer_panel

def rr(value,neg):
    left=int(np.count_nonzero(neg>value));num=int(np.count_nonzero(neg==value))+1
    return sum((Fraction(1,i+1) for i in range(left,left+num)),Fraction())/num

def winners(order,ranks):
    # Assign proxy slots using the original sequence of tied intervals.
    proxy=sorted(Fraction(1,1+int(x)) for x in ranks)
    slot=rankdata(order,method='min').astype(int)-1
    values=[sum(proxy[i:i+int(np.sum(slot==i))],Fraction())/int(np.sum(slot==i)) for i in slot]
    return [j for j,x in enumerate(values) if x==max(values)]

def redo(row,b,v,rank,letters,sg,k):
    ix=b['maps'][row['image']];p0=ix['native'][0]
    pos=row['positive_ids'];held=row['held'];permitted=[j for j in pos if j!=held]
    alt=permitted[1:];texts=b['native'];hard=b['hard']
    keep=np.ones(len(texts),bool);keep[pos]=False
    s=v@texts.T
    neg=s[:,keep&(b['even'] if sg==0 else ~b['even'])]
    test=s[:,keep&(~b['even'] if sg==0 else b['even'])]
    tr=[[rr(x[c],n) for c in permitted] for x,n in zip(s,neg)]
    target=[rr(x[held],n) for x,n in zip(s,test)]
    m=s[:,p0]-neg.max(1);hs=v@hard.T
    d=np.mean([hs[:,ix[l][0]]>hs[:,ix[l][1]] for l in letters],axis=0)
    good=np.array([t>target[-1] and a>d[-1] for t,a in zip(target[:-1],d[:-1])])
    h=hs[:,ix['A'][0]]-hs[:,ix['A'][1]]
    r2=winners(m[:16],rank);dense=rankdata((m+k*h)[:16],method='dense')
    central=max(alt,key=lambda a:sum(texts[a]@texts[c] for c in permitted if c!=a)/(len(permitted)-1))
    diverse=min(alt,key=lambda a:texts[p0]@texts[a])
    variants={'hash_two':permitted[:2],'first':[p0],'central_two':[p0,central],'diverse_two':[p0,diverse],'mean_all':permitted,'min_all':permitted}
    saved=row['original' if sg==0 else 'swapped']
    for name,refs in variants.items():
        assert saved[name]['refs']==refs
        values=[]
        for real in tr[:16]:
            a=[real[permitted.index(c)] for c in refs]
            values.append(min(a) if name=='min_all' else sum(a,Fraction())/len(a))
        floor=sum((values[c] for c in r2),Fraction())/len(r2)
        mask=np.array([x>=floor for x in values]);ids=winners(dense+17*mask,rank)
        assert saved[name]['winners']==ids and saved[name]['passed']==np.flatnonzero(mask).tolist()
        signs=[]
        for i in range(16):
            for j in range(i):
                if target[i]!=target[j]:
                    signs.append(.5 if values[i]==values[j] else float((values[i]>values[j])==(target[i]>target[j])))
        z=saved[name];assert abs(z['joint']-good[ids].mean())<1e-12 and abs(z['mrr']-np.mean([float(target[i]) for i in ids]))<1e-12
        assert (z['concordance'] is None and not signs) or (bool(signs) and abs(z['concordance']-np.mean(signs))<1e-12)

def main():
    proof={};k=read(OLD/'inputs/continuous_lock.json')['kappa']
    for tag in ('development','repeat','transfer'):
        p=ROOT/'records'/(tag+'_folds.jsonl.gz');s=read(ROOT/'results'/(tag+'.json'));assert sha(p)==s['fold_sha']
        focus={};first=None;n=0;ids=set()
        with gzip.open(p,'rt') as f:
            for line in f:
                z=json.loads(line);n+=1
                if first is None:first=z['pair']
                key=z['image'],z['seed'],z['mode'],z['held'];assert key not in ids;ids.add(key)
                for side in ('original','swapped'):
                    for rule,v in z[side].items():
                        assert z['held'] not in v['refs'] and v['refs'] and v['winners'] and all(t in z['positive_ids'] for t in v['refs'])
                if z['pair']==first:focus[key]=z
        raw=transfer_panel() if tag=='transfer' else question_panels(tag);verified=0
        for c,seed,mode,b,v,rank,letters,_,_ in raw:
            if c['pair']!=first:continue
            for h in set(b['maps'][c['image']]['native'][1:])-{b['maps'][c['image']]['native'][0]}:
                row=focus.pop((c['image'],seed,mode,h))
                for sg in range(2):redo(row,b,v,rank,letters,sg,k);verified+=1
        assert not focus
        proof[tag]=dict(folds_checked=n,independent_fold_directions=verified,history_error=s['historical_max_error'])
    dump(ROOT/'results/audit.json',proof)
if __name__=='__main__':main()
