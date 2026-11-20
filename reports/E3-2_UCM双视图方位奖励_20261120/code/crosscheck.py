"""Check exact-tie recalls by ranking the best correct score, then verify E1 paired deltas."""
from collections import defaultdict
from math import comb
import csv
import numpy as np
import torch
from shared import ROOT, dump, read
from metrics import csvwrite, ci
from analyze import study_states, E1SETS

def check_direction(scores,owner):
    out=[]
    for one,yes in zip(scores,owner):
        peak=one[yes].max()
        better=int((one>peak).sum())
        ties=int((one==peak).sum())
        good=int(((one==peak)&yes).sum())
        def expected(k):
            if better>=k:return 0.
            take=min(k-better,ties)
            return 1.-comb(ties-good,take)/comb(ties,take)
        out.append([expected(k) for k in (1,5,10)])
    return np.asarray(out)

def main():
    r=ROOT/'results'
    sums=read(r/'states.json')
    text=torch.load(ROOT/'inputs/retrieval_text.pt',weights_only=True,map_location='cpu')
    allobs=list(csv.DictReader((r/'retrieval_by_source.csv').open(encoding='utf-8-sig')))
    states=study_states()
    owner=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:]
    errs=[];checked=0; gram_max=0.
    for name in states:
        for key in ('native','generic'):
            if key not in sums[name]:continue
            d=ROOT/'eval'/name
            vf=torch.load(d/(key+'.pt'),weights_only=True,map_location='cpu')['visual'].double().numpy()
            tf=text['native'].double().numpy()
            # Every product here is an exactly representable float32 x float32 product in float64.
            sc=np.einsum('ik,jk->ij',vf,tf,optimize=False)[:,text['caption_order']]
            targets={r0['image']:r0 for r0 in allobs if r0['state']==name and r0['part']==key}
            rows=read(d/(key+'.json'))['records']
            for prefix,ar in [('I2T',check_direction(sc,owner)),('T2I',check_direction(sc.T,owner.T).reshape(210,5,3).mean(1))]:
                expected=np.array([[float(targets[e['image']][f'{prefix}_R{k}']) for k in (1,5,10)] for e in rows])
                errs.append(float(abs(ar-expected).max()));checked+=expected.size
            gram_max=max(gram_max,abs(vf@tf.T - sc[:,np.array([text['caption_order'].index(i) for i in range(len(tf))])]).max())
    assert max(errs)<1e-12
    e1=list(csv.DictReader((r/'e1_by_source.csv').open(encoding='utf-8-sig')))
    ets=defaultdict(dict)
    for row in e1:
        block=row['image'] if row['group'].startswith('quantity_') else row['pair']
        arr=ets[row['state'],row['part'],row['group']].setdefault(block,[])
        arr.append((row['image'],float(row['correct']),float(row['margin'])))
    result=[]
    for part in ('native','generic'):
        for g in E1SETS:
            comparisons=[]
            for left,right in [('joint','separate'),('joint','single'),('joint','base')]:
                pairs=[(s+'_'+left+'_16',s+'_'+right+'_16' if right!='base' else 'base_00') for s in 'ab']
                for seed,pair in zip('ab',pairs):comparisons.append((seed+' '+left+'-'+right,[pair]))
                comparisons.append(('two-seed mean '+left+'-'+right,pairs))
            for term,arm_pairs in comparisons:
                d=[]
                for a,b in arm_pairs:
                    aa=ets[a,part,g];bb=ets[b,part,g]; keys=sorted(aa)
                    assert keys==sorted(bb)
                    vals=[]
                    for k in keys:
                        v=sorted(aa[k]);w=sorted(bb[k]);assert [x[0] for x in v]==[x[0] for x in w]
                        sa=np.asarray([x[1:] for x in v]);sb=np.asarray([x[1:] for x in w])
                        vals.append([*(sa-sb).mean(0),float(sa[:,0].all())-float(sb[:,0].all())])
                    d.append(vals)
                avg=np.mean(d,axis=0)
                for i,metric in enumerate(('PairAcc','Margin','both_photos')):
                    if metric=='both_photos' and g.startswith('quantity_'):continue
                    result.append(dict(part=part,group=g,contrast=term,metric=metric,**ci(avg[:,i])))
    csvwrite(r/'e1_paired_contrasts.csv',result)
    dump(r/'alternative_scoring.json',dict(checked_query_recall_values=checked,max_recall_difference=max(errs),maximum_gram_rounding_difference=float(gram_max),total_states=len(states),e1_interval_rows=len(result)))
if __name__=='__main__': main()
