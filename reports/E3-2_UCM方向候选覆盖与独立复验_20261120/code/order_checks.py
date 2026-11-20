"""Count natural forced-choice decisions before option-order averaging and verify exact token semantics."""
import csv
from collections import defaultdict
import numpy as np
from shared import ROOT,read,dump
from cases import cases
from metrics import csvwrite

def main():
    rows=[];agg=[]
    for group in ('train','fresh'):
        for source in cases(group):
            name=source['image']
            original=read(ROOT/'choice'/group/(name+'.json'))
            ids=original['choice_ids']
            for method,rr in [('enabled',original['records']),('disabled',read(ROOT/'choice_backbone'/group/(name+'.json'))['records'])]:
                index={(r['word'],r['view'],r['order']):r for r in rr}
                for w in 'AB':
                    margins=np.empty((2,2));chosen=np.empty((2,2),dtype=bool);mass=[];valid=[]
                    for v in (0,1):
                        for order in (0,1):
                            rec=index[w,v,order]
                            margins[v,order]=(rec['logit_a']-rec['logit_b'])*(1-2*order)
                            chosen[v,order]=rec['top']==ids[order^v]
                            mass.append(rec['valid_mass']);valid.append(rec['top'] in ids)
                    rows.append(dict(group=group,image=name,word=w,method=method,both_averaged=int(margins[0].mean()>0 and margins[1].mean()<0),
                        top_correct_4=int(chosen.all()),top_correct_order0=int(chosen[:,0].all()),top_correct_order1=int(chosen[:,1].all()),
                        stable_views=int((margins.prod(1)>0).sum()),top_correct=int(chosen.sum()),top_letter=sum(valid),mass=float(np.mean(mass))))
        for method in ('enabled','disabled'):
            x=[r for r in rows if r['group']==group and r['method']==method]
            agg.append(dict(group=group,method=method,items=len(x),views=2*len(x),presentations=4*len(x),
                     **{k:sum(r[k] for r in x) for k in ['both_averaged','top_correct_4','top_correct_order0','top_correct_order1','stable_views','top_correct','top_letter']},
                     mean_valid_mass=np.mean([r['mass'] for r in x])))
    csvwrite(ROOT/'results/option_order_by_source.csv',rows)
    csvwrite(ROOT/'results/option_order.csv',agg)
if __name__=='__main__':main()
