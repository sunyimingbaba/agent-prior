"""Independent rank calculation using the best positive score, and paired sign checks."""
import math
import numpy as np
import torch
from shared import ROOT,read,dump,sha
OUT=ROOT/'results'
def load(name):return torch.load(OUT/name,weights_only=True,map_location='cpu')
def hits(scores,pos):
    r={k:[] for k in (1,5,10)}
    for row,p in zip(scores,pos):
        best=row[p].max()
        larger=np.count_nonzero(row>best)
        tied=np.count_nonzero(row==best)
        good=np.count_nonzero(row[p]==best)
        for k in r:
            slots=min(k-larger,tied)
            if slots<=0:v=0.
            elif tied-good<slots:v=1.
            else:v=1-math.prod((tied-good-j)/(tied-j) for j in range(slots))
            r[k].append(v)
    return {k:np.mean(v)*100 for k,v in r.items()}
def main():
    result=read(OUT/'paired_results.json')
    err=0.;rank_checks=0;sign_checks=0
    attrs=read(ROOT/'inputs/attributes.json');ab=load('attribute_text.pt');w={s:i for i,s in enumerate(ab['texts'])}
    for dn in ('ucm','rsitmd'):
        cases=read(ROOT/'inputs'/f'{dn}.json')
        n=len(cases);gallery=load(dn+'_text.pt');order=gallery['caption_order']
        ids={r['image']:i for i,r in enumerate(cases)}
        targets=np.arange(n).repeat(5);img_pos=[np.flatnonzero(targets==i) for i in range(n)]
        for arm,expected in result[dn]['all_native']['levels'].items():
            x=load(f'{dn}_{arm}.pt')['features'].double()
            scores=(x@gallery['native'].double().T)[:,order].numpy()
            for name,r in [('I2T',hits(scores,img_pos)),('T2I',hits(scores.T,targets[:,None]))]:
                for k,v in r.items():
                    err=max(err,abs(expected[name+'_R'+str(k)]-v));rank_checks+=1
            if dn=='ucm':
                for cat,rows in [('all',attrs),('old_extension',[r for r in attrs if r['vocabulary']=='old_extension'])]:
                    decisions=[(x[ids[r['image']]]@(ab['features'][w[r['positive']]].double()-ab['features'][w[r['negative']]].double())).item()>0 for r in rows]
                    assert sum(decisions)==result['attributes'][cat]['levels'][arm]['correct']
                    sign_checks+=len(decisions)
    assert err<1e-9
    dump(OUT/'verification.json',dict(six_recall_checks=rank_checks,max_difference_percentage_points=err,
            attribute_checks_including_subset=sign_checks,using='FP64 matrix multiply, best positive rank with tied positives; attributes from direct dot product with text-vector difference.'))
if __name__=='__main__':main()
