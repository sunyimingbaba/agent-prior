"""Reconstruct the illustrative 374.tif fold from its actual saved ranks and withheld captions."""
import gzip,json
import numpy as np
from rosters import ROOT,PRIOR,read,dump
from report_scores import bank,load
def main():
    c=next(r for r in read(ROOT/'inputs/second.json') if r['image']=='374.tif')
    r=next(json.loads(line) for line in gzip.open(ROOT/'records/second_folds.jsonl.gz','rt') if json.loads(line)['image']==c['image'] and json.loads(line)['seed']=='a')
    b=bank(PRIOR/'reward_confirm');idx=b['maps'][c['image']]['native'];held=261
    eligible=~b['even'].copy();eligible[idx]=False;cols=np.flatnonzero(eligible)
    obj=load(ROOT/'records/second/a/features/374.tif.pt')
    result=dict(image=c['image'],seed='a',attribute=c['attribute'],neutral_question=c['question'],correct_A=c['A'][0],counterfactual_A=c['A'][1],held_caption=c['captions'][idx.index(held)],held=held)
    for k in ('scene','question'):
        f=next(v for v in r[k] if v['held']==held);z=f['baseline'];s=obj['modes'][k]['visual'].double().numpy()@b['native'].T;details=[]
        for i in z['winners']:
            better=int((s[i,cols]>s[i,held]).sum());ties=int((s[i,cols]==s[i,held]).sum())
            details.append(dict(slot=i,rank_interval=[1+better,1+better+ties]))
        result[k]=dict(detail=z['detail'],mrr=z['mrr'],joint=z['hit'],winners=details)
    dump(ROOT/'results/beach_example.json',result)
if __name__=='__main__':main()
