"""Show two traceable image-level outcomes selected by image identifier and opposing observed effects."""
import sys
from fractions import Fraction
from rosters import ROOT,PRIOR,read,dump
from report_scores import load,bank,image_folds
import numpy as np

def describe(tag,sign):
    scored=read(ROOT/'records'/('controlled_timing_'+tag+'.json'))
    groups={}
    for r in scored:groups.setdefault(r['image'],[]).append(r)
    def diff(rs):return np.mean([r['free']['joint']-r['locked']['joint'] for r in rs])
    candidates=sorted(nm for nm,rs in groups.items() if sign*diff(rs)>1e-10)
    if not candidates:return None
    nm=candidates[0];by_id={r['image']:r for r in read(ROOT/'inputs'/(tag+'.json'))};c=by_id[nm]
    b=bank(PRIOR/('guard' if tag=='third' else 'reward_confirm'));ix=b['maps'][nm]
    by_caption=dict(zip(ix['native'],c['captions']))
    for seed in 'ab':
        folder=ROOT/'records/prefix'/tag/seed/'features';z=load(folder/(nm+'.pt'))['modes'];locked=z['locked']
        free=z['free'] if tag=='third' else load(ROOT/'records/second'/seed/'features'/(nm+'.pt'))['modes']['scene']
        fold={}
        for mode,f in [('locked',locked),('free',free)]:
            v=f['visual'].double().numpy();v[-1]=locked['visual'][-1].double().numpy()
            fs,_=image_folds(c,seed,b,v,np.asarray(f['ranks'][:16]));fold[mode]={r['held']:r for r in fs}
        for j in fold['locked']:
            old=fold['locked'][j];new=fold['free'][j]
            if sign*(new['baseline']['hit']-old['baseline']['hit'])<=1e-10:continue
            ix0=new['baseline']['winners']
            return dict(tag=tag,image=nm,seed=seed,correct_A=c['A'][0],opposite_A=c['A'][1],
                unseen_C=c['C'],unseen_D=c['D'],unseen_E=c['E'],withheld_native=by_caption[j],
                original_greedy=dict(detail=old['anchor_detail'],mrr=old['anchor_mrr']),
                fixed={k:old['baseline'][k] for k in ('detail','mrr','hit','winners')},
                free={k:new['baseline'][k] for k in ('detail','mrr','hit','winners')},
                seed_averaged_joint_change=float(diff(groups[nm])))
    raise AssertionError(nm)
def main():
    out={t:{n:describe(t,s) for n,s in [('increased',1),('decreased',-1)]} for t in ('second','third')}
    dump(ROOT/'results/illustrations.json',out)
if __name__=='__main__':main()
