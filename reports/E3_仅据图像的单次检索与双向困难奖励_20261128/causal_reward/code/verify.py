"""Independently reproduce score permutations, tie probabilities, and paired holdout metrics."""
from fractions import Fraction
from collections import defaultdict
from pathlib import Path
import itertools
import numpy as np
import torch
from prepare import ROOT,SOURCE,SIZE,read,sha,dump

ARMS=('r2','strict','prior_contrast')
def reassign(row,effects,held,arm):
    scores=row['r2'][:]
    if arm=='r2':return scores
    top=[k for k,v in enumerate(scores) if v==max(scores)]
    xs=row['items']; ab=[x['hard_margins'] for x in xs]; counts=[sum(a>0 for a in x) for x in ab]
    weak=list(map(min,ab));signal=weak if arm=='strict' else list(map(min,effects))
    rr=[x['other_caption_rr'] for x in xs]; cols=[i for i in range(len(rr[0])) if i!=held]
    n=len(top);lower=[]
    for i in range(8):
        if scores[i]>=max(scores) or n*counts[i]<=sum(counts[k] for k in top) or weak[i]<=max(weak[k] for k in top):
            continue
        if arm=='prior_contrast' and (signal[i]<=0 or signal[i]<=max(signal[k] for k in top)):
            continue
        if all(n*Fraction(rr[i][c])>=sum(Fraction(rr[t][c]) for t in top) for c in cols):
            lower.append(i)
    if not lower or not cols:return scores
    idx=top+lower;prior=sorted(scores[i] for i in idx);j=0
    for strength in sorted({signal[i] for i in idx}):
        same=[i for i in idx if signal[i]==strength]
        v=float(np.mean(prior[j:j+len(same)]));j+=len(same)
        for i in same:scores[i]=v
    return scores

def main():
    results=read(ROOT/'results/先验贡献检验.json');proof={}
    for tag,n in SIZE.items():
        by_image={r['image']:r for r in read(ROOT/'results'/('逐图_'+tag+'.json'))}
        src=SOURCE[tag];texts=torch.load(src/'inputs/vectors.pt',weights_only=True);groups=read(ROOT/'inputs/groups.json')[tag]['pairs']
        max_err=0.;prob_err=0.;groups_checked=0;per=defaultdict(list);positive=[];stable=[];nonzero=[]
        for pair,rows in groups.items():
            raw=torch.load(ROOT/'records'/tag/(pair+'.pt'),weights_only=True)['visual'].double()
            pred=[];rrs=[];test_scores=[]
            for side,row in enumerate(rows):
                name=row['image'];r=by_image[name];saved=read(src/'scores'/(name+'.json')); W=texts[name].double()
                z=raw@W.T;diff=z[:,:,:,0::2]-z[:,:,:,1::2]
                pa=torch.stack([diff[x,side]-diff[x,1-side].mean(dim=0) for x in (side,1-side)])
                c=pa.mean(0)[:,:2].numpy()
                max_err=max(max_err,float(np.abs(c-r['AB_candidate_effect']).max()))
                assert max_err<1e-12
                # Use the archived A/B effect in the second calculation so exact zero / native double ties match the locked reader.
                prior=r['AB_candidate_effect']; probs={a:[] for a in ARMS}; hidden={a:[] for a in ARMS}
                for h,rec in enumerate(r['folds']):
                    for a in ARMS:
                        v=reassign(saved,prior,h,a);np.testing.assert_array_equal(v,rec[a]['rewards'])
                        w=np.flatnonzero(np.asarray(v)==max(v));assert w.tolist()==rec[a]['winners']
                        p=np.zeros(8);p[w]=1/len(w);probs[a].append(p)
                        hidden[a].append(np.mean([saved['items'][k]['other_caption_rr'][h] for k in w]))
                p={a:np.mean(v,axis=0) for a,v in probs.items()}
                assert all(np.array_equal(p[a],r['winners'][a]) for a in ARMS)
                pred.append(p);rrs.append({a:np.mean(v) for a,v in hidden.items()});test_scores.append(z[side,side].numpy())
                groups_checked+=len(r['folds'])*len(ARMS)
                # Descriptive baseline: fraction of uniform real candidates that favor their own side in A/B on both fixed pixels.
                positive.append(float((c>0).all(1).mean()))
                stable.append(float((pa[:,:,:2].numpy()>0).all((0,2)).mean()))
                nonzero.append(float(np.max(np.abs(pa.numpy()))))
            for a in ARMS:
                point=defaultdict(list)
                for i,(row,sel) in enumerate(zip(rows,pred)):
                    reference=by_image[row['image']]['scores'][a];p=sel[a];q=pred[1-i][a]
                    assert abs(reference['MRR']-rrs[i][a])<1e-12;point['MRR'].append(rrs[i][a])
                    l=test_scores[i];other=test_scores[1-i]
                    for base,key in ((4,'DE'),(8,'FG')):
                        ix=[];tx=[]
                        for k in (base,base+2):
                            ix.append(sum(p[j]*float(l[j,k]>l[j,k+1]) for j in range(8)))
                            tx.append(sum(p[j]*q[v]* (1. if l[j,k]>other[v,k+1] else .5 if l[j,k]==other[v,k+1] else 0.) for j,v in itertools.product(range(8),repeat=2)))
                        for direction,value in (('I2T',np.mean(ix)),('T2I',np.mean(tx))):
                            metric=key+'_'+direction
                            prob_err=max(prob_err,abs(value-reference[metric]));point[metric].append(value)
                for metric,v in point.items():per[a,metric].append(np.mean(v))
        assert prob_err<1e-12
        draws=np.random.default_rng(281128).multinomial(n,np.full(n,1/n),size=20000)
        for arm in ('r2','strict'):
            for m in ('MRR','DE_I2T','DE_T2I','FG_I2T','FG_T2I'):
                a=np.asarray(per['prior_contrast',m])-per[arm,m];ref=results[tag]['contrasts'][arm][m]
                np.testing.assert_allclose([a.mean(),*np.quantile(draws@a/n,[.025,.975])],[ref['mean'],*ref['interval']],atol=1e-12,rtol=0)
        proof[tag]=dict(pairs=n,images=2*n,leave_one_caption_arm_folds=groups_checked,maximum_independent_AB_effect_error=max_err,
            maximum_holdout_probability_error=prob_err,all_reallocated_R2_scores_and_selected_slots_verified=True,all_registered_metrics_and_intervals_reproduced=True,
            unselected_uniform_candidates_with_positive_mean_effect_on_both_AB=float(np.mean(positive)),
            unselected_uniform_candidates_with_positive_effect_on_each_pixel_and_both_AB=float(np.mean(stable)),largest_observed_absolute_effect=max(nonzero))
    dump(ROOT/'results/独立复算.json',proof)
if __name__=='__main__':
    torch.set_num_threads(1);main()
