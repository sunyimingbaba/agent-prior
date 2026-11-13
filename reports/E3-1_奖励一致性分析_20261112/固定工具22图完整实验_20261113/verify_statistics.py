"""Independent SciPy rho, pair-count and reported image-bootstrap checks."""
import argparse
import collections
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args();r=a.run
plan=json.loads((r/'plan.json').read_text())
full=json.loads((r/'完整逐图统计与重采样.json').read_text())
scores=json.loads((r/'scoring/scores.json').read_text())
candidates=[x for x in scores if x['role']=='candidate']
metrics={'utility_rr':1e-12,'utility_margin':1e-6,'E1_margin_A':1e-6,'E1_margin_B':1e-6}
defined=undefined=pair_checks=ci_checks=0
for image in full['per_image']:
    group=[x for x in candidates if x['row']==image['row']]
    for subset in ['all','quality_pass']:
        values=group if subset=='all' else [x for x in group if x['quality_usable']]
        for reward in ['r_retrieval','r_total']:
            for metric,epsilon in metrics.items():
                stats=image['subsets'][subset][reward][metric]
                xs=[x[reward] for x in values]
                ys=[x[metric] if metric=='utility_rr' else round(x[metric],6) for x in values]
                if len(set(xs))<2 or len(set(ys))<2:
                    assert stats['spearman'] is None;undefined+=1
                else:
                    actual=float(spearmanr(xs,ys).statistic)
                    assert abs(actual-stats['spearman'])<1e-12;defined+=1
                counts=collections.Counter()
                for u,v in itertools.combinations(values,2):
                    dr=u[reward]-v[reward];du=u[metric]-v[metric]
                    rt=dr==0;ut=abs(du)<=epsilon
                    cat='both_tied' if rt and ut else 'reward_tie_only' if rt else 'utility_tie_only' if ut else 'concordant' if dr*du>0 else 'discordant'
                    counts[cat]+=1;pair_checks+=1
                assert all(counts[k]==n for k,n in stats['pair_counts'].items())
for label,rows in plan['image_groups'].items():
    images=[x for x in full['per_image'] if x['row'] in rows]
    rng=np.random.default_rng(plan['full_analysis']['bootstrap']['seed'])
    draw=rng.integers(0,len(images),size=(10000,len(images)))
    for subset in ['all','quality_pass']:
        for reward in ['r_retrieval','r_total']:
            for metric in metrics:
                group=full['groups'][label]['subsets'][subset][reward][metric]
                stats=[x['subsets'][subset][reward][metric] for x in images]
                for field,output in [('concordance','image_mean_concordance_bootstrap'),('spearman','image_mean_spearman_bootstrap')]:
                    x=np.array([np.nan if v[field] is None else v[field] for v in stats])
                    selected=x[draw];mask=np.isfinite(selected);n=mask.sum(axis=1)
                    resampled=np.sum(np.where(mask,selected,0),axis=1)/np.maximum(n,1)
                    usable=resampled[n>0];declared=group[output]
                    assert declared['defined_resamples']==len(usable)
                    if len(usable):assert np.allclose(np.percentile(usable,[2.5,97.5]),declared['interval95'],atol=1e-12,rtol=0)
                    finite=x[np.isfinite(x)]
                    if len(finite):assert abs(float(finite.sum()/len(finite))-declared['point_mean'])<1e-12
                    else:assert declared['point_mean'] is None
                    ci_checks+=1
with (r/'pairwise_comparisons.csv').open() as stream:primary=list(csv.DictReader(stream))
assert len(primary)==22*28*8
result={'ok':True,'defined_scipy_rho_checks':defined,'undefined_rho_checks':undefined,
    'all_and_quality_subset_pair_checks':pair_checks,'primary_pair_csv_rows':len(primary),
    'bootstrap_interval_recomputations':ci_checks,'bootstrap_resamples':10000}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
