"""Locate lost retrieval-useful candidates at the ordinary-caption gate or the subsequent ranking."""
import gzip,json
import numpy as np
from rosters import ROOT,read,dump
from report_scores import load
from run_stage import boot
def main():
    reports={}
    for tag in ('first','second'):
        by={c['image']:c for c in read(ROOT/'inputs'/(tag+'.json'))};observations=[]
        with gzip.open(ROOT/'records'/(tag+'_folds.jsonl.gz'),'rt') as fh:
            for line in fh:
                row=json.loads(line);nm=row['image']
                g=dict(image=nm,seed=row['seed'],pair=by[nm]['pair'])
                for arm in ('scene','question','mixed_1','mixed_2'):
                    folds=row[arm];z={k:np.mean([f['baseline'][k] for f in folds]) for k in ('hit','blocked_miss','other_miss')}
                    z['opportunity']=np.mean([bool(f['good']) for f in folds])
                    assert abs(z['opportunity']-z['hit']-z['blocked_miss']-z['other_miss'])<1e-12;g[arm]=z
                g['mixed']={k:(g['mixed_1'][k]+g['mixed_2'][k])/2 for k in g['scene']};observations.append(g)
        outcomes={};labels=[r['pair'] for r in observations]
        for arm in ('scene','mixed','question'):
            parts={k:np.array([r[arm][k] for r in observations]) for k in observations[0][arm]}
            outcomes[arm]={k:boot(parts[k],parts['opportunity'],labels) for k in ('hit','blocked_miss','other_miss')}
            misses=parts['blocked_miss']+parts['other_miss']
            outcomes[arm]['gate_share_among_misses']=boot(parts['blocked_miss'],misses,labels)
            outcomes[arm]['remaining_share_among_misses']=boot(parts['other_miss'],misses,labels)
        reports[tag]=outcomes
    dump(ROOT/'results/missed_opportunities.json',reports)
if __name__=='__main__':main()
