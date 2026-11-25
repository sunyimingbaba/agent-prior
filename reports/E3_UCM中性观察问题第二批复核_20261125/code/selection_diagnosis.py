"""Descriptive selection diagnosis, restricted to pools with a verified joint improvement."""
from collections import Counter
import numpy as np
from analyze import ROOT,read,dump,detail

def main():
    archive={}
    for seed in 'ab':
        for mode in ('scene','question'):
            totals=Counter()
            for info in read(ROOT/'inputs/lock.json')['rows']:
                record=read(ROOT/'records'/seed/(info['image']+'.json'))['groups']
                base=record['scene']['scores'][-1];group=record[mode];sc=group['scores'][:-1]
                good=np.array([detail(s)>detail(base) and s['rr']>base['rr']+1e-12 for s in sc])
                if not good.any():continue
                totals['opportunities']+=1
                totals['good_draws']+=int(good.sum())
                passes=np.array([s['keep']+1e-14>=group['floor'] for s in sc])
                totals['any_good_meets_native_floor']+=int((good&passes).any())
                totals['all_good_below_floor']+=int(not (good&passes).any())
                for reward in ('R2','R3'):
                    values=np.array(group['retrieval'][reward]);top=values==values.max()
                    hits=int((top&good).sum())
                    totals[reward+'_has_good_at_top']+=int(hits>0)
                    totals[reward+'_exclusively_good_at_top']+=int(hits==int(top.sum()))
                    totals[reward+'_all_tied']+=int(top.all())
                    totals[reward+'_expected_selections']+=hits/int(top.sum())
            archive[seed+'_'+mode]=dict(totals)
    dump(ROOT/'results/selection_diagnosis.json',archive)
if __name__=='__main__':main()
