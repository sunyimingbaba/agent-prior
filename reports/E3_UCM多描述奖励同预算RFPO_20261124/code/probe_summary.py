from collections import defaultdict
import numpy as np
from report_io import ROOT,read,sha,dump
def main():
    proof=read(ROOT/'inputs/probe_lock.json');held={(r['seed'],r['image']):r for r in read(ROOT/'results/probe_pair_readouts.json')}
    summary={};pooled={}
    for seed in 'ab':
        f={m:read(ROOT/'records'/('probe_'+seed+'_'+m+'.json')) for m in ('R2','R3')}
        assert all(v['cases']==proof['cases'] and v['adapter']==read(ROOT/'pilot'/(seed+'_'+m)/'complete.json')['final'] for m,v in f.items())
        shifts=[];ds=[];rs=[];changes=defaultdict(int)
        for x,y in zip(f['R2']['rows'],f['R3']['rows'],strict=True):
            assert all(x[k]==y[k] for k in ('image','prefix','target','alternative','common_length'))
            shifts.append(y['delta']-x['delta']);h=held[seed,x['image']];ds.append(h['detail_gap']);rs.append(h['rr_gap'])
            for m,v in [('R2',x),('R3',y)]:
                changes[m+'_both_in_top50']+=int(v['target_kept'] and v['other_kept'])
                changes[m+'_target_argmax']+=int(v['argmax']==v['target'])
                changes[m+'_target_preferred']+=int(v['delta']>0)
        z=np.array(shifts);summary[seed]=dict(n=len(z),mean=float(z.mean()),median=float(np.median(z)),higher=int((z>0).sum()),tied=int((z==0).sum()),
            mean_held_detail_gap=np.mean(ds),mean_held_rr_gap=np.mean(rs),**changes)
    dump(ROOT/'results/probe_summary.json',summary)
if __name__=='__main__':main()
