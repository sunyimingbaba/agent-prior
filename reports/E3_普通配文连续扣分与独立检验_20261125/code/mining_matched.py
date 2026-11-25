"""Descriptive permutation check preserving scene, fine attribute, side, sample batch and both model seeds."""
from collections import defaultdict
import numpy as np
from compare import OUT,read,dump

def main():
    roster={r['image']:r for r in read(OUT/'inputs/real_roster.json')}
    obs=read(OUT/'records/mining_images.json');cells=defaultdict(lambda:defaultdict(dict))
    for z in obs:
        r=roster[z['image']];key=(r['tag'],r['scene_id'],r['attribute'])
        cells[key][r['pair']][r['side'],z['to_seed']]=(int(z['marked']),int(z['target']['changed']))
    rng=np.random.default_rng(20261125);null=np.zeros(20000);pred=hit=total=pairs=cells_used=0;other_hits=0.
    for key,groups in sorted(cells.items()):
        if len(groups)<2:continue
        z=np.array([[row[j,s] for j in (0,1) for s in 'ab'] for _,row in sorted(groups.items())]);x,y=z[:,:,0],z[:,:,1];n=len(z)
        pairs+=n;cells_used+=1;total+=y.size;pred+=x.sum();hit+=(x*y).sum()
        other_hits+=(x*(y.sum(0)-y)/(n-1)).sum()
        perm=np.argsort(rng.random((len(null),n)),axis=1)
        null+=np.einsum('dij,ij->d',y[perm],x)
    assert pred>0
    dump(OUT/'results/mining_matched.json',dict(pairs=pairs,cells=cells_used,groups=total,marked=int(pred),same_photo=int(hit)/int(pred),
        matched_other_photo=other_hits/pred,increment=(hit-other_hits)/pred,
        permutation_null_interval=np.quantile(null/pred,[.025,.975]).tolist(),
        descriptive_upper_tail=(1+int((null>=hit).sum()))/(1+len(null)),draws=len(null)))
if __name__=='__main__':main()
