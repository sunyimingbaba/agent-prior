"""Pool already sampled candidates by training photo to locate which view lacks a positive example."""
import numpy as np
from shared import ROOT, read, dump
from metrics import csvwrite

def main():
    arms=[s+'_'+mode for s in 'ab' for mode in ('separate','joint')]
    groups={arm:read(ROOT/('policy_'+arm)/'steps.json') for arm in arms}
    meta=read(ROOT/'inputs/train.json')
    rows=[]
    for k,item in enumerate(meta):
        up=[];rot=[];both=0
        for arm in arms:
            g=groups[arm][k]
            assert g['image']==item['image']
            weak=np.asarray(g['caption_and_direction_margins'])[:,1]
            up.extend(weak[:4]);rot.extend(weak[4:])
            both+=int(np.sum(np.minimum(weak[:4],weak[4:])>0))
        u=np.asarray(up);v=np.asarray(rot)
        nu=int((u>0).sum());nv=int((v>0).sum())
        status='both_available' if nu and nv else 'upright_only' if nu else 'rotated_only' if nv else 'neither'
        rows.append(dict(image=item['image'],pair=item['pair'],upright_positive=nu,rotated_positive=nv,
                         sampled_per_view=len(u),paired_positive=both,status=status,
                         upright_min=u.min(),upright_max=u.max(),rotated_min=v.min(),rotated_max=v.max()))
    from collections import Counter
    stats=dict(sources=len(rows),availability=dict(Counter(r['status'] for r in rows)),matched_positive_pairs=sum(r['paired_positive'] for r in rows),
               total_pair_slots=4*4*len(rows),rows_with_original_and_rotated_available=sum(r['status']=='both_available' for r in rows),
               positive_upright=sum(r['upright_positive'] for r in rows),positive_rotated=sum(r['rotated_positive'] for r in rows),
               note='Pooled logged slots, with shared first-step samples across matched arms. These are dependent draws, not a claim about all possible continuations.')
    csvwrite(ROOT/'results/candidate_coverage_by_source.csv',rows);dump(ROOT/'results/candidate_coverage.json',stats)
if __name__=='__main__':main()
