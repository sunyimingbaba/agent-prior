"""Count gained/lost questions and evaluate how much a single scene changes the paired direction result."""
import csv
from collections import Counter,defaultdict
import numpy as np
from shared import ROOT,read,dump
from metrics import csvwrite
def main():
    r=ROOT/'results'
    with (r/'detail_by_source.csv').open(encoding='utf-8-sig') as f:
        data={(row['state'],row['image']):row for row in csv.DictReader(f) if row['part']=='detail'}
    meta=read(ROOT/'inputs/detail.json');rows=[];out={}
    for seed in 'ab':
        cnt=Counter();by=defaultdict(list)
        for m in meta:
            a=data[seed+'_joint_16',m['image']];b=data[seed+'_separate_16',m['image']]
            for w in 'CD':
                x,y=int(a['both_'+w]),int(b['both_'+w]);status=('gained' if x>y else 'lost' if y>x else 'both_right' if x else 'both_wrong')
                cnt[status]+=1;by[m['block']].append(x-y)
                rows.append(dict(seed=seed,image=m['image'],scene=m['block'],word=w,joint=x,separate=y,transition=status))
        lod={str(k):float(np.mean([v for group,ds in by.items() if group!=k for v in ds])) for k in by}
        out[seed]=dict(counts=dict(cnt),leave_one_scene_out=lod)
    csvwrite(r/'direction_transitions.csv',rows);dump(r/'scene_sensitivity.json',out)
if __name__=='__main__':main()
