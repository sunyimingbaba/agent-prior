from collections import Counter
import numpy as np
from score import ROOT,OLD,read,dump,boot
import csv
def main():
    rows=read(ROOT/'inputs/panel.json');train={c['image']:c for sub in ('','confirm') for c in read(OLD/sub/'inputs/panel.json')}
    ro=read(ROOT/'inputs/rosters.json')
    def stratum(r):return (r['scene_id'],r['attribute'])
    seen=set.intersection(*[{stratum(train[n]) for n in r['selected']} for r in ro.values()])
    groups={n:[r for r in rows if (stratum(r) in seen)==flag] for n,flag in [('covered',True),('remaining',False)]}
    with (ROOT/'results/per_image.csv').open(encoding='utf-8-sig') as f:
        per={(r['tag'],r['image'],r['metric']):float(r['score']) for r in csv.DictReader(f) if r['metric'] in ('C','D','E')}
    out={}
    for ctrl in ('regular_R3','selected_R3','focused_R2','start'):
        left='focused_R3';parts={}
        for name,ss in {'all':rows,**groups}.items():
            delta=[];up=0;down=0;steady=0
            for seed in ('a','b'):
                a=np.array([[per[seed+'_'+left,r['image'],k] for k in 'CDE'] for r in ss])
                b=np.array([[per[seed+'_'+ctrl,r['image'],k] for k in 'CDE'] for r in ss])
                d=a-b;delta.append(d.mean(1));up+=int((d>0).sum());down+=int((d<0).sum());steady+=int((d==0).sum())
            delta=np.array(delta)*100
            parts[name]=dict(n_images=len(ss),gain=up,loss=down,unchanged=steady,seeds=delta.mean(1).tolist(),**boot(delta.mean(0),[r['pair'] for r in ss]))
        out[ctrl]=parts
    dump(ROOT/'results/coverage.json',dict(common=sorted(seen),comparisons=out))
if __name__=='__main__':main()
