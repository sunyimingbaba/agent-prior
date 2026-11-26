import sys
import numpy as np
import torch
from local import ROOT,read,dump,load,sha,banks,bank_for

def main(mode="base"):
    stage=ROOT if mode=="base" else ROOT/"reachable"
    arms=("parent","ordinary","guard","teacher") if mode=="base" else ("parent","ordinary","teacher")
    packs=banks();roster=read(ROOT/'inputs/train.json');tr=read(ROOT/'inputs/teacher_trace.json');ws={(r['seed'],r['image'],r['view']):r['winner'] for r in tr}
    out={};data=[]
    for s in 'ab':
        targets=load(stage/'inputs'/('targets_'+s+'.pt'));rows={r['image']:r for r in roster[s]}
        for arm in arms:
            feature_stage=ROOT if arm=='parent' else stage
            dest=feature_stage/'fitcheck'/(s+'_'+arm+'.pt');proof=read(dest.with_suffix('.json'));assert sha(dest)==proof['features']
            z=load(dest);assert len(z['rows'])==48
            expected=read(ROOT/'inputs/lock.json')['starting'][s] if arm=='parent' else read(stage/'models'/(s+'_'+arm)/'complete.json')['final'];assert z['adapter']==expected
            for rec in z['rows']:
                n,view,v=rec['image'],rec['view'],rec['visual'];r=rows[n];assert rec['rgb']==r['rgb_sha256'] and rec['query']==r['scene_query' if view=='scene' else 'orig_query']
                g=targets[n,view];h=g['hard_dir']@v;f=bank_for(r,packs);ix=f['maps'][n]
                fine={k:float(v.double()@(f['hard'][ix[k][0]].double()-f['hard'][ix[k][1]].double())) for k in 'ABCDE'}
                data.append(dict(seed=s,image=n,arm=arm,view=view,winner=ws[s,n,view],hard_short=float(torch.relu(g['teacher']-h).mean()),
                    ordinary_short=float(torch.relu(g['ordinary_target']-g['ordinary_dir']@v).mean()),
                    trust=float((v-g['base']).square().sum()),AB=np.mean([fine[k]>0 for k in 'AB']),CDE=np.mean([fine[k]>0 for k in 'CDE'])))
    for g in ('all','qualified','fallback'):
        out[g]={}
        for a in arms:
            xs=[r for r in data if r['arm']==a and (g=='all' or (r['winner']<16)==(g=='qualified'))]
            out[g][a]=dict(conditions=len(xs),**{k:float(np.mean([r[k] for r in xs])) for k in ('hard_short','ordinary_short','trust','AB','CDE')})
    dump(stage/'results/fit_scores.json',out);dump(stage/'results/fit_images.json',data)
if __name__=='__main__':main(*sys.argv[1:])
