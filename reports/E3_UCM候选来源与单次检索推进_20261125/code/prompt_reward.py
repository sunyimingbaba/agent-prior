"""Compare the same three reward selectors on each already-generated prompt source."""
import sys
import numpy as np
from attribute_exploration import score_group
from rosters import ROOT,OLD,PRIOR,read,dump
from report_scores import load,bank
from run_stage import boot
def main(tag):
    b=bank(OLD if tag=='first' else PRIOR/'reward_confirm');rows=read(ROOT/'inputs'/(tag+'.json'))
    ref={(r['image'],r['seed']):r for r in read(ROOT/'records'/(tag+'_images.json'))};out=[];totals={}
    for seed in 'ab':
        for c in rows:
            nm=c['image']
            if (nm,seed) not in ref:continue
            f=load(ROOT/'records'/tag/seed/'features'/(nm+'.pt'))['modes'];anchor=f['scene']['visual'][-1].double().numpy()
            row=dict(image=nm,seed=seed,pair=c['pair'])
            for mode in ('scene','question'):
                v=f[mode]['visual'].double().numpy();v[-1]=anchor
                q=score_group(c,b,v,np.asarray(f[mode]['ranks'][:16]))
                row[mode]={k:q['16_'+k] for k in (a+'_'+m for a in ('random','R2','R3') for m in ('joint','mrr','detail'))}
                for m in ('joint','mrr','detail'):assert abs(row[mode]['R3_'+m]-ref[nm,seed][mode][m])<1e-12
            out.append(row)
    labels=[r['pair'] for r in out]
    for mode in ('scene','question'):
        labels0={k:np.array([r[mode][k] for r in out]) for k in out[0][mode]}
        for m in ('joint','mrr','detail'):
            for base in ('random','R2'):labels0['R3_minus_'+base+'_'+m]=labels0['R3_'+m]-labels0[base+'_'+m]
        totals[mode]={k:boot(z,np.ones(len(z)),labels) for k,z in labels0.items()}
    dump(ROOT/'results'/('prompt_reward_'+tag+'.json'),totals);dump(ROOT/'records'/('prompt_reward_'+tag+'_images.json'),out)
if __name__=='__main__':main(sys.argv[1])
