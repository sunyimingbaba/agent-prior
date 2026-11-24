import numpy as np
from pathlib import Path
from dev_metrics import ROOT,HIST,read,sha,dump
def main():
    old=read(HIST/'records/runtime.json')['init'];out={};base_rows=read(ROOT/'inputs/candidates.json')
    for seed in 'ab':
        rec={r['image']:r for r in base_rows[seed]}
        for arm in ['hard','native']:
            tag=seed+'_'+arm;f=ROOT/'pilot'/tag;ss=read(f/'steps.json');cp=read(f/'complete.json')
            assert cp['initial']==old[seed] and cp['steps']==len(ss)==32 and cp['ref']==sha(ROOT/'records'/(seed+'_observed_ref.pt'))
            for a in ss:
                assert a['image'] in rec and a['norm']>0
                expect=float(np.logaddexp(0,-.2*a['sign']*(a['p']-a['q']-(a['p0']-a['q0']))))
                assert abs(expect-a['loss'])<2e-5
            out[tag]=dict(steps=len(ss),images=len({a['image'] for a in ss}),positive_gradient_steps=sum(a['norm']>0 for a in ss))
        hs,ns=[read(ROOT/'pilot'/(seed+'_'+a)/'steps.json') for a in ('hard','native')]
        assert [(a['image'],a['p0'],a['q0']) for a in hs]==[(a['image'],a['p0'],a['q0']) for a in ns]
        out[seed+'_observed_disagreements']=sum(a['sign']!=b['sign'] for a,b in zip(hs,ns))
        for arm in ['joint','continuous_native']:
            tag=seed+'_'+arm;f=ROOT/'pilot'/tag;ss=read(f/'steps.json');cp=read(f/'complete.json');assert cp['initial']==old[seed] and len(ss)==cp['steps']==84
            seen=set();cos=[];k=cp['kappa']
            for s in ss:
                assert len(s['records'])==4 and len({r['image'] for r in s['records']})==2 and s['norm']>0
                for r in s['records']:
                    key=r['image'],r['field'];assert key not in seen;seen.add(key);cos.append(r['cos'])
                    assert r['field'] in ('scene_query','orig_query') and np.isclose(r['hard_weight'],2*k if arm=='joint' and r['field']=='scene_query' else 0)
                    assert np.isclose(r['hard'],np.logaddexp(0,-50*np.array(r['AB'])).mean(),atol=2e-6)
                    assert np.isfinite(r['gradnorm']) and r['cos']>.98
            out[tag]=dict(pairs=len(ss),greedy_training_trajectories=len(seen),min_replay_cos=min(cos),mean_replay_cos=np.mean(cos),actual_updates=sum(a['norm']>0 for a in ss))
        j,n=[read(ROOT/'pilot'/(seed+'_'+a)/'steps.json') for a in ('joint','continuous_native')]
        assert [r['pair'] for r in j]==[r['pair'] for r in n]
        for a,b in zip(j[0]['records'],n[0]['records']):
            assert all(a[k]==b[k] for k in ('image','field','rounds','tools')) and np.isclose(a['native'],b['native']) and np.allclose(a['AB'],b['AB'])
    dump(ROOT/'results/fit_audit.json',out)
if __name__=='__main__':main()
