"""Evaluate the fixed final warm-start update against both same-budget controls and retained R2."""
from fractions import Fraction
import numpy as np
from compare import OUT, read, dump, sha, load
from single_scores import FORM,RFPO,calculate,summary_by_seed
from exact_scores import rank_payout,paired
from orders import advantages

ARMS=('selected_ordinary','selected_joint','uniform_joint')

def verify():
    lock=read(OUT/'inputs/mined_lock.json');path=OUT/'inputs/mined_roster.json';plans=read(path)
    assert sha(path)==lock['training'] and sha(OUT/'inputs/mined_training_bank.pt')==lock['bank']
    assert all(sha(OUT/'code'/k)==h for k,h in lock['implementation'].items())
    checks={}
    for s in 'ab':
        first=None
        for arm in ARMS:
            group,obj=arm.split('_');folder=OUT/'mined_policy'/(s+'_'+arm)
            proof=read(folder/'complete.json');data=read(folder/'steps.json')
            assert proof['initial']==load(RFPO/'vectors'/(s+'_deployed_primary.pt'))['adapter']
            assert proof['logs']==sha(folder/'steps.json') and proof['receipt']==sha(OUT/'inputs/mined_lock.json')
            assert proof['steps']==len(data)==24 and proof['updated']==sum(z['norm']>0 for z in data)
            counts=dict(opportunities=0,changed=0,winners=0,slots=0)
            for j,(r,z) in enumerate(zip(plans[s+'_'+group],data)):
                rng=(91000000 if s=='a' else 101000000)+j*1000
                assert z['rng']==[rng,rng+100] and r['image']==z['image'] and r['orig_query']==z['query']
                scores=z['scores'];vals=[[Fraction(*x) for x in row] for row in scores['rr']];ds=scores['detail']
                assert len(vals)==len(ds)==17 and len(z['paths'])==16 and z['step']==j+1
                ordinary=[sum(v)/len(v) for v in vals[:-1]]
                joint=[Fraction(sum(a>b for a,b in zip(row,vals[-1])),len(row)) if ds[i]>ds[-1] else Fraction() for i,row in enumerate(vals[:-1])]
                p={'ordinary':rank_payout(ordinary),'joint':rank_payout(list(zip(joint,ordinary)))}
                assert np.allclose(z['joint_fraction'],[float(x) for x in joint],atol=1e-12,rtol=0)
                assert all(np.allclose(v,z['rewards'][k],rtol=0,atol=1e-12) for k,v in p.items())
                assert np.allclose(z['adv'],advantages(p[obj]),atol=1e-8,rtol=0)
                counts['opportunities']+=any(joint);counts['slots']+=sum(x>0 for x in joint)
                counts['changed']+=not np.array_equal(p['ordinary'],p['joint'])
                counts['winners']+=not np.array_equal(p['ordinary']==p['ordinary'].max(),p['joint']==p['joint'].max())
                if j==0 and group=='selected':
                    matched={k:z[k] for k in ('rng','paths','anchor','scores','rewards')}
                    if first is None:first=matched
                    else: assert first==matched
            checks[s+'_'+arm]=dict(**proof,**counts,replayed=sum(z['replayed'] for z in data),max_logp_error=max(z['max_error'] for z in data))
    return checks

def main():
    audit=verify();result={'audit':audit}
    for tag,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        labels=[r.get('pair',r['image']) for r in rows];ids=[r['image'] for r in rows]
        vals={k:{} for k in (*ARMS,'retained')}
        for arm in vals:
            for s in 'ab':
                if arm=='retained':z=load(RFPO/'vectors'/(s+'_deployed_'+tag+'.pt'))
                else:
                    f=OUT/'mined_policy'/(s+'_'+arm)/'evaluate';p=f/(tag+'.pt');assert sha(p)==read(f/tag/'complete.json')['features']
                    z=load(p);assert z['adapter']==audit[s+'_'+arm]['final']
                assert ids==z['images'];v=calculate(tag,z['features'],rows)
                if tag=='primary':v['DE']=(v['D']+v['E'])/2
                vals[arm][s]=v
        result[tag]=dict(images=len(rows),levels={k:summary_by_seed(v,labels) for k,v in vals.items()},
            joint_minus={k:paired(vals['selected_joint'],v,labels) for k,v in vals.items() if k!='selected_joint'})
    endpoints=[result[tag]['joint_minus']['retained'][key] for tag,key in [('primary','CDE'),('official','mR')]]
    result['directional']=all(e['mean']>0 and min(e['seeds'])>0 for e in endpoints)
    result['promote']=result['directional'] and all(e['interval'][0]>0 for e in endpoints)
    dump(OUT/'results/mined_results.json',result)
if __name__=='__main__':main()
