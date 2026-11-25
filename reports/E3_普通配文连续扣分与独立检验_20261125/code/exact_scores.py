"""Reconstruct RFPO feedback from all logged training-image observations and score final greedy outputs."""
from fractions import Fraction
import numpy as np
from compare import OUT, SOURCE, read, dump, sha, load
from single_scores import FORM, RFPO, calculate, summary_by_seed
from orders import advantages

ARMS = ('ordinary', 'joint')
ALL = (*ARMS, 'untrained', 'retained')

def rank_payout(order):
    # Integrate a uniform random ordering within each tied block.
    out = []
    for a in order:
        above = sum(b > a for b in order)
        tied = sum(b == a for b in order)
        out.append(float(sum(Fraction(1, i) for i in range(above + 1, above + tied + 1)) / tied))
    return np.array(out)

def verify_training():
    plan = read(SOURCE / 'inputs/training_roster.json')['rows']
    check = read(OUT / 'inputs/exact_lock.json')
    assert sha(SOURCE / 'inputs/training_roster.json') == check['training']
    assert all(sha(OUT / 'code' / name) == digest for name, digest in check['implementation'].items())
    audit, first = {}, {}
    for seed in 'ab':
        for arm in ARMS:
            folder = OUT / 'exact_policy' / (seed + '_' + arm)
            proof, steps = read(folder / 'complete.json'), read(folder / 'steps.json')
            assert proof['receipt'] == sha(OUT / 'inputs/exact_lock.json') and proof['logs'] == sha(folder / 'steps.json')
            assert len(plan) == len(steps) == proof['steps'] == 24
            init = load(RFPO / 'vectors' / (seed + '_R0_primary.pt'))['adapter']
            assert proof['initial'] == init
            assert proof['updated'] == sum(z['norm'] > 0 for z in steps)
            chances, changed, winners, slots = 0, 0, 0, 0
            for j, (row, z) in enumerate(zip(plan, steps)):
                assert z['image'] == row['image'] and z['step'] == j+1
                rng = (71000000 if seed == 'a' else 81000000) + 1000*j
                assert z['rng'] == [rng, rng+100] and len(z['paths']) == 16
                obs = z['scores']
                vals = [[Fraction(*x) for x in v] for v in obs['rr']]
                d = obs['detail']
                assert len(vals) == len(d) == 17 and set(d) <= {0,1,2}
                assert len(obs['positive']) == len(vals[0]) > 0
                ordinary = [sum(r)/len(r) for r in vals[:-1]]
                joint = [Fraction(sum(a > b for a,b in zip(r, vals[-1])),len(r)) if d[i] > d[-1] else Fraction() for i,r in enumerate(vals[:-1])]
                pay = {'ordinary':rank_payout(ordinary), 'joint':rank_payout(list(zip(joint,ordinary)))}
                assert np.allclose(z['joint_fraction'],[float(x) for x in joint],atol=1e-12,rtol=0)
                assert all(np.allclose(pay[k],z['rewards'][k],atol=1e-12,rtol=0) for k in ARMS)
                assert np.allclose(advantages(pay[arm]),z['adv'],atol=1e-8,rtol=0)
                assert np.isfinite(z['norm']) and np.isfinite(z['max_error'])
                chances += any(joint)
                slots += sum(x>0 for x in joint)
                changed += not np.array_equal(pay['ordinary'],pay['joint'])
                winners += not np.array_equal(pay['ordinary']==pay['ordinary'].max(),pay['joint']==pay['joint'].max())
                if j == 0:
                    same = {k:z[k] for k in ('image','query','rng','anchor','paths','scores','rewards')}
                    if arm == 'ordinary': first[seed] = same
                    else: assert first[seed] == same
            audit[seed+'_'+arm] = dict(**proof, opportunity_groups=chances, joint_slots=slots,
                changed_reward_groups=changed,changed_winner_groups=winners,
                replayed=sum(z['replayed'] for z in steps),max_logp_error=max(z['max_error'] for z in steps))
    return audit

def paired(left,right,labels):
    return summary_by_seed({s:{k:v-right[s][k] for k,v in left[s].items()} for s in 'ab'},labels)

def main():
    report = {'audit':verify_training()}
    for tag,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        labels = [r.get('pair',r['image']) for r in rows]
        ids = [r['image'] for r in rows]
        values = {arm:{} for arm in ALL}
        for s in 'ab':
            for arm in ALL:
                if arm in ARMS:
                    dest = OUT/'exact_policy'/(s+'_'+arm)
                    path = dest/'evaluate'/(tag+'.pt')
                    assert sha(path) == read(dest/'evaluate'/tag/'complete.json')['features']
                    obj = load(path)
                    assert obj['adapter'] == report['audit'][s+'_'+arm]['final']
                else:
                    obj = load(RFPO/'vectors'/(s+'_'+('R0' if arm=='untrained' else 'deployed')+'_'+tag+'.pt'))
                assert obj['images'] == ids
                z = calculate(tag,obj['features'],rows)
                if tag=='primary': z['DE']=(z['D']+z['E'])/2
                values[arm][s] = z
        report[tag] = dict(images=len(rows),levels={a:summary_by_seed(v,labels) for a,v in values.items()},
            joint_minus={a:paired(values['joint'],values[a],labels) for a in ALL if a!='joint'})
    dump(OUT/'results/exact_results.json',report)

if __name__=='__main__':
    main()
