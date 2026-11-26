from collections import defaultdict
import numpy as np
from study import STAGE, PREV, ARMS, read, dump, sha
from strict_reward import strict_rewards


def check():
    lock = read(STAGE/'inputs/lock.json')
    result = {}
    for seed, rows in read(STAGE/'inputs/train.json').items():
        starts = {}
        for arm in ARMS[1:]:
            folder = STAGE/'models'/(seed+'_'+arm)
            cp = read(folder/'complete.json'); steps = read(folder/'steps.json')
            assert cp['initial']==lock['starts'][seed] and cp['receipt']==sha(STAGE/'inputs/lock.json')
            assert len(steps)==cp['steps']==24 and sha(folder/'steps.json')==cp['logs']
            starts[arm]=steps[0]['views']
            by_block = [defaultdict(int),defaultdict(int)]; errors=[]; eligible_both=0
            for j,(row,step) in enumerate(zip(rows,steps)):
                assert step['image']==row['image'] and step['step']==j+1 and len(step['views'])==2
                both=[]
                for b,view in enumerate(step['views']):
                    expected='scene_query' if b==1 and arm.startswith('dual') else 'orig_query'
                    assert view['field']==expected and view['query']==row[expected]
                    assert view['rng']==(91000000 if seed=='a' else 101000000)+j*1000+b*100
                    assert len(view['paths'])==8 and len(view['scores']['rr'])==9 and len(view['scores']['detail'])==9
                    scores,p= strict_rewards(view['scores'])
                    assert p==view['joint']
                    for k in scores: assert np.array_equal(scores[k],view['rewards'][k])
                    rewards=scores['joint' if arm.endswith('joint') else 'ordinary']
                    expected_adv=(rewards-rewards.mean())/(rewards.std(ddof=1)+1e-8) if len(set(rewards))>1 else np.zeros(8)
                    assert np.allclose(expected_adv,view['advantage'],rtol=0,atol=1e-12)
                    different=not np.array_equal(scores['joint'],scores['ordinary'])
                    different_top=not np.array_equal(scores['joint']==max(scores['joint']),scores['ordinary']==max(scores['ordinary']))
                    stats=by_block[b];stats['groups']+=1;stats['candidates']+=8;stats['eligible']+=int(max(p)>0)
                    stats['reordered']+=int(different);stats['top_changed']+=int(different_top)
                    both.append(max(p)>0);errors.append(view)
                eligible_both += int(all(both))
            result[seed+'_'+arm] = dict(**cp, blocks=[dict(x) for x in by_block],both_eligible=eligible_both,
                max_logp_error=max(e['max_error'] for e in errors),
                mean_absolute_logp_error=sum(e['mean_error']*e['replayed'] for e in errors)/sum(e['replayed'] for e in errors))
        for block, arms in [(0,list(ARMS[1:])),(1,['singleplain','singlejoint']),(1,['dualplain','dualjoint'])]:
            base=starts[arms[0]][block]
            for arm in arms[1:]:
                for k in ('scores','paths','anchor','calls','rng'):assert starts[arm][block][k]==base[k]
        old=read(PREV/'strict_rfpo/models'/(seed+'_coldordinary')/'steps.json')[0]
        for b,g in enumerate(starts['singleplain']):
            assert g['paths']==old['paths'][8*b:8*(b+1)] and g['anchor']==old['anchor']
            for k in ('rr','detail'):
                assert g['scores'][k]==old['scores'][k][8*b:8*(b+1)]+old['scores'][k][-1:]
    dump(STAGE/'results/training.json',result)


if __name__=='__main__':
    check()
