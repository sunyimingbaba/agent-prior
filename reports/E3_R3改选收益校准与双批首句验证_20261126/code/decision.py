"""Resolve registered training conditions, including uncertainty for the data-only control."""
import gzip
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from data import ROOT,read,sha,PRIOR_RUN

def adjust(rows,arm):
    group=defaultdict(list)
    for r in rows:
        group[r['pair'],r['rgb'],r['seed']].append(r)
    pairs=sorted({p for p,_,_ in group})
    w=np.random.default_rng(261126).multinomial(len(pairs),np.ones(len(pairs))/len(pairs),size=20000)
    den=w@np.array([sum(p==q for p,_,_ in group) for q in pairs])
    out={}
    for k in ('detail','mrr','net'):
        num=[sum(np.mean([r['outcomes'][arm][k]-r['outcomes']['old'][k] for r in g]) for (p,_,_),g in group.items() if p==q) for q in pairs]
        out[k]=np.quantile((w@num)/den,[.0125,.9875]).tolist()
    return out


def main():
    rs=ROOT/'results'
    with gzip.open(rs/'pooled.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    adjusted={k:adjust(rows,k) for k in ('old3','switch')}
    pooled=read(rs/'pooled.json')
    def passes(method,flag):
        arm='local3' if method=='neighbor' else method
        prefix='neighbor_' if method=='neighbor' else ''
        stats=read(rs/(prefix+'pooled.json'))['against']['old'][arm]
        intervals=adjusted[method] if method!='neighbor' else {k:v['interval_975'] for k,v in stats.items()}
        for main,other in [('detail','mrr'),('mrr','detail')]:
            if stats[other]['mean']<=0 or intervals[main][0]<=0:continue
            if all(read(rs/(prefix+tag+'.json'))['against']['old'][arm][main]['mean']>0 for tag in ('v122','v148')):return True
        return False
    flags=dict(switch=bool(read(rs/'首次判定.json')['improvement']),neighbor=passes('neighbor',True),old3=passes('old3',True))
    ref=pooled['against']['ordinary']['old']
    reference=(all(read(rs/(tag+'.json'))['against']['ordinary']['old'][k]['mean']>0 for tag in ('v122','v148') for k in ('detail','mrr')) and
               any(ref[k]['interval'][0]>0 for k in ('detail','mrr')))
    nominee=next((k for k in ('switch','neighbor','old3') if flags[k]),None)
    arms=['ordinary','old',nominee] if nominee else (['ordinary','old'] if reference else [])
    selected=dict(train_arms=arms,new_nominee=nominee,candidate_flags=flags,reference_pilot=reference,adjusted=adjusted,
                  starts=read(PRIOR_RUN/'inputs/lock.json')['starts'])
    files=list((ROOT/'inputs/pilot').glob('*'))+[ROOT/'code'/name for name in ('policy_worker.py','online_reward.py','train_pilot.py','eval_pilot.py')]
    selected['lock']={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file() and p.name!='decision.json'}
    (ROOT/'inputs/pilot/decision.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('pilot',arms,flush=True)

if __name__=='__main__':main()
