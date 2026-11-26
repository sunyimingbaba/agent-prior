"""Rebuild pooled effects and decompose held-caption rank changes from the saved choices."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    reported=json.loads((ROOT/'校准结果.json').read_text())
    output={}
    for tag,files in [('148',['148']),('122',['122']),('pooled',['148','122'])]:
        rows=defaultdict(list)
        for file in files:
            with gzip.open(ROOT/('校准_'+file+'.jsonl.gz'),'rt') as f:
                for line in f:
                    r=json.loads(line)
                    rows[r['rgb'],r['pair'],r['seed']].append(r)
        effects=[]
        for (_,pair,seed),entries in rows.items():
            scores=[]
            for r in entries:
                a=r['outcomes']['ordinary']
                b=r['outcomes']['calibrated']
                delta=b['mrr']-a['mrr']
                ceiling=np.isclose(a['mrr'],1,rtol=0,atol=1e-12)
                scores.append([b['detail']-a['detail'],delta,b['net'],delta if ceiling else 0.,0. if ceiling else delta])
            effects.append((pair,seed,np.mean(scores,axis=0)))
        point=np.mean([v for _,_,v in effects],axis=0)
        summary=reported[tag] if tag=='pooled' else reported[tag]['scene']
        block=summary['against']['ordinary']['calibrated']
        assert np.allclose(point[:3],[block[k]['mean'] for k in ('detail','mrr','net')],rtol=0,atol=1e-12)
        assert abs(point[3]+point[4]-point[1])<1e-12
        seed_means=[float(np.mean([v[0] for _,s,v in effects if s==seed])) for seed in 'ab']
        assert np.allclose(seed_means,block['detail']['seeds'],rtol=0,atol=1e-12)
        output[tag]=dict(images=len(rows)//2,pairs=len({p for p,_,_ in effects}),
                         detail_delta=point[0],mrr_delta=point[1],net_delta=point[2],
                         mrr_contribution_when_original_first=point[3],mrr_contribution_elsewhere=point[4])
    (ROOT/'逐项核对.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
