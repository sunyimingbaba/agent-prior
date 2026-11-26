"""Describe how the 64-image and 78-image responses differ, including shared source/attribute cells."""
from collections import defaultdict
import numpy as np
from study import STAGE, OLD, read, dump


def main():
    logs=read(STAGE/'results/caption_outcomes.json')
    if (STAGE/'reference/caption_outcomes.json').exists():
        logs=[r for r in logs if r['arm']!='retained']+read(STAGE/'reference/caption_outcomes.json')
    ix={(r['panel'],r['image'],r['seed'],r['arm']):np.array([np.mean(r['CDE']),np.mean(r['mrr_by_caption'])]) for r in logs}
    db={r['image']:r for r in read(OLD/'inputs/real_roster.json')};out={}
    for view in ('scene','first'):
        for ref in ('retained','singlejoint','dualplain'):
            cells={};detail={}
            for panel,p in [('repeat','repeat'),('hold','holdout')]:
                groups=defaultdict(list)
                for r in read(STAGE/'inputs'/(p+'.json')):
                    key=panel+'_'+view;name=r['image']
                    dif=np.mean([ix[key,name,s,'dualjoint']-ix[key,name,s,ref] for s in 'ab'],axis=0)
                    groups[r['tag']+'/'+db[name]['attribute'],r['pair']].append(dif)
                assert all(len(v)==2 for v in groups.values())
                rows={k:np.mean(v,axis=0) for k,v in groups.items()};cells[panel]=rows
                detail[panel]={key:dict(pairs=sum(g==key for g,_ in rows),delta=np.mean([v for (g,_),v in rows.items() if g==key],axis=0).tolist())
                               for key in sorted({g for g,_ in rows})}
            common=sorted(set(detail['repeat']) & set(detail['hold']))
            n=sum(detail['repeat'][k]['pairs']+detail['hold'][k]['pairs'] for k in common)
            standard={p:sum(np.array(detail[p][k]['delta'])*(detail['repeat'][k]['pairs']+detail['hold'][k]['pairs'])/n for k in common).tolist() for p in cells}
            rng=np.random.default_rng(20261126)
            samples={p:np.stack(list(v.values())) for p,v in cells.items()}
            resampled={p:v[rng.integers(len(v),size=(20000,len(v)))].mean(1) for p,v in samples.items()}
            diff=resampled['repeat']-resampled['hold']
            out[view+'_'+ref]=dict(metrics=['CDE','mrr'],repeat_minus_hold=(samples['repeat'].mean(0)-samples['hold'].mean(0)).tolist(),
                interval=np.quantile(diff,[.025,.975],axis=0).T.tolist(),common_cells=len(common),common_pairs={p:sum(detail[p][k]['pairs'] for k in common) for p in cells},
                standardized=standard,by_cell=detail)
    dump(STAGE/'results/panel_comparison.json',out)


if __name__=='__main__':
    main()
