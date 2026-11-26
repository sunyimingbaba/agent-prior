from collections import Counter
import numpy as np
from study import STAGE, OLD, read, dump, banks, bank_for, digest


def main():
    text=banks(); roster={r['image']:r for r in read(OLD/'inputs/real_roster.json')}
    records={(v['arm'],v['seed'],v['panel'],v['image']):v for v in read(STAGE/'results/caption_outcomes.json') if v['panel'] in ('hold_scene','hold_first')}
    out={}
    for ref in ('singlejoint','dualplain'):
        cases={'improved':[],'worse':[]}
        for (arm,seed,panel,name),v in records.items():
            if arm!='dualjoint':continue
            baseline=records[ref,seed,panel,name]
            dc=np.subtract(v['CDE'],baseline['CDE']);dr=np.subtract(v['mrr_by_caption'],baseline['mrr_by_caption'])
            direction='improved' if dc.sum()>0 and dr.mean()>0 else 'worse' if dc.sum()<0 and dr.mean()<0 else None
            if direction is None:continue
            sign=1 if direction=='improved' else -1
            row=roster[name]; pos=bank_for(row,text)['maps'][name]['native']
            col=sorted(set(pos)-{pos[0]})[int(np.flatnonzero(sign*dr>0)[0])]
            item=int(np.flatnonzero(sign*dr>0)[0]);k=int(np.flatnonzero(sign*dc>0)[0])
            cases[direction].append(dict(seed=seed,image=name,panel=panel,attribute=row['attribute'],
                  prompt=row['orig_query' if panel=='hold_first' else 'scene_query'],
                  detail_wording='CDE'[k],positive_detail=row['CDE'[k]][0],negative_detail=row['CDE'[k]][1],
                  old_correct=baseline['CDE'],new_correct=v['CDE'],caption=row['captions'][pos.index(col)],
                  old_reciprocal=baseline['mrr_by_caption'][item],new_reciprocal=v['mrr_by_caption'][item],
                  old_mean=float(np.mean(baseline['mrr_by_caption'])),new_mean=float(np.mean(v['mrr_by_caption']))))
        out[ref]={k:dict(count=len(v),example=min(v,key=lambda r:digest('dual26-trace',r['seed'],r['panel'],r['image'])) if v else None)
                  for k,v in cases.items()}
    dump(STAGE/'results/examples.json',out)


if __name__=='__main__':
    main()
