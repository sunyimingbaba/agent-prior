"""Inspect all losses of an ordinary first rank under the calibrated R3 on new captions."""
import json
import numpy as np
import torch
from guard import STAGE, LAST, PRIOR, choose, load_bank, read, real_stream, with_gaps
from audit import read_rows


def main():
    rows=read_rows('first98')
    wanted={(r['image'],r['seed'],r['block'],r['held']):r for r in rows if r['ceiling']
            and r['outcomes']['calibrated']['mrr']<1}
    archive=read(LAST/'锁定评分参数.json')['model']
    weights=np.array(archive['coef'])/np.array(archive['scale_x'])[:,None]
    bank=load_bank(PRIOR/'reward_confirm')
    roster={r['image']:r for r in read(STAGE/'inputs/real_roster.json')}
    cases=[]
    required={(n,s) for n,s,_,_ in wanted}
    for p in with_gaps(v for v in real_stream() if (v[0]['image'],v[1]) in required):
        key=tuple(p[f] for f in ('image','seed','block','held'))
        if key not in wanted:continue
        record=wanted[key]
        top,_=choose(p,weights)
        floor=record['outcomes']['calibrated']['detail']
        joint=(p['mrr']==1)&(p['fine']>=floor)
        baseline=p['baseline']
        preds=p['x']@weights
        gains=preds-preds[baseline].mean(0)
        visible=p['gaps']
        keep=(visible[baseline]>0).all(0)
        elig=(gains>1e-12).all(1)&(visible[:,keep]>0).all(1)
        ids=bank['maps'][p['image']]['native']
        held=sorted(set(ids)-{ids[0]})[p['held']]
        r=roster[p['image']]
        cases.append(dict(image=p['image'],seed=p['seed'],block=p['block'],held=p['held'],
                     caption=r['captions'][ids.index(held)],prompt=r['orig_query'],
                     D=r['D'],E=r['E'],old=top['calibrated'],guard=top['guard'],
                     restored=record['outcomes']['guard']['mrr']==1,
                     eligible_joint=np.flatnonzero(joint&elig).tolist(),
                     passes_visible=np.flatnonzero(joint&(visible[:,keep]>0).all(1)).tolist(),
                     passes_both_predictions=np.flatnonzero(joint&(gains>1e-12).all(1)).tolist(),
                     predicted_gains=gains.tolist(),baseline=baseline,
                     per_candidate_mrr=p['mrr'].tolist(),per_candidate_detail=p['fine'].tolist(),
                     can_retain_first_and_old_detail=np.flatnonzero(joint).tolist()))
    assert len(cases)==len(wanted)
    (STAGE/'results/原首位退位案例.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    for c in cases:print(c['image'],c['restored'],c['can_retain_first_and_old_detail'],c['passes_visible'],c['passes_both_predictions'])


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
