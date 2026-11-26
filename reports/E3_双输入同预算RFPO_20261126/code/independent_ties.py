from collections import defaultdict
import numpy as np
import torch
from study import STAGE, OLD, FORM, RFPO, read, dump, rr, summary_by_seed, paired
from panels import PRIOR, transfer_panel
from candidate_transfer import choices


def main():
    torch.set_num_threads(1)
    skip={r['rgb_sha256'] for p in (OLD/'inputs/real_roster.json',FORM/'inputs/ucm.json',RFPO/'inputs/panel.json') for r in read(p)}
    grouped=defaultdict(list)
    for r in read(PRIOR/'reward_confirm/inputs/panel.json'):grouped[r['pair']].append(r)
    rows=[r for pair in grouped.values() if all(r['rgb_sha256'] not in skip for r in pair) for r in pair]
    wanted={r['rgb_sha256'] for r in rows}; assert len(wanted)==98 and len(set(r['pair'] for r in rows))==49
    data={}; logs=[]
    for row,seed,view,bank,vs,*_ in transfer_panel():
        if row['rgb_sha256'] not in wanted:continue
        assert view=='scene' and vs.shape==(17,768)
        ix=bank['maps'][row['image']];pos=ix['native'];alts=sorted(set(pos)-{pos[0]})
        native=vs@bank['native'].T; hard=vs@bank['hard'].T
        odd=~bank['even'].copy();even=bank['even'].copy();odd[pos]=False;even[pos]=False
        grid=[[[rr(q[j],q[m]) for j in alts] for q in native] for m in (odd,even)]
        det=np.column_stack([hard[:,ix[k][0]]>hard[:,ix[k][1]] for k in 'ABCDE'])
        for tag,folds in [('all',[None]),('leave_one',range(len(alts)))]:
            rec=choices(grid,det,folds)
            if rec is not None:
                data[seed,row['image'],tag]=rec
                logs.append(dict(image=row['image'],rgb=row['rgb_sha256'],seed=seed,panel=tag,**rec))
    out={}
    for tag in ('all','leave_one'):
        used=[r for r in rows if ('a',r['image'],tag) in data]
        assert all(('b',r['image'],tag) in data for r in used)
        labels=[r['pair'] for r in used]
        vals={a:{s:dict(zip(('CDE','mrr','win','loss','net'),np.array([data[s,r['image'],tag][a] for r in used]).T)) for s in 'ab'}
              for a in ('ordinary','joint','tiebreak')}
        rec=[data[s,r['image'],tag] for s in 'ab' for r in used]
        out[tag]=dict(images=len(used),pairs=len(set(labels)),groups=sum(v['groups'] for v in rec),
            tie_changes=sum(v['tie_changes'] for v in rec),strict_changes=sum(v['changed_groups'] for v in rec),
            levels={a:summary_by_seed(x,labels) for a,x in vals.items()},
            diff={a:paired(vals[a],vals['ordinary'],labels) for a in ('joint','tiebreak')})
    z=out['leave_one']['diff']['tiebreak']
    out['run_tiebreak']=min(z['CDE']['seeds'])>0 and z['CDE']['interval'][0]>0 and z['mrr']['mean']>=-1e-12 and z['net']['mean']>=-1e-12
    dump(STAGE/'results/independent_ties.json',out);dump(STAGE/'results/independent_tie_images.json',logs)


if __name__=='__main__':
    main()
