import numpy as np
import torch
from study import STAGE, PREV, OLD, RFPO, FORM, read, dump, banks, digest, summary_by_seed
from dual_scores import features, panel_outcomes


def main():
    torch.set_num_threads(1); plan=read(STAGE/'inputs/train.json'); new=read(STAGE/'inputs/holdout.json'); repeat=read(STAGE/'inputs/repeat.json')
    prior={r['pair'] for rows in read(OLD/'inputs/mined_roster.json').values() for r in rows}
    repeat_pairs={r['pair'] for r in repeat}
    remaining=[r for r in read(OLD/'inputs/real_roster.json') if r['pair'] not in prior|repeat_pairs]
    assert [r['rgb_sha256'] for r in new]==[r['rgb_sha256'] for r in remaining]
    hashes=lambda rows:{r['rgb_sha256'] for r in rows}
    trained=hashes(plan['a']+plan['b']); formal=hashes(read(RFPO/'inputs/panel.json')+read(FORM/'inputs/ucm.json'))
    assert not trained & (formal|hashes(new)|hashes(repeat)) and not formal & (hashes(new)|hashes(repeat)) and not hashes(new)&hashes(repeat)
    packs=banks(); text=[(p['native'].double().numpy(),p['hard'].double().numpy(),p['maps']) for p in packs]
    mask=[np.array([int(digest(','.join(map(str,t)))[-1],16)%2==0 for t in p['native_tokens'].tolist()]) for p in packs]
    result={};orig=read(PREV/'strict_rfpo/results/score.json')
    for panel,rows in [('hold',new),('repeat',repeat)]:
        labels=[r['pair'] for r in rows]
        for view in ('first','scene'):
            stats={}
            for seed in 'ab':
                vals=panel_outcomes(rows,features(panel+'_'+view,'retained',seed,rows),text,mask)
                stats[seed]={k:np.array([np.mean(q[j]) for q in vals]) for j,k in enumerate(('CDE','mrr'))}
            grouped=summary_by_seed(stats,labels)
            if panel=='repeat':
                previous=orig['hold_'+view]['levels']['retained']
                for k in grouped:assert np.allclose(grouped[k]['seeds'],previous[k]['seeds'],rtol=0,atol=1e-12)
            result[panel+'_'+view]=grouped
    dump(STAGE/'results/design_checks.json',dict(training_distinct=len(trained),new=len(new),new_pairs=len({r['pair'] for r in new}),
        old_repeat=len(repeat),repeat_pairs=len(repeat_pairs),formal_distinct=len(formal),baseline=result))


if __name__=='__main__':
    main()
