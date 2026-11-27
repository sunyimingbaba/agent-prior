"""Enumerate every equally likely rank order among distinct protected terminal states."""
from collections import defaultdict
from fractions import Fraction
from itertools import permutations
import json

import numpy as np
import torch

from independent import OUT,POOL,infer,load,pool,read,reprojection,stats


def expected_winners(base,raw,rr,held):
    t=[i for i in range(8) if base[i]==max(base)]
    visible=[k for k in range(len(rr[0])) if k!=held]
    valid=[i for i in range(8) if i not in t and all(Fraction(rr[i][k])*len(t)>=sum(Fraction(rr[j][k]) for j in t) for k in visible)]
    classes=defaultdict(list)
    for i in t+valid:
        classes[raw[i].double().numpy().tobytes()].append(i)
    blocks=list(classes.values())
    for same in blocks:
        assert len({base[i] for i in same})==1
        assert all(rr[j]==rr[same[0]] for j in same)
    baseline=np.zeros(8);baseline[t]=1/len(t)
    if not valid:
        return baseline,0.,1
    count=0
    changed=0
    weights=np.zeros(8)
    for order in permutations(range(len(blocks))):
        # Ascending random ranks. Only lower-R2 blocks above every top block enter.
        last_top=max(j for j,k in enumerate(order) if blocks[k][0] in t)
        lifted=order[last_top+1:]
        if not lifted:
            weights+=baseline
        else:
            selected=[k for k in order if blocks[k][0] in t or k in lifted]
            rewards=np.array(base).copy()
            sorted_base=sorted(base[i] for k in selected for i in blocks[k])
            pos=0
            for k in selected:
                end=pos+len(blocks[k])
                rewards[blocks[k]]=np.mean(sorted_base[pos:end])
                pos=end
            winners=np.flatnonzero(rewards==max(rewards))
            weights[winners]+=1/len(winners)
            changed+=int(set(t)!=set(winners))
        count+=1
    assert np.isclose(weights.sum(),count)
    return weights/count,changed/count,count


def main():
    bank,cohort,_=pool()
    rows=[];perms=0;trials=0
    for r in cohort:
        if len(r['alt_ids'])<2:continue
        sample=defaultdict(list)
        for seed in 'ab':
            name=r['image']
            g=read(POOL/'scores'/seed/(name+'.json'))
            h=load(POOL/'sampling/raw'/seed/(name+'.pt'))['hidden']
            base=g['r2'];t=g['r2_top']
            q=infer(h,bank[r['attribute']],r['side'])
            legacy=[min(v['hard_margins']) for v in g['items']]
            rr=[x['other_caption_rr'] for x in g['items']]
            de,fg=[np.mean(np.array([v[k] for v in g['items']])>0,axis=1) for k in ['DE','FG']]
            for held in range(len(rr[0])):
                w,ch,num=expected_winners(base,h,rr,held);perms+=num;trials+=1
                sample['random_changed'].append(ch)
                chosen=reprojection(base,q,rr,held);old=reprojection(base,legacy,rr,held)
                for lab,z in [('DE',de),('FG',fg),('rr',np.array([v[held] for v in rr]))]:
                    expected=w@z
                    sample['random_minus_R2_'+lab].append(expected-z[t].mean())
                    sample['direction_minus_random_'+lab].append(z[chosen].mean()-expected)
                    sample['native_minus_random_'+lab].append(z[old].mean()-expected)
        rows.append(((r['cohort'],r['pair']),{k:np.mean(y) for k,y in sample.items()}))
    result={**stats(rows),'trials':trials,'enumerated_rank_orders':perms}
    (OUT/'同配文保护随机排序对照.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    torch.set_num_threads(1);main()
