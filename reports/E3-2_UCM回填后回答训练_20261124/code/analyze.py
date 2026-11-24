"""Compute the registered endpoints from the saved deployment outputs and paired contrasts."""
from collections import defaultdict
import numpy as np
import torch
from shared import ROOT, REBUILD, read, dump, sha, setup
from metrics import fixed, recalls, csvwrite, ci
from agent.grpo.reward import r_format

import sys
ARMS = ['R0','R1','R2'] + (['first_R2','old_R2'] if '--controls' in sys.argv else [])
GROUPS = ['quantity_A','quantity_B','attribute','relation']

def e1group(row):
    if row['cohort']=='previous' and row['dimension']=='quantity' and row['template'] in ('A','B'):
        return 'quantity_'+row['template']
    if row['cohort']=='extension':
        return 'relation' if row['template']=='shadow_direction' else row['dimension']
    return ''

def main():
    setup()
    inputs=REBUILD/'paired_views_20261120/inputs'
    val=read(inputs/'validation.json')
    pos={r['image']:i for i,r in enumerate(val)}
    bank=torch.load(inputs/'retrieval_text.pt',map_location='cpu',weights_only=True)
    e1=torch.load(inputs/'e1/features.pt',map_location='cpu',weights_only=True)
    words={s:i for i,s in enumerate(e1['texts'])}
    cases=read(inputs/'e1/cases.json')
    n=len(val)
    owner=np.arange(n)[:,None]==np.repeat(np.arange(n),5)[None,:]
    retrieval,detail,behavior=[],[],[]
    states,compiled,clusters={},{},{}
    for arm in ARMS:
        evaldir=ROOT/arm/'eval'
        proof=read(evaldir/'complete.json')
        entries=[]
        for j,r in enumerate(val):
            path=evaldir/f'{j:03d}.pt'
            assert sha(path)==proof['rows'][j]
            z=torch.load(path,map_location='cpu',weights_only=True)
            assert z['adapter']==proof['adapter'] and z['row']==j and z['image']==r['image'] and z['query']==r['orig_query']
            entries.append(z)
            behavior.append(dict(arm=arm,image=r['image'],rounds=len(z['rounds']),tool_calls=len(z['tools']),
                answer_format=r_format('\n'.join(q['text'] for q in z['rounds']))>0,
                first_eos_only=z['rounds'][0]['eos'] and z['rounds'][0]['length']==1,
                last_eos=z['rounds'][-1]['eos'],tokens=sum(q['length'] for q in z['rounds'])))
        feats=torch.stack([z['feature'] for z in entries])
        scores=fixed(feats,bank['native'])[:,bank['caption_order']]
        i2t=recalls(scores,owner)
        t2i=recalls(scores.T,owner.T)
        by={f'I2T_R{k}':v for k,v in i2t.items()}
        by.update({f'T2I_R{k}':v.reshape(n,5).mean(1) for k,v in t2i.items()})
        by['mR']=np.mean(list(by.values()),axis=0)
        for j,r in enumerate(val):
            retrieval.append(dict(arm=arm,image=r['image'],**{k:float(v[j]) for k,v in by.items()}))
        fine=fixed(feats,e1['text_features'])
        groups=defaultdict(list)
        for case in cases:
            g=e1group(case)
            if g not in GROUPS:
                continue
            scores=fine[pos[case['image']]]
            margin=float(scores[words[case['positive']]]-scores[words[case['negative']]])
            row=dict(arm=arm,group=g,image=case['image'],pair=case['pair'],margin=margin,correct=int(margin>0))
            detail.append(row)
            groups[g].append(row)
        state={'retrieval_percent':{k:100*float(v.mean()) for k,v in by.items()},'e1':{}}
        b=[v for v in behavior if v['arm']==arm]
        state['behavior']={k:float(np.mean([z[k] for z in b])) for k in ('rounds','tool_calls','answer_format','first_eos_only','last_eos','tokens')}
        cluster={}
        for g,ncases in zip(GROUPS,[22,22,36,36],strict=True):
            rows=groups[g]
            assert len(rows)==ncases
            pairs=defaultdict(list)
            for row in rows:
                key=row['pair'] if g in ('relation','attribute') else row['image']
                pairs[key].append(row['correct'])
            counts={len(v) for v in pairs.values()}
            assert counts==({2} if g in ('relation','attribute') else {1})
            cluster[g]={k:np.mean(v) for k,v in pairs.items()}
            state['e1'][g]=dict(correct=sum(r['correct'] for r in rows),n=ncases,
                 margin=float(np.mean([r['margin'] for r in rows])))
            if g in ('relation','attribute'):
                state['e1'][g].update(both=sum(all(v) for v in pairs.values()),pairs=len(pairs))
        compiled[arm]=by
        clusters[arm]=cluster
        states[arm]=state
    contrasts={}
    for a,b in [('R2','R1'),('R2','R0'),('R1','R0')]:
        result={k:ci(100*(v-compiled[b][k])) for k,v in compiled[a].items()}
        for g in GROUPS:
            ca,cb=clusters[a][g],clusters[b][g]
            assert ca.keys()==cb.keys()
            result[g]=ci(100*np.array([ca[k]-cb[k] for k in ca]))
        contrasts[f'{a}-{b}']=result
    for arm in (a for a in ARMS[1:] if a!='old_R2'):
        steps=read(ROOT/arm/'steps.json')
        diffs=[s['replay_mean_abs_logp_difference'] for s in steps]
        signal=[np.std(s['retrieval'],ddof=1) for s in steps]
        fmt=[np.std(s['format'],ddof=1) for s in steps]
        states[arm]['training']=dict(updates=sum(s['norm']>0 for s in steps),groups=len(steps),
          varied_retrieval=sum(float(x)>1e-12 for x in signal),mixed_format=sum(float(x)>0 for x in fmt),
          mean_within_group_retrieval_sd=float(np.mean(signal)),mean_within_group_format_sd=float(np.mean(fmt)),
          varied_format_larger_sd=sum(f>r and f>0 for f,r in zip(fmt,signal)).item(),
          mean_group_sampling_logp_error=float(np.mean(diffs)),max_token_logp_error=max(z['replay_max_abs_logp_difference'] for z in steps),
          eos=sum(tr['ended'] for s in steps for tr in s['trajectories']),generated=sum(len(s['trajectories']) for s in steps))
    s1,s2=[read(ROOT/arm/'steps.json') for arm in ('R1','R2')]
    assert [s['image'] for s in s1]==[s['image'] for s in s2]
    assert s1[0]['trajectories']==s2[0]['trajectories'] and s1[0]['proxy_rank']==s2[0]['proxy_rank']
    assert all(len({r['image'] for r in retrieval if r['arm']==arm})==210 for arm in ARMS)
    out=ROOT/'results'
    out.mkdir(exist_ok=True)
    csvwrite(out/'retrieval_by_image.csv',retrieval)
    csvwrite(out/'detail_by_case.csv',detail)
    csvwrite(out/'behavior_by_image.csv',behavior)
    dump(out/'states.json',states)
    dump(out/'contrasts_percentage_points.json',contrasts)
    print('Completed all three arms',flush=True)
if __name__=='__main__':
    main()
