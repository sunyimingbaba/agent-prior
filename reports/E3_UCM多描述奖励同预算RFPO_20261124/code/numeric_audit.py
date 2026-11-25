import json
import numpy as np
from report_io import ROOT,read,dump,steps,sha
def main():
    saved=ROOT/'numeric_legacy/pilot/a_R2'
    old=[json.loads(line) for line in (saved/'steps.jsonl').read_text().splitlines()]
    current=steps('a_R2');assert len(current)==len(old)
    events=[];first_change=None
    for a,b in zip(old,current):
        assert a['image']==b['image'] and a['rng']==b['rng']
        ranks=[s['rank'] for s in a['scores']]
        if len(set(ranks))==1 and a['norm']>0:
            events.append(dict(step=a['step'],rank=ranks[0],legacy_reward_range=float(np.ptp(a['retrieval']['R2'])),legacy_norm=a['norm'],
                new_same_sample=a['records']==b['records'],new_norm=b['norm'],new_rank_count=len({s['rank'] for s in b['scores']})))
        if not first_change and any(x['ids']!=y['ids'] for x,y in zip(a['records'],b['records'])):first_change=a['step']
    assert events and sha(saved/'steps.jsonl')==read(saved/'complete.json')['steps_sha256']
    flat={}
    for seed in 'ab':
        for arm in ('R2','R3'):
            tag=seed+'_'+arm;ss=steps(tag);st=[q for q in ss if np.ptp(q['retrieval'][arm])==0]
            assert all(q['norm']==0 and all(a==0 for a in q['advantages']) for q in st)
            flat[tag]=len(st)
    dump(ROOT/'results/numeric_audit.json',dict(legacy_a_R2_spurious_steps=events,first_changed_sampled_tokens=first_change,first_group=dict(exact_same_token_records=old[0]['records']==current[0]['records'],exact_same_advantages=old[0]['advantages']==current[0]['advantages'],legacy_gradient_norm=old[0]['norm'],corrected_gradient_norm=current[0]['norm']),corrected_zero_signal_no_update=flat))
if __name__=='__main__':main()
