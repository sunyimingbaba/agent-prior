"""Independent midrank/credit checks from the logged raw scores and sampled actions."""
from pathlib import Path
from itertools import permutations
from collections import Counter
import hashlib
import numpy as np
import torch
from shared import ROOT,read,dump,sha
from metrics import csvwrite

def midranks(x):
    return np.asarray([1+sum(y<v for y in x)+(sum(y==v for y in x)-1)/2 for v in x])

def reconstructed(cap,direction,arm):
    groups=[slice(None)] if arm=='single' else [slice(0,4),slice(4,8)]
    signed=direction if arm!='joint' else np.tile(np.minimum(direction[:4],direction[4:]),2)
    values=[];advantages=[]
    for sl in groups:
        z=(midranks(cap[sl])+midranks(signed[sl]))/2
        values.extend(z)
        z=z.astype(np.float32)
        advantages.extend((z-z.mean())/(z.std(ddof=1)+1e-8))
    return np.asarray(values),np.asarray(advantages)

def run():
    inputs=ROOT/'inputs'
    for f,h in read(inputs/'lock.json').items():assert sha(inputs/f)==h
    plan=read(inputs/'protocol.json');cases=read(inputs/'train.json')
    all_steps={};summary={};credit=[];weights={}
    for arm in plan['arms']:
        seed,kind=arm.split('_')
        folder=ROOT/('policy_'+arm);rows=read(folder/'steps.json');u=read(folder/'update.json')
        assert len(rows)==u['steps']==16 and u['order']==[r['image'] for r in cases]
        resume=torch.load(folder/'resume.pt',mmap=True,map_location='cpu',weights_only=True)
        assert resume['log']==rows and resume['arm']==arm and resume['source']==sha(ROOT/'code/train.py')
        assert resume['reward_code']==sha(ROOT/'code/rewards.py') and resume['input_lock']==sha(inputs/'lock.json')
        active=hashlib.sha256()
        for w in resume['weights']: active.update(w.contiguous().view(torch.uint8).numpy().tobytes())
        assert active.hexdigest()==rows[-1]['active_params']==u['final_params']
        moments=resume['optim']['state'];optimizer_steps={int(v['step']) for v in moments.values()}
        assert optimizer_steps=={u['nonzero']} and len(moments)==len(resume['weights'])
        total=Counter();max_adv=0.;max_lp=0.;arm_groups=[]
        for i,(src,z) in enumerate(zip(cases,rows,strict=True)):
            assert src['image']==z['image'] and z['step']==i+1 and len(z['traces'])==8
            cap=[];direction=[]
            for k,t in enumerate(z['traces']):
                view=int(kind!='single' and k>=4)
                assert t['view']==view and t['seed']==plan['schedules'][seed]+8*i+k and t['filtered']==0
                assert t['slot']==(k if kind=='single' else k%4)
                cap.append(t['caption_positive']-t['caption_max_negative'])
                direction.append((t['upright_text_pos']-t['upright_text_neg'])*(1-2*view))
                for q in t['rounds']:
                    ids=q['tokens'];assert len(ids)==len(q['logp']) and ids and np.isfinite(q['logp']).all()
                    assert not any(t0 in (151643,151645) for t0 in ids[:-1]);total['generated_tokens']+=len(ids)
                total['sampled']+=1;total['positive_single_view']+=direction[-1]>0
            cap=np.asarray(cap);direction=np.asarray(direction)
            assert np.array_equal(np.column_stack([cap,direction]),z['caption_and_direction_margins'])
            v,a=reconstructed(cap,direction,kind)
            assert np.array_equal(v,z['rewards'])
            max_adv=max(max_adv,float(np.max(abs(a-z['advantages']))));max_lp=max(max_lp,z['max_logp_error'])
            assert max_adv<1e-5
            if kind!='single':
                up,rot=direction[:4],direction[4:]
                weak=np.minimum(up,rot);ok=(up>0)&(rot>0)
                separate_v,separate_a=reconstructed(cap,direction,'separate');joint_v,joint_a=reconstructed(cap,direction,'joint')
                hits=[int(np.sum((up>0)&(rot[list(perm)]>0))) for perm in permutations(range(4))]
                row=dict(arm=arm,step=i+1,image=z['image'],paired_both=int(ok.sum()),upright_correct=int((up>0).sum()),rotated_correct=int((rot>0).sum()),
                    any_possible=int((up>0).any() and (rot>0).any()),mean_weak=float(weak.mean()),max_weak=float(weak.max()),
                    upright_weaker=int((up<rot).sum()),rotated_weaker=int((rot<up).sum()),equal=int((up==rot).sum()),
                    all_pairings_min=min(hits),all_pairings_max=max(hits),all_pairings_mean=float(np.mean(hits)),
                    advantage_changed=int(np.max(abs(separate_a-joint_a))>1e-6),changed_signs=int(np.sum(np.sign(separate_a)!=np.sign(joint_a))),
                    top_set_changed_views=sum(set(np.flatnonzero(separate_v[sl]==separate_v[sl].max()))!=set(np.flatnonzero(joint_v[sl]==joint_v[sl].max())) for sl in (slice(0,4),slice(4,8))))
                credit.append(row);arm_groups.append(row)
        checks={}
        for step in plan['steps']:
            path=folder/f'step_{step:02d}';data=read(path/'update.json');file=path/'adapter/adapter_model.safetensors'
            assert sha(file)==data['adapter_hash'] and data['final_params']==rows[step-1]['active_params'] and data['initial_params']==resume['initial']
            checks[str(step)]={**data,'bytes':file.stat().st_size}
        weights[arm]=dict(checkpoints=checks,resume_sha=sha(folder/'resume.pt'),resume_bytes=(folder/'resume.pt').stat().st_size)
        summary[arm]=dict(total,nonzero=u['nonzero'],tensors=len(resume['weights']),max_adv_error=max_adv,max_logp_error=max_lp,
                          paired_both=sum(g['paired_both'] for g in arm_groups) if arm_groups else None,
                          groups_any_possible=sum(g['any_possible'] for g in arm_groups) if arm_groups else None,
                          groups_positive_pair=sum(g['paired_both']>0 for g in arm_groups) if arm_groups else None,
                          groups_different_advantages=sum(g['advantage_changed'] for g in arm_groups) if arm_groups else None)
        all_steps[arm]=rows;del resume
    paired={}
    assert len({w['checkpoints']['16']['initial_params'] for w in weights.values()})==1
    for seed in 'ab':
        s,j=all_steps[seed+'_separate'],all_steps[seed+'_joint']
        idx=next((i for i,(x,y) in enumerate(zip(s,j)) if x['advantages']!=y['advantages']),len(s))
        for i in range(min(idx+1,len(s))):
            assert s[i]['traces']==j[i]['traces'] and s[i]['caption_and_direction_margins']==j[i]['caption_and_direction_margins']
        assert all(s[i]['active_params']==j[i]['active_params'] for i in range(idx))
        # The upright-only control uses the same random stream in its first four original-view slots.
        assert s[0]['traces'][:4]==all_steps[seed+'_single'][0]['traces'][:4]
        paired[seed]=dict(first_distinct_advantage=(idx+1 if idx<len(s) else None),
                          first_distinct_weights=next((i+1 for i,(x,y) in enumerate(zip(s,j)) if x['active_params']!=y['active_params']),None),
                          identical_sample_groups_through_first_effect=min(idx+1,len(s)),single_first_four_upright_match=True)
    target=ROOT/'results';dump(target/'training_audit.json',dict(arms=summary,matched_prefix=paired))
    dump(target/'checkpoint_index.json',weights);csvwrite(target/'same_candidate_credit.csv',credit)
if __name__=='__main__':run()
