"""Verify logged rewards against decoded actions, optimizer checkpoints and matched seed controls."""
from collections import Counter
from pathlib import Path
import re
import numpy as np
import torch
from scipy.stats import rankdata
from safetensors import safe_open
from transformers import AutoProcessor
from shared import ASSETS, REBUILD, ROOT, dump, read, sha
from agent.grpo.reward import r_format
from metrics import csvwrite

def main():
    torch.set_num_threads(4)
    config=read(ROOT/'inputs/protocol.json')
    processor=AutoProcessor.from_pretrained(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'))
    init=ASSETS/'ucm_main_lora/adapter_model.safetensors'
    prev=REBUILD/'matched_policy_20261120'/'policy_hard'
    historical=read(prev/'steps.json')
    archive_keys=['image','rewards','advantages','traces','caption_and_hard_margins','max_logp_error','unclipped_grad_norm']
    train=read(ROOT/'inputs/train.json')
    all_rows={};audit={};groups=[];checkpoints={}
    for arm in config['arms']:
        p=ROOT/('policy_'+arm);steps=read(p/'steps.json');update=read(p/'update.json')
        seed,on=arm.split('_');count=Counter();tokens=0;maxerr=0.
        state=torch.load(p/'resume.pt',mmap=True,map_location='cpu',weights_only=True)
        assert state['log']==steps and state['arm']==arm and state['source']==sha(ROOT/'code/train.py') and len(steps)==24
        assert state['initial']==update['initial_params'] and len(state['optim']['state'])==len(state['weights'])
        for i,(z,r) in enumerate(zip(steps,train,strict=True)):
            assert z['image']==r['image'] and z['step']==i+1
            a,b=np.array(z['caption_and_hard_margins']).T
            semantic=(rankdata(a)+rankdata(b))/2
            fmt=[];has_content=0;immediate=0
            for k,tr in enumerate(z['traces']):
                assert tr['seed']==config['schedule'][seed]+i*8+k and tr['filtered']==0
                texts=[]
                for run in tr['rounds']:
                    ts=run['tokens'];assert len(ts)==len(run['logp']) and np.isfinite(run['logp']).all() and not any(t in (151645,151643) for t in ts[:-1])
                    tokens+=len(ts);texts.append(processor.decode(ts,skip_special_tokens=True))
                text='\n'.join(texts);fmt.append(r_format(text))
                has_content+=any(s.strip() for s in re.findall(r'<answer>(.*?)</answer>',text,flags=re.DOTALL))
                immediate+=int(tr['rounds'][0]['tokens'] in ([151645],[151643]))
            expected=semantic+(7*np.array(fmt) if on=='on' else 0)
            assert np.array_equal(fmt,z['format_score']) and np.array_equal(semantic,z['semantic_midrank']) and np.array_equal(expected,z['rewards'])
            rr=torch.tensor(expected,dtype=torch.float32)
            assert torch.allclose(torch.tensor(z['advantages']), (rr-rr.mean())/(rr.std()+1e-8), atol=1e-6,rtol=1e-5)
            mixed=len(set(fmt))>1;hits=sum(s>0 for s in fmt)
            count.update(dict(formatted=hits,nonempty_formatted=has_content,mixed_format_groups=int(mixed),eos=immediate,nonzero=int(z['unclipped_grad_norm']>0)))
            maxerr=max(maxerr,z['max_logp_error'])
            groups.append(dict(arm=arm,step=i+1,image=z['image'],format_candidates=hits,nonempty=has_content,format_changes_advantage=int(mixed),first_eos=immediate,gradient=z['unclipped_grad_norm'],replay_max_error=z['max_logp_error']))
        cp={}
        for step in (6,12,24):
            folder=p/f'step_{step:02d}';f=folder/'adapter/adapter_model.safetensors';ref=read(folder/'update.json')
            assert sha(f)==ref['adapter_hash'] and ref['step']==step and ref['initial_params']==update['initial_params']
            with safe_open(str(init),framework='pt',device='cpu') as aa,safe_open(str(f),framework='pt',device='cpu') as bb:
                assert aa.keys()==bb.keys()
                changed=sum(not torch.equal(aa.get_tensor(k),bb.get_tensor(k)) for k in aa.keys())
            cp[str(step)]={**ref,'changed_tensors':changed,'tensors':len(state['weights']),'bytes':f.stat().st_size}
        checkpoints[arm]=dict(steps=cp,resume_sha256=sha(p/'resume.pt'),resume_bytes=(p/'resume.pt').stat().st_size)
        audit[arm]=dict(count,groups=24,trajectories=192,action_tokens=tokens,max_replay_error=maxerr)
        all_rows[arm]=steps
        del state
    paired={}
    for s in ('a','b'):
        off,on=all_rows[s+'_off'],all_rows[s+'_on']
        first=next((i for i,(x,y) in enumerate(zip(off,on)) if len(set(x['format_score']))>1 or len(set(y['format_score']))>1),None)
        end=24 if first is None else first+1
        assert all(all(off[i][key]==on[i][key] for key in ('traces','caption_and_hard_margins','semantic_midrank','format_score')) for i in range(end))
        before_signal=24 if first is None else first
        assert all(off[i]['active_params']==on[i]['active_params'] for i in range(before_signal))
        first_params=next((i+1 for i,(a,b) in enumerate(zip(off,on)) if a['active_params']!=b['active_params']),None)
        first_tokens=next((i+1 for i,(a,b) in enumerate(zip(off,on)) if [[q['tokens'] for q in t['rounds']] for t in a['traces']]!=[[q['tokens'] for q in t['rounds']] for t in b['traces']]),None)
        paired[s]=dict(first_differential_format_group=None if first is None else first+1,identical_candidate_groups_through_first_exposure=end,
                       identical_parameter_updates_before_signal=before_signal,first_different_trained_weights=first_params,first_different_sampled_tokens=first_tokens)
    matches=[all(x[k]==y[k] for k in archive_keys) for x,y in zip(all_rows['a_off'],historical,strict=True)]
    a_off=read(ROOT/'policy_a_off/update.json');old=read(prev/'update.json')
    ref=dict(identical_original_groups=sum(matches),groups=24,active_params_match=a_off['final_params']==old['final_params'],same_initial_params=a_off['initial_params']==old['initial_params'])
    assert ref['same_initial_params']
    dump(ROOT/'results/training_checks.json',dict(arms=audit,paired=paired,previous_hard_replication=ref))
    dump(ROOT/'results/checkpoint_index.json',checkpoints)
    csvwrite(ROOT/'results/training_groups.csv',groups)

if __name__=='__main__':main()
