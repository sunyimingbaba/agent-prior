"""Audit each saved adapter and all 72 group updates against the original tokens, sampling logs, weights and replayed first group."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from safetensors import safe_open
from shared import ROOT,ASSETS,read,dump,sha
def main():
    r=read(ROOT/'inputs/train.json');order=[x['image'] for x in r];probe=read(ROOT/'probe/scores.json')['cases'];result={}
    epochs=[];initial=set()
    for arm in ('old','caption','hard'):
        out=ROOT/('policy_'+arm);u=read(out/'update.json');st=read(out/'steps.json');assert len(st)==len(order)==u['steps']==24 and u['order']==order
        trained=out/'adapter/adapter_model.safetensors';base=ASSETS/'ucm_main_lora/adapter_model.safetensors'
        assert sha(trained)==u['adapter_hash'] and u['initial_params']!=u['final_params']
        initial.add(u['initial_params']);change=[];absolute=0;norm=0;dtype=set()
        with safe_open(str(base),framework='pt',device='cpu') as x,safe_open(str(trained),framework='pt',device='cpu') as y:
            assert x.keys()==y.keys()
            for key in x.keys():
                a,b=x.get_tensor(key),y.get_tensor(key);dtype.add(str(b.dtype));diff=(a.double()-b.double())
                assert torch.isfinite(b).all();change.append(bool(torch.count_nonzero(diff)));absolute=max(absolute,float(diff.abs().max()))
                norm+=float(diff.square().sum())
        ep=0.;tokens=0;steps=0
        for i,z in enumerate(st):
            assert z['image']==order[i] and z['step']==i+1 and len(z['traces'])==len(z['rewards'])==8
            rewards=np.array(z['rewards']);rr_cpu=torch.tensor(z['rewards'],dtype=torch.float32)
            ref=((rr_cpu-rr_cpu.mean())/(rr_cpu.std()+1e-8)).numpy()
            assert np.max(np.abs(ref-z['advantages']))<5e-5
            for k,tr in enumerate(z['traces']):
                assert tr['seed']==284000+i*8+k
                if i==0:
                    assert [q['tokens'] for q in tr['rounds']]==probe[k]['token_ids'] and abs(rewards[k]-probe[k][arm])<1e-10
                for q in tr['rounds']:
                    assert len(q['tokens'])==len(q['logp']) and np.isfinite(q['logp']).all()
                    assert all(t not in (151645,151643) for t in q['tokens'][:-1]);tokens+=len(q['tokens'])
            if arm!='old':
                a,b=np.array(z['caption_and_hard_margins']).T;rr=rankdata(a) if arm=='caption' else (rankdata(a)+rankdata(b))/2
                assert np.array_equal(rr,rewards)
            ep=max(ep,z['max_logp_error']);steps+=int(z['unclipped_grad_norm']>0)
            epochs.append(dict(arm=arm,step=i+1,image=z['image'],reward_min=float(min(rewards)),reward_max=float(max(rewards)),
                               actions=8,grad_norm=z['unclipped_grad_norm'],logprob_replay_error=z['max_logp_error']))
        result[arm]=dict(steps=24,nonzero_updates=steps,samples=192,tokens=tokens,max_replay_error=ep,changed_tensors=sum(change),total_tensors=len(change),
                    parameter_l2_delta=norm**.5,max_parameter_change=absolute,dtype=sorted(dtype),adapter_hash=sha(trained),initial_file_sha=sha(base),
                    step1_all_tokens_and_rewards_match=True)
    assert len(initial)==1 and len(set(z['adapter_hash'] for z in result.values()))==3
    (ROOT/'results').mkdir(exist_ok=True)
    dump(ROOT/'results/training_audit.json',result);dump(ROOT/'results/training_groups.json',epochs)
if __name__=='__main__':main()
