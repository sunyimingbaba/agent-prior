"""Paired full-trajectory RFPO pilot; only unlocked by the preregistered nomination."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from remote_common import ROOT,BASE,read,dump,sha,setup,table,rgb,savept
from runtime import initial,pool,OLD,BASE as LEGACY
from policy import policy
from orders import score,advantages,calibrate
from train import sequence_logp
from full_traces import trace_batch
from prefixes import greedy

def train(seed,condition,reward):
    setup();torch.set_num_threads(1)
    assert condition in ('scene','selected','locked','free','ordinal','resolutionbase') and reward in ('R2','R3') and seed in ('a','b')
    timing=condition in ('locked','free')
    resolution=condition in ('ordinal','resolutionbase')
    nomination=ROOT/'results'/('resolution_nomination.json' if resolution else ('prefix_nomination.json' if timing else 'nomination.json'))
    decision=read(nomination)
    if timing or resolution:
        assert decision['eligible']
        if timing:assert decision['protocol']==sha(ROOT/'首轮探索复核口径.md')
        else:assert reward=='R3' and decision['protocol']==sha(ROOT/'奖励并列与训练信号复核口径.md')
        if (ROOT/'logs/cross_first').is_file():
            while not (ROOT/'logs/cross_gpu.done').is_file():
                if time.time()>=float(os.environ.get('E3_END_EPOCH','inf')):raise SystemExit(75)
                time.sleep(3)
    else:
        assert decision['selected'] in ('mixed','question') and decision['protocol']==sha(ROOT/'实验口径.md')
    plan=read(ROOT/'inputs/training_roster.json')
    assert plan['protocol']==sha(ROOT/'入围后同预算短训口径.md')
    dest=ROOT/'pilot'/(seed+'_'+condition+'_'+reward);dest.mkdir(exist_ok=True,parents=True)
    p=initial(seed);start=sha(p/'adapter/adapter_model.safetensors')
    if (dest/'complete.json').exists():
        cp=read(dest/'complete.json');assert cp['initial']==start and cp['final']==sha(dest/'adapter/adapter_model.safetensors');return
    model,proc=policy(p);params={k:v.requires_grad_() for k,v in model.named_parameters() if '.lora_' in k}
    opt=torch.optim.AdamW(params.values(),lr=5e-6)
    state=dest/'resume.pt';logs=[];roster_sig=sha(ROOT/'inputs/training_roster.json')
    if state.exists():
        saved=torch.load(state,map_location='cpu',weights_only=True)
        assert saved['roster']==roster_sig and saved['initial']==start
        for k,v in params.items():v.data.copy_(saved['weights'][k].to(v.device))
        opt.load_state_dict(saved['optimizer']);logs=saved['logs']
    cases=pool();data=table();k=read(OLD/'inputs/continuous_lock.json')['kappa']
    worker=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u',str(BASE/'multicap_rfpo_20261124/code/score_worker.py')],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert worker.stdout.readline().strip()=='READY'
        for j,row in enumerate(plan['rows']):
            if j<len(logs):continue
            if time.time() >= float(os.environ.get('E3_END_EPOCH','inf')):
                if logs:
                    savept(state,dict(initial=start,roster=roster_sig,weights={k:v.detach().cpu() for k,v in params.items()},
                                      optimizer=opt.state_dict(),logs=logs))
                dump(dest/'window_end.json',dict(completed_steps=len(logs),roster=roster_sig))
                raise SystemExit(75)
            nm=row['image'];c=cases[nm];assert c['rgb_sha256']==row['rgb'];im=rgb(data,c)
            source=decision['selected'] if condition=='selected' else 'scene'
            prompts=['scene','scene'] if source=='scene' else (['question','question'] if source=='question' else
                (['scene','question'] if j%2==0 else ['question','scene']))
            batches=[];states=[];rngs=[];actions=[];sampled_tokens=0
            fixed=(greedy(model,proc,im,c['scene_query'],c['feedback']) if condition=='locked' else None)
            for n,mode in enumerate(prompts):
                query=c['scene_query']+(' '+row['question'] if mode=='question' else '')
                rng=(13100000 if seed=='a' else 23100000)+1000*j+100*n
                if fixed is not None and fixed[3] is None:
                    ctx={};paths=[[] for _ in range(8)];hs=[fixed[0]]*8
                else:
                    prefix=None if fixed is None else fixed[3]
                    ctx,paths,hs=trace_batch(model,proc,im,query,c['feedback'],rng,initial_prefix=prefix)
                batches.append((ctx,paths));states.extend(hs);rngs.append(rng)
                for trace in paths:
                    ids=[z[1][0] for z in trace];sampled_tokens+=sum(map(len,ids))
                    actions.append(ids if fixed is None else [fixed[1][0]]+ids)
            # The worker expects 17 slots; its extra diagnostic slot is excluded before reward construction.
            packet=base64.b64encode(torch.stack(states+[states[0]]).float().numpy().tobytes()).decode()
            worker.stdin.write(json.dumps(dict(image=nm,data=packet))+'\n');worker.stdin.flush()
            scores=json.loads(worker.stdout.readline());assert len(scores)==17 and scores[0]==scores[-1]
            rw,floor=score(scores[:16],k)
            if condition=='ordinal':
                native,hard,keep=(np.array([v[key] for v in scores[:16]]) for key in ('native','A','keep'))
                dense=np.unique(native+k*hard,return_inverse=True)[1]
                rw['original_R3']=rw['R3'];rw['R3']=calibrate(dense+17*(keep+1e-14>=floor),np.arange(16))
            adv=advantages(rw[reward]);opt.zero_grad(set_to_none=True)
            errors=[];used=0
            for n,(ctx,paths) in enumerate(batches):
                repeated={}
                for i,trace in enumerate(paths):
                    a=float(adv[8*n+i])
                    if abs(a)<1e-12:continue
                    for ci,rec in trace:
                        # Share an exact gradient term only if context, sampled tokens and support are identical.
                        key=(ci,tuple(rec[0]),tuple(tuple(v.tolist()) for v in rec[3]))
                        if key in repeated:repeated[key][0]+=a
                        else:repeated[key]=[a,ci,rec]
                for weight,ci,rec in repeated.values():
                    if abs(weight)<1e-12:continue
                    logp,err=sequence_logp(model,ctx[ci],rec)
                    (-logp*(weight/16)).backward();errors.extend(err);used+=1
            norm=float(torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            logs.append(dict(step=j+1,image=nm,prompts=prompts,rngs=rngs,paths=actions,scores=scores[:16],
                rewards={k:v.tolist() for k,v in rw.items()},floor=floor,advantages=adv.tolist(),grad_norm=norm,
                sampled_tokens=sampled_tokens,unique_backward_terms=used,replay_tokens=len(errors),mean_abs_logp_error=float(np.mean(errors)) if errors else 0.,
                max_abs_logp_error=max(errors,default=0.)))
            if (j+1)%2==0:
                savept(state,dict(initial=start,roster=roster_sig,weights={k:v.detach().cpu() for k,v in params.items()},
                                  optimizer=opt.state_dict(),logs=logs))
            print(dest.name,j+1,len(plan['rows']),norm,flush=True)
    finally:
        worker.stdin.close();worker.wait()
    assert worker.returncode==0 and len(logs)==len(plan['rows'])
    model.save_pretrained(dest/'adapter')
    dump(dest/'steps.json',logs)
    dump(dest/'complete.json',dict(initial=start,final=sha(dest/'adapter/adapter_model.safetensors'),steps=len(logs),
        updated=sum(z['grad_norm']>0 for z in logs),roster=roster_sig,nomination=sha(nomination),
        steps_sha=sha(dest/'steps.json'),entry=sha(ROOT/'code/train_rfpo.py'),sampler=sha(ROOT/'code/full_traces.py')))
if __name__=='__main__':
    from gpu_turn import lease
    with lease():train(*sys.argv[1:])
