"""Matched warm-start RFPO on uniformly sampled or independently flagged training images."""
from pathlib import Path
import base64,json,os,subprocess,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;OLD_RUN=BASE/'source_transfer_2h_20261125'
sys.path.insert(0,str(OLD_RUN/'code'))
from remote_common import setup,table,rgb,read,dump,sha,savept
from runtime import BASE as LEGACY
from eval_io import weights
from policy import policy
from full_traces import trace_batch
from prefixes import greedy
from train import sequence_logp
from orders import advantages
from exact_reward import exact_rewards

def main(seed,arm):
    group,objective=arm.split('_')
    assert arm in ('selected_ordinary','selected_joint','uniform_joint') and seed in 'ab'
    receipt=ROOT/'inputs/mined_lock.json';lock=read(receipt)
    assert lock['protocol']==sha(ROOT/'筛图后从现有R2继续训练.md') and lock['training']==sha(ROOT/'inputs/mined_roster.json')
    for name,expected in lock['implementation'].items():assert sha(ROOT/'code'/name)==expected
    plan=read(ROOT/'inputs/mined_roster.json')[seed+'_'+group];assert len(plan)==24
    setup();torch.set_num_threads(1);dest=ROOT/'mined_policy'/(seed+'_'+arm);dest.mkdir(exist_ok=True,parents=True)
    start,initial_sha=weights(seed+'_deployed')
    if (dest/'complete.json').exists():assert read(dest/'complete.json')['final']==sha(dest/'adapter/adapter_model.safetensors');return
    m,proc=policy(start);pars={k:p.requires_grad_() for k,p in m.named_parameters() if '.lora_' in k}
    opt=torch.optim.AdamW(pars.values(),lr=1e-6);t=table()
    worker=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u',str(ROOT/'code/mined_worker.py'),seed,group],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    history=[];state=dest/'resume.pt'
    if state.exists():
        z=torch.load(state,weights_only=True,map_location='cpu');assert z['initial']==initial_sha and z['decision']==sha(receipt)
        for name,p in pars.items():p.data.copy_(z['weights'][name].to(p.device))
        opt.load_state_dict(z['opt']);history=z['history']
    try:
        assert worker.stdout.readline().strip()=='READY'
        for j,c0 in enumerate(plan):
            if j<len(history):continue
            c=c0
            im=rgb(t,c);query=c['orig_query'];fb=c['feedback'];anchor,aa,ac,_=greedy(m,proc,im,query,fb)
            batches=[];states=[];rng=[];paths=[]
            for kk in (0,1):
                num=(91000000 if seed=='a' else 101000000)+j*1000+kk*100;rng.append(num)
                ctx,hs,z=trace_batch(m,proc,im,query,fb,num);batches.append((ctx,hs));states+=z
                paths += [[rec[1][0] for rec in trace] for trace in hs]
            packet=base64.b64encode(torch.stack(states+[anchor]).float().numpy().tobytes()).decode()
            worker.stdin.write(json.dumps(dict(image=c['image'],data=packet))+'\n');worker.stdin.flush()
            sc=json.loads(worker.stdout.readline());rew,joint=exact_rewards(sc)
            adv=advantages(rew[objective]);opt.zero_grad(set_to_none=True);errors=[];terms=0
            for kk,(ctx,hs) in enumerate(batches):
                ag={}
                for i,trace in enumerate(hs):
                    for ci,rec in trace:
                        key=(ci,tuple(rec[0]),tuple(tuple(v.tolist()) for v in rec[3]))
                        if key not in ag:ag[key]=[0.,ci,rec]
                        ag[key][0]+=float(adv[8*kk+i])
                for weight,ci,rec in ag.values():
                    if abs(weight)<1e-12:continue
                    lp,err=sequence_logp(m,ctx[ci],rec);(-weight*lp/16).backward();terms+=1;errors+=err
            norm=float(torch.nn.utils.clip_grad_norm_(pars.values(),1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            history.append(dict(image=c['image'],step=j+1,query=query,rng=rng,anchor=aa,calls=ac,paths=paths,scores=sc,
                rewards={k:v.tolist() for k,v in rew.items()},joint_fraction=joint,adv=adv.tolist(),norm=norm,replay_terms=terms,
                replayed=len(errors),mean_error=float(np.mean(errors)) if errors else 0.,max_error=max(errors,default=0.)))
            if (j+1)%4==0:
                savept(state,dict(initial=initial_sha,decision=sha(receipt),weights={k:v.detach().cpu() for k,v in pars.items()},opt=opt.state_dict(),history=history))
            print(seed,arm,j+1,norm,flush=True)
    finally:
        worker.stdin.close();worker.wait()
    assert worker.returncode==0 and len(history)==24;m.save_pretrained(dest/'adapter');dump(dest/'steps.json',history)
    dump(dest/'complete.json',dict(initial=initial_sha,final=sha(dest/'adapter/adapter_model.safetensors'),steps=len(history),updated=sum(r['norm']>0 for r in history),receipt=sha(receipt),logs=sha(dest/'steps.json')))
if __name__=='__main__':main(*sys.argv[1:])
