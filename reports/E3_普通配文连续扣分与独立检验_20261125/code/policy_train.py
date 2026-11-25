"""Cold-start RFPO on first captions with the selected reward and two matched controls."""
from pathlib import Path
import base64,json,os,subprocess,sys
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;OLD_RUN=BASE/'source_transfer_2h_20261125'
sys.path.insert(0,str(OLD_RUN/'code'))
from remote_common import setup,table,rgb,read,dump,sha,savept
from runtime import initial,pool,OLD,BASE as LEGACY
from policy import policy
from full_traces import trace_batch
from prefixes import greedy
from train import sequence_logp
from orders import score,calibrate,advantages

def main(seed,arm):
    assert arm in ('R2','old','new') and seed in 'ab'
    receipt=ROOT/'results/real_confirmation.json';decision=read(receipt)
    assert decision['eligible'] and decision['nomination']==sha(ROOT/'inputs/multi_nomination.json') and decision['plan']==sha(ROOT/'inputs/real_lock.json')
    plan=read(OLD_RUN/'inputs/training_roster.json')['rows'];assert len(plan)==24 and {r['rgb'] for r in plan}.isdisjoint(z['rgb_sha256'] for z in read(ROOT/'inputs/real_roster.json'))
    setup();torch.set_num_threads(1);dest=ROOT/'policy'/(seed+'_'+arm);dest.mkdir(exist_ok=True,parents=True)
    start=initial(seed);initial_sha=sha(start/'adapter/adapter_model.safetensors')
    if (dest/'complete.json').exists():assert read(dest/'complete.json')['final']==sha(dest/'adapter/adapter_model.safetensors');return
    m,proc=policy(start);pars={k:p.requires_grad_() for k,p in m.named_parameters() if '.lora_' in k}
    opt=torch.optim.AdamW(pars.values(),lr=5e-6);t=table();cases=pool();k=read(OLD/'inputs/continuous_lock.json')['kappa']
    worker=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u',str(ROOT/'code/policy_worker.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    history=[];state=dest/'resume.pt'
    if state.exists():
        z=torch.load(state,weights_only=True,map_location='cpu');assert z['initial']==initial_sha and z['decision']==sha(receipt)
        for name,p in pars.items():p.data.copy_(z['weights'][name].to(p.device))
        opt.load_state_dict(z['opt']);history=z['history']
    try:
        assert worker.stdout.readline().strip()=='READY'
        for j,c0 in enumerate(plan):
            if j<len(history):continue
            c=cases[c0['image']];assert c['rgb_sha256']==c0['rgb']
            im=rgb(t,c);query=c['orig_query'];fb=c['feedback'];anchor,aa,ac,_=greedy(m,proc,im,query,fb)
            batches=[];states=[];rng=[];paths=[]
            for kk in (0,1):
                num=(71000000 if seed=='a' else 81000000)+j*1000+kk*100;rng.append(num)
                ctx,hs,z=trace_batch(m,proc,im,query,fb,num);batches.append((ctx,hs));states+=z
                paths += [[rec[1][0] for rec in trace] for trace in hs]
            packet=base64.b64encode(torch.stack(states+[anchor]).float().numpy().tobytes()).decode()
            worker.stdin.write(json.dumps(dict(image=c['image'],data=packet))+'\n');worker.stdin.flush()
            sc=json.loads(worker.stdout.readline());records=[{k:sc[k][i] for k in ('rank','native','A','keep')} for i in range(16)]
            rewards,floor=score(records,k);rew={'R2':rewards['R2'],'old':rewards['R3'],'new':calibrate(np.array(sc['order']),sc['rank'][:16])}
            adv=advantages(rew[arm]);opt.zero_grad(set_to_none=True);errors=[];terms=0
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
                rewards={k:v.tolist() for k,v in rew.items()},floor=floor,adv=adv.tolist(),norm=norm,replay_terms=terms,
                replayed=len(errors),mean_error=float(np.mean(errors)) if errors else 0.,max_error=max(errors,default=0.)))
            if (j+1)%4==0:
                savept(state,dict(initial=initial_sha,decision=sha(receipt),weights={k:v.detach().cpu() for k,v in pars.items()},opt=opt.state_dict(),history=history))
            print(seed,arm,j+1,norm,flush=True)
    finally:
        worker.stdin.close();worker.wait()
    assert worker.returncode==0 and len(history)==24;m.save_pretrained(dest/'adapter');dump(dest/'steps.json',history)
    dump(dest/'complete.json',dict(initial=initial_sha,final=sha(dest/'adapter/adapter_model.safetensors'),steps=len(history),updated=sum(r['norm']>0 for r in history),receipt=sha(receipt),logs=sha(dest/'steps.json')))
if __name__=='__main__':main(*sys.argv[1:])
