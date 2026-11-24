import base64,os,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from base import RUN,PREVIOUS,read,dump,sha,table,rgb,setup,model_path,weight,savept
from shared import BASE
from policy import policy
from train import sample,finish,sequence_logp,matched_scores
from agent.grpo.reward import r_format
def run(seed,arm):
    setup();assert arm in ('R2','R3')
    cpath=RUN/'pilot'/(seed+'_contexts.pt');ctx=torch.load(cpath,weights_only=True);cases=read(RUN/'pilot/train.json')
    kappa=read(RUN/'inputs/continuous_lock.json')['kappa'];src=weight(seed)
    m,proc=policy(model_path(seed));params=[p.requires_grad_() for n,p in m.named_parameters() if '.lora_' in n]
    opt=torch.optim.AdamW(params,lr=5e-6);t=table();dest=RUN/'pilot'/(seed+'_'+arm);dest.mkdir(exist_ok=True)
    worker=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(Path(__file__).with_name('pilot_worker.py'))],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    assert worker.stdout.readline().strip()=='READY';logs=[]
    try:
        for j,c in enumerate(cases):
            rng=(3511000 if seed=='a' else 4511000)+j;torch.manual_seed(rng);nm=c['image']
            prefix,tr=sample(m,proc,rgb(t,c),c['scene_query'],ctx[nm]);hs,extra=finish(m,proc,prefix,tr,ctx[nm]['answer'])
            hidden=torch.stack(hs);packet=base64.b64encode(hidden.float().numpy().tobytes()).decode()
            worker.stdin.write(json.dumps(dict(image=nm,hidden=packet))+'\n');worker.stdin.flush();sc=json.loads(worker.stdout.readline())
            rank,native,hard=(np.array([r[k] for r in sc]) for k in ('rank','native','A'))
            order=native if arm=='R2' else native+kappa*hard
            fmt=r_format(proc.decode(ctx[nm]['first'],skip_special_tokens=True));rw=fmt+matched_scores(order,1/(1+rank))
            adv=(rw-rw.mean())/(rw.std(ddof=1)+1e-8);opt.zero_grad(set_to_none=True);errs=[]
            for i,record in enumerate(tr):
                if abs(adv[i])>1e-10:
                    lp,err=sequence_logp(m,prefix,record);(-lp*float(adv[i])/len(tr)).backward();errs.extend(err)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            logs.append(dict(image=nm,step=j+1,rng=rng,ids=[r[0] for r in tr],third=extra,scored=sc,format=fmt,
                       reward=rw.tolist(),advantages=adv.tolist(),norm=norm,mean_abs_logp_err=float(np.mean(errs)) if errs else 0.))
            dump(dest/'steps.json',logs);print(seed,arm,j+1,nm,norm,flush=True)
    finally:
        worker.stdin.close();assert worker.wait()==0
    m.save_pretrained(dest/'adapter')
    dump(dest/'complete.json',dict(initial=src,final=sha(dest/'adapter/adapter_model.safetensors'),groups=len(logs),
            train=sha(RUN/'pilot/train.json'),contexts=sha(cpath),updated=sum(r['norm']>0 for r in logs),kappa=kappa if arm=='R3' else 0.))
if __name__=='__main__':run(*sys.argv[1:])
