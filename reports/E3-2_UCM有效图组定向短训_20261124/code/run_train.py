import base64,json,os,sys,subprocess
from pathlib import Path
import numpy as np,torch
from common import ROOT,OLD,ARMS,pool,model_path,weight,read,sha,dump,table,rgb,setup
from shared import BASE
from policy import policy
from train import sample,finish,sequence_logp,matched_scores
from agent.grpo.reward import r_format
def mixed(sc):
    return any(z['A']>0 and z['B']>0 for z in sc) and any(z['A']<0 and z['B']<0 for z in sc)
def run(seed,arm):
    setup();assert arm in ARMS
    roster_path=ROOT/'inputs/rosters.json';ctxfile=ROOT/'records'/(seed+'_contexts.pt')
    src=weight(seed);dest=ROOT/'pilot'/(seed+'_'+arm);dest.mkdir(exist_ok=True)
    if (dest/'complete.json').is_file():
        cp=read(dest/'complete.json');assert cp['initial']==src and cp['roster']==sha(roster_path) and sha(dest/'adapter/adapter_model.safetensors')==cp['final'];return
    ctx=torch.load(ctxfile,weights_only=True);rows=pool();t=table()
    cases=read(roster_path)[seed]['regular' if arm=='regular_R3' else 'selected']
    kappa=read(OLD/'inputs/continuous_lock.json')['kappa']
    m,proc=policy(model_path(seed));params=[p.requires_grad_() for n,p in m.named_parameters() if '.lora_' in n]
    opt=torch.optim.AdamW(params,lr=5e-6)
    w=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(Path(__file__).with_name('worker.py'))],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    assert w.stdout.readline().strip()=='READY';logs=[]
    try:
        for j,nm in enumerate(cases):
            c=rows[nm];contexts=ctx[nm];im=rgb(t,c);bs=[];hidden=[];batchlog=[]
            for b in range(3):
                rng=(7811000 if seed=='a' else 8811000)+j*3+b
                torch.manual_seed(rng);prefix,tr=sample(m,proc,im,c['scene_query'],contexts)
                hs,extra=finish(m,proc,prefix,tr,contexts['answer'])
                hidden.extend(hs);bs.append((prefix,tr));batchlog.append(dict(rng=rng,ids=[v[0] for v in tr],third=extra))
            packet=base64.b64encode(torch.stack(hidden).float().numpy().tobytes()).decode()
            w.stdin.write(json.dumps(dict(image=nm,hidden=packet))+'\n');w.stdin.flush()
            scores=json.loads(w.stdout.readline());assert len(scores)==24
            for b,g in enumerate(batchlog):
                g['scored']=scores[8*b:8*b+8];g['mixed']=mixed(g['scored'])
            use=next((b for b,g in enumerate(batchlog) if g['mixed']),0) if arm.startswith('focused') else 0
            sc=batchlog[use]['scored'];prefix,tr=bs[use]
            rank,native,A=(np.array([v[k] for v in sc]) for k in ('rank','native','A'))
            fmt=r_format(proc.decode(contexts['first'],skip_special_tokens=True))
            def rewards(order):return fmt+matched_scores(order,1/(1+rank))
            def advantages(r):return (r-r.mean())/(r.std(ddof=1)+1e-8)
            r2=rewards(native);r3=rewards(native+kappa*A);rw=r2 if arm.endswith('R2') else r3;adv=advantages(rw)
            opt.zero_grad(set_to_none=True);errs=[]
            for i,record in enumerate(tr):
                if abs(adv[i])>1e-10:
                    lp,err=sequence_logp(m,prefix,record);(-lp*float(adv[i])/len(tr)).backward();errs.extend(err)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            logs.append(dict(image=nm,step=j+1,batches=batchlog,picked=use,format=fmt,reward=rw.tolist(),advantages=adv.tolist(),reward_changes=bool(np.max(abs(advantages(r3)-advantages(r2)))>1e-8),norm=norm,mean_abs_logp_err=float(np.mean(errs)) if errs else 0.))
            dump(dest/'steps.json',logs);print(seed,arm,j+1,nm,use,norm,flush=True)
    finally:
        w.stdin.close();assert w.wait()==0
    m.save_pretrained(dest/'adapter')
    dump(dest/'complete.json',dict(initial=src,final=sha(dest/'adapter/adapter_model.safetensors'),groups=len(logs),roster=sha(roster_path),contexts=sha(ctxfile),updated=sum(r['norm']>0 for r in logs),kappa=kappa if arm.endswith('R3') else 0.))
if __name__=='__main__':run(*sys.argv[1:])
