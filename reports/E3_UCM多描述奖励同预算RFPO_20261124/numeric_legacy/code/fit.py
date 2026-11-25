import sys,subprocess,os,base64,json
from pathlib import Path
import numpy as np,torch
from runtime import ROOT,OLD,BASE,pool,table,rgb,setup,initial,read,dump,sha
from prefixes import greedy
from policy import policy
from train import sample,finish,sequence_logp,matched_scores
from agent.grpo.reward import r_format
from orders import score

def main(seed,arm):
    setup();assert arm in ('R2','R3') and seed in 'ab'
    roster=ROOT/'inputs/roster.json';plan=read(roster);ctxfile=ROOT/'records'/(seed+'_contexts.pt')
    assert sha(ctxfile)==plan['contexts'][seed] and sha(ROOT/'实验口径.md')==plan['protocol']
    seedpath=initial(seed);init=sha(seedpath/'adapter/adapter_model.safetensors');assert init==plan['initial'][seed]
    tag=seed+'_'+arm;dest=ROOT/'pilot'/tag;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        cp=read(dest/'complete.json');assert cp['initial']==init and cp['roster']==sha(roster) and cp['final']==sha(dest/'adapter/adapter_model.safetensors');return
    ctx=torch.load(ctxfile,weights_only=True)['rows'];m,proc=policy(seedpath)
    params=[p.requires_grad_() for n,p in m.named_parameters() if '.lora_' in n];opt=torch.optim.AdamW(params,lr=5e-6)
    rows=pool();t=table();kappa=read(OLD/'inputs/continuous_lock.json')['kappa'];logs=[]
    worker=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(Path(__file__).with_name('score_worker.py'))],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert worker.stdout.readline().strip()=='READY'
        for j,nm in enumerate(plan['order']):
            c=rows[nm];im=rgb(t,c);hd,actions,calls,_=greedy(m,proc,im,c['scene_query'],c['feedback'])
            draws=[];hs=[];rngs=[];stored=[]
            for batch in range(2):
                rng=(1871000 if seed=='a' else 2871000)+j*2+batch;rngs.append(rng);torch.manual_seed(rng)
                single,trs=sample(m,proc,im,c['scene_query'],ctx[nm])
                states,extras=finish(m,proc,single,trs,c['feedback']);hs+=states;draws.append((single,trs))
                stored+=[dict(ids=tr[0],sampled_logp=tr[2],third=e) for tr,e in zip(trs,extras)]
            packet=base64.b64encode(torch.stack(hs+[hd]).float().numpy().tobytes()).decode()
            worker.stdin.write(json.dumps(dict(image=nm,data=packet))+'\n');worker.stdin.flush()
            sc=json.loads(worker.stdout.readline());assert len(sc)==17;anchor=sc.pop()
            rw,floor=score(sc,kappa)
            nr=np.array([z['native'] for z in sc]);pr=1/(1+np.array([z['rank'] for z in sc]))
            assert np.array_equal(rw['R2'],matched_scores(nr,pr))
            fmt=float(r_format(proc.decode(ctx[nm]['first'],skip_special_tokens=True)))
            total=rw[arm]+fmt;adv=(total-total.mean())/(total.std(ddof=1)+1e-8)
            opt.zero_grad(set_to_none=True);errs=[]
            for b,(single,trs) in enumerate(draws):
                for i,tr in enumerate(trs):
                    a=adv[8*b+i]
                    if abs(a)>1e-10:
                        lp,err=sequence_logp(m,single,tr);(-lp*float(a)/len(adv)).backward();errs.extend(err)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            logs.append(dict(step=j+1,image=nm,rng=rngs,records=stored,scores=sc,anchor=dict(scores=anchor,actions=actions,tools=calls),first_matches_current=ctx[nm]['first']==actions[0],
                             format=fmt,retrieval={k:v.tolist() for k,v in rw.items()},floor=floor,advantages=adv.tolist(),norm=norm,mean_abs_logp_error=float(np.mean(errs)) if errs else 0.))
            # One durable step per line; the final checkpoint alone is evaluated.
            mode='w' if j==0 else 'a'
            with (dest/'steps.jsonl').open(mode) as f:f.write(json.dumps(logs[-1],allow_nan=False)+'\n')
            if (j+1)%10==0:print(tag,j+1,len(plan['order']),flush=True)
    finally:worker.stdin.close();worker.wait()
    assert worker.returncode==0 and len(logs)==len(plan['order'])
    m.save_pretrained(dest/'adapter')
    dump(dest/'complete.json',dict(initial=init,final=sha(dest/'adapter/adapter_model.safetensors'),roster=sha(roster),steps=len(logs),steps_sha256=sha(dest/'steps.jsonl'),
         updated=sum(x['norm']>0 for x in logs),kappa=kappa,sources={x:sha(Path(__file__).with_name(x)) for x in ('fit.py','prefixes.py','runtime.py','orders.py','score_worker.py')}))
if __name__=='__main__':main(*sys.argv[1:])
