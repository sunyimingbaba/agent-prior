"""Paired cold-start RFPO pilot with one eight-candidate group per image."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'dual_input_rfpo_20261126/code'))
from train_dual import setup,table,rgb,read,sha,savept,dump,initial,policy,trace_batch,advantages,replay,LEGACY


def main(seed,arm):
    receipt=ROOT/'inputs/pilot/decision.json';cfg=read(receipt);assert arm in cfg['train_arms']
    for rel,signature in cfg['lock'].items():assert sha(ROOT/rel)==signature
    setup();torch.set_num_threads(1)
    rows=read(ROOT/'inputs/pilot/train.json')[seed]
    start=initial(seed);sig=sha(start/'adapter/adapter_model.safetensors')
    assert sig==cfg['starts'][seed] and len(rows)==24
    dest=ROOT/'models'/(seed+'_'+arm);dest.mkdir(exist_ok=True,parents=True)
    if (dest/'complete.json').exists():
        cp=read(dest/'complete.json');assert cp['lock']==sha(receipt) and cp['final']==sha(dest/'adapter/adapter_model.safetensors')
        return
    model,processor=policy(start)
    params={k:p.requires_grad_() for k,p in model.named_parameters() if '.lora_' in k}
    opt=torch.optim.AdamW(params.values(),lr=1e-6)
    checkpoint=dest/'resume.pt';history=[]
    if checkpoint.exists():
        z=torch.load(checkpoint,map_location='cpu',weights_only=True);assert z['lock']==sha(receipt)
        for k,p in params.items():p.data.copy_(z['weights'][k].to(p.device))
        opt.load_state_dict(z['opt']);history=z['history'];del z
    worker=subprocess.Popen([str(LEGACY/'env/bin/python'),'-Bu',str(ROOT/'code/policy_worker.py'),seed],
             env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert worker.stdout.readline().strip()=='READY'
        tb=table()
        for j,r in enumerate(rows):
            if j<len(history):continue
            opt.zero_grad(set_to_none=True)
            rng=(91000000 if seed=='a' else 101000000)+j*1000
            ctx,traces,hs=trace_batch(model,processor,rgb(tb,r),r['orig_query'],r['feedback'],rng)
            buf=base64.b64encode(torch.stack(hs).float().numpy().tobytes()).decode()
            worker.stdin.write(json.dumps(dict(image=r['image'],data=buf))+'\n');worker.stdin.flush()
            score=json.loads(worker.stdout.readline());scores=np.zeros(8)
            scores[score['winners'][arm]]=1.
            adv=advantages(scores)
            # The reused replay routine divides by 16 for a two-group update.
            errors=replay(model,ctx,traces,adv*2.)
            norm=float(torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            history.append(dict(image=r['image'],step=j+1,query=r['orig_query'],rng=rng,norm=norm,
                                **score,paths=[[rec[1][0] for rec in path] for path in traces],
                                states=__import__('hashlib').sha256(base64.b64decode(buf)).hexdigest(),
                                advantages=adv.tolist(),mean_replay_error=float(np.mean(errors)) if errors else 0.,max_replay_error=max(errors,default=0.)))
            if (j+1)%4==0:
                savept(checkpoint,dict(lock=sha(receipt),weights={k:p.detach().cpu() for k,p in params.items()},opt=opt.state_dict(),history=history))
            del ctx,traces,hs
            print(seed,arm,j+1,norm,flush=True)
    finally:
        worker.stdin.close();worker.wait()
    assert worker.returncode==0
    model.save_pretrained(dest/'adapter')
    dump(dest/'steps.json',history)
    dump(dest/'complete.json',dict(lock=sha(receipt),initial=sig,steps=len(history),nonzero=sum(r['norm']>0 for r in history),
                                    final=sha(dest/'adapter/adapter_model.safetensors'),scores=sha(dest/'steps.json')))

if __name__=='__main__':main(*sys.argv[1:])
