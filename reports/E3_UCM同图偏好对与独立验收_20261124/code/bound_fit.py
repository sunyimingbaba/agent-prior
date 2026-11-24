"""On-policy terminal hidden-state gradient through the fixed CUP readout; matched balanced image pairs."""
import base64,json,os,sys,subprocess
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,torch.nn.functional as F
from env_pair import ROOT,OLD,REBUILD,model_path,weight,read,dump,sha,setup,table,rgb,inputs,rows,hashkey
from shared import BASE
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
def run(seed,arm):
    setup();assert arm=='bounded';m,proc=policy(model_path(seed));original=m.generate
    dest=ROOT/'pilot'/(seed+'_'+arm);dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        assert sha(dest/'adapter/adapter_model.safetensors')==read(dest/'complete.json')['final'];return
    params=[p.requires_grad_() for n,p in m.named_parameters() if '.lora_' in n];opt=torch.optim.AdamW(params,lr=1e-6)
    allrows=rows();t=table();pairs=defaultdict(list);answers={}
    for c in allrows.values():pairs[c['pair']].append(c)
    for sub in ('','confirm'):answers.update({n:a[0] for n,a in read(OLD/sub/'records/tools.json').items()})
    assert len(pairs)==84 and all(len(v)==2 and {c['side'] for c in v}=={0,1} and v[0]['scene_query']==v[1]['scene_query'] for v in pairs.values())
    kappa=read(OLD/'inputs/continuous_lock.json')['kappa']
    worker=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(Path(__file__).with_name('bound_reader.py')),'train',seed],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    assert worker.stdout.readline().strip()=='READY';steps=[]
    try:
        for j,key in enumerate([k for epoch in (0,1) for k in sorted(pairs,key=lambda k:hashkey('continuous',seed,k) if epoch==0 else hashkey('bounded-repeat',seed,k))]):
            entries=[];opt.zero_grad(set_to_none=True)
            for c in sorted(pairs[key],key=lambda x:x['side']):
                nm=c['image'];im=rgb(t,c)
                for field in ('scene_query','orig_query'):
                    ids=[];rounds=[];calls=[]
                    def capture(**kw):
                        z=original(**kw);act=z.sequences[0,kw['input_ids'].shape[1]:].tolist()
                        rounds.append(act);ids[:]=kw['input_ids'][0].tolist()+act[:-1];return z
                    m.generate=capture
                    try:
                        with torch.no_grad():gen,n,f=rollout_one(m,proc,im,c[field],{'scene_classify':lambda _:calls.append(answers[nm]) or answers[nm]},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                    finally:m.generate=original
                    assert n==len(calls) and f==0
                    inp=inputs(proc,im,c[field],torch.tensor(ids)[None])
                    z=m(**inp,return_dict=True,output_hidden_states=True,use_cache=False,logits_to_keep=1)
                    h=z.hidden_states[-1][0,-1].float();cos=float(F.cosine_similarity(h.detach()[None],gen.cuda().float()[None]));assert cos>.98
                    # Score exactly the greedily generated prior. Use the teacher-forced position for its derivative.
                    packet=base64.b64encode(gen.float().cpu().numpy().tobytes()).decode()
                    hard=2*kappa if field=='scene_query' else 0.
                    worker.stdin.write(json.dumps(dict(image=nm,field=field,data=packet,hard=hard))+'\n');worker.stdin.flush()
                    out=json.loads(worker.stdout.readline());g=np.frombuffer(base64.b64decode(out.pop('grad')),dtype=np.float32).copy()
                    h.backward(torch.from_numpy(g).cuda());entries.append(dict(image=nm,field=field,rounds=rounds,tools=calls,cos=cos,hard_weight=hard,**out))
                    del z,h,inp
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm>0:opt.step()
            steps.append(dict(pair=key,step=j+1,records=entries,norm=norm));dump(dest/'steps.json',steps)
            if j%10==0:print(seed,arm,j+1,flush=True)
    finally:worker.stdin.close();assert worker.wait()==0
    m.save_pretrained(dest/'adapter');dump(dest/'complete.json',dict(initial=weight(seed),final=sha(dest/'adapter/adapter_model.safetensors'),steps=len(steps),kappa=kappa,source=sha(Path(__file__)),reader=sha(Path(__file__).with_name('bound_reader.py')),ref=sha(ROOT/'records'/(seed+'_bound_ref.json'))))
if __name__=='__main__':run(*sys.argv[1:])
