"""Paired full-policy sampling under scene-only and attribute-question prompts."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
import torch

ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'multicap_rfpo_20261124'
sys.path.insert(0,str(PREV/'code'))
from runtime import BASE,OLD,read,dump,sha,setup,table,rgb,pool,savept
from eval_io import weights
from policy import policy,prompt,cut
from rollout import parse_tool_call
from prefixes import greedy
from orders import score

def collect(model,proc,inputs,slots,seed,feedback):
    paths={i:[] for i in slots};states={};pending=[(slots,inputs.input_ids)]
    for turn in range(3):
        following=[]
        for group,prefix in pending:
            torch.manual_seed(seed+turn*11+min(group))
            features=dict(inputs,input_ids=prefix,attention_mask=torch.ones_like(prefix))
            batched={k:v.repeat(len(group),*([1]*(v.ndim-1))) for k,v in features.items()}
            z=model.generate(**batched,max_new_tokens=64,do_sample=True,temperature=1.,
                output_hidden_states=True,return_dict_in_generate=True)
            grouped=defaultdict(list)
            for j,slot in enumerate(group):
                ids=cut(z.sequences[j,prefix.shape[1]:].tolist(),model.generation_config.eos_token_id)
                paths[slot].append(ids)
                answer=proc.decode(ids,skip_special_tokens=True)
                if turn<2 and parse_tool_call(answer):
                    grouped[answer].append(slot)
                else:
                    states[slot]=z.hidden_states[len(ids)-1][-1][j,-1].cpu().clone()
            del z
            for action,members in grouped.items():
                add=proc(text=[action+f'\n<result>{feedback}</result>\n'],return_tensors='pt').input_ids.to(prefix.device)
                following.append((members,torch.cat([prefix,add],dim=1)))
        pending=following
    assert len(states)==len(slots) and not pending
    return [states[i] for i in slots],[paths[i] for i in slots]

@torch.inference_mode()
def main(seed):
    setup()
    plan=read(ROOT/'inputs/lock.json')
    assert sha(ROOT/'实验口径.md')==plan['protocol'] and sha(ROOT/'inputs/questions.json')==plan['questions']
    for sub,digest in plan['panels'].items():assert sha(OLD/sub/'inputs/panel.json')==digest
    cases=pool();data=table();path,sig=weights(seed+'_deployed');model,proc=policy(path)
    cfg=model.generation_config
    assert cfg.top_k==50 and cfg.top_p==1 and cfg.repetition_penalty==1.05
    k=read(OLD/'inputs/continuous_lock.json')['kappa'];folder=ROOT/'records'/seed;folder.mkdir(exist_ok=True)
    worker=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(PREV/'code/score_worker.py')],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert worker.stdout.readline().strip()=='READY'
        for j,row in enumerate(plan['rows']):
            nm=row['image'];c=cases[nm];image=rgb(data,c)
            assert row['rgb']==c['rgb_sha256'] and row['scene']==c['scene_query'] and row['attribute']==c['attribute']
            dest=folder/(nm+'.pt');rec=folder/(nm+'.json')
            if rec.exists():
                assert read(rec)['states']==sha(dest) and read(rec)['adapter']==sig
                continue
            all_states={};records={}
            for mode,query in [('scene',row['scene']),('question',row['scene']+' '+row['question'])]:
                anchor,actions,calls,_=greedy(model,proc,image,query,c['feedback'])
                inp=proc(text=[prompt(proc,query)],images=[image],return_tensors='pt').to('cuda')
                hs=[];draws=[]
                for batch in range(2):
                    rng=(10000000 if seed=='a' else 20000000)+j*1000+100*batch
                    states,paths=collect(model,proc,inp,list(range(8)),rng,c['feedback'])
                    hs+=states;draws+=paths
                h=torch.stack(hs+[anchor]);assert h.shape==(17,3584) and h.isfinite().all()
                all_states[mode]=h;packet=base64.b64encode(h.float().numpy().tobytes()).decode()
                worker.stdin.write(json.dumps(dict(image=nm,data=packet))+'\n');worker.stdin.flush()
                sc=json.loads(worker.stdout.readline());assert len(sc)==17
                reward,floor=score(sc[:-1],k)
                records[mode]=dict(query=query,actions=draws,greedy=actions,greedy_tools=calls,scores=sc,floor=floor,
                    retrieval={key:v.tolist() for key,v in reward.items()})
            savept(dest,all_states)
            dump(rec,dict(adapter=sig,image=nm,rgb=row['rgb'],feedback=c['feedback'],lock=sha(ROOT/'inputs/lock.json'),states=sha(dest),groups=records))
            print(seed,j+1,len(plan['rows']),flush=True)
    finally:
        worker.stdin.close();worker.wait()
    dump(folder/'complete.json',dict(adapter=sig,source=sha(Path(__file__)),lock=sha(ROOT/'inputs/lock.json'),
        records={r['image']:sha(folder/(r['image']+'.json')) for r in plan['rows']}))
if __name__=='__main__':main(sys.argv[1])
