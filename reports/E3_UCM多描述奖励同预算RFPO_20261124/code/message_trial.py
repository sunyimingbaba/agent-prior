"""Compare two encodings of identical real tool feedback on locked training images."""
import base64
import json
import os
import subprocess
from pathlib import Path
import torch
from runtime import ROOT,BASE,OLD,initial,pool,table,rgb,setup,read,dump,savept,sha
from policy import policy,prompt
from train import sample,finish
from rollout import SYSTEM_PROMPT,parse_tool_call
from orders import score

def chat_turns(query):
    return [
        {'role':'system','content':SYSTEM_PROMPT},
        {'role':'user','content':[{'type':'text','text':'查询文本: There is a harbor with several ships.'}]},
        {'role':'assistant','content':'<tool_call>scene_classify</tool_call>'},
        {'role':'user','content':'<result>harbor (0.61), port (0.24), ship (0.08)</result>'},
        {'role':'assistant','content':'<answer>图中是一座港口，与查询文本相符。</answer>'},
        {'role':'user','content':[{'type':'image'},{'type':'text','text':f'查询文本: {query}'}]},
    ]

def append_feedback(turns,action,feedback):
    assert parse_tool_call(action)=='scene_classify'
    return turns+[{'role':'assistant','content':action},{'role':'user','content':f'<result>{feedback}</result>'}]

def render(proc,image,turns):
    text=proc.apply_chat_template(turns,tokenize=False,add_generation_prompt=True)
    return proc(text=[text],images=[image],return_tensors='pt').to('cuda')

def generate(model,inputs):
    result=model.generate(**inputs,max_new_tokens=64,do_sample=False,temperature=1.,
        output_hidden_states=True,return_dict_in_generate=True)
    ids=result.sequences[0,inputs.input_ids.shape[1]:].tolist()
    return ids,result.hidden_states[-1][-1][0,-1].cpu().clone()

def last_structured(model,proc,image,turns,draws,feedback):
    states=[];extras=[];repeats={}
    for ids,hidden,_,_,text in draws:
        if parse_tool_call(text):
            if text not in repeats:
                inputs=render(proc,image,append_feedback(turns,text,feedback))
                repeats[text]=generate(model,inputs)
            third,hidden=repeats[text]
        else:
            third=[]
        states.append(hidden);extras.append(third)
    return states,extras

@torch.inference_mode()
def main():
    setup()
    plan=read(ROOT/'inputs/message_lock.json')
    assert plan['protocol']==sha(ROOT/'工具回填对照口径.md') and plan['roster']==sha(ROOT/'inputs/roster.json')
    ctx=torch.load(ROOT/'records/a_contexts.pt',weights_only=True)
    model,proc=policy(initial('a'));data=table();cases=pool()
    kappa=read(OLD/'inputs/continuous_lock.json')['kappa']
    worker=subprocess.Popen([str(BASE/'env/bin/python'),'-u',str(Path(__file__).with_name('score_worker.py'))],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    outputs={};vectors={}
    try:
        assert worker.stdout.readline().strip()=='READY'
        for i,nm in enumerate(plan['images']):
            c=cases[nm];image=rgb(data,c);q=c['scene_query'];old=ctx['rows'][nm]
            first=proc.decode(old['first'],skip_special_tokens=True)
            turns=chat_turns(q)
            assert proc.apply_chat_template(turns,tokenize=False,add_generation_prompt=True)==prompt(proc,q)
            turns=append_feedback(turns,first,c['feedback'])
            orig=render(proc,image,chat_turns(q))
            raw=torch.cat([orig.input_ids,proc(text=[first+f"\n<result>{c['feedback']}</result>\n"],return_tensors='pt').input_ids.cuda()],dim=1)
            assert torch.equal(raw.cpu(),old['prefix'])
            outputs[nm]={};vectors[nm]={}
            for mode in ('raw','chat'):
                args=render(proc,image,turns) if mode=='chat' else type(orig)(dict(orig,input_ids=raw,attention_mask=torch.ones_like(raw)))
                prefix=args.input_ids.cpu()
                ids,h=generate(model,args);t=proc.decode(ids,skip_special_tokens=True)
                one=[(ids,h,[],None,t)]
                def terminal(inputs,draws):
                    if mode=='raw':return finish(model,proc,inputs,draws,c['feedback'])
                    return last_structured(model,proc,image,turns,draws,c['feedback'])
                h1,third=terminal(args,one)
                if mode=='raw':
                    assert torch.equal(h1[0],old['prior'])
                    assert [old['first'],ids]+([third[0]] if third[0] else [])==old['actions']
                states=[];records=[]
                for batch in range(2):
                    torch.manual_seed(40190200+2*i+batch)
                    arg,draws=sample(model,proc,image,q,dict(prefix=prefix))
                    hs,ex=terminal(arg,draws);states+=hs
                    records.extend(dict(ids=r[0],text=r[4],third=e) for r,e in zip(draws,ex,strict=True))
                states=torch.stack(states+h1);vectors[nm][mode]=states
                packet=base64.b64encode(states.float().numpy().tobytes()).decode()
                worker.stdin.write(json.dumps(dict(image=nm,data=packet))+'\n');worker.stdin.flush()
                sc=json.loads(worker.stdout.readline());assert len(sc)==17
                rewards,_=score(sc[:-1],kappa)
                outputs[nm][mode]=dict(first_text=first,feedback=c['feedback'],prefix=prefix[0].tolist(),records=records,
                    greedy=dict(ids=ids,text=t,third=third[0]),scores=sc,retrieval={k:v.tolist() for k,v in rewards.items()})
            print('paired',i+1,len(plan['images']),flush=True)
    finally:
        worker.stdin.close();worker.wait()
    savept(ROOT/'records/message_priors.pt',vectors)
    dump(ROOT/'records/message_trial.json',dict(lock=sha(ROOT/'inputs/message_lock.json'),adapter=ctx['adapter'],
        priors=sha(ROOT/'records/message_priors.pt'),cases=outputs))
if __name__=='__main__':main()
