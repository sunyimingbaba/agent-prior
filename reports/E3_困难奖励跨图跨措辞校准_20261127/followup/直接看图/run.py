"""Query a frozen R0 with the same explicit mirrored D alternatives on both images."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import torch
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2]
POOL=BASE/'hard_reward_calibration_20261127/followup_pool'
SRC=BASE/'second_answer_5090_20261124'
sys.path.insert(0,str(SRC/'code'))
from shared import setup,rgb,table,read,sha
from policy import policy

def save(path,obj):
    part=path.with_suffix('.part')
    part.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    part.replace(path)

def digest(x):return hashlib.sha256(x).hexdigest()

@torch.inference_mode()
def main():
    rows=read(POOL/'inputs/rows.json');proof=read(POOL/'d_complete.json')
    groups=defaultdict(list)
    for row in rows:
        if proof['eligibility'][row['image']]:groups[row['pair']].append(row)
    groups={k:sorted(v,key=lambda z:z['side']) for k,v in sorted(groups.items()) if len(v)==2}
    assert len(rows)==118 and len(groups)==57
    setup(); model,proc=policy('R0'); dataset=table('train')
    labels=[proc.tokenizer.encode(t,add_special_tokens=False) for t in ('A','B')]
    assert all(len(x)==1 for x in labels) and labels[0]!=labels[1]
    ids=[x[0] for x in labels]
    assert [proc.tokenizer.decode([j]) for j in ids]==['A','B']
    results={}; counts={}
    for key,pair in groups.items():
        assert [r['side'] for r in pair]==[0,1] and pair[0]['D']==pair[1]['D'][::-1]
        baseline=pair[0]['D']
        entries=[]
        for row in pair:
            image=rgb(dataset,row);records=[]
            for order,options in enumerate((baseline,baseline[::-1])):
                text=f"Which statement describes the image? A. {options[0]} B. {options[1]} Answer with A or B."
                message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
                prompt=proc.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
                dat=proc(text=[prompt],images=[image],return_tensors='pt').to('cuda')
                logit=model(**dat,use_cache=False,return_dict=True).logits[0,-1].float()
                a,b=logit[ids].tolist()
                winner=int(logit.argmax())
                records.append(dict(order=order,prompt=prompt,prompt_tokens_sha256=digest(dat.input_ids.cpu().numpy().tobytes()),
                    logit_A=a,logit_B=b,top_token=winner,top_text=proc.tokenizer.decode([winner])))
            margin=((records[0]['logit_A']-records[0]['logit_B'])-(records[1]['logit_A']-records[1]['logit_B']))/2.
            entries.append(dict(image=row['image'],side=row['side'],attribute=row['attribute'],rgb=row['rgb_sha256'],
                                options=baseline,observations=records,margin=margin))
        for c in (0,1):
            for fld in ('prompt','prompt_tokens_sha256'):
                assert entries[0]['observations'][c][fld]==entries[1]['observations'][c][fld]
        results[key]=entries
        print(len(results),key,flush=True)
    save(ROOT/'逐图原始logits.json',results)
    save(ROOT/'回执.json',dict(pairs=len(results),images=2*len(results),forward_passes=4*len(results),
        label_ids=ids,gpu=torch.cuda.get_device_name(),version=torch.__version__,adapter=sha(SRC/'R0/adapter/adapter_model.safetensors'),
        source_inputs=sha(POOL/'inputs/rows.json'),eligible=sha(POOL/'d_complete.json'),source=sha(Path(__file__)),
        protocol=sha(ROOT/'口径.md'),raw=sha(ROOT/'逐图原始logits.json'),same_prompt_and_tokens=2*len(results)))

if __name__=='__main__':main()
