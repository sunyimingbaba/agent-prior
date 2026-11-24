import sys,hashlib
import torch
from runtime import ROOT,LAST,read,sha,dump,setup,rgb,table
from eval_io import weights
from policy import policy,prompt

@torch.inference_mode()
def main(tag):
    seed,arm=tag.split('_');assert arm in ('R2','R3')
    setup();p=ROOT/'inputs/probe_pairs.json';lock=read(ROOT/'inputs/probe_lock.json')
    assert sha(p)==lock['cases'] and sha(ROOT/'回答偏好复核口径.md')==lock['protocol']
    cases=read(p)[seed];meta={r['image']:r for r in read(LAST/'reward_confirm/inputs/panel.json')};t=table()
    path,sig=weights(tag);model,proc=policy(path);res=[]
    for case in cases:
        nm=case['image'];c=meta[nm];kwargs=proc(text=[prompt(proc,c['scene_query'])],images=[rgb(t,c)],return_tensors='pt').to('cuda')
        first=proc.decode(case['first'],skip_special_tokens=True)
        addition=proc(text=[first+'\n<result>'+case['tool']+'</result>\n'],return_tensors='pt').input_ids
        common=torch.tensor(case['common'],dtype=torch.long)[None]
        tokens=torch.cat((kwargs.input_ids.cpu(),addition,common),dim=1)
        kwargs['input_ids']=tokens.cuda();kwargs['attention_mask']=torch.ones_like(kwargs['input_ids'])
        logits=model(**kwargs,use_cache=False,return_dict=True,logits_to_keep=1).logits[0,-1].float()
        # Same repetition adjustment as sequence_logp; temperature 1. Top-k censoring is reported separately.
        seen=torch.unique(kwargs['input_ids']);penalty=model.generation_config.repetition_penalty;assert penalty==1.05 and model.generation_config.top_k==50
        adjusted=logits.clone();adjusted[seen]=torch.where(logits[seen]<0,logits[seen]*penalty,logits[seen]/penalty)
        good,alt=case['target'],case['alternative'];cut=adjusted.topk(50).values[-1]
        res.append(dict(image=nm,prefix=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),target=good,alternative=alt,delta=float(adjusted[good]-adjusted[alt]),raw=float(logits[good]-logits[alt]),
            target_kept=bool(adjusted[good]>=cut),other_kept=bool(adjusted[alt]>=cut),argmax=int(adjusted.argmax()),common_length=len(case['common'])))
    dump(ROOT/'records'/('probe_'+tag+'.json'),dict(adapter=sig,cases=sha(p),rows=res))
if __name__=='__main__':main(sys.argv[1])
