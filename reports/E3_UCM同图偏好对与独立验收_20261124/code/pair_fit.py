import sys,numpy as np,torch
import torch.nn.functional as F
from env_pair import ROOT,HIST,OLD,model_path,weight,read,dump,sha,savept,table,rgb,setup,fixtures,rows,inputs,captured_prefix
from policy import policy
from rollout import rollout_one,SYSTEM_PROMPT
def extend(inp,seq):
    x=torch.tensor(seq,device=inp.input_ids.device,dtype=torch.long)[None];ids=torch.cat((inp.input_ids,x),1)
    return dict(inp,input_ids=ids,attention_mask=torch.ones_like(ids))
def logp(m,inp,seq):
    assert seq;out=m(**extend(inp,seq),use_cache=False,return_dict=True,logits_to_keep=len(seq)+1)
    row=out.logits[0,:-1].float().log_softmax(-1)
    return row[torch.arange(len(seq),device=row.device),torch.tensor(seq,device=row.device)].sum()
def last(m,inp):
    out=m(**inp,output_hidden_states=True,return_dict=True,use_cache=False,logits_to_keep=1)
    return out.logits[0,-1].float(),out.hidden_states[-1][0,-1].float()
def anchor_loss(logits,hidden,ref):
    kl=F.kl_div(F.log_softmax(logits,-1),ref['logits'].cuda().float().softmax(-1),reduction='sum')
    cos=1-F.cosine_similarity(hidden[None],ref['hidden'].cuda().float()[None]).squeeze()
    return kl+5*cos
def ref_file(seed):return ROOT/'records'/(seed+'_observed_ref.pt')
def run(seed,arm):
    setup();src=weight(seed);m,proc=policy(model_path(seed))
    t=table();rs=rows();plan=fixtures(seed);ctx=torch.load(HIST/'records'/(seed+'_contexts.pt'),weights_only=True)
    chosen={r['image']:r for r in read(ROOT/'inputs/candidates.json')[seed]};original=m.generate
    if arm=='ref':
        anchors={};steprefs=[]
        with torch.inference_mode():
            for rec in plan:
                nm=rec['image'];c=rs[nm];im=rgb(t,c);rr=chosen[nm];fb=ctx[nm]['prefix']
                assert torch.equal(captured_prefix(proc,inputs(proc,im,c['scene_query']),rr),fb)
                i=inputs(proc,im,c['scene_query'],fb)
                steprefs.append([float(logp(m,i,rec[k])) for k in ('pos','neg')])
                if nm in anchors:continue
                last_ids=[];calls=[]
                def capture(**kw):
                    z=original(**kw);ids=z.sequences[0,kw['input_ids'].shape[1]:].tolist();last_ids[:]=kw['input_ids'][0].tolist()+ids[:-1];return z
                m.generate=capture
                try:
                    h,n,f=rollout_one(m,proc,im,c['orig_query'],{'scene_classify':lambda _:calls.append(rr['tool']) or rr['tool']},3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                finally:m.generate=original
                assert n==len(calls) and f==0
                ids=torch.tensor(last_ids)[None];inp=inputs(proc,im,c['orig_query'],ids)
                l,h2=last(m,inp);anchors[nm]=dict(ids=ids,hidden=h2.cpu(),logits=l.bfloat16().cpu(),greedy_cos=float(F.cosine_similarity(h2[None],h.cuda().float()[None])))
        savept(ref_file(seed),dict(plan=plan,logps=steprefs,anchors=anchors,adapter=src));return
    ref=torch.load(ref_file(seed),weights_only=True);assert ref['plan']==plan and ref['adapter']==src
    params=[p.requires_grad_() for n,p in m.named_parameters() if '.lora_' in n];opt=torch.optim.AdamW(params,lr=1e-6)
    dest=ROOT/'pilot'/(seed+'_'+arm);dest.mkdir(exist_ok=True);steps=[]
    for j,rec in enumerate(plan):
        nm=rec['image'];c=rs[nm];im=rgb(t,c);i=inputs(proc,im,c['scene_query'],ctx[nm]['prefix'])
        # Two independent forwards keep the exact unpadded condition for each observed answer.
        with torch.no_grad():p,q=[logp(m,i,rec[k]) for k in ('pos','neg')]
        rp,rq=ref['logps'][j];sgn=1 if arm=='hard' else rec['native_sign']
        pref=-F.logsigmoid(.2*sgn*(p-q-(rp-rq)));opt.zero_grad(set_to_none=True)
        slope=-.2*sgn*torch.sigmoid(-.2*sgn*(p-q-(rp-rq)))
        for label,coeff in [('pos',slope),('neg',-slope)]:
            (coeff*logp(m,i,rec[label])).backward()
        anchor=ref['anchors'][nm];l,h=last(m,inputs(proc,im,c['orig_query'],anchor['ids']))
        a=anchor_loss(l,h,anchor);a.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
        steps.append(dict(image=nm,p=float(p.detach()),q=float(q.detach()),p0=rp,q0=rq,sign=sgn,loss=float(pref.detach()),anchor=float(a.detach()),norm=norm))
        dump(dest/'steps.json',steps)
    m.save_pretrained(dest/'adapter')
    dump(dest/'complete.json',dict(initial=src,final=sha(dest/'adapter/adapter_model.safetensors'),ref=sha(ref_file(seed)),steps=32,code=sha(ROOT/'code/pair_fit.py')))
if __name__=='__main__':run(*sys.argv[1:])
