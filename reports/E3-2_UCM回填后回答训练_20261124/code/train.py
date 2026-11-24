"""Matched on-policy cold-start pilot. Only R2 changes the retrieval preference order."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from shared import ROOT, REBUILD, read, dump, sha, setup, table, rgb
from reader_client import Reader
from policy import policy, prompt, cut
from agent.grpo.reward import r_format
from rollout import parse_tool_call

G = 8
MAX_TOKENS = 64
SAMPLES = REBUILD/'matched_policy_20261120/inputs/train.json'
NATIVE = REBUILD/'relation_candidates_20261120/native'

@torch.no_grad()
def sample(model, proc, image, query, context):
    text = prompt(proc, query)
    single = proc(text=[text],images=[image],return_tensors='pt').to('cuda')
    prefix=context['prefix'].cuda() if context is not None else single.input_ids
    assert torch.equal(prefix[:,:single.input_ids.shape[1]],single.input_ids)
    single['input_ids']=prefix
    single['attention_mask']=torch.ones_like(prefix)
    inputs={k:v.repeat(G,*([1]*(v.ndim-1))) for k,v in single.items()}
    n = inputs['input_ids'].shape[1]
    gc = model.generation_config
    assert gc.top_k == 50 and gc.top_p == 1 and gc.repetition_penalty == 1.05
    ans = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=1.0,
                         output_scores=True, output_hidden_states=True, return_dict_in_generate=True)
    # In batched generation rows finishing early contain synthetic padding after their first EOS.
    token_rows = ans.sequences[:,n:].tolist()
    support = torch.stack([x.topk(gc.top_k, dim=-1).indices for x in ans.scores], dim=1)
    traces = []
    for i, row in enumerate(token_rows):
        ids = cut(row, gc.eos_token_id)
        assert all(x==gc.pad_token_id for x in row[len(ids):])
        end = len(ids)-1
        h = ans.hidden_states[end][-1][i,-1].detach().cpu().clone()
        ref = [torch.log_softmax(ans.scores[j][i].float(),-1)[tok].item() for j,tok in enumerate(ids)]
        keep = support[i,:len(ids)].clone()
        # At a tied top-k boundary Transformers can retain >k tokens. Keep the exact live support.
        if any(int(torch.isfinite(ans.scores[j][i]).sum()) != gc.top_k for j in range(len(ids))):
            keep = [torch.where(torch.isfinite(ans.scores[j][i]))[0] for j in range(len(ids))]
        traces.append((ids, h, ref, keep, proc.decode(ids,skip_special_tokens=True)))
    return single, traces

@torch.no_grad()
def finish(model,proc,single,traces,answer):
    states=[];extras=[];cache={}
    for ids,h,_,_,text in traces:
        third=[]
        if parse_tool_call(text):
            key=tuple(ids)
            if key not in cache:
                feedback=proc(text=[text+f'\n<result>{answer}</result>\n'],return_tensors='pt').input_ids.cuda()
                prefix=torch.cat([single.input_ids,feedback],dim=1)
                ans=model.generate(**dict(single,input_ids=prefix,attention_mask=torch.ones_like(prefix)),
                    max_new_tokens=64,do_sample=False,temperature=1.,output_hidden_states=True,return_dict_in_generate=True)
                cache[key]=(ans.hidden_states[-1][-1][0,-1].cpu().clone(),ans.sequences[0,prefix.shape[1]:].tolist())
            h,third=cache[key]
        states.append(h)
        extras.append(third)
    return states,extras

def sequence_logp(model, single, record):
    ids, _, reference, support, _ = record
    prefix = single.input_ids
    actions = torch.tensor(ids,device='cuda')
    all_ids = torch.cat([prefix, actions[None]],dim=1)
    inputs = dict(single, input_ids=all_ids, attention_mask=torch.ones_like(all_ids))
    # The teacher-forced row ends at the last sampled action. Only predicting positions enter the loss.
    pred = model(**inputs,use_cache=False,return_dict=True,logits_to_keep=len(ids)+1).logits[0,:-1].float()
    seen = set(prefix[0].tolist())
    lp = []
    for j, tok in enumerate(ids):
        valid = support[j]
        values = pred[j,valid]
        repeated = torch.tensor([v in seen for v in valid.tolist()], device='cuda')
        penalized = torch.where(values < 0, values*1.05, values/1.05)
        normed = torch.log_softmax(torch.where(repeated,penalized,values),dim=0)
        which = torch.nonzero(valid==tok).item()
        lp.append(normed[which])
        seen.add(tok)
    lps = torch.stack(lp)
    err = (lps.detach().cpu()-torch.tensor(reference)).abs()
    assert torch.isfinite(lps).all()
    return lps.sum(), err.tolist()

def matched_scores(margins, scores):
    ranked = np.argsort(margins,kind='stable')
    result = np.empty(len(scores))
    values = np.sort(scores)
    for m in np.unique(margins):
        ix = np.where(np.asarray(margins)[ranked] == m)[0]
        result[ranked[ix]] = values[ix].mean()
    return result

def run(arm):
    assert arm in ('R1','R2','first_R2')
    setup()
    init = read(ROOT/'initial.json')
    assert sha(SAMPLES)==init['train'] and sha(ROOT/'R0/adapter/adapter_model.safetensors')==init['adapter']
    cases = read(SAMPLES)
    model, proc = policy('R0')
    params = [p.requires_grad_() for name,p in model.named_parameters() if '.lora_' in name]
    optim = torch.optim.AdamW(params, lr=5e-6)
    reader=Reader('R2' if arm=='first_R2' else arm)
    prep=read(ROOT/'prepare.json')
    assert sha(ROOT/'contexts.pt')==prep['contexts']
    contexts=torch.load(ROOT/'contexts.pt',weights_only=True,map_location='cpu')
    assert set(contexts)=={case['image'] for case in cases}
    t = table()
    out = ROOT/arm
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        proof=read(out/'complete.json')
        assert proof['initial']==init['adapter'] and proof['source']==sha(Path(__file__))
        assert proof['final']==sha(out/'adapter/adapter_model.safetensors')
        return
    log = []
    for j, case in enumerate(cases):
        im = rgb(t,case)
        seed = {20261123:284000,20261124:385000}[init['seed']]+j
        torch.manual_seed(seed)
        ctx=contexts[case['image']]
        if arm=='first_R2':
            from first_continuation import finish_first
            single,traces=sample(model,proc,im,case['scene_query'],None)
            states,extras=finish_first(model,proc,im,case['scene_query'],single,traces,ctx['answer'])
            formats=[r_format(t[-1]) for t in traces]
        else:
            single,traces=sample(model,proc,im,case['scene_query'],ctx)
            states,extras=finish(model,proc,single,traces,ctx['answer'])
            formats=[r_format(proc.decode(ctx['first'],skip_special_tokens=True))]*G
        scored=reader.score(case['image'],states)
        ranks=[z['rank'] for z in scored]
        margins=[z['margin'] for z in scored] if arm!='R1' else []
        old = 1/(1+np.asarray(ranks))
        comp = old if arm=='R1' else matched_scores(margins,old)
        rewards = formats + comp
        adv = (rewards-rewards.mean())/(rewards.std(ddof=1)+1e-8)
        optim.zero_grad(set_to_none=True)
        errors = []
        for i,tr in enumerate(traces):
            if abs(adv[i])>1e-10:
                lp, es = sequence_logp(model,single,tr)
                (-lp*float(adv[i])/G).backward()
                errors.extend(es)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        if norm>0:
            optim.step()
        trace = [dict(ids=ids, sampled_logp=ref, text=txt, ended=ids[-1] in model.generation_config.eos_token_id) for ids,_,ref,_,txt in traces]
        log.append(dict(step=j+1,image=case['image'],row=case['row'],seed=seed,proxy_rank=ranks,
            native_margin=margins,format=formats,retrieval=comp.tolist(),total=rewards.tolist(),advantage=adv.tolist(),
            continuations=extras,norm=norm,replay_mean_abs_logp_difference=float(np.mean(errors)) if errors else 0.,
            replay_max_abs_logp_difference=max(errors,default=0.),trajectories=trace))
        dump(out/'steps.json',log)
        print(arm,j+1,case['image'],norm,log[-1]['replay_mean_abs_logp_difference'],flush=True)
    reader.close()
    assert any(z['norm']>0 for z in log)
    model.save_pretrained(out/'adapter')
    dump(out/'complete.json',dict(initial=init['adapter'],final=sha(out/'adapter/adapter_model.safetensors'),
        updates=sum(z['norm']>0 for z in log),groups=len(log),train=init['train'],source=sha(Path(__file__))))

if __name__=='__main__':
    run(sys.argv[1])
