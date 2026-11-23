"""Matched on-policy cold-start pilot. Only R2 changes the retrieval preference order."""
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from shared import ROOT, REBUILD, read, dump, sha, setup, table, rgb
from readouts import load_model, scorer_for
from policy import policy, prompt, cut
from agent.grpo.reward import r_format

G = 8
MAX_TOKENS = 64
SAMPLES = REBUILD/'matched_policy_20261120/inputs/train.json'
NATIVE = REBUILD/'relation_candidates_20261120/native'

@torch.no_grad()
def sample(model, proc, image, query):
    text = prompt(proc, query)
    inputs = proc(text=[text]*G, images=[image]*G, return_tensors='pt').to('cuda')
    n = inputs.input_ids.shape[1]
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
    single = proc(text=[text],images=[image],return_tensors='pt').to('cuda')
    return single, traces

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
    assert arm in ('R1','R2')
    setup()
    init = read(ROOT/'initial.json')
    assert sha(SAMPLES)==init['train'] and sha(ROOT/'R0/adapter/adapter_model.safetensors')==init['adapter']
    cases = read(SAMPLES)
    model, proc = policy('R0')
    params = [p.requires_grad_() for name,p in model.named_parameters() if '.lora_' in name]
    optim = torch.optim.AdamW(params, lr=5e-6)
    proxy = scorer_for(load_model(True), REBUILD/'relation_candidates_20261120/gallery/train.pt')
    if arm=='R2':
        reader = load_model(False)
        index = read(NATIVE/'index.json')
        assert sha(NATIVE/'features.pt')==index['sha256']
        bank = torch.load(NATIVE/'features.pt',weights_only=True,map_location='cpu')
        features = bank['features'].double()
        even = np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['tokens'].tolist()])
    t = table()
    out = ROOT/arm
    out.mkdir(exist_ok=True)
    log = []
    for j, case in enumerate(cases):
        im = rgb(t,case)
        seed = 284000+j
        torch.manual_seed(seed)
        single, traces = sample(model,proc,im,case['scene_query'])
        ranks, margins, formats = [], [], []
        with torch.no_grad():
            for ids,h,ref,support,text in traces:
                rank,_ = proxy.score(h.float().tolist(),case['row'])
                ranks.append(rank)
                formats.append(r_format(text))
                if arm=='R2':
                    mapping = index['maps'][case['image']]
                    assert mapping['captions']==t['caption'][case['row']].as_py()
                    allowed = even.copy()
                    allowed[mapping['positive']]=False
                    vis=reader.img_embeds(reader.preprocess(im)[None].cuda(),agent_hidden=h[None].cuda())['image_feats'].cpu().double()
                    sims = (vis @ features.T)[0]
                    margins.append(float(sims[mapping['first']]-sims[allowed].max()))
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
            norm=norm,replay_mean_abs_logp_difference=float(np.mean(errors)) if errors else 0.,
            replay_max_abs_logp_difference=max(errors,default=0.),trajectories=trace))
        dump(out/'steps.json',log)
        print(arm,j+1,case['image'],norm,log[-1]['replay_mean_abs_logp_difference'],flush=True)
    assert any(z['norm']>0 for z in log)
    model.save_pretrained(out/'adapter')
    dump(out/'complete.json',dict(initial=init['adapter'],final=sha(out/'adapter/adapter_model.safetensors'),
        updates=sum(z['norm']>0 for z in log),groups=len(log),train=init['train'],source=sha(Path(__file__))))

if __name__=='__main__':
    run(sys.argv[1])
