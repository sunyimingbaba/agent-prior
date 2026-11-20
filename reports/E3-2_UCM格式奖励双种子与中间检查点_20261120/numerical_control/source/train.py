"""Train paired switches for the historical format reward with identical rank scales."""
import sys,random,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor,TopKLogitsWarper
from shared import ROOT,ASSETS,REBUILD,table,rgb,read,dump,sha,setup
from readouts import load_model,RetrievalScorer,clip
from rollout import load_mllm,rollout_one,SYSTEM_PROMPT
from tools import scene_classify_with_conf
from agent.grpo import reward
def run(arm):
    protocol=read(ROOT/'inputs/protocol.json')
    assert arm in protocol['arms']
    schedule,condition=arm.split('_')
    seed_base=protocol['schedule'][schedule]
    setup();out=ROOT/('policy_'+arm);out.mkdir(exist_ok=True)
    cases={r['image']:r for r in read(ROOT/'inputs/train.json')}
    order=list(cases);assert len(order)==24
    t=table();model,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
    model.eval()
    for n,p in model.named_parameters():p.requires_grad_('.lora_' in n)
    params=[p for p in model.parameters() if p.requires_grad]
    def signature():
        d=hashlib.sha256()
        for p in params:d.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return d.hexdigest()
    before=signature()
    optim=torch.optim.AdamW(params,lr=5e-6)
    tool,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False);tool=tool.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True);head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head']);head=head.eval().cuda()
    frozen=load_model(False)
    prior=REBUILD/'relation_candidates_20261120/native'
    ix=read(prior/'index.json');assert sha(prior/'features.pt')==ix['sha256']
    nat=torch.load(prior/'features.pt',weights_only=True,map_location='cpu')
    capbank=nat['features'].double()
    parity=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2 for q in nat['tokens'].tolist()])
    words=sorted({text for r in cases.values() for text in r['A']})
    with torch.no_grad():
        bank=torch.cat([frozen.txt_embeds(z.cuda())['text_feats'].cpu() for z in clip.tokenize(words,context_length=77-frozen.prompt_length).split(24)]).double()
    for name,h in read(ROOT/'inputs/lock.json').items():assert sha(ROOT/'inputs'/name)==h
    core=model.base_model.model
    gc=core.generation_config
    assert gc.top_k==50 and gc.top_p==1 and gc.repetition_penalty==1.05 and gc.num_beams==1
    rep=RepetitionPenaltyLogitsProcessor(gc.repetition_penalty);trunc=TopKLogitsWarper(gc.top_k)
    original=model.generate
    def capture(**kw):
        kw['output_scores']=True;ans=original(**kw)
        ids=ans.sequences[0,kw['input_ids'].shape[1]:].tolist()
        lp=[float(torch.log_softmax(q[0].float(),dim=-1)[token]) for q,token in zip(ans.scores,ids)]
        assert len(ids)==len(lp) and ids
        captures.append(({n:v for n,v in kw.items() if isinstance(v,torch.Tensor)},ids,lp))
        return ans
    model.generate=capture
    def gradient_lps(record):
        inp,tokens,reference=record
        ids=inp['input_ids'];n=ids.shape[1]
        kw={k:v for k,v in inp.items() if k!='input_ids'}
        kw.update(use_cache=True,past_key_values=DynamicCache(config=core.config),cache_position=torch.arange(n,device='cuda'),logits_to_keep=1)
        vals=[]
        for tok in tokens:
            prepared=core.prepare_inputs_for_generation(ids,**kw)
            res=core(**prepared,return_dict=True)
            logits=res.logits[:,-1,:].float().clone()
            # The generation processors run after conversion to FP32.
            scores=trunc(ids,rep(ids,logits))
            vals.append(torch.log_softmax(scores[0],dim=0)[tok])
            kw=core._update_model_kwargs_for_generation(res,kw)
            ids=torch.cat([ids,torch.tensor([[tok]],device='cuda')],dim=1)
        lp=torch.stack(vals)
        e=float((lp.detach().cpu()-torch.tensor(reference)).abs().max())
        assert torch.isfinite(lp).all() and e<0.06,(tokens,e)
        return lp.sum(),e
    log=[]
    progress=out/'resume.pt'
    if progress.exists():
        state=torch.load(progress,weights_only=True,map_location='cpu')
        assert state['initial']==before and state['arm']==arm and state['source']==sha(Path(__file__))
        with torch.no_grad():
            for par,weight in zip(params,state['weights'],strict=True):par.copy_(weight)
        optim.load_state_dict(state['optim']);log=state['log']
    for step,name in enumerate(order):
        if step<len(log):continue
        r=cases[name];im=rgb(t,r);results=[];rewards=[];margins=[]
        tracks=[];traces=[];formats=[]
        answer=scene_classify_with_conf(im,model=tool,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))[0]
        for k in range(8):
            seed=seed_base+step*8+k;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            captures=[]
            with torch.no_grad():
                h,used,filtered=rollout_one(model,proc,im,r['scene_query'],{'scene_classify':lambda im:answer},3,64,'cuda',temperature=1.,system_prompt=SYSTEM_PROMPT)
                text='\n'.join(proc.decode(ids,skip_special_tokens=True) for _,ids,_ in captures)
                formats.append(reward.r_format(text))
                vis=frozen.img_embeds(frozen.preprocess(im)[None].cuda(),agent_hidden=h[None].cuda())['image_feats'].cpu().double()
                sims=(vis@bank.T)[0]; pos,neg=[words.index(q) for q in r['A']]
                natives=(vis@capbank.T)[0];mapping=ix['maps'][name]
                assert mapping['captions']==t['caption'][r['row']].as_py()
                allowed=parity==0;allowed[mapping['positive']]=False
                mar=float(natives[mapping['first']]-natives[allowed].max())
                margins.append([mar,float(sims[pos]-sims[neg])])
            tracks.append(captures)
            traces.append(dict(seed=seed,rounds=[{'tokens':ids,'logp':lp} for _,ids,lp in captures],used=used,filtered=filtered))
        a,b=np.asarray(margins).T
        semantic=(rankdata(a)+rankdata(b))/2
        rewards=(semantic + (7*np.asarray(formats) if condition=='on' else 0)).tolist()
        rew=torch.tensor(rewards);adv=(rew-rew.mean())/(rew.std()+1e-8)
        optim.zero_grad(set_to_none=True);maxerr=0
        for i,track in enumerate(tracks):
            if float(adv[i])==0:continue
            for record in track:
                lp,e=gradient_lps(record);(-lp*float(adv[i])/len(tracks)).backward();maxerr=max(e,maxerr)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optim.step()
        log.append(dict(step=step+1,image=name,rewards=rewards,semantic_midrank=semantic.tolist(),format_score=formats,caption_and_hard_margins=margins,advantages=adv.tolist(),unclipped_grad_norm=norm,max_logp_error=maxerr,traces=traces))
        if (step+1)%6==0:
            if step+1 in (6,12,24):
                folder=out/f'step_{step+1:02d}'
                model.save_pretrained(folder/'adapter',safe_serialization=True)
                dump(folder/'update.json',dict(step=step+1,initial_params=before,final_params=signature(),
                     adapter_hash=sha(folder/'adapter/adapter_model.safetensors')))
            tmp=out/'resume.part'
            torch.save(dict(initial=before,arm=arm,source=sha(Path(__file__)),weights=[p.detach().cpu() for p in params],optim=optim.state_dict(),log=log),tmp)
            tmp.replace(progress)
        dump(out/'steps.json',log);print(arm,step+1,name,norm,maxerr,flush=True)
    after=signature();assert before!=after
    last=read(out/'step_24/update.json')
    assert last['final_params']==after
    dump(out/'update.json',dict(**last,steps=len(log),nonzero=sum(z['unclipped_grad_norm']>0 for z in log),order=order))
if __name__=='__main__':run(sys.argv[1])
