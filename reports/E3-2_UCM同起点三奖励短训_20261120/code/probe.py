"""Independently regenerate the first shared group and reconstruct each arm's rewards from the actual flags and vectors."""
import hashlib
import numpy as np
import torch
from scipy.stats import rankdata
from shared import ROOT,ASSETS,REBUILD,table,rgb,read,sha,dump,setup
from readouts import load_model,RetrievalScorer,clip
from rollout import load_mllm,rollout_one,SYSTEM_PROMPT
from tools import scene_classify_with_conf
from agent.grpo import reward
def main():
    setup();out=ROOT/'probe';out.mkdir(exist_ok=True);r=read(ROOT/'inputs/train.json')[0]
    t=table();im=rgb(t,r);m,proc=load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(ASSETS/'ucm_main_lora'))
    tool,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False);tool=tool.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']));head.load_state_dict(cp['head']);head=head.eval().cuda()
    original=m.generate;rounds=[]
    def capture(**kw):
        z=original(**kw);rounds.append(z.sequences[0,kw['input_ids'].shape[-1]:].tolist());return z
    m.generate=capture
    traces=[];hs=[];recs=[]
    with torch.inference_mode():
        text,conf=scene_classify_with_conf(im,model=tool,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
        for k in range(8):
            seed=284000+k;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rounds.clear()
            h,used,filtered=rollout_one(m,proc,im,r['scene_query'],{'scene_classify':lambda x:text},3,64,'cuda',temperature=1.,system_prompt=SYSTEM_PROMPT)
            hs.append(h);traces.append('\n'.join(proc.decode(q,skip_special_tokens=True) for q in rounds))
            recs.append(dict(seed=seed,token_ids=rounds.copy(),used=used,filtered=filtered))
    del m,original,tool
    proxy=load_model(True);sc=RetrievalScorer.__new__(RetrievalScorer);sc.pl=proxy;sc.device='cuda'
    prior=REBUILD/'relation_candidates_20261120'
    sc.cache_path=str(prior/'gallery/train.pt');sc.image_features,sc.image_indices=sc._load_or_build_cache()
    assert sc.image_indices==list(range(1680))
    actual=load_model(False);ix=read(prior/'native/index.json')
    bundle=torch.load(prior/'native/features.pt',weights_only=True,map_location='cpu')
    even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bundle['tokens'].tolist()])
    mp=ix['maps'][r['image']];even[mp['positive']]=False
    # Encode all selected train relation texts in the original batches to avoid tiny batching differences.
    words=sorted({s for z in read(ROOT/'inputs/train.json') for s in z['A']})
    with torch.inference_mode():
        ft=torch.cat([actual.txt_embeds(q.cuda())['text_feats'].cpu() for q in clip.tokenize(words,context_length=77-actual.prompt_length).split(24)]).double()
        for k,h in enumerate(hs):
            rank,rr=sc.score(h.float().tolist(),r['row'])
            v=actual.img_embeds(actual.preprocess(im)[None].cuda(),agent_hidden=h[None].cuda())['image_feats'].cpu().double()
            sims=(v@bundle['features'].double().T)[0];s=(v@ft.T)[0]
            p,n=[words.index(q) for q in r['A']];ri=recs[k]
            ri.update(rank0=rank,format=reward.r_format(traces[k]),old=reward.total_reward(traces[k],rank,lam=1.,mode='soft'),
                      positive=float(sims[mp['first']]),hardest_native_negative=float(sims[even].max()),true_relation=float(s[p]),false_relation=float(s[n]))
    a=rankdata([z['positive']-z['hardest_native_negative'] for z in recs]);b=rankdata([z['true_relation']-z['false_relation'] for z in recs])
    for i,z in enumerate(recs):z.update(caption=float(a[i]),hard=float((a[i]+b[i])/2))
    torch.save(torch.stack(hs),out/'hidden.pt');dump(out/'scores.json',dict(image=r['image'],tool_result=text,confidence=float(conf),cases=recs,sha256=sha(out/'hidden.pt')))
if __name__=='__main__':main()
