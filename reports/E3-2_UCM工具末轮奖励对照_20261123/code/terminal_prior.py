"""Finish each sampled first action through the deployment tool loop; cache identical greedy continuations."""
from types import SimpleNamespace
import hashlib
import torch
from tools import scene_classify_with_conf
from rollout import rollout_one, SYSTEM_PROMPT, parse_tool_call
from readouts import clip
from shared import ASSETS

def signature(h):
    return hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def scene_tool():
    model,preprocess=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    model=model.float().eval().cuda()
    state=torch.load(ASSETS/'ucm_scene_cls.pt',weights_only=True,map_location='cpu')
    head=torch.nn.Linear(state['feat_dim'],len(state['classes']))
    head.load_state_dict(state['head'])
    head=head.eval().cuda()
    def classify(image):
        return scene_classify_with_conf(image,model=model,preprocess=preprocess,cls_head=head,cls_names=list(state['classes']))
    return classify

@torch.no_grad()
def finish_group(model,proc,image,query,single,traces,classify):
    generate=model.generate
    answer=None
    if any(parse_tool_call(t[-1]) for t in traces):
        answer,confidence=classify(image)
        assert answer is not None
    cache={}
    states,logs=[],[]
    for ids,first,_,_,text in traces:
        if parse_tool_call(text) is None:
            final,events,continuations=first,0,[]
        else:
            key=tuple(ids)
            if key not in cache:
                rounds=[]
                def follow(**kw):
                    if not rounds:
                        assert torch.equal(single.input_ids,kw['input_ids'])
                        rounds.append(ids)
                        seq=torch.cat([kw['input_ids'],torch.tensor([ids],device='cuda')],dim=1)
                        return SimpleNamespace(sequences=seq,hidden_states=((first[None,None,:],),))
                    assert not kw['do_sample']
                    out=generate(**kw)
                    rounds.append(out.sequences[0,kw['input_ids'].shape[1]:].tolist())
                    return out
                model.generate=follow
                try:
                    h,used,filtered=rollout_one(model,proc,image,query,{'scene_classify':lambda _:answer},
                         3,64,'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
                finally:
                    model.generate=generate
                assert used>=1 and filtered==0 and len(rounds)==min(used+1,3)
                cache[key]=(h,used,rounds[1:])
            final,events,continuations=cache[key]
        assert final.shape==(3584,) and torch.isfinite(final).all()
        states.append(final)
        logs.append(dict(first_sha=signature(first),final_sha=signature(final),tool_calls=events,
            tool_response=answer if events else None,confidence=float(confidence) if events else None,
            continuation_ids=continuations))
    return states,logs
