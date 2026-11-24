from types import SimpleNamespace
import torch
from rollout import rollout_one,SYSTEM_PROMPT,parse_tool_call

@torch.no_grad()
def finish_first(model,proc,image,query,single,traces,answer):
    generate=model.generate
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
        logs.append(continuations)
    return states,logs
