"""Validate all actual generation inputs and archive only those sampled actions for replay."""
import torch
from common import QUERY, PromptCheck, tensor_sha


def generate(sampler, model, proc, image, feedback, seed, reference):
    audited=PromptCheck(proc)
    original=model.generate
    received=[]
    def observe(**kw):
        ids=kw["input_ids"].detach().cpu()
        count=ids.shape[0]
        assert torch.equal(ids,ids[:1].expand_as(ids))
        assert torch.equal(kw["attention_mask"],torch.ones_like(kw["input_ids"]))
        assert torch.equal(kw["image_grid_thw"].cpu(),reference["grid"].repeat(count,1))
        pixels=kw["pixel_values"].chunk(count,dim=0)
        assert len(pixels)==count and all(torch.equal(pixels[0],v) for v in pixels)
        assert tensor_sha(pixels[0])==reference["pixel_values_sha"]
        if not received:
            assert count==8 and torch.equal(ids[:1],reference["first_input_ids"])
            assert torch.equal(audited.first_ids,reference["first_input_ids"])
            assert audited.template==reference["template"]
        received.append((count,ids[:1].clone()))
        return original(**kw)
    model.generate=observe
    try:
        inputs,history,states=sampler(model,audited,image,QUERY,feedback,seed,initial_prefix=None)
    finally:
        model.generate=original
    assert len(states)==len(history)==8 and len(inputs)==len(received)
    for k,(count,ids) in enumerate(received):
        assert torch.equal(inputs[k].input_ids.cpu(),ids)
        assert sum(ci==k for trajectory in history for ci,_ in trajectory)==count
    serial=[]
    for trajectory in history:
        assert trajectory[0][0]==0 and 1<=len(trajectory)<=3
        actions=[]
        for turn,(key,tr) in enumerate(trajectory):
            ids,_,logps,support,text=tr
            assert 0<len(ids)==len(logps)==len(support)<=64
            if turn+1<len(trajectory):
                from rollout import parse_tool_call
                assert parse_tool_call(text)=="scene_classify"
                nxt=trajectory[turn+1][0]
                addition=proc(text=[text+f"\n<result>{feedback}</result>\n"],return_tensors="pt").input_ids
                assert torch.equal(received[nxt][1],torch.cat([received[key][1],addition],dim=1))
            elif turn<2:
                from rollout import parse_tool_call
                assert parse_tool_call(text) is None
            actions.append(dict(context=key,ids=ids,text=text,sampled_logp=logps,
                      support=[t.detach().cpu().to(torch.int32) for t in support]))
        serial.append(actions)
    compact=dict(context_ids=[ids for count,ids in received],context_batch_sizes=[count for count,ids in received],
                 actions=serial,terminal_hidden=torch.stack(states),feedback=feedback,
                 query=QUERY,image_input_token_hash=tensor_sha(reference["first_input_ids"]),pixel_values=reference["pixel_values_sha"])
    return inputs,history,states,compact


def signature(actions):
    return [[dict(context=r["context"],ids=r["ids"],text=r["text"],sampled_logp=r["sampled_logp"]) for r in trajectory]
            for trajectory in actions]
