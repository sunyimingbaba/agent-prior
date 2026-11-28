"""Verify real scene-tool returns, then freeze each seed's actual image-only first response."""
import hashlib
import sys
from pathlib import Path
import torch
from common import QUERY, STAGE, PromptCheck, locked, pixels, read, sha, tensor_sha

SEED=sys.argv[1]
assert SEED in ("a","b","feedback")
BASE=STAGE.parents[1]
SOURCE=BASE/"second_answer_5090_20261124"
if SEED=="b": SOURCE=SOURCE/"repeat"
sys.path.insert(0,str(SOURCE/"code"))
from shared import ASSETS, table, setup, dump
from policy import policy, prompt
from rollout import SYSTEM_PROMPT, parse_tool_call, rollout_one


def verify(table_data,rows,lock):
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    head_path=ASSETS/"ucm_scene_cls.pt"
    clip_path=ASSETS/"models/ViT-L-14.pt"
    prior_path=BASE/"aftertool_joint_pool_20261127/sampling/feedback_verified.json"
    previous=read(prior_path)
    assert sha(prior_path)==lock["prior_feedback"] and sha(head_path)==previous["classifier"] and sha(clip_path)==previous["clip"]
    m,preprocess=clip.load(str(clip_path),device="cpu",jit=False)
    m=m.float().eval().cuda()
    ck=torch.load(head_path,map_location="cpu",weights_only=True)
    labels=list(ck["classes"])
    head=torch.nn.Linear(ck["feat_dim"],len(labels));head.load_state_dict(ck["head"]);head=head.eval().cuda()
    checks={}
    for row in rows:
        im=pixels(table_data,row)
        result,conf=scene_classify_with_conf(im,model=m,preprocess=preprocess,cls_head=head,cls_names=labels)
        old=previous["images"][row["image"]]
        assert (old["rgb"],old["feedback"])==(row["rgb_sha256"],result)
        checks[row["image"]]=dict(rgb=row["rgb_sha256"],feedback=result,confidence=float(conf))
    dump(STAGE/"inputs/feedback_verified.json",dict(images=checks,head=previous["classifier"],clip=previous["clip"],
         previous=sha(prior_path),lock=sha(STAGE/"inputs/train_lock.json")))


class GotSecondRound(Exception):
    pass


def capture(model,proc,image,feedback):
    checked=PromptCheck(proc)
    reference_template=prompt(proc,QUERY)
    expected=proc(text=[reference_template],images=[image],return_tensors="pt")
    original=model.generate
    first=[];prefix=None;calls=[]

    def answer(im):
        assert im is image and not calls
        calls.append(feedback)
        return feedback

    def wrapped(**kwargs):
        nonlocal prefix
        actual=kwargs["input_ids"].cpu()
        assert torch.equal(expected["pixel_values"],kwargs["pixel_values"].cpu())
        assert torch.equal(expected["image_grid_thw"],kwargs["image_grid_thw"].cpu())
        if not first:
            assert checked.template==reference_template and torch.equal(actual,checked.first_ids)
            assert torch.equal(actual,expected.input_ids)
            output=original(**kwargs)
            first.extend(output.sequences[0,actual.shape[1]:].tolist())
            return output
        assert calls==[feedback]
        rendering=proc.decode(first,skip_special_tokens=True)
        suffix=proc(text=[rendering+f'\n<result>{feedback}</result>\n'],return_tensors="pt").input_ids
        assert torch.equal(actual,torch.cat([expected.input_ids,suffix],dim=1))
        prefix=actual.clone()
        raise GotSecondRound

    model.generate=wrapped
    try:
        rollout_one(model,checked,image,QUERY,{"scene_classify":answer},3,64,"cuda",temperature=0.,system_prompt=SYSTEM_PROMPT)
    except GotSecondRound:
        pass
    finally:
        model.generate=original
    text=proc.decode(first,skip_special_tokens=True)
    assert first and ((prefix is not None)==(parse_tool_call(text)=="scene_classify")==bool(calls))
    return dict(query=QUERY,template=reference_template,first_input_ids=expected.input_ids,first=first,
        token_hash=tensor_sha(expected.input_ids),pixel_values_sha=tensor_sha(expected["pixel_values"]),
        grid=expected["image_grid_thw"],first_text=text,calls=calls,answer=feedback if calls else None,
        prefix=prefix,eligible=prefix is not None)


@torch.inference_mode()
def main():
    setup();torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    lock,rows=locked();tb=table("train")
    if SEED=="feedback":return verify(tb,rows,lock)
    feedback=read(STAGE/"inputs/feedback_verified.json")
    assert feedback["lock"]==sha(STAGE/"inputs/train_lock.json")
    adapter=sha(SOURCE/"R0/adapter/adapter_model.safetensors")
    assert adapter==lock["starts"][SEED]
    model,proc=policy("R0")
    directory=STAGE/"contexts"/SEED;directory.mkdir(parents=True,exist_ok=True)
    checks={};eligible={}
    for row in rows:
        name=row["image"];im=pixels(tb,row);f=directory/(name+".pt")
        assert feedback["images"][name]["rgb"]==row["rgb_sha256"]
        data=capture(model,proc,im,feedback["images"][name]["feedback"])
        data.update(image=name,rgb=row["rgb_sha256"],row=row["row"],adapter=adapter)
        tmp=f.with_suffix(".part");torch.save(data,tmp);tmp.replace(f)
        eligible[name]=data["eligible"];checks[name]=sha(f)
    assert len(checks)==24
    template=hashlib.sha256(data["template"].encode()).hexdigest()
    original=STAGE.parent/"image_only/records/a_r2/official/complete.json"
    assert template==read(original)["template"] and read(original)["query"]==QUERY
    dump(directory/"complete.json",dict(seed=SEED,adapter=adapter,roster=lock["roster"],files=checks,
         eligible=eligible,available=sum(eligible.values()),template=template,eval_reference=sha(original),
         feedback=sha(STAGE/"inputs/feedback_verified.json"),lock=sha(STAGE/"inputs/train_lock.json")))
    print(SEED,sum(eligible.values()),"/24",flush=True)


if __name__=="__main__":main()
