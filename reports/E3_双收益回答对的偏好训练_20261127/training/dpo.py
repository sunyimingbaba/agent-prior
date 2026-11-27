"""Train two fixed preferences on identical real tool-output prefixes and response pairs."""
from pathlib import Path
import json
import os
import sys

import torch
import torch.nn.functional as F

STAGE=Path(__file__).resolve().parents[1]
REBUILD=STAGE.parent
POOL=REBUILD/"aftertool_joint_pool_20261127"
SEED,ARM=sys.argv[1:]
assert SEED in ("a","b") and ARM in ("ref","r2","r3")
SOURCE=REBUILD/"second_answer_5090_20261124"
if SEED=="b":
    SOURCE=SOURCE/"repeat"
sys.path.insert(0,str(SOURCE/"code"))
from shared import read, dump, sha, setup, rgb, table
from policy import policy, prompt

BETA=0.1
LR=5e-6


def load_training():
    lockfile=STAGE/"training/lock.json"
    lock=read(lockfile)
    for file,h in lock["stage"].items():
        assert sha(STAGE/file)==h,file
    for file,h in lock["historical"][SEED].items():
        assert sha(SOURCE/file)==h,file
    assert sha(POOL/"sampling/a_complete.json")==lock["sampling"]
    pairs=read(STAGE/"training/data.json")
    assert len(pairs)==lock["pairs"]
    assert len({r["image"] for r in pairs})==len(pairs)
    return lockfile,lock,pairs


def build_input(proc,tb,case,source_seed="a"):
    assert source_seed in ("a","b")
    path=POOL/"sampling/raw"/source_seed/(case["image"]+".pt")
    assert sha(path)==case["raw_sha256"]
    raw=torch.load(path, map_location="cpu", weights_only=True)
    assert raw["eligible"] and raw["seed"]==source_seed and raw["image"]==case["image"]
    assert (raw["rgb"],raw["query"],raw["row"],raw["feedback"])==(case["rgb_sha256"],case["orig_query"],case["row"],case["feedback"])
    image=rgb(tb,case)
    prefix=raw["context"]["prefix"]
    assert raw["context"]["first"]==raw["first"] and raw["context"]["answer"]==raw["feedback"]
    inputs=proc(text=[prompt(proc,case["orig_query"])],images=[image],return_tensors="pt").to("cuda")
    plain=inputs.input_ids.cpu()
    assert torch.equal(prefix[:,:plain.shape[1]],plain)
    suffix=proc(text=[raw["first_text"]+f'\n<result>{case["feedback"]}</result>\n'],return_tensors="pt").input_ids
    assert torch.equal(prefix,torch.cat([plain,suffix],dim=1))
    inputs["input_ids"]=prefix.to("cuda")
    inputs["attention_mask"]=torch.ones_like(inputs.input_ids)
    paths=[raw["paths"][n] for n in case["slots"]]
    assert len(paths)==2 and paths[0]!=paths[1] and all(1<=len(ids)<=64 for ids in paths)
    return inputs,paths


def sequence_logp(model,inputs,ids):
    actions=torch.tensor(ids,device=inputs.input_ids.device,dtype=torch.long)
    tokens=torch.cat([inputs.input_ids,actions[None]],dim=1)
    picture=dict(inputs,input_ids=tokens,attention_mask=torch.ones_like(tokens))
    logits=model(**picture,use_cache=False,return_dict=True,logits_to_keep=len(ids)+1).logits[0,:-1].float()
    assert logits.shape[0]==len(ids)
    per_token=-F.cross_entropy(logits,actions,reduction="none")
    assert torch.isfinite(per_token).all()
    return per_token.sum(),per_token.detach().cpu().tolist()


def log_snapshot(model,inputs,paths):
    out=[]
    for ids in paths:
        total,values=sequence_logp(model,inputs,ids)
        out.append(dict(tokens=ids,logp=total.item(),per_token=values))
        del total
    return out


def run():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8" and os.environ["PYTHONHASHSEED"]=="0"
    setup()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    lockfile,lock,pairs=load_training()
    lk=sha(lockfile)
    initial=sha(SOURCE/"R0/adapter/adapter_model.safetensors")
    assert initial==read(SOURCE/"initial.json")["adapter"]
    reference=STAGE/"training"/("reference_"+SEED+".json")
    out=STAGE/"models"/(SEED+"_"+ARM)
    if ARM=="ref" and reference.exists():
        z=read(reference)
        assert z["lock"]==lk and z["initial"]==initial and len(z["pairs"])==len(pairs)
        return
    if ARM!="ref" and (out/"complete.json").exists():
        done=read(out/"complete.json")
        assert done["lock"]==lk and done["reference"]==sha(reference)
        assert done["final"]==sha(out/"adapter/adapter_model.safetensors") and done["scores"]==sha(out/"steps.json")
        return
    model,proc=policy("R0")
    named={k:p for k,p in model.named_parameters() if ".lora_" in k}
    assert len(named)>0 and all(v.dtype==torch.float32 for v in named.values())
    for p in named.values():
        p.requires_grad_(True)
    tb=table()
    if ARM=="ref":
        rows=[]
        for j,case in enumerate(pairs):
            inputs,paths=build_input(proc,tb,case)
            # Use the exact same grad-enabled forward path as the policy and release each graph immediately.
            values=log_snapshot(model,inputs,paths)
            rows.append(dict(image=case["image"],slots=case["slots"],responses=values))
            print(SEED,"ref",j+1,flush=True)
        dump(reference,dict(lock=lk,initial=initial,beta=BETA,pairs=rows))
        return
    out.mkdir(parents=True,exist_ok=True)
    frozen=read(reference)
    assert frozen["lock"]==lk and frozen["initial"]==initial and len(frozen["pairs"])==len(pairs)
    refhash=sha(reference)
    optim=torch.optim.AdamW(list(named.values()),lr=LR)
    history=[]
    checkpoint=out/"resume.pt"
    if checkpoint.exists():
        z=torch.load(checkpoint,map_location="cpu",weights_only=True)
        assert (z["arm"],z["lock"],z["initial"],z["reference"])==(ARM,lk,initial,refhash)
        for name,param in named.items():
            param.data.copy_(z["weights"][name].to(param.device))
        optim.load_state_dict(z["opt"])
        history=z["history"]
        del z
    for j,case in enumerate(pairs):
        if j<len(history):
            continue
        inputs,paths=build_input(proc,tb,case)
        ref=frozen["pairs"][j]
        assert ref["image"]==case["image"] and ref["slots"]==case["slots"]
        assert all(v["tokens"]==p for v,p in zip(ref["responses"],paths))
        chosen=case["preferences"][ARM]
        rejected=1-chosen
        optim.zero_grad(set_to_none=True)
        response=[None,None]
        for pos,sign in ((chosen,-1),(rejected,1)):
            total,per_token=sequence_logp(model,inputs,paths[pos])
            response[pos]=dict(logp=total.item(),per_token=per_token,tokens=paths[pos])
            (sign*total).backward()
            del total
        if j==0:
            assert response==ref["responses"]
        policy_gap=response[chosen]["logp"]-response[rejected]["logp"]
        ref_gap=ref["responses"][chosen]["logp"]-ref["responses"][rejected]["logp"]
        logit=torch.tensor(BETA*(policy_gap-ref_gap),dtype=torch.float64)
        factor=float(BETA*torch.sigmoid(-logit))
        # The accumulated gradient is grad(log p_rejected - log p_chosen).
        # The scalar gives exactly the gradient of -log sigmoid(beta*(policy_gap-ref_gap)).
        for param in named.values():
            if param.grad is not None:
                param.grad.mul_(factor)
        norm=float(torch.nn.utils.clip_grad_norm_(named.values(),1.,error_if_nonfinite=True))
        if norm>0:
            optim.step()
        history.append(dict(step=j+1,image=case["image"],row=case["row"],slots=case["slots"],chosen_position=chosen,
                            chosen_slot=case["slots"][chosen],rejected_slot=case["slots"][rejected],
                            responses=response,policy_gap=policy_gap,reference_gap=ref_gap,logit=logit.item(),
                            loss=F.softplus(-logit).item(),gradient_scale=factor,norm=norm))
        temp=checkpoint.with_suffix(".part")
        torch.save(dict(lock=lk,reference=refhash,initial=initial,arm=ARM,weights={k:v.detach().cpu() for k,v in named.items()},
                        opt=optim.state_dict(),history=history),temp)
        temp.replace(checkpoint)
        dump(out/"steps.json",history)
        print(SEED,ARM,j+1,history[-1]["loss"],norm,flush=True)
    assert [r["image"] for r in history]==[r["image"] for r in pairs]
    model.save_pretrained(out/"adapter")
    dump(out/"steps.json",history)
    dump(out/"complete.json",dict(initial=initial,final=sha(out/"adapter/adapter_model.safetensors"),
        lock=lk,reference=refhash,steps=len(history),nonzero=sum(r["norm"]>0 for r in history),
        scores=sha(out/"steps.json"),train=sha(STAGE/"training/data.json")))


if __name__=="__main__":
    run()
