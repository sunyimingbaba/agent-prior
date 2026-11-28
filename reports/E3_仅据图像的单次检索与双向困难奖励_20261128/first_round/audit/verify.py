"""Reconstruct frame inputs, action masks, training rewards and optimizer records independently."""
import hashlib
import json
import sys
from collections import Counter
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT.parents[1]
MATCHED=ROOT.parent/"matched"

def read(path):return json.loads(path.read_bytes())
def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
def module(path,name):
    s=spec_from_file_location(name,path);m=module_from_spec(s);s.loader.exec_module(m);return m

previous=module(REPORTS/"E3_视觉细节复核与RFPO_20261127/training/audit.py","strict_audit")
format_path=REPORTS.parent/"CUP/agent/grpo/reward.py"
assert sha(format_path)=="c6cad8a097ad4ac7749aa58f569d27c58065157133b2d169fa1838caf4a9c052"
fmt=module(format_path,"original_format")


def full_group(raw,h,context,query):
    assert raw["step"]==h["step"] and raw["rng"]==h["seed"] and raw["image"]==h["image"] and raw["query"]==h["query"]==query
    assert raw["terminal_hidden"].shape==(8,3584) and torch.isfinite(raw["terminal_hidden"]).all()
    assert torch.equal(raw["context_ids"][0],context["first_input_ids"]) and raw["pixel_values"]==context["pixel_values_sha"]
    assert raw["image_input_token_hash"]==context["token_hash"] and raw["reference"]==h["context"]
    native=np.array(h["retrieval"]); marg=np.array(h["native_margin"]); bym=np.argsort(marg,kind="stable")
    sorted_scores=np.sort(1/(1+np.array(h["proxy_rank"])))
    expected=np.empty(8)
    for v in np.unique(marg):
        mask=marg[bym]==v; expected[bym[mask]]=sorted_scores[mask].mean()
    np.testing.assert_array_equal(native,expected)
    _,strict=previous.recompute(h);formats=np.array([fmt.r_format(x[0]["text"]) for x in h["actions"]])
    np.testing.assert_array_equal(formats,h["format"])
    r2=formats+native;r3=formats+strict
    np.testing.assert_array_equal(h["r2"],r2);np.testing.assert_array_equal(h["r3"],r3)
    np.testing.assert_array_equal(h["bonus"],strict-native)
    for src,dst in ((r2,"adv2"),(r3,"adv3")):
        adv=(src-src.mean())/(src.std(ddof=1)+1e-8)
        np.testing.assert_array_equal(adv,h[dst])
    slot_counts=[];round_counts=[]; per_ctx=Counter()
    assert len(raw["actions"])==len(h["actions"])==8
    for actual,listed in zip(raw["actions"],h["actions"]):
        assert 1<=len(actual)==len(listed)<=3 and actual[0]["context"]==0
        count=0
        for turn,(record,old) in enumerate(zip(actual,listed)):
            ci=record["context"];per_ctx[ci]+=1
            assert torch.all(raw["context_ids"][ci][0,:context["first_input_ids"].shape[1]]==context["first_input_ids"][0])
            assert len(record["ids"])==len(record["sampled_logp"])==len(record["support"])
            assert np.isfinite(record["sampled_logp"]).all()
            assert {k:record[k] for k in ("context","ids","text","sampled_logp")}==old
            for token,legal in zip(record["ids"],record["support"]):
                assert legal.ndim==1 and legal.numel()>=50 and torch.any(legal==token)
            assert 0<len(record["ids"])<=64
            if turn<len(actual)-1:
                assert fmt.TOOL_CALL_PAT.search(record["text"])[1]=="scene_classify"
                nxt=raw["context_ids"][actual[turn+1]["context"]];oldids=raw["context_ids"][ci]
                assert nxt.shape[1]>oldids.shape[1] and torch.equal(oldids,nxt[:,:oldids.shape[1]])
            count+=len(record["ids"])
        slot_counts.append(count);round_counts.append(len(actual))
    assert raw["context_batch_sizes"]==[per_ctx[i] for i in range(len(raw["context_ids"]))]
    used=sum(n for n,a in zip(slot_counts,h["advantage"]) if abs(a)>1e-10)
    assert used==h["replayed_tokens"]
    return dict(token_count=sum(slot_counts),replayed_tokens=used,calls=sum(n-1 for n in round_counts),
                multi_round_paths=sum(n>1 for n in round_counts),format_varies=len(set(formats))>1,
                retrieval_top_changed=not np.array_equal(native==native.max(),np.array(strict)==max(strict)),
                total_top_changed=not np.array_equal(r2==r2.max(),r3==r3.max()),
                advantage_changed=h["adv2"]!=h["adv3"],global_zero=all(abs(v)<=1e-10 for v in h["advantage"]))


def main():
    backup=Path(sys.argv[1]).resolve();modelproof=read(ROOT/"audit/远端模型摘要.json")
    lock=read(ROOT/"inputs/train_lock.json");roster=read(ROOT/"inputs/roster.json")["rows"]
    assert all(sha(ROOT/n)==h for n,h in lock["files"].items())
    out={};modelout={}
    for seed in "ab":
        cdir=MATCHED/"contexts"/seed;cpctx=read(cdir/"complete.json")
        arms={}
        for arm in ("r2","strict"):
            tag=seed+"_"+arm;d=backup/tag;digests=modelproof[tag]
            assert set(digests)=={p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
            for n,h in digests.items():assert sha(d/n)==h
            cp=read(d/"complete.json");history=read(d/"steps.json")
            assert cp["steps"]==cp["scheduled_slots"]==len(history)==24 and cp["scores"]==sha(d/"steps.json")
            assert cp["initial"]==lock["starts"][seed] and cp["train"]==lock["roster"] and cp["lock"]==sha(ROOT/"inputs/train_lock.json")
            assert [s["image"] for s in history]==[r["image"] for r in roster]
            stats=[]
            for j,h in enumerate(history):
                name=h["image"];state_path=d/"trajectories"/(name+".pt");ref=cdir/(name+".pt")
                assert sha(state_path)==h["trajectory"] and sha(ref)==h["context"]==cpctx["files"][name]
                assert h["seed"]==lock["seed_schedule"][seed]+1000*j and h["step"]==j+1
                data=torch.load(state_path,weights_only=True,map_location="cpu")
                x=full_group(data,h,torch.load(ref,weights_only=True,map_location="cpu"),lock["query"])
                assert h["advantage"]==h["adv2" if arm=="r2" else "adv3"]
                stats.append(x)
            state=torch.load(d/"resume.pt",weights_only=True,map_location="cpu",mmap=True)
            assert state["history"]==history and state["lock"]==cp["lock"] and state["initial"]==cp["initial"] and state["arm"]==arm
            weights={k.replace(".default.","."):v for k,v in state["weights"].items()}
            with safe_open(d/"adapter/adapter_model.safetensors",framework="pt") as comparable:
                assert set(comparable.keys())==set(weights)
                assert all(torch.equal(comparable.get_tensor(k),v) for k,v in weights.items())
            optsteps={int(s["step"].item()) for s in state["opt"]["state"].values()}
            assert optsteps==({cp["nonzero"]} if cp["nonzero"] else set())
            assert cp["nonzero"]==sum(h["norm"]>0 for h in history)
            modelout[tag]=dict(verified_final_tensors=len(weights),optimizer_steps=sorted(optsteps),final=cp["final"],files=digests)
            del state,weights
            arms[arm]=history
            out[tag]=dict(steps=24,slots=192,nonzero_gradient_updates=cp["nonzero"],
                counts={k:sum(r[k] for r in stats) for k in stats[0]},summary_by_step=stats,
                max_token_logp_replay_delta=max(h["replay_max_abs_logp_difference"] for h in history))
        equal=[]
        fields=("actions","retrieval","proxy_rank","native_margin","format","hard","other_caption_rr","r2","adv2")
        for j,(a,b) in enumerate(zip(arms["r2"],arms["strict"])):
            assert all(a[k]==b[k] for k in fields);equal.append(j+1)
            if b["adv2"]!=b["adv3"]:break
            assert a["norm"]==b["norm"]
        out[seed+"_matched_first_groups"]=equal
    save(ROOT/"audit/训练核查.json",out);save(ROOT/"audit/权重备份核对.json",dict(path=str(backup),models=modelout))


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
