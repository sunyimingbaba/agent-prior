"""Check both reward arms independently against the fixed sample order and resume tensors."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from safetensors import safe_open
import torch

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT.parents[1]/"E3_视觉细节复核与RFPO_20261127/training/audit.py"
spec=spec_from_file_location("first_strict_audit",REFERENCE)
ref=module_from_spec(spec)
spec.loader.exec_module(ref)
read,sha=ref.read,ref.sha


def trace_lock(step):
    keys=("step","image","row","seed","proxy_rank","native_margin","format","retrieval","hard",
          "other_caption_rr","trajectories","continuations","r2","adv2")
    return {k:step[k] for k in keys}


def verify_files(folder,block):
    assert set(block)=={str(f.relative_to(folder)) for f in folder.rglob("*") if f.is_file()}
    for rel,checksum in block.items():
        assert sha(folder/rel)==checksum


def main():
    archive=Path(sys.argv[1]).resolve()
    inputs=ROOT/"inputs/train_lock.json"
    lock=read(inputs)
    rows=read(ROOT/"inputs/roster.json")["rows"]
    assert lock["roster"]==sha(ROOT/"inputs/roster.json") and all(sha(ROOT/f)==h for f,h in lock["files"].items())
    remote=read(ROOT/"audit/远端模型摘要.json")
    report={}
    modelinfo={}
    torch.set_num_threads(2)
    for seed in "ab":
        saved={}
        for arm in ("r2","strict"):
            tag=seed+"_"+arm
            folder=archive/tag
            verify_files(folder,remote[tag])
            context_dir=ROOT/"contexts"/seed
            context_proof=read(context_dir/"complete.json")
            candidates=[(j+1,row) for j,row in enumerate(rows) if context_proof["eligible"][row["image"]]]
            if not candidates:
                assert read(folder/"no_eligible.json")["steps"]==0
                report[tag]={"groups":0,"slots":0,"no_eligible":True}
                continue
            cp=read(folder/"complete.json");history=read(folder/"steps.json")
            assert cp["steps"]==len(history)==len(candidates) and cp["initial"]==lock["starts"][seed]
            assert cp["scheduled_slots"]==24 and cp["contexts"]==sha(context_dir/"complete.json")
            assert cp["scores"]==sha(folder/"steps.json") and cp["final"]==sha(folder/"adapter/adapter_model.safetensors")
            assert cp["lock"]==sha(inputs) and cp["train"]==lock["roster"]
            assert [v["image"] for v in history]==[r["image"] for _,r in candidates]
            z=torch.load(folder/"resume.pt",weights_only=True,map_location="cpu",mmap=True)
            assert z["lock"]==cp["lock"] and z["initial"]==cp["initial"] and z["arm"]==arm and z["history"]==history
            finals={k.replace(".default.","."):v for k,v in z["weights"].items()}
            with safe_open(folder/"adapter/adapter_model.safetensors",framework="pt") as f:
                assert set(f.keys())==set(finals) and all(torch.equal(f.get_tensor(k),v) for k,v in finals.items())
            opt=z["opt"]
            steps={int(rec["step"].item()) for rec in opt["state"].values()}
            assert steps==({cp["nonzero"]} if cp["nonzero"] else set())
            modelinfo[tag]=dict(files=remote[tag],equal_resume_final_tensors=len(finals),optimizer_updates=sorted(steps))
            del opt, finals, z
            changes=[];updates=[];av_diffs=[]
            for (slot,row),h in zip(candidates,history):
                j=slot-1
                assert h["step"]==slot and h["image"]==row["image"] and h["query"]==lock["query"]
                context_path=context_dir/(row["image"]+".pt")
                ctx=torch.load(context_path,map_location="cpu",weights_only=True)
                assert h["context"]==sha(context_path)==context_proof["files"][row["image"]]
                trajectory=folder/"trajectories"/(row["image"]+".pt")
                assert sha(trajectory)==h["trajectory"]
                tr=torch.load(trajectory,weights_only=True,map_location="cpu")
                assert tr["step"]==slot and tr["rng"]==h["seed"] and tr["image"]==h["image"] and tr["query"]==h["query"]
                assert tr["context"]==h["context"] and tr["continuations"]==h["continuations"]
                assert tr["candidate_tokens"]==[v["ids"] for v in h["trajectories"]]
                assert tr["single_prefix"]==hashlib.sha256(ctx["prefix"].contiguous().numpy().tobytes()).hexdigest()
                assert tr["sampled_batch_prefix"]==hashlib.sha256(ctx["prefix"].repeat(8,1).contiguous().numpy().tobytes()).hexdigest()
                for field in ("terminal_hidden","sampled_last_hidden"):
                    assert tr[field].shape==(8,3584) and torch.isfinite(tr[field]).all()
                assert h["seed"]=={"a":231000000,"b":241000000}[seed]+j*1000 and len(h["trajectories"])==len(h["continuations"])==8
                assert all(0<len(x["ids"])==len(x["sampled_logp"])<=64 for x in h["trajectories"])
                margins=np.array(h["native_margin"]); proxy_score=1/(1+np.array(h["proxy_rank"]))
                sorted_values=np.sort(proxy_score); by_margin=np.argsort(margins,kind="stable")
                expected=np.empty(8)
                for value in np.unique(margins):
                    selected=margins[by_margin]==value
                    expected[by_margin[selected]]=sorted_values[selected].mean()
                np.testing.assert_array_equal(expected,np.array(h["retrieval"]))
                _,recomputed=ref.recompute(h); fmt=np.asarray(h["format"])
                native=np.asarray(h["retrieval"]); r2=fmt+native; r3=fmt+recomputed
                np.testing.assert_array_equal(h["r2"],r2)
                np.testing.assert_array_equal(h["r3"],r3)
                np.testing.assert_array_equal(h["bonus"],recomputed-native)
                adv2=(r2-r2.mean())/(r2.std(ddof=1)+1e-8)
                adv3=(r3-r3.mean())/(r3.std(ddof=1)+1e-8)
                np.testing.assert_array_equal(h["adv2"],adv2)
                np.testing.assert_array_equal(h["adv3"],adv3)
                np.testing.assert_array_equal(h["advantage"],adv2 if arm=="r2" else adv3)
                if not np.array_equal(r2==r2.max(),r3==r3.max()):changes.append(h["step"])
                if not np.array_equal(adv2,adv3):av_diffs.append(h["step"])
                if h["norm"]>0:updates.append(h["step"])
            assert len(updates)==cp["nonzero"]
            saved[arm]=history
            report[tag]=dict(groups=len(history),slots=len(history)*8,correctly_reproduced_selected_reward_groups=len(history),
                             strict_rule_changes_top_steps=changes,strict_rule_changes_advantage_steps=av_diffs,
                             nonzero_updates=len(updates),adapter=cp["final"])
        if not saved:continue
        prefix=[]
        for a,b in zip(saved["r2"],saved["strict"]):
            assert trace_lock(a)==trace_lock(b)
            prefix.append(a["step"])
            if b["adv2"]!=b["adv3"]:break
            assert a["norm"]==b["norm"]
        report[seed+"_paired"]=dict(initial_matching_full_sample_groups=prefix)
    (ROOT/"audit/训练核查.json").write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"audit/权重备份核对.json").write_text(json.dumps(dict(path=str(archive),models=modelinfo),indent=2)+"\n")


if __name__=="__main__":main()
