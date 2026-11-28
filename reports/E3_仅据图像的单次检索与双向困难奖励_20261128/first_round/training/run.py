"""Replay the historical answer-round R2 trainer with a separately logged, optional detail bonus."""
import os
import sys
from pathlib import Path
import subprocess
from common import QUERY, pixels, tensor_sha
from batch import generate,signature

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
HIST = STAGE.parents[1] / "second_answer_5090_20261124"
POOL = STAGE.parents[1] / "aftertool_joint_pool_20261127"
CAL = STAGE.parents[1] / "hard_reward_calibration_20261127"
sys.path.insert(0,str(CAL/"code"))
from reward import redistribute
GRADED = STAGE.parents[1] / "graded_r2_detail_20261127"
sys.path.insert(0, str(GRADED/"code"))
from detail_reward import graded_bonus

SEED, ARM = sys.argv[1:]
assert SEED in ("a","b") and ARM in ("strict","r2")
SOURCE = HIST if SEED == "a" else HIST/"repeat"
sys.path.insert(0, str(SOURCE/"code"))
from train import sequence_logp, matched_scores, G, r_format, policy
TRACE_SRC=STAGE.parents[1]/"source_transfer_2h_20261125/code"
sys.path.insert(0,str(TRACE_SRC))
from full_traces import trace_batch
from reader_client import Reader
from shared import BASE, ROOT, read, dump, sha, setup, table


class PoolReader(Reader):
    def __init__(self):
        self.proc = subprocess.Popen([str(BASE/"env/bin/python"),"-Bu",str(POOL/"scoring/code/pool_reader.py"),"--service"],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=""), stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert self.proc.stdout.readline().strip() == "READY"

    def score(self, image, states):
        response = super().score(image, states)
        rows = response["items"]
        assert len(rows) == G
        return rows


def advantages(rewards):
    return (rewards-rewards.mean())/(rewards.std(ddof=1)+1e-8)


def main():
    assert ROOT == SOURCE and G==8 and os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
    setup()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    lock_path = POOL/"training/train_lock.json"
    lock = read(lock_path)
    for file, digest in lock["training_inputs"].items():
        assert sha(POOL/file) == digest, file
    for file, digest in lock["historical"][SEED].items():
        assert sha(SOURCE/file) == digest, file
    assert sha(GRADED/"code/detail_reward.py") == lock["graded_bonus"]
    initial = read(SOURCE/"initial.json")
    start = sha(SOURCE/"R0/adapter/adapter_model.safetensors")
    assert start == initial["adapter"]
    roster_path = STAGE/"inputs/roster.json"
    cases = read(roster_path)["rows"]
    assert len(cases) == lock["n"] and 2 <= len(cases) <= 24
    assert len(set(r["rgb_sha256"] for r in cases)) == len(cases)
    assert cases == sorted(cases,key=lambda r:r["rgb_sha256"])
    out = STAGE/"models"/f"{SEED}_{ARM}"
    out.mkdir(parents=True,exist_ok=True)
    arm_lock_path = STAGE/"inputs/train_lock.json"
    arm_lock = read(arm_lock_path)
    assert arm_lock["pool_lock"] == sha(lock_path)
    assert all(sha(STAGE/f)==h for f,h in arm_lock["files"].items())
    assert sha(CAL/"code/reward.py")==arm_lock["redistribution"]
    lock_hash=sha(arm_lock_path)
    assert sha(roster_path) == arm_lock["roster"]
    if (out/"complete.json").exists():
        cp=read(out/"complete.json")
        assert cp["lock"]==lock_hash and cp["final"]==sha(out/"adapter/adapter_model.safetensors")
        return
    reference_dir=STAGE.parent/"matched/contexts"/SEED
    ctx_proof=read(reference_dir/"complete.json")
    assert ctx_proof["adapter"]==start and arm_lock["reference_contexts"][SEED]==sha(reference_dir/"complete.json")
    verified=read(STAGE.parent/"matched/inputs/feedback_verified.json")["images"]
    assert sha(STAGE.parent/"matched/inputs/feedback_verified.json")==arm_lock["feedback"]
    assert sha(TRACE_SRC/"full_traces.py")==arm_lock["sampler"]
    model,proc = policy("R0")
    named = {name:p.requires_grad_() for name,p in model.named_parameters() if ".lora_" in name}
    params = list(named.values())
    optim = torch.optim.AdamW(params,lr=5e-6)
    log=[]
    ck = out/"resume.pt"
    if ck.exists():
        z = torch.load(ck,weights_only=True,map_location="cpu")
        assert z["lock"]==lock_hash and z["initial"]==start and z["arm"]==ARM
        for k,p in named.items():
            p.data.copy_(z["weights"][k].to(p.device))
        optim.load_state_dict(z["opt"])
        log=z["history"]
        del z
    reader = PoolReader()
    images = table()
    try:
        for j,case in enumerate(cases):
            if j<len(log):
                continue
            seed = {"a":231000000,"b":241000000}[SEED] + j*1000
            image=pixels(images,case)
            raw_ref=reference_dir/(case["image"]+".pt")
            ctx_hash=sha(raw_ref)
            assert ctx_hash==ctx_proof["files"][case["image"]]
            reference=torch.load(raw_ref,map_location="cpu",weights_only=True)
            assert reference["rgb"]==case["rgb_sha256"]==verified[case["image"]]["rgb"]
            assert reference["query"]==QUERY and reference["adapter"]==start
            answer=verified[case["image"]]["feedback"]
            inputs,history,states,raw=generate(trace_batch,model,proc,image,answer,seed,reference)
            raw.update(image=case["image"],step=j+1,rng=seed,reference=ctx_hash)
            fmt=[r_format(trajectory[0][1][4]) for trajectory in history]
            rows = reader.score(case["image"],states)
            ranks=[r["rank"] for r in rows]
            margins=[r["margin"] for r in rows]
            assert len(ranks) == G and min(ranks) >= 0 and np.isfinite(margins).all()
            native=matched_scores(margins,1/(1+np.asarray(ranks)))
            hard=np.array([r["hard_margins"] for r in rows])
            weak=hard.min(axis=1)
            correct=(hard>0).sum(axis=1)
            top=native==native.max()
            valid=correct*top.sum()>correct[top].sum()
            signal=np.where(valid|top,weak,weak.min())
            rr=[r["other_caption_rr"] for r in rows]
            updated=redistribute(native,signal,rr)
            bonus=updated-native
            r2=fmt+native
            r3=fmt+updated
            adv2=advantages(r2)
            adv3=advantages(r3)
            adv=adv2 if ARM=="r2" else adv3
            actions=signature(raw["actions"])
            if ARM=="strict" and j==0:
                counterpart_dir=STAGE/"models"/(SEED+"_r2")
                counterpart=read(counterpart_dir/"steps.json")[0]
                other_path=counterpart_dir/"trajectories"/(case["image"]+".pt")
                assert sha(other_path)==counterpart["trajectory"]
                other=torch.load(other_path,map_location="cpu",weights_only=True)
                assert all(torch.equal(a,b) for a,b in zip(raw["context_ids"],other["context_ids"]))
                assert raw["context_batch_sizes"]==other["context_batch_sizes"] and torch.equal(raw["terminal_hidden"],other["terminal_hidden"])
                assert actions==counterpart["actions"] and (rr,ranks,margins,hard.tolist(),fmt,native.tolist())==(
                    counterpart["other_caption_rr"],counterpart["proxy_rank"],counterpart["native_margin"],
                    counterpart["hard"],counterpart["format"],counterpart["retrieval"])
                for first,second in zip(raw["actions"],other["actions"]):
                    for turn_a,turn_b in zip(first,second):
                        assert all(torch.equal(a,b) for a,b in zip(turn_a["support"],turn_b["support"]))
                del other
            optim.zero_grad(set_to_none=True)
            errors=[]
            for i,trajectory in enumerate(history):
                if abs(adv[i])>1e-10:
                    for ci,trace in trajectory:
                        lp,err=sequence_logp(model,inputs[ci],trace)
                        (-lp*float(adv[i])/G).backward()
                        errors.extend(err)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm>0:
                optim.step()
            state_folder=out/"trajectories";state_folder.mkdir(exist_ok=True)
            state_file=state_folder/(case["image"]+".pt")
            temp=state_file.with_suffix(".part");torch.save(raw,temp)
            temp.replace(state_file)
            log.append(dict(step=j+1,trajectory=sha(state_file),context=ctx_hash,query=QUERY,image=case["image"],row=case["row"],seed=seed,proxy_rank=ranks,
                            native_margin=margins,format=fmt,retrieval=native.tolist(),hard=hard.tolist(),bonus=bonus.tolist(),
                            r2=r2.tolist(),r3=r3.tolist(),adv2=adv2.tolist(),adv3=adv3.tolist(),advantage=adv.tolist(),
                            other_caption_rr=[r["other_caption_rr"] for r in rows],norm=norm,actions=actions,replayed_tokens=len(errors),
                            replay_mean_abs_logp_difference=float(np.mean(errors)) if errors else 0.,
                            replay_max_abs_logp_difference=max(errors,default=0.)))
            temp=ck.with_suffix(".part")
            torch.save(dict(lock=lock_hash,initial=start,arm=ARM,weights={k:p.detach().cpu() for k,p in named.items()},
                            opt=optim.state_dict(),history=log),temp)
            temp.replace(ck)
            dump(out/"steps.json",log)
            print(SEED,ARM,j+1,norm,flush=True)
    finally:
        reader.close()
    assert [z["image"] for z in log] == [c["image"] for c in cases]
    model.save_pretrained(out/"adapter")
    dump(out/"steps.json",log)
    dump(out/"complete.json",dict(lock=lock_hash,initial=start,final=sha(out/"adapter/adapter_model.safetensors"),
        steps=len(log),scheduled_slots=24,reference_contexts=sha(reference_dir/"complete.json"),nonzero=sum(z["norm"]>0 for z in log),scores=sha(out/"steps.json"),train=sha(roster_path)))


if __name__=="__main__":
    main()
