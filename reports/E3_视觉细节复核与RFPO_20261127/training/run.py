"""Replay the historical answer-round R2 trainer with a separately logged, optional detail bonus."""
import os
import sys
from pathlib import Path
import subprocess

import numpy as np
import torch

STAGE = Path(__file__).resolve().parents[1]
HIST = STAGE.parent / "second_answer_5090_20261124"
POOL = STAGE.parent / "aftertool_joint_pool_20261127"
CAL = STAGE.parent / "hard_reward_calibration_20261127"
sys.path.insert(0,str(CAL/"code"))
from reward import redistribute
GRADED = STAGE.parent / "graded_r2_detail_20261127"
sys.path.insert(0, str(GRADED/"code"))
from detail_reward import graded_bonus

SEED, ARM = sys.argv[1:]
assert SEED in ("a","b") and ARM in ("strict",)
SOURCE = HIST if SEED == "a" else HIST/"repeat"
sys.path.insert(0, str(SOURCE/"code"))
from train import sample, finish, sequence_logp, matched_scores, G, r_format, policy
from reader_client import Reader
from shared import BASE, ROOT, read, dump, sha, setup, table, rgb


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
    roster_path = POOL/"scoring/locks/training.json"
    cases = read(roster_path)["rows"]
    assert len(cases) == lock["n"] and 2 <= len(cases) <= 24
    assert len(set(r["rgb_sha256"] for r in cases)) == len(cases)
    assert cases == sorted(cases,key=lambda r:r["rgb_sha256"])
    gate = read(POOL/"scoring/results/summary.json")
    assert gate["replicated"] and gate["primary"]["difference"]["interval"][0] > 0
    out = STAGE/"models"/f"{SEED}_{ARM}"
    out.mkdir(parents=True,exist_ok=True)
    arm_lock_path = STAGE/"training"/(ARM+"_lock.json")
    arm_lock = read(arm_lock_path)
    assert arm_lock["pool_lock"] == sha(lock_path)
    assert all(sha(STAGE/f)==h for f,h in arm_lock["files"].items())
    assert sha(CAL/"code/reward.py")==arm_lock["redistribution"]
    baseline_path=POOL/"models"/(SEED+"_r2")/"steps.json"
    baseline=read(baseline_path)
    assert sha(baseline_path)==arm_lock["baseline_history"][SEED]
    lock_hash=sha(arm_lock_path)
    if (out/"complete.json").exists():
        cp=read(out/"complete.json")
        assert cp["lock"]==lock_hash and cp["final"]==sha(out/"adapter/adapter_model.safetensors")
        return
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
            image = rgb(images,case)
            raw_path = POOL/"sampling/raw"/SEED/(case["image"]+".pt")
            assert sha(raw_path) == lock["contexts"][SEED][case["image"]]
            payload = torch.load(raw_path,map_location="cpu",weights_only=True)
            assert payload["eligible"] and payload["image"] == case["image"] and payload["adapter"] == start
            assert payload["rgb"] == case["rgb_sha256"] and payload["query"] == case["orig_query"]
            assert payload["row"] == case["row"] and payload["seed"] == SEED
            assert payload["feedback"] == case["feedback"]
            ctx = payload["context"]
            assert ctx["answer"] == payload["feedback"] and len(ctx["first"]) and ctx["prefix"].shape[0]==1
            del payload
            torch.manual_seed(seed)
            single,traces=sample(model,proc,image,case["orig_query"],ctx)
            states,extras=finish(model,proc,single,traces,ctx["answer"])
            fmt = [r_format(proc.decode(ctx["first"],skip_special_tokens=True))]*G
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
            adv=adv3
            if j==0:
                original=baseline[0]
                assert [tr[0] for tr in traces]==[v["ids"] for v in original["trajectories"]]
                assert [tr[2] for tr in traces]==[v["sampled_logp"] for v in original["trajectories"]]
                assert extras==original["continuations"] and rr==original["other_caption_rr"]
                assert (ranks,margins,hard.tolist(),fmt,native.tolist(),r2.tolist())==(
                    original["proxy_rank"],original["native_margin"],original["hard"],original["format"],
                    original["retrieval"],original["r2"])
            optim.zero_grad(set_to_none=True)
            errors=[]
            for i,tr in enumerate(traces):
                if abs(adv[i])>1e-10:
                    lp,es=sequence_logp(model,single,tr)
                    (-lp*float(adv[i])/G).backward()
                    errors.extend(es)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm>0:
                optim.step()
            trace=[dict(ids=ids,sampled_logp=ref,text=txt,ended=ids[-1] in model.generation_config.eos_token_id)
                   for ids,_,ref,_,txt in traces]
            log.append(dict(step=j+1,image=case["image"],row=case["row"],seed=seed,proxy_rank=ranks,
                            native_margin=margins,format=fmt,retrieval=native.tolist(),hard=hard.tolist(),bonus=bonus.tolist(),
                            r2=r2.tolist(),r3=r3.tolist(),adv2=adv2.tolist(),adv3=adv3.tolist(),advantage=adv.tolist(),
                            other_caption_rr=[r["other_caption_rr"] for r in rows],norm=norm,trajectories=trace,continuations=extras,
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
    assert [z["image"] for z in log] == [r["image"] for r in cases]
    model.save_pretrained(out/"adapter")
    dump(out/"steps.json",log)
    dump(out/"complete.json",dict(lock=lock_hash,initial=start,final=sha(out/"adapter/adapter_model.safetensors"),
        steps=len(log),nonzero=sum(z["norm"]>0 for z in log),scores=sha(out/"steps.json"),train=sha(roster_path)))


if __name__=="__main__":
    main()
