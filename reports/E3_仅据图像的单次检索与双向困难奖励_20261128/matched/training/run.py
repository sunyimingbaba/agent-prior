"""Replay the historical answer-round R2 trainer with a separately logged, optional detail bonus."""
import os
import sys
from pathlib import Path
import subprocess
from common import QUERY, pixels, tensor_sha

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
from train import sample, finish, sequence_logp, matched_scores, G, r_format, policy
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
    ctx_proof_path = STAGE/"contexts"/SEED/"complete.json"
    ctx_proof = read(ctx_proof_path)
    assert ctx_proof["lock"]==lock_hash and ctx_proof["adapter"]==start
    scheduled = [(j,c) for j,c in enumerate(cases) if ctx_proof["eligible"][c["image"]]]
    assert len(scheduled)==ctx_proof["available"]
    if not scheduled:
        dump(out/"no_eligible.json",dict(seed=SEED,arm=ARM,initial=start,scheduled_slots=24,steps=0,contexts=sha(ctx_proof_path)))
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
        for ordinal,(j,case) in enumerate(scheduled):
            if ordinal<len(log):
                continue
            seed = {"a":231000000,"b":241000000}[SEED] + j*1000
            image = pixels(images,case)
            raw_path = STAGE/"contexts"/SEED/(case["image"]+".pt")
            ctx_hash = sha(raw_path)
            assert ctx_hash==ctx_proof["files"][case["image"]]
            ctx=torch.load(raw_path,map_location="cpu",weights_only=True)
            assert (ctx["rgb"],ctx["image"],ctx["query"],ctx["adapter"])==(case["rgb_sha256"],case["image"],QUERY,start)
            assert ctx["eligible"] and ctx["calls"]==[ctx["answer"]]
            torch.manual_seed(seed)
            original_generate=model.generate
            audited=[]
            def observe(**kwargs):
                assert not audited
                required=ctx["prefix"].to("cuda")
                assert torch.equal(kwargs["input_ids"],required.repeat(G,1))
                assert torch.equal(kwargs["attention_mask"],torch.ones_like(kwargs["input_ids"]))
                assert torch.equal(kwargs["image_grid_thw"],ctx["grid"].to("cuda").repeat(G,1))
                visual=kwargs["pixel_values"]
                chunks=visual.chunk(G,dim=0)
                assert len(chunks)==G and all(torch.equal(chunks[0],v) for v in chunks)
                assert tensor_sha(chunks[0])==ctx["pixel_values_sha"]
                audited.append(tensor_sha(kwargs["input_ids"]))
                return original_generate(**kwargs)
            model.generate=observe
            try:
                single,traces=sample(model,proc,image,QUERY,ctx)
            finally:
                model.generate=original_generate
            assert len(audited)==1 and torch.equal(single.input_ids.cpu(),ctx["prefix"])
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
            adv=adv2 if ARM=="r2" else adv3
            if ARM=="strict" and ordinal==0:
                counterpart=read(STAGE/"models"/(SEED+"_r2")/"steps.json")[0]
                assert [tr[0] for tr in traces] == [v["ids"] for v in counterpart["trajectories"]]
                assert [tr[2] for tr in traces] == [v["sampled_logp"] for v in counterpart["trajectories"]]
                assert (extras,rr,ranks,margins,hard.tolist(),fmt,native.tolist()) == (
                    counterpart["continuations"],counterpart["other_caption_rr"],counterpart["proxy_rank"],
                    counterpart["native_margin"],counterpart["hard"],counterpart["format"],counterpart["retrieval"])
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
            state_folder=out/"trajectories";state_folder.mkdir(exist_ok=True)
            state_file=state_folder/(case["image"]+".pt")
            temp=state_file.with_suffix(".part")
            torch.save(dict(image=case["image"],step=j+1,rng=seed,query=QUERY,context=ctx_hash,
                       single_prefix=tensor_sha(single.input_ids),sampled_batch_prefix=audited[0],
                       candidate_tokens=[tr[0] for tr in traces],continuations=extras,
                       terminal_hidden=torch.stack(states),sampled_last_hidden=torch.stack([tr[1] for tr in traces])),temp)
            temp.replace(state_file)
            log.append(dict(step=j+1,trajectory=sha(state_file),context=ctx_hash,query=QUERY,image=case["image"],row=case["row"],seed=seed,proxy_rank=ranks,
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
    assert [z["image"] for z in log] == [c["image"] for _,c in scheduled]
    model.save_pretrained(out/"adapter")
    dump(out/"steps.json",log)
    dump(out/"complete.json",dict(lock=lock_hash,initial=start,final=sha(out/"adapter/adapter_model.safetensors"),
        steps=len(log),scheduled_slots=24,contexts=sha(ctx_proof_path),nonzero=sum(z["norm"]>0 for z in log),scores=sha(out/"steps.json"),train=sha(roster_path)))


if __name__=="__main__":
    main()
