"""Replay the historical answer-round R2 trainer with a separately logged, optional detail bonus."""
import os
import sys
from pathlib import Path
import subprocess

import numpy as np
import torch

from detail_reward import graded_bonus

STAGE = Path(__file__).resolve().parents[1]
HIST = STAGE.parent / "second_answer_5090_20261124"
SEED, ARM = sys.argv[1:]
assert SEED in ("a","b") and ARM in ("r2","r3")
SOURCE = HIST if SEED == "a" else HIST/"repeat"
sys.path.insert(0, str(SOURCE/"code"))
from train import sample, finish, sequence_logp, matched_scores, SAMPLES, G, r_format, policy
from reader_client import Reader
from shared import BASE, ROOT, read, dump, sha, setup, table, rgb


class GradedReader(Reader):
    def __init__(self):
        self.proc = subprocess.Popen([str(BASE/"env/bin/python"),"-Bu",str(STAGE/"code/graded_reader.py"),str(SOURCE)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=""), stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert self.proc.stdout.readline().strip() == "READY"


def advantages(rewards):
    return (rewards-rewards.mean())/(rewards.std(ddof=1)+1e-8)


def main():
    assert ROOT == SOURCE and G==8 and os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
    setup()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    lock_path = STAGE/"inputs/train_lock.json"
    lock = read(lock_path)
    for file, digest in lock["stage"].items():
        assert sha(STAGE/file)==digest
    for file,digest in lock[SEED].items():
        assert sha(SOURCE/file)==digest
    initial = read(SOURCE/"initial.json")
    start = sha(SOURCE/"R0/adapter/adapter_model.safetensors")
    assert sha(SAMPLES)==initial["train"] and start==initial["adapter"]
    cases = read(SAMPLES)
    contexts = torch.load(SOURCE/"contexts.pt",map_location="cpu",weights_only=True)
    assert sha(SOURCE/"contexts.pt")==read(SOURCE/"prepare.json")["contexts"]
    assert len(cases)==24 and set(contexts)=={r["image"] for r in cases}
    out = STAGE/"models"/f"{SEED}_{ARM}"
    out.mkdir(parents=True,exist_ok=True)
    lock_hash = sha(lock_path)
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
    reader = GradedReader()
    images = table()
    try:
        for j,case in enumerate(cases):
            if j<len(log):
                continue
            seed = {20261123:284000,20261124:385000}[initial["seed"]] + j
            image = rgb(images,case)
            ctx = contexts[case["image"]]
            torch.manual_seed(seed)
            single,traces=sample(model,proc,image,case["scene_query"],ctx)
            states,extras=finish(model,proc,single,traces,ctx["answer"])
            fmt = [r_format(proc.decode(ctx["first"],skip_special_tokens=True))]*G
            rows = reader.score(case["image"],states)
            ranks=[r["rank"] for r in rows]
            margins=[r["margin"] for r in rows]
            if j==0:
                original=Reader("R2")
                gold=original.score(case["image"],states)
                original.close()
                assert gold==[dict(rank=r,margin=m) for r,m in zip(ranks,margins)]
            native=matched_scores(margins,1/(1+np.asarray(ranks)))
            hard=np.array([r["hard_margins"] for r in rows])
            bonus=graded_bonus(native,hard)
            r2=fmt+native
            r3=r2+bonus
            adv2=advantages(r2)
            adv3=advantages(r3)
            adv=adv2 if ARM=="r2" else adv3
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
            if (j+1)%4==0:
                temp=ck.with_suffix(".part")
                torch.save(dict(lock=lock_hash,initial=start,arm=ARM,weights={k:p.detach().cpu() for k,p in named.items()},
                                opt=optim.state_dict(),history=log),temp)
                temp.replace(ck)
                dump(out/"steps.json",log)
            print(SEED,ARM,j+1,norm,flush=True)
    finally:
        reader.close()
    assert len(log)==24 and any(z["norm"]>0 for z in log)
    model.save_pretrained(out/"adapter")
    dump(out/"steps.json",log)
    dump(out/"complete.json",dict(lock=lock_hash,initial=start,final=sha(out/"adapter/adapter_model.safetensors"),
        steps=len(log),nonzero=sum(z["norm"]>0 for z in log),scores=sha(out/"steps.json"),train=initial["train"]))


if __name__=="__main__":
    main()
