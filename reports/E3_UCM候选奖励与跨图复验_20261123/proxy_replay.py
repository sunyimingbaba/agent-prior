"""Complete the missing original-RFPO control on 14 saved candidate groups."""
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiment import CROSS, EARLY, FRESH, OUT, read, rows, save, sha, table, tensor

CUP = OUT.parent.parent / "CUP"
sys.path.insert(0,str(CUP))
from agent.grpo.reward import total_reward
from agent.grpo.scorer import RetrievalScorer


class ConfigDict(dict):
    pass


class ConfigList(list):
    pass


class CheckpointCallback:
    pass


class FinetuningCallback:
    pass


def scorer():
    path = EARLY / "评分器续训_20261113/检查点备份/冻结评分器_last_50轮.ckpt"
    assert sha(path) == read(CROSS/"val/binding.json")["assets"]["/private/AGSE_E3_1_rebuild_20261112/scorer_resume_20261113/run_01/checkpoints/last.ckpt"]
    safe = [(ConfigDict,"sacred.config.custom_containers.ReadOnlyDict"),
            (ConfigList,"sacred.config.custom_containers.ReadOnlyList"),
            (CheckpointCallback,"pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"),
            (FinetuningCallback,"run_frozen.MyBackboneFinetuning")]
    with torch.serialization.safe_globals(safe):
        s=torch.load(path,mmap=True,map_location="cpu",weights_only=True)["state_dict"]
    spec=importlib.util.spec_from_file_location("cup_prompt_model", CUP/"prompt_clip/model.py")
    mdl=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mdl)
    device="mps" if torch.backends.mps.is_available() else "cpu"
    with torch.device("meta"):
        visual=mdl.VisionTransformer(224,14,1024,24,16,768)
    visual.load_state_dict({k.removeprefix("model.visual."):v for k,v in s.items() if k.startswith("model.visual.")},
                           strict=True,assign=True)
    visual=visual.float().eval().to(device)
    pl=nn.Module()
    pl.model=SimpleNamespace(encode_image=visual)
    pl.prompt_length=s["visual_prompt_embeddings"].shape[1]
    pl.visual_prompt_embeddings=nn.Parameter(s["visual_prompt_embeddings"])
    pl.agent_proj=nn.Sequential(nn.Linear(3584,1024),mdl.LayerNorm(1024))
    for name in ["agent_proj","visual_prompt_proj","visual_prompt_proj_gather","cluster_mapping"]:
        group={k.removeprefix(name+"."):v for k,v in s.items() if k.startswith(name+".")}
        if name!="agent_proj":
            out,ins=group["weight"].shape
            setattr(pl,name,nn.Linear(ins,out))
        getattr(pl,name).load_state_dict(group,strict=True)
    obj=RetrievalScorer.__new__(RetrievalScorer)
    obj.device=device
    obj.pl=pl.float().eval().to(device)
    obj.image_features=tensor(CROSS/"val/features.pt")["proxy_gallery"]
    assert obj.image_features.shape==(210,768)
    return obj


def main():
    obj=scorer()
    names=read(CROSS/"val/index.json")["names"]
    known=rows(CROSS/"val/candidates.csv")
    cache={}
    def score(h,row):
        import hashlib
        key=(hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).digest(),row)
        if key not in cache:
            cache[key]=obj.score(h.flatten().float().tolist(),row)[0]
        return cache[key]
    reference=[]
    for z in known:
        if z["mode"]!="scene" or int(z["k"])<0:
            continue
        path=CROSS/"samples/scene"/z["image"].split(".")[0]/f"sample{z['k']}.pt"
        rank=score(tensor(path),names.index(z["image"]))
        reference.append(dict(image=z["image"],k=int(z["k"]),r=rank,archived=int(z["rank0"]),
                              format=float(z["fmt"])))
    errors=[abs(v["r"]-v["archived"]) for v in reference]
    for n in {v["image"] for v in reference}:
        group=[z for z in reference if z["image"]==n]
        def best(key):
            values=np.array([z["format"]+1/(1+z[key]) for z in group])
            return set(np.flatnonzero(values==values.max()))
        assert best("r")==best("archived"),n
    assert len(errors)==128
    out=[]
    for image in read(FRESH/"inputs/fresh.json"):
        name=image["image"]
        prefix=FRESH/"outputs/fresh"/name.split(".")[0]
        bundle=tensor(prefix.with_suffix(".pt"))
        meta=read(prefix.with_suffix(".json"))
        for i,z in enumerate(meta["logs"]):
            if z["cue"]!="scene" or z["view"]!=0 or z["slot"]==0:
                continue
            rank=score(bundle["own"][i],names.index(name))
            fmt="\n".join(part["text"] for part in z["rounds"])
            out.append(dict(image=name,k=z["slot"]-1,rank0=rank,old=total_reward(fmt,rank)))
    assert len(out)==112
    table("原奖励重放已知对照.csv",reference)
    table("原奖励补齐14图.csv",out)
    save("原奖励复核.json",dict(device=obj.device,known=len(errors),known_exact=sum(e==0 for e in errors),
                             max_rank_delta=max(errors),known_winner_sets_exact=16,new=len(out)))
    print("Verified",len(errors),"old candidates and restored",len(out),"new control scores")


if __name__=="__main__":
    main()
