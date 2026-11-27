"""Verify both frozen LoRA initializations using CPU tensor values."""
from pathlib import Path
from safetensors import safe_open
import hashlib
import json
import torch

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

root=Path("/private/AGSE_E3_1_rebuild_20261112/second_answer_5090_20261124")
paths=[x/"R0/adapter/adapter_model.safetensors" for x in [root,root/"repeat"]]
torch.set_num_threads(1)
configs=[json.loads(p.with_name("adapter_config.json").read_bytes()) for p in paths]
for config in configs: config["target_modules"]=sorted(config["target_modules"])
assert configs[0]==configs[1]
out=dict(seeds=[json.loads((p.parents[2]/"initial.json").read_bytes())["seed"] for p in paths],
         sha256=[sha(p) for p in paths],config=configs[0],a_tensors=0,b_tensors=0,
         same_a=0,both_b_zero=0,other_tensors=0)
with safe_open(paths[0],framework="pt",device="cpu") as a, safe_open(paths[1],framework="pt",device="cpu") as b:
    assert a.keys()==b.keys()
    for k in a.keys():
        x,y=a.get_tensor(k),b.get_tensor(k)
        if ".lora_A." in k:
            out["a_tensors"]+=1
            out["same_a"]+=int(torch.equal(x,y))
        elif ".lora_B." in k:
            out["b_tensors"]+=1
            out["both_b_zero"]+=int(x.count_nonzero()==0 and y.count_nonzero()==0)
        else:
            out["other_tensors"]+=1
assert out["a_tensors"]==out["b_tensors"]==out["both_b_zero"] and out["other_tensors"]==0
print(json.dumps(out,indent=2))
