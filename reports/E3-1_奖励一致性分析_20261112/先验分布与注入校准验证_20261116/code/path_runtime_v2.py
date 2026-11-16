import hashlib
import torch

def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def model_hash(model):
    h=hashlib.sha256()
    for n,p in sorted(model.state_dict().items()):h.update(n.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def intervene(model,variant,offset):
    hooks=[]
    if variant.startswith('gate_'):
        value={'gate_0':0.,'gate_0p5':.5,'gate_1':1.}[variant]
        hooks.append(model.cluster_mapping.register_forward_hook(lambda module,args,out:torch.full_like(out,value)))
    elif variant=='mean_shift_auto':
        hooks.append(model.agent_proj[1].register_forward_pre_hook(lambda module,args:(args[0]+offset.to(args[0]),)))
    else:assert variant=='auto'
    return hooks

def clear(hooks):
    for h in hooks:h.remove()
