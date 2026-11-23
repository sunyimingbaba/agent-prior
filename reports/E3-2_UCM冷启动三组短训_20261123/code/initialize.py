"""Create one replayable cold start and lock the image lists before training."""
from shared import ROOT, ASSETS, REBUILD, read, dump, setup, sha
import torch
from rollout import load_mllm
from peft import LoraConfig, TaskType, get_peft_model

if __name__ == '__main__':
    setup()
    m, _ = load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'), 'cuda')
    torch.manual_seed(20261123)
    model = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        lora_dropout=0.05, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    blocks = [(name, p) for name,p in model.named_parameters() if '.lora_' in name]
    for _, param in blocks:
        param.data = param.data.float()
    assert blocks and all(p.dtype == torch.float32 for _,p in blocks)
    assert all(torch.count_nonzero(p) == 0 for name,p in blocks if '.lora_B.' in name)
    p = ROOT/'R0'/'adapter'
    assert not (p/'adapter_model.safetensors').exists()
    model.save_pretrained(p)
    old = REBUILD/'matched_policy_20261120/inputs'
    a,b = [read(old/k) for k in ('train.json','validation.json')]
    assert len(a)==24 and len(b)==210 and {x['image'] for x in a}.isdisjoint(x['image'] for x in b)
    assert {x['rgb_sha256'] for x in a}.isdisjoint(x['rgb_sha256'] for x in b)
    dump(ROOT/'initial.json', dict(seed=20261123,train=sha(old/'train.json'),val=sha(old/'validation.json'),
        adapter=sha(p/'adapter_model.safetensors'),parameters=sum(v.numel() for _,v in blocks), tensors=len(blocks)))
    print('COLD START READY', flush=True)
