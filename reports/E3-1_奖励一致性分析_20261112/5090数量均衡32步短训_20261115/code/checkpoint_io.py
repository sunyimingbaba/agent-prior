"""Restore saved float32 adapter values after creating the PEFT modules."""
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file


def restore_adapter_exact(model,path):
    saved=load_file(str(path/'adapter_model.safetensors'),device='cpu')
    for name,p in model.named_parameters():
        if 'lora_' in name:p.data=p.data.float()
    set_peft_model_state_dict(model,saved,adapter_name='default')
    restored=get_peft_model_state_dict(model,adapter_name='default')
    assert set(restored)==set(saved)
    assert all(torch.equal(restored[k].detach().cpu().float(),saved[k].float()) for k in saved)
    return {'adapter_tensors':len(saved),'saved_float32_adapter_restored_exactly':True}
