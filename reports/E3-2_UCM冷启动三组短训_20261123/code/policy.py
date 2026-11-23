"""Use the saved FP32 LoRA exactly, including when the backbone loader chose BF16."""
from shared import ROOT, ASSETS, read
import torch
from safetensors.torch import load_file
from rollout import load_mllm, SYSTEM_PROMPT

def policy(arm):
    path = ROOT/arm/'adapter'
    model, proc = load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'), 'cuda', str(path))
    weights = load_file(str(path/'adapter_model.safetensors'))
    seen = set()
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if '.lora_' in name:
            key = name.replace('.default.', '.')
            assert weights[key].dtype == torch.float32
            param.data = weights[key].to(param.device)
            seen.add(key)
    assert seen == set(weights)
    return model.eval(), proc

def prompt(proc, query):
    messages = [
        {'role':'system', 'content':SYSTEM_PROMPT},
        {'role':'user','content':[{'type':'text','text':'查询文本: There is a harbor with several ships.'}]},
        {'role':'assistant','content':'<tool_call>scene_classify</tool_call>'},
        {'role':'user','content':'<result>harbor (0.61), port (0.24), ship (0.08)</result>'},
        {'role':'assistant','content':'<answer>图中是一座港口，与查询文本相符。</answer>'},
        {'role':'user','content':[{'type':'image'},{'type':'text','text':f'查询文本: {query}'}]},
    ]
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def cut(ids, eos):
    for i,token in enumerate(ids):
        if token in eos:
            return ids[:i+1]
    return ids
