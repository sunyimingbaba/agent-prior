from pathlib import Path
import sys
import torch
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'soft_guard_20261125'
sys.path.insert(0,str(OLD/'code'))
sys.path.insert(0,str(ROOT.parent/'source_transfer_2h_20261125/code'))
from remote_common import setup,table,rgb,read,dump,sha,savept
from eval_io import datasets,weights
from policy import policy,prompt
from prefixes import greedy
from head_data import reader
TRANS=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
def locked():
    z=read(ROOT/'inputs/lock.json')
    for n in ('train','holdout'):assert sha(ROOT/'inputs'/(n+'.json'))==z[n]
    assert sha(ROOT/'实验口径.md')==z['protocol']
    for name,h in z['implementation'].items():assert sha(ROOT/'code'/name)==h
    return z
def parent(seed):return OLD/'mined_policy'/(seed+'_selected_joint')
def hidden_replay(m,proc,image,query,feedback,actions,calls):
    inp=proc(text=[prompt(proc,query)],images=[image],return_tensors='pt').to('cuda')
    assert len(calls)>=len(actions)-1
    ids=inp.input_ids
    for tokens,fb in zip(actions[:-1],calls):
        assert fb==feedback
        decoded=proc.decode(tokens,skip_special_tokens=True)
        add=proc(text=[decoded+f'\n<result>{fb}</result>\n'],return_tensors='pt').input_ids.cuda()
        ids=torch.cat((ids,add),1)
    last=torch.tensor(actions[-1][:-1],device=ids.device,dtype=ids.dtype)[None]
    ids=torch.cat((ids,last),1)
    fw=m(**dict(inp,input_ids=ids,attention_mask=torch.ones_like(ids)),output_hidden_states=True,use_cache=False,logits_to_keep=1,return_dict=True)
    return fw.hidden_states[-1][0,-1].float()
