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
    for n,h in z['manifests'].items():assert sha(ROOT/'inputs'/n)==h
    assert sha(ROOT/'实验口径.md')==z['protocol']
    for name,h in z['implementation'].items():assert sha(ROOT/'code'/name)==h
    return z
