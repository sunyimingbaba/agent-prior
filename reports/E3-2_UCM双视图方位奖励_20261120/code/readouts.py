"""Strict model loads and native historical scoring methods."""
from shared import ROOT,ASSETS,ARROW,SCORER,sha
import torch
from prompt_clip import clip
import __main__
from run_frozen import MyBackboneFinetuning
# Historical Lightning checkpoints pickle this callback under the training entry point.
__main__.MyBackboneFinetuning = MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS
orig=clip.load
def local_clip(name,device='cpu',jit=False,download_root=None):
    return orig(str(ASSETS/'models/ViT-L-14.pt') if name=='ViT-L/14' else name,device='cpu',jit=jit,download_root=download_root)
clip.load=local_clip
def load_model():
    path=ASSETS/'ucm_main_domain_tool.ckpt'
    expected='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
    assert sha(path)==expected
    obj=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    cfg=dict(obj['hyper_parameters']['config']);cfg.update(test_only=False,load_path='')
    m=ViLTransformerSS(cfg)
    m.load_state_dict(obj['state_dict'],strict=True)
    assert m.use_agent_prior
    return m.float().eval().requires_grad_(False).cuda()
