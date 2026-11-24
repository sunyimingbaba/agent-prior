"""Strict model loads and native historical scoring methods."""
from shared import ROOT,ASSETS,ARROW,SCORER,sha
import torch
from prompt_clip import clip
from run_frozen import MyBackboneFinetuning
from vilt.modules.vilt_module import ViLTransformerSS
from agent.grpo.scorer import RetrievalScorer
orig=clip.load
def local_clip(name,device='cpu',jit=False,download_root=None):
    return orig(str(ASSETS/'models/ViT-L-14.pt') if name=='ViT-L/14' else name,device='cpu',jit=jit,download_root=download_root)
clip.load=local_clip
def load_model(proxy):
    path=SCORER if proxy else ASSETS/'ucm_main_domain_tool.ckpt'
    expected='e36801d50807519273408e4caf18604ad2f43db2c7b520cee98675631eac9a59' if proxy else '09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
    assert sha(path)==expected
    obj=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    cfg=dict(obj['hyper_parameters']['config']);cfg.update(test_only=False,load_path='')
    m=ViLTransformerSS(cfg)
    m.load_state_dict(obj['state_dict'],strict=True)
    assert m.use_agent_prior
    return m.float().eval().requires_grad_(False)
def scorer_for(m,cache):
    s=RetrievalScorer.__new__(RetrievalScorer)
    s.pl=m;s.device='cpu';s.val_arrow_path=str(ARROW);s.cache_path=str(cache)
    s.image_features,s.image_indices=s._load_or_build_cache()
    assert s.image_indices==list(range(1680)) and s.image_features.shape==(1680,768)
    return s
