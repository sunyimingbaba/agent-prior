import os,sys,hashlib,json
from pathlib import Path
RUN=Path(os.environ.get('E3_PANEL_ROOT',Path(__file__).resolve().parents[1]))
PREVIOUS=Path('/private/AGSE_E3_1_rebuild_20261112/second_answer_5090_20261124')
sys.path.insert(0,str(PREVIOUS/'code'))
from shared import ASSETS,ARROW,REBUILD,dump,read,rgb,sha,table,setup
os.environ.update(TMPDIR=str(RUN/'scratch'),HF_HOME=str(RUN/'scratch/hf'),MPLCONFIGDIR=str(RUN/'scratch/mpl'),XDG_CACHE_HOME=str(RUN/'scratch/cache'))
def panel():
    lock=read(RUN/'inputs/lock.json')
    assert sha(RUN/'inputs/panel.json')==lock['panel'] and sha(RUN/'实验口径.md')==lock['protocol'] and sha(ARROW)==lock['train_arrow']
    return read(RUN/'inputs/panel.json')
def model_path(seed):
    assert seed in ('a','b')
    return PREVIOUS/('repeat' if seed=='b' else '')/'R2'
def weight(seed):
    path=model_path(seed)
    sig=sha(path/'adapter/adapter_model.safetensors')
    assert sig==read(path/'complete.json')['final']
    return sig
def savept(path,obj):
    import torch
    temp=path.with_suffix('.part');torch.save(obj,temp);temp.replace(path)
