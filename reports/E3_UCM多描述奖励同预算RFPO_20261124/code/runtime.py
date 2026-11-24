from pathlib import Path
import os,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
REBUILD=Path('/private/AGSE_E3_1_rebuild_20261112')
TRAIN=REBUILD/'second_answer_5090_20261124'
OLD=REBUILD/'r3_after_tool_trainpanel_20261124'
LAST=REBUILD/'paired_answer_preference_20261124'
HIST=REBUILD/'r3_informative_groups_20261124'
FORM=REBUILD/'after_tool_holdout_20261124'
sys.path.insert(0,str(TRAIN/'code'))
from shared import BASE,ASSETS,read,dump,sha,setup,table,rgb
os.environ.update(TMPDIR=str(ROOT/'scratch'),HF_HOME=str(ROOT/'scratch/hf'),MPLCONFIGDIR=str(ROOT/'scratch/mpl'),XDG_CACHE_HOME=str(ROOT/'scratch/cache'))
def pool():
    rows={}
    for sub in ('','confirm'):
        path=OLD/sub
        assert sha(path/'inputs/panel.json')==read(path/'inputs/lock.json')['panel']
        tools=read(path/'records/tools.json')
        for row in read(path/'inputs/panel.json'):
            nm=row['image'];assert nm not in rows
            rows[nm]=dict(row,sub=sub,feedback=tools[nm][0])
    return rows
def initial(seed):
    path=TRAIN/('repeat' if seed=='b' else '')
    assert sha(path/'R0/adapter/adapter_model.safetensors')==read(path/'initial.json')['adapter']
    return path/'R0'
def savept(path,obj):
    import torch
    temp=path.with_suffix('.part');torch.save(obj,temp);temp.replace(path)
def key(*x):return hashlib.sha256(('|'.join(map(str,x))).encode()).hexdigest()
