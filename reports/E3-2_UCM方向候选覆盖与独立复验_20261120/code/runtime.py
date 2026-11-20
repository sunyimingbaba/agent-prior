import importlib.metadata,subprocess
from pathlib import Path
from shared import ROOT,BASE,ASSETS,REBUILD,read,dump,sha
previous=REBUILD/'relation_candidates_20261120/运行环境.json'
files={}
for name,info in read(previous)['files'].items():
    file=Path(name)
    assert sha(file)==info['sha256'] and file.stat().st_size==info['bytes'],name
    files[name]=info
for file in [ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow',REBUILD/'relation_candidates_20261120/gallery/train.pt',
             REBUILD/'relation_candidates_20261120/native/features.pt',REBUILD/'relation_transfer_20261120/val/features.pt']:
    files[str(file)]=dict(bytes=file.stat().st_size,sha256=sha(file))
dump(ROOT/'environment.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
    private_mount=subprocess.check_output(['findmnt','-n','-T','/private','-o','SOURCE,TARGET,FSTYPE'],text=True).strip(),
    packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','pyarrow','scipy','numpy','safetensors']},
    files=files,previous_receipt=sha(previous)))
