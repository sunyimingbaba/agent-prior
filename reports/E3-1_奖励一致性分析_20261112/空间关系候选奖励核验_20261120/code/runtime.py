"""Pin live generation and retrieval artifacts to the E1 environment and this train split."""
import importlib.metadata
import subprocess
from pathlib import Path
from shared import ROOT,BASE,ASSETS,read,dump,sha
previous=BASE.parent/'e1_ucm_validation_delivery_20261120/运行环境.json'
receipt=read(previous)
files={}
for name,info in receipt['files'].items():
    p=Path(name)
    if 'ucm_ccb_clustering_prior' in name or name.endswith('_val.arrow'):continue
    assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes'],name
    files[name]=info
for name,value in read(ROOT/'scoring/binding.json')['assets'].items():
    p=Path(name);assert sha(p)==value
    files[name]=dict(bytes=p.stat().st_size,sha256=value)
dump(ROOT/'运行环境.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        mount=subprocess.check_output(['findmnt','-n','-T','/private','-o','SOURCE,TARGET,FSTYPE'],text=True).strip(),
        packages={m:importlib.metadata.version(m) for m in ('torch','transformers','peft','pyarrow')},files=files,
        original_qwen_generation_config=read(ASSETS/'models/Qwen2.5-VL-7B-Instruct/generation_config.json'),
        explicit_rollout_parameters=dict(sample_temperature=1.,greedy_temperature=0.,max_new_tokens=64,max_rounds=3),
        previous_receipt_sha256=sha(previous)))
print(len(files))
