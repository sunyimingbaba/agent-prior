"""Re-attest live frozen model sources and the validation Arrow against the prior run."""
import importlib.metadata
import subprocess
from pathlib import Path
from environment import ROOT, PREV, read, dump, digest
old=read(PREV/'asset_bindings.json')
files={}
for name, meta in old['files'].items():
    if 'ucm_prior_archive/agent_cache' in name or name.endswith('_train.arrow'):
        continue
    p=Path(name)
    actual=digest(p)
    assert actual==meta['sha256'] and p.stat().st_size==meta['bytes']
    files[name]=meta
dump(ROOT/'运行环境.json',dict(files=files,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
     mount=subprocess.check_output(['findmnt','-n','-T','/private','-o','SOURCE,TARGET,FSTYPE'],text=True).strip(),
     packages={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','pyarrow')},
     source_receipt=digest(PREV/'asset_bindings.json'),notes='Scored archival priors are individually bound by outputs/AGSE_archived/settings.json and replay records.'))
print(len(files),'live source assets verified')
