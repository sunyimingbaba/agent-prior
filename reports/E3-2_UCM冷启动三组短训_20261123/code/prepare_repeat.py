"""Materialize the independent paired rerun before viewing the primary outcomes."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

def create(target):
    source=Path(__file__).resolve().parent
    dest=target/'code'
    dest.mkdir(parents=True,exist_ok=False)
    (target/'scratch/cache/torch/kernels').mkdir(parents=True)
    changed={}
    for p in source.glob('*.py'):
        data=p.read_text(encoding='utf-8')
        if p.name=='initialize.py':
            assert data.count('20261123')==2
            data=data.replace('20261123','20261124')
        if p.name=='train.py':
            assert data.count('284000+j')==1
            data=data.replace('284000+j','385000+j')
        (dest/p.name).write_text(data,encoding='utf-8')
        changed[p.name]=hashlib.sha256(data.encode()).hexdigest()
    (target/'repeat_lock.json').write_text(json.dumps(dict(initial_seed=20261124,sampling_seed_start=385000,code=changed),indent=2),encoding='utf-8')

if __name__=='__main__':
    create(Path(sys.argv[1]))
