"""Re-run both query conditions on the first locked image from clean storage."""
import copy
from pathlib import Path
import torch
import generate
from shared import ROOT,locked,read,dump

cases=locked()
original_root=ROOT
generate.ROOT=ROOT/'repeat'
generate.locked=lambda: cases[:1]
generate.main()
checks=[]
for p in sorted((generate.ROOT/'samples').rglob('*.json')):
    src=original_root/p.relative_to(generate.ROOT)
    replay=read(p);first=read(src)
    assert replay['rounds']==first['rounds'] and replay['events']==first['events']
    assert replay['seed']==first['seed'] and replay['vector_sha256']==first['vector_sha256']
    assert torch.equal(torch.load(src.with_suffix('.pt'),weights_only=True),torch.load(p.with_suffix('.pt'),weights_only=True))
    checks.append(dict(image=first['image'],mode=first['mode'],k=first['k'],seed=first['seed'],vector_sha256=first['vector_sha256']))
assert len(checks)==18
dump(generate.ROOT/'verified.json',checks)
print('REPLAY_OK',len(checks),flush=True)
