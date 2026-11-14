import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def dump(p,j):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def tensor_sha(x):
    import torch
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def hydrate(r=ROOT):
    plan=read(r/'plan.json');archive=r/'parent_evidence'/plan['parent_archive_name'];assert sha(archive)==plan['parent_archive_sha256']
    target=r/'scratch/parent';target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:
        ms=t.getmembers();assert len(ms)==1788 and len({m.name for m in ms})==1788
        for m in ms:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            p=target/m.name
            if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
    manifest=read(target/'file_checksums.json');assert len(manifest)==1787
    for n,i in manifest.items():assert sha(target/n)==i['sha256'],n
    return target
def paired_records(parent):
    new=read(parent/'scoring/all_scores.json');old={(x['image'],x['role'],x['seed']):x for x in read(parent/'reference_run/scoring/all_scores.json') if x['arm']=='sft_strict'}
    assert len(new)==len(old)==425
    return new,old
