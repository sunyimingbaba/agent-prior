import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def unpack(archive,dest,digest,count):
    assert sha(archive)==digest;dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:
        ms=t.getmembers();assert len(ms)==count+1 and len({m.name for m in ms})==count+1
        for m in ms:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            p=dest/m.name
            if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
    manifest=read(dest/'file_checksums.json');assert len(manifest)==count
    for n,i in manifest.items():assert sha(dest/n)==i['sha256'],n
    return dest
def parents(r=ROOT):
    p=read(r/'plan.json');d=unpack(r/'parent_evidence'/p['parent_archive_name'],r/'scratch/diagnostic',p['parent_archive_sha256'],152)
    e=unpack(d/'parent_evidence'/p['source_eos_archive_name'],r/'scratch/eos',p['source_eos_archive_sha256'],1787)
    return d,e
