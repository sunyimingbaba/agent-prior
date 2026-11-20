"""Portable SHA-256 inventory for the distributed result tree; ignore runtime caches."""
import hashlib
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=root/'文件校验.json'
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
if sys.argv[1]=='make':
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {manifest.name,'双端校验.json','.DS_Store'}
           and not {'__pycache__','scratch'} & set(p.relative_to(root).parts)]
    manifest.write_text(json.dumps({p.relative_to(root).as_posix():[p.stat().st_size,sha(p)] for p in files},indent=2,ensure_ascii=False))
elif sys.argv[1]!='check':
    raise SystemExit('choose make or check')
data=json.loads(manifest.read_bytes())
for name,(size,expected) in data.items():
    p=root/name
    if p.stat().st_size!=size or sha(p)!=expected:
        raise ValueError(name)
print(json.dumps(dict(files=len(data),bytes=sum(info[0] for info in data.values()),manifest_sha256=sha(manifest))))
