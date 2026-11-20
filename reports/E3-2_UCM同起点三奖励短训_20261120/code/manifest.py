"""Compute an identical local/server manifest over the complete reusable experiment."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
target=root/'文件校验.json'
def skip(p):
    a=p.relative_to(root)
    return p.name in ('.DS_Store','文件校验.json','双端校验.json') or any(s in ('__pycache__','scratch') for s in a.parts) or p.suffix in ('.pyc','.part')
def digest(p):
    with p.open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()
files={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and not skip(p)}
if '--verify' in sys.argv:
    ref=json.loads(target.read_bytes())
    assert ref==files,sorted(k for k in set(ref)|set(files) if ref.get(k)!=files.get(k))[:8]
else:
    target.write_text(json.dumps(files,ensure_ascii=False,indent=2))
print(len(files),sum(z['bytes'] for z in files.values()),digest(target))
