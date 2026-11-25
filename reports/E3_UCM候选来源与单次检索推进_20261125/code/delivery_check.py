"""Build or verify a relative SHA-256 file inventory without relying on locale-specific path output."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'results'/'delivery_sha256.json'
def digest(path):
    with path.open('rb') as fh:return hashlib.file_digest(fh,'sha256').hexdigest()
def included(path):
    rel=path.relative_to(ROOT)
    return path.is_file() and path!=INDEX and rel.parts[0]!='logs' and not (set(rel.parts)&{'__pycache__','.git','.DS_Store'}) and path.suffix not in ('.pyc','.part') and not path.name.startswith('._')
def main():
    if sys.argv[1]=='build':
        paths=sorted((p for p in ROOT.rglob('*') if included(p)),key=lambda p:p.as_posix())
        payload={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
        INDEX.write_text(json.dumps(payload,ensure_ascii=False,indent=1)+'\n')
    data=json.loads(INDEX.read_bytes())
    for name,z in data.items():
        p=ROOT/name;assert p.is_file() and p.stat().st_size==z['bytes'] and digest(p)==z['sha256'],name
    print(len(data),'files',sum(z['bytes'] for z in data.values()),'bytes; manifest',digest(INDEX))
if __name__=='__main__':main()
