"""Verify byte identity of the report, scores, priors and scoring materials."""
from pathlib import Path
import argparse,json,hashlib
base=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');args=p.parse_args()
target=base/'result_files.json'
excluded={'scratch','models','__pycache__','.aris'}
paths=[v for v in base.rglob('*') if v.is_file() and not excluded.intersection(v.relative_to(base).parts) and v not in [target,base/'remote_verification.json']]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest={str(v.relative_to(base)):{'bytes':v.stat().st_size,'sha256':sha(v)} for v in sorted(paths)}
if args.register:
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
else:
    expected=json.loads(target.read_bytes())
    assert expected==manifest,[k for k in set(expected)|set(manifest) if manifest.get(k)!=expected.get(k)][:15]
    print(json.dumps({'ok':True,'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'manifest_sha256':sha(target)}))
