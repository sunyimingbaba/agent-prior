"""Bind the small delivery mirror; raw matrices live in the complete evidence archive."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
excluded={'scratch','__pycache__','原始运行记录','完成记录_20261117','归档分卷'}
sidecars={'交付文件索引.json','file_checksums.json','备份校验.json','服务器最终备份核验.json','最终归档审阅_20261117.json','最终归档审阅_20261117.md'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
files={}
for p in sorted(r.rglob('*')):
 if not p.is_file() or set(p.relative_to(r).parts)&excluded or p.name in sidecars or p.name.endswith(('.pyc','.partial','.tar.gz','.tar.xz')):continue
 assert p.stat().st_size<100*1024*1024,p
 files[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':sha(p)}
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'delivery_files':len(files),'scope':'friendly source/input/result/audit/report mirrors; raw complete archive and final backup sidecars are bound separately'}
(r/'交付文件索引.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('DELIVERY_FILES_BOUND',len(files))
