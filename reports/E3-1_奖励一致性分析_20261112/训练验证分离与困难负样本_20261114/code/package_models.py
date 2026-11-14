"""Create separate, verified model archives for the actual training branches."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

r=Path(__file__).resolve().parent.parent;assert str(r).startswith('/private/')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

branch=json.loads((r/'pilot/branch_decision.json').read_text());policies={}
for arm in branch['arms']:
    checkpoint=r/'training'/arm/'checkpoint';result=json.loads((checkpoint.parent/'result.json').read_text());assert result['ok']
    archive=r.parent/f'separated_{arm}_checkpoint_20261114.tar.gz';assert not archive.exists()
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(checkpoint.iterdir()) if p.is_file()}
    assert len(files)==12
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for name in files:t.add(checkpoint/name,arcname=name)
    with tarfile.open(archive) as t:
        assert {m.name for m in t.getmembers()}==set(files)
        for m in t.getmembers():assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==files[m.name]['sha256']
    policies[arm]={'archive_name':archive.name,'remote_archive':str(archive),'archive_bytes':archive.stat().st_size,
        'archive_sha256':sha(archive),'files':files,'lora_tensor_hash':result['final_lora_tensor_hash'],
        'attempted_groups':result['attempted_steps'],'optimizer_updates':result['optimizer_updates']}
out={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policies':policies}
(r/'模型备份清单.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({arm:{k:v for k,v in info.items() if k!='files'} for arm,info in policies.items()}))
