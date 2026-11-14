"""Package the new adapter separately from the reproducibility evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

r=Path(__file__).resolve().parent.parent
assert str(r).startswith('/private/')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

checkpoint=r/'training/aligned_margin/checkpoint'
result=json.loads((r/'training/aligned_margin/result.json').read_text());assert result['ok']
archive=r.parent/'reward_alignment_aligned_margin_checkpoint_20261114.tar.gz'
assert not archive.exists()
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(checkpoint.iterdir()) if p.is_file()}
assert len(files)==12
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
    for name in files:t.add(checkpoint/name,arcname=name)
with tarfile.open(archive) as t:
    assert {p.name for p in t.getmembers()}==set(files)
    for member in t.getmembers():
        data=t.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest()==files[member.name]['sha256']
info={'archive_name':archive.name,'remote_archive':str(archive),'archive_bytes':archive.stat().st_size,
      'archive_sha256':sha(archive),'files':files,'lora_tensor_hash':result['final_lora_tensor_hash'],
      'attempted_groups':result['attempted_steps'],'optimizer_updates':result['optimizer_updates']}
manifest={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'policies':{'aligned_margin':info},
          'inherited_models':{'source_stage':'training_loop_20261114',
                              'source_manifest':'../训练闭环与短程对照_20261114/模型备份清单.json',
                              'policies':json.loads((r/'plan.json').read_text())['inherited_policies']}}
(r/'模型备份清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in info.items() if k!='files'}))
