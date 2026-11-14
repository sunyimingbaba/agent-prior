"""Verify downloaded model archives against file and in-memory adapter hashes."""
import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import tarfile

from safetensors.torch import load
import torch
torch.set_num_threads(2)

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
path=r/'模型备份清单.json';manifest=json.loads(path.read_text())
for arm,info in manifest['policies'].items():
    archive=r/'模型检查点'/info['archive_name']
    assert archive.stat().st_size==info['archive_bytes']
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    assert h.hexdigest()==info['archive_sha256']
    model_hash=None;model_tensors=0
    with tarfile.open(archive) as t:
        members=t.getmembers();assert {x.name for x in members}==set(info['files'])
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            data=t.extractfile(member).read();expected=info['files'][member.name]
            assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
            if member.name=='adapter_model.safetensors':
                tensors=load(data);names={}
                for name,value in tensors.items():
                    assert value.dtype==torch.float32 and torch.isfinite(value).all()
                    actual=name.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
                    assert actual!=name and actual not in names;names[actual]=value
                tensor_hash=hashlib.sha256()
                for name,value in sorted(names.items()):
                    tensor_hash.update(name.encode());tensor_hash.update(value.contiguous().view(torch.uint8).numpy().tobytes())
                model_hash=tensor_hash.hexdigest();model_tensors=len(names)
                assert model_hash==info['lora_tensor_hash']
                del names,tensors,data;gc.collect()
    info.update(local_archive_sha256_equal=True,local_member_file_hashes_verified=len(members),
                local_adapter_tensor_hash=model_hash,local_adapter_tensors_verified=model_tensors,
                local_all_adapter_values_finite=True)
manifest.update(local_verification='passed',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'local_verification':'passed','policies':{arm:{k:info[k] for k in
    ['archive_bytes','local_member_file_hashes_verified','local_adapter_tensors_verified','local_adapter_tensor_hash']}
    for arm,info in manifest['policies'].items()}},ensure_ascii=False))
