import argparse
import datetime
import gc
import hashlib
import io
import json
from pathlib import Path
import tarfile
import torch
from safetensors.torch import load

torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'模型备份清单.json').read_text());archive=r/'模型检查点'/j['archive_name'];h=hashlib.sha256()
with archive.open('rb') as f:
    for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()==j['archive_sha256'] and archive.stat().st_size==j['archive_bytes']
adapter_hash=None;optimizer_info=None;members_verified=0
with tarfile.open(archive) as t:
    members=t.getmembers();assert len(members)==len(j['files']) and {m.name for m in members}==set(j['files'])
    for member in members:
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        data=t.extractfile(member).read();info=j['files'][member.name]
        assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'];members_verified+=1
        if member.name=='adapter_model.safetensors':
            ts=load(data);actual={}
            for name,value in ts.items():
                assert value.dtype==torch.float32 and torch.isfinite(value).all()
                live=name.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
                assert live!=name and live not in actual;actual[live]=value
            h=hashlib.sha256()
            for name,value in sorted(actual.items()):h.update(name.encode());h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            adapter_hash=h.hexdigest();assert adapter_hash==j['lora_tensor_hash'] and len(actual)==584
            del actual,ts;gc.collect()
        elif member.name=='optimizer.pt':
            state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True);steps=[];values=0
            for item in state['state'].values():
                steps.append(float(item['step']))
                for name,value in item.items():
                    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all();values+=value.numel()
                assert (item['exp_avg_sq']>=0).all()
            assert steps and max(steps)==30 and min(steps)>=1 and max(steps)<=30
            for group in state['param_groups']:
                assert group['lr']==1e-5 and group['weight_decay']==0. and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
            optimizer_info={'parameter_states':len(steps),'min_parameter_step':min(steps),'max_parameter_step':max(steps),'finite_tensor_values':values}
            del state;gc.collect()
        del data
assert adapter_hash and optimizer_info
out={'ok':True,'archive_sha256':j['archive_sha256'],'archive_bytes':j['archive_bytes'],'member_hashes_verified':members_verified,
 'adapter_tensors_verified':584,'all_adapter_values_float32_finite':True,'live_adapter_parameter_hash':adapter_hash,
 'optimizer':optimizer_info,'optimizer_updates':30,'optimizer_and_rng_files_preserved':True,
 'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(r/'模型本机核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
