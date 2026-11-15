import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import torch

r=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
spec=json.loads((r/'environment_spec.json').read_text());assert Path(sys.prefix)==r/'env'
versions={n:metadata.version(n) for n in spec['target_versions']};assert versions==spec['target_versions'],versions
assert torch.cuda.get_device_capability()==(12,0) and 'sm_120' in torch.cuda.get_arch_list()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(0);torch.cuda.manual_seed_all(0)
x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
assert abs(float(loss.detach())-35.21732711791992)<1e-6 and torch.isfinite(x.grad).all()
for n,h in json.loads((r/'before_environment_hashes.json').read_text()).items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h
out={'ok':True,'versions':versions,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'capability':[12,0],'kernel_loss':float(loss.detach()),'finite_gradient':True,'original_environment_hashes_preserved':True,'spec_sha256':hashlib.sha256((r/'environment_spec.json').read_bytes()).hexdigest()}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))
