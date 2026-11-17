"""Independent actual-device group normalization reconstruction; no model load or optimization."""
from pathlib import Path
import hashlib
import json
import sys
import torch
r=Path(sys.argv[1]).resolve();arm=sys.argv[2];root=r/'training_v2'/arm;assert json.loads((root/'result.json').read_text())['ok'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
rows=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()];updates=[json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()];result=[];assert len(rows)==256 and len(updates)==32
for i,u in enumerate(updates):
 rewards=torch.tensor([x['reward'] for x in rows[i*8:(i+1)*8]],device='cuda',dtype=torch.float32);mean=rewards.mean();std=rewards.std(unbiased=True);adv=(rewards-mean)/(std+1e-8) if float(std)>1e-8 else torch.zeros_like(rewards);assert float(mean)==u['reward_mean'] and float(std)==u['reward_std'] and torch.equal(adv.cpu(),torch.tensor(u['advantages'],dtype=torch.float32));result.append({'attempt':i+1,'rewards_float32':rewards.cpu().tolist(),'mean_float32':float(mean),'std_float32':float(std),'advantages_float32':adv.cpu().tolist(),'all_recorded_values_bit_exact':True})
out={'ok':True,'arm':arm,'groups':32,'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'optimizer_updates':0,'model_loaded':False,'source_samples_sha256':hashlib.sha256((root/'samples.jsonl').read_bytes()).hexdigest(),'source_updates_sha256':hashlib.sha256((root/'updates.jsonl').read_bytes()).hexdigest(),'records':result};path=root/'independent_cuda_normalization.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'}))
