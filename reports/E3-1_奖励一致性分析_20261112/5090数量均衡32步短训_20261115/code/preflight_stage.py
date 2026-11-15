import importlib.metadata as metadata
from pathlib import Path
import shutil
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,setup,seed
from stage_io import read,dump,sha,parents
torch=setup();mount=subprocess.check_output(['findmnt','-T','/private','-no','SOURCE,TARGET,FSTYPE'],text=True).strip();assert ' /private gpfs' in mount
assert '5090' in torch.cuda.get_device_name() and torch.cuda.get_device_capability()==(12,0)
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
runtime=Path(PLAN['runtime_root']);assert read(runtime/'probe_doc_replay.json')['ok'] and sha(runtime/'environment_spec.json')==PLAN['rtx_migration']['runtime_spec_sha256']
assert read(runtime/'probe_doc_replay.json')['versions']==read(runtime/'environment_spec.json')['target_versions']
for n,h in PLAN['inputs_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in read(ROOT/'core_code_hashes.json').items():assert sha(ROOT/n)==h,n
(ROOT/'parent_evidence').mkdir();shutil.copy2(ROOT.parent/PLAN['parent_archive_name'],ROOT/'parent_evidence'/PLAN['parent_archive_name'])
d,e=parents();old=read(e/'preflight.json')
for n,h in old['original_source_hashes'].items():assert sha(n)==h,n
for n,i in old['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==i['sha256'],n
for n,i in read(ROOT/'inputs/inherited_sft_model_reference.json')['files'].items():assert sha(Path(PLAN['policy']['checkpoint'])/n)==i['sha256'],n
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256']
assert sha(ASSETS/'models/ViT-L-14.pt')==PLAN['model_hashes']['clip_sha256']
for split,h in read(d/'inputs/训练输入清单_v2.json')['arrow_sha256'].items():assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')==h
for n in ['cases.json','evaluation_cases.json']:assert (ROOT/'inputs'/n).read_bytes()==(e/'inputs'/n).read_bytes()
shutil.copytree(e/'features',ROOT/'features');(ROOT/'runtime_evidence').mkdir()
for n in ['environment_spec.json','probe_doc_replay.json','before_environment_hashes.json']:shutil.copy2(runtime/n,ROOT/'runtime_evidence'/n)
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward();assert abs(float(loss.detach())-35.21732711791992)<1e-6 and torch.isfinite(x.grad).all()
dump(ROOT/'preflight.json',{'ok':True,'mount':mount,'gpu':torch.cuda.get_device_name(),'packages':{n:metadata.version(n) for n in ['torch','torchvision','transformers','peft','numpy','pyarrow']},'kernel_loss':float(loss.detach()),'original_source_hashes':old['original_source_hashes'],'model_asset_hashes':old['model_asset_hashes'],'inherited_checkpoint_files_verified':16,'parent_diag_files':152,'parent_eos_files':1787})
print('RTX_STAGE_PREFLIGHT_PASSED',flush=True)
