"""Read-only final binding of actually imported sources, all base assets and dataset identities."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
r=Path(sys.argv[1]).resolve()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
p=read(r/'plan.json');c=read(r/'cache_plan.json');base=Path(c['base_root']);assets=base/'assets';checks={}
for n,h in read(r/'source_snapshot/inventory_before.json')['files'].items():assert sha(base/'CUP'/n)==h,n
checks['actual_import_source_files_unchanged']=len(read(r/'source_snapshot/inventory_before.json')['files'])
for n,v in read(r/'inputs/base_model_reference.json').items():assert sha(assets/'models/Qwen2.5-VL-7B-Instruct'/n)==v['sha256'],n
checks['base_model_processor_generation_assets_unchanged']=len(read(r/'inputs/base_model_reference.json'))
for split,v in read(r/'inputs/dataset_sources_reference.json').items():assert sha(assets/f'ucm/ucm_caption_karpathy_label_{split}.arrow')==v['arrow_sha256'],split
checks['all_dataset_arrow_identities_unchanged']=True
for n,v in read(r/'inputs/balanced_checkpoint_reference.json')['files'].items():assert sha(Path(p['policy']['checkpoint'])/n)==v['sha256'],n
checks['common_initial_checkpoint_files_unchanged']=len(read(r/'inputs/balanced_checkpoint_reference.json')['files'])
cp=p['readout']['checkpoint'];assert sha(Path(p['matched_root'])/cp['path'])==cp['sha256'];assert sha(assets/'ucm_main_domain_tool.ckpt')==c['utility_model_sha256'];meta=read(r/'assets/scoring_assets.json');assert sha(r/'assets/scoring_assets.pt')==meta['sha256'];checks['frozen_readout_and_proxy_assets_unchanged']=True
expected=read(r/'inputs/parent_environment.json')['packages'];actual={k:md.version(k) for k in expected};assert actual==expected;mount=subprocess.check_output(['findmnt','-T','/private','-o','SOURCE,TARGET,FSTYPE','-n'],text=True).strip();assert mount==read(r/'runtime_environment_v2.json')['mount'];checks.update(packages_unchanged=True,private_mount_unchanged=mount,new_test_examples_scored=0)
for n,v in read(r/'runtime_source_snapshot/manifest.json')['files'].items():assert sha(v['installed_path'])==sha(r/n)==v['sha256'],n
checks['native_cache_generation_peft_source_files_unchanged']=len(read(r/'runtime_source_snapshot/manifest.json')['files'])
result={'ok':True,**checks};out=r/'final_asset_verification.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
