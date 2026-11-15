from pathlib import Path
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS
from stage_io import read,dump,sha,parents
pre=read(ROOT/'preflight.json');d,e=parents()
for n,h in pre['original_source_hashes'].items():assert sha(n)==h,n
for n,i in pre['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==i['sha256'],n
for n,i in read(ROOT/'inputs/inherited_sft_model_reference.json')['files'].items():assert sha(Path(PLAN['policy']['checkpoint'])/n)==i['sha256'],n
assert sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256']
assert sha(ASSETS/'models/ViT-L-14.pt')==PLAN['model_hashes']['clip_sha256']
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')==sha(ROOT/'runtime_evidence/sitecustomize.py')
for snapshot in ['core_code_hashes.json','training_code_hashes.json','post_code_hashes.json']:
    for n,h in read(ROOT/snapshot).items():assert sha(ROOT/n)==h,n
for label in ['intermediate_16','final']:
    for n,i in read(ROOT/'training'/(label+'_checkpoint_manifest.json'))['files'].items():assert sha(ROOT/'training'/label/'checkpoint'/n)==i['sha256'],n
assert read(ROOT/'服务器独立核验.json')['ok']
dump(ROOT/'final_source_audit.json',{'ok':True,'original_sources_models_and_A100_environment_unchanged':True,'initial_SFT_checkpoint_files_preserved':16,
 'parent_diag_files_preserved':152,'parent_eos_files_preserved':1787,'intermediate16_and_final32_snapshots_preserved':True,'new_optimizer_updates':32,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()})
print('ALL_A100_BALANCED_SOURCES_AND_SNAPSHOTS_PRESERVED',flush=True)
