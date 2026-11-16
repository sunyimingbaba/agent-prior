"""Seal diagnostic cohorts, feedback, permutations and existing model references."""
import datetime
import hashlib
import importlib.metadata as md
import io
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image
from milestone_common import ROOT,PLAN,ASSETS,BASE,setup
from stage_io import read,dump,sha

torch=setup();p=Path(PLAN['matched_parent_root']);d=read(ROOT/'diagnostic_plan.json')
assert not (ROOT/'inputs/prepared.json').exists()
for n,v in read(p/'file_checksums.json').items():assert sha(p/n)==v['sha256'],n
for n,h in read(ROOT/'source_snapshot/inventory_before.json')['files'].items():assert sha(BASE/'CUP'/n)==h,n
expected=read(ROOT/'inputs/parent_environment.json')['packages'];actual={n:md.version(n) for n in expected}
assert actual==expected
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
x=torch.randn(16,16,device='cuda',requires_grad=True);w=torch.randn(16,16,device='cuda',requires_grad=True)
loss=(x@w).square().mean();loss.backward();assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
dump(ROOT/'runtime_environment.json',{'python':sys.version,'executable':sys.executable,'packages':actual,'packages_unchanged':True,'seed0_cuda_loss':float(loss.detach()),'finite_grad':True,'env_ledger_sha256':sha(ROOT/'environment_context/school-a100-legacy-20261026.md'),'previous_environment_sha256':sha(p/'runtime_environment.json'),'mount':subprocess.check_output(['findmnt','-T','/private','-o','SOURCE,TARGET,FSTYPE','-n'],text=True).strip(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv,noheader'],text=True).strip(),'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
old=read(p/'cache/records.json');bykey={(c['split'],c['row']):c for c in old}
q=read(ROOT/'inputs/quantity_cases.json');assert len(q)==85 and sum(c['split']=='train' for c in q)==63
keys=[('val',row) for row in range(210)]+[('train',c['row']) for c in q if c['split']=='train']
assert len(set(keys))==273
tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
cohort=[]
for split,row in keys:
 c=bykey[split,row];table=tables[split];img=Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')
 assert c['image']==table['path'][row].as_py() and hashlib.sha256(img.tobytes()).hexdigest()==c['rgb_sha256']
 tr=read(p/c['trace_file']);assert len(tr['tool_events'])==1 and tr['tool_events'][0]['origin']=='model_requested_actual_tool'
 e=tr['tool_events'][0]
 cohort.append({**c,'key':split+'_'+str(row).zfill(4),'size':list(img.size),'feedback':{'text':e['text'],'confidence':e['confidence'],'accepted':e['accepted']},'feedback_original_trace_sha256':sha(p/c['trace_file'])})
constant=next(c for c in cohort if c['image']=='102.tif' and c['split']=='train')
dump(ROOT/'inputs/cohort.json',cohort);dump(ROOT/'inputs/constant_feedback.json',constant)
for c in q:
 assert bykey[c['split'],c['row']]['rgb_sha256']==c['image_rgb_sha256'] and bykey[c['split'],c['row']]['image']==c['image']
perms={}
for split,n in [('train',1680),('val',210)]:
 rng=np.random.default_rng(d['permutations']['seed']+(0 if split=='train' else 1));xs=[]
 for k in range(20):
  while True:
   a=rng.permutation(n)
   if np.all(a!=np.arange(n)):break
  assert sorted(a.tolist())==list(range(n));xs.append(a.tolist())
 assert len({tuple(x) for x in xs})==20;perms[split]=xs
dump(ROOT/'inputs/permutations.json',perms)
lock=read(ROOT/'inputs/parent_selection_lock.json');assert sha(p/'selection_lock.json')==sha(ROOT/'inputs/parent_selection_lock.json')
for arm,v in lock['arms'].items():assert sha(p/v['checkpoint']['path'])==v['checkpoint']['sha256'],arm
shutil.copy2(p/'cache/dataset_sources.json',ROOT/'inputs/dataset_sources.json')
dump(ROOT/'inputs/parent_evidence_reference.json',{'root':str(p),'manifest_sha256':sha(p/'file_checksums.json'),'manifest_files':len(read(p/'file_checksums.json')),'selection_lock_sha256':sha(p/'selection_lock.json')})
dump(ROOT/'inputs/prepared.json',{'ok':True,'cohort_images':273,'val_images':210,'train_quantity_images':63,'conditions':8,'logical_state_slots':2184,'new_generation_cap':1911,'all20_split_permutations_sealed':True,'all_four_checkpoint_hashes_verified':True,'parent_manifest_files_verified':len(read(p/'file_checksums.json')),'test_data_opened':False,'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print('DIAGNOSTIC_INPUTS_SEALED',flush=True)
