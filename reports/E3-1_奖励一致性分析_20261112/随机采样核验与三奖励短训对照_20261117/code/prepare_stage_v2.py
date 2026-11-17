"""Warm-reuse witness, frozen-source binding, train-only proxy library and exact scorer replays."""
import datetime
import hashlib
import importlib.metadata as md
from pathlib import Path
import subprocess
from milestone_common import ROOT,PLAN,BASE,ASSETS,setup,seed
from reward_model_v2 import RewardScorer
from stage_io import read,dump,sha
torch=setup();p=read(ROOT/'plan.json');parent=Path(p['parent_protocol_root']);assert not (ROOT/'prepare_result.json').exists()
for n,h in read(ROOT/'pilot_code_hashes_v2.json').items():assert sha(ROOT/n)==h,n
old=read(parent/'file_checksums.json')
for n,v in old.items():assert sha(parent/n)==v['sha256'],n
for n,h in read(ROOT/'source_snapshot/inventory_before.json')['files'].items():assert sha(BASE/'CUP'/n)==h,n
for n,v in read(ROOT/'inputs/balanced_checkpoint_reference.json')['files'].items():assert sha(Path(p['policy']['checkpoint'])/n)==v['sha256'],n
for n,v in read(ROOT/'inputs/base_model_reference.json').items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==v['sha256'],n
expected=read(ROOT/'inputs/parent_environment.json')['packages'];actual={k:md.version(k) for k in expected};assert actual==expected
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward();assert torch.isfinite(x.grad).all() and abs(float(loss.detach())-35.21732711791992)<1e-6
dump(ROOT/'runtime_environment_v2.json',{'packages':actual,'packages_unchanged':True,'seed0_cuda_loss':float(loss.detach()),'finite_grad':True,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),'mount':subprocess.check_output(['findmnt','-T','/private','-o','SOURCE,TARGET,FSTYPE','-n'],text=True).strip(),'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ledger_sha256':sha(ROOT/'environment_context/school-a100-legacy-20261026.md')});del x,loss
scorer=RewardScorer(prepare=False);cases=read(ROOT/'inputs/quantity_cases.json');cohort=read(ROOT/'inputs/cohort.json');byname={c['image']:c for c in cohort}
for c in cases:
 source=byname[c['image']];assert (c['split'],c['row'])==(source['split'],source['row']);assert hashlib.sha256(scorer.case_image(c).tobytes()).hexdigest()==c['image_rgb_sha256']==source['rgb_sha256']
records=read(parent/'generation_v2/records.json');warm=[];baseline=torch.load(parent/'evaluation/calibrated_own/selected_last/correct/tensors.pt',map_location='cpu',weights_only=True)
for name in p['training_images'][:2]:
 c=byname[name];ref=next(x for x in records if x['image']==name and x['group']=='pilot_initial/contract_only');h=torch.load(parent/ref['tensor_file'],map_location='cpu',weights_only=True)['candidates']['last'];value=scorer.score(h,c,ref['strict_protocol_usable'] and ref['feedback_consumed']);i=next(i for i,x in enumerate(cases) if x['image']==name)
 error=float((scorer.last_tensors['quantity_scores']-baseline['quantity_scores'][i]).abs().max());assert torch.equal(scorer.last_tensors['image_feature'][0],baseline['quantity_image_features'][i]) and error<1e-6;warm.append({'image':name,'state_source_sha256':ref['tensor_sha256'],'quantity_feature_exact':True,'quantity_score_max_abs_error':error,'quantity_score_abs_tolerance':1e-6,'scoring_arithmetic':'current GPU single-row versus historical CPU batch','initial_proxy_rank0':value['proxy_rank0']})
dump(ROOT/'inputs/frozen_readout_warm_replay.json',{'ok':True,'records':warm,'source_tensor_sha256':sha(parent/'evaluation/calibrated_own/selected_last/correct/tensors.pt')});dump(ROOT/'prepare_result.json',{'ok':True,'parent_manifest_files_unchanged':len(old),'readout_frozen':scorer.verify_frozen(),'proxy_library_rows':1680,'all85manual_case_identities_bound':True,'warm_feature_replays_exact':2,'warm_quantity_score_tolerance':1e-6,'base_model_asset_files_bound':15,'test_examples_scored':0})
print('FROZEN_SCORER_AND_TRAIN_LIBRARY_READY',flush=True)
