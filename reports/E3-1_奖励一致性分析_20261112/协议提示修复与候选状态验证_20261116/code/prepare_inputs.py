"""Warm environment, executed-supervision masks, byte alignment and fixed cohorts."""
import datetime
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
from PIL import Image
from transformers import AutoTokenizer
from milestone_common import ROOT,PLAN,BASE,ASSETS,setup,seed
from stage_io import read,dump,sha
from state_contract import body_layout

torch=setup();d=read(ROOT/'protocol_plan.json');parent=Path(d['diagnostic_parent']);sft=Path(d['sft_parent']);rl=Path(d['rl_parent'])
assert not (ROOT/'inputs/prepared.json').exists()
for n,h in read(ROOT/'generation_code_hashes.json').items():assert sha(ROOT/n)==h,n
for n,v in read(parent/'file_checksums.json').items():assert sha(parent/n)==v['sha256'],n
for n,h in read(ROOT/'source_snapshot/inventory_before.json')['files'].items():assert sha(BASE/'CUP'/n)==h,n
expected=read(ROOT/'inputs/parent_environment.json')['packages'];actual={n:md.version(n) for n in expected};assert actual==expected
seed(0);x=torch.randn(16,16,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward();assert torch.isfinite(x.grad).all() and abs(float(loss.detach())-35.21732711791992)<1e-6
dump(ROOT/'runtime_environment.json',{'python':sys.version,'executable':sys.executable,'packages':actual,'packages_unchanged':True,'seed0_cuda_loss':float(loss.detach()),'finite_grad':True,'ledger_sha256':sha(ROOT/'environment_context/school-a100-legacy-20261026.md'),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv,noheader'],text=True).strip(),'mount':subprocess.check_output(['findmnt','-T','/private','-o','SOURCE,TARGET,FSTYPE','-n'],text=True).strip(),'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
for n in ['teacher_examples.json','actual_teacher_tools.json','result.json']:
 shutil.copy2(sft/'training'/n,ROOT/'inputs'/('sft_'+n))
shutil.copy2(sft/'inputs/cases.json',ROOT/'inputs/sft_all_cases.json')
for path in [sft,rl]:
 for name in ['training_code_hashes.json','core_code_hashes.json']:
  if not (path/name).exists():continue
  for n,h in read(path/name).items():assert sha(path/n)==h,n
tokenizer=AutoTokenizer.from_pretrained(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),local_files_only=True);assert tokenizer.is_fast
decoder=json.loads(tokenizer.backend_tokenizer.decoder.__getstate__());assert decoder['type']=='ByteLevel'
dump(ROOT/'inputs/tokenizer_decoder.json',decoder);(ROOT/'inputs/tokenizer_snapshot.json').write_text(tokenizer.backend_tokenizer.to_str())
spec=importlib.util.spec_from_file_location('audited_supervision',ROOT/'source_executed/sft/supervised_format.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
teachers=read(ROOT/'inputs/teacher_sources.json');bodies={x['image']:x['source_body'] for x in teachers};examples=read(ROOT/'inputs/sft_teacher_examples.json');rows=[]
for example in examples:
 target=module.target_tokens(tokenizer,example['kind'],bodies[example['image']] if example['kind']=='answer' else '')
 assert target['target_ids']==example['target_ids'] and target['loss_mask']==example['loss_mask'] and target['offsets']==example['offsets']
 assert 'There is a harbor with several ships.' in example['prompt_text'] and example['prompt_text'].endswith('<|im_start|>assistant\n')
 if example['kind']=='answer':assert example['message_roles'][-2:]==['assistant','user'] and '<result>' in example['prompt_text']
 rows.append({'id':example['id'],'kind':example['kind'],'active_tokens':sum(target['loss_mask']),'target_tokens':len(target['target_ids']),'active_ids':[x for x,m in zip(target['target_ids'],target['loss_mask']) if m],'user_feedback_loss_positions':0,'answer_body_loss_positions':0})
assert len(rows)==30
dump(ROOT/'supervision_mask_audit.json',{'ok':True,'examples':rows,'all30_legacy_example_prompts':True,'per_epoch_active_labels':sum(x['active_tokens'] for x in rows),'two_epoch_active_labels':2*sum(x['active_tokens'] for x in rows),'scope':'executed SFT targets only; earlier inherited adapter supervision provenance not established here','tokenizer_snapshot_sha256':sha(ROOT/'inputs/tokenizer_snapshot.json')})
cohort=read(ROOT/'inputs/cohort.json');assert len(cohort)==273 and sum(x['split']=='train' for x in cohort)==63
tables={s:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
contracts=[]
for c in cohort:
 table=tables[c['split']];assert table['path'][c['row']].as_py()==c['image'];img=Image.open(io.BytesIO(table['image'][c['row']].as_py())).convert('RGB');assert hashlib.sha256(img.tobytes()).hexdigest()==c['rgb_sha256']
 tr=read(Path(PLAN['matched_parent_root'])/c['trace_file']);step=tr['rounds'][-1];tokens=tokenizer.convert_ids_to_tokens(step['generated_ids']);layout=body_layout(step['generated_ids'],tokens,step['eos_token_ids'],step['generated_text'])
 contracts.append({'image':c['image'],'key':c['key'],'layout':layout,'original_last_numeric_sha256':c['numeric_state_sha256']})
assert all(x['layout']['available'] for x in contracts), 'Original valid-answer byte alignment changed'
dump(ROOT/'inputs/previous_body_alignment.json',contracts)
tools=read(ROOT/'inputs/sft_actual_teacher_tools.json');rot=[];seen=set()
for name in sorted(tools):
 text=tools[name]['text'];top=text.split(',')[0].split(' (')[0].strip()
 if top in seen:continue
 seen.add(top);rot.append({'source_image':name,'top1':top,'feedback':tools[name],'source_file_sha256':sha(ROOT/'inputs/sft_actual_teacher_tools.json'),'role':'training-only textual format demonstration, target classifier output replayed separately'})
 if len(rot)==3:break
assert len(rot)==3 and not set(x['source_image'] for x in rot)&set(c['image'] for c in cohort);dump(ROOT/'inputs/rotating_examples.json',rot)
old=read(parent/'inputs/permutations.json')['val'];assert len(old)==20;random=np.random.default_rng(2026111609);new=[]
for _ in range(20):
 while True:
  z=random.permutation(63)
  if (z!=np.arange(63)).all():break
 new.append(z.tolist())
dump(ROOT/'inputs/permutations.json',{'train':new,'val':old})
dump(ROOT/'inputs/prepared.json',{'ok':True,'cohort':273,'train_prompt_selection_images':63,'val_confirmation_images':210,'rotating_examples':3,'teacher_target_masks_verified':30,'prior_body_alignment_available':sum(x['layout']['available'] for x in contracts),'parent_files_unchanged':len(read(parent/'file_checksums.json')),'test_examples_scored':0})
print('PROTOCOL_MASK_AND_TOKEN_CONTRACT_PREFLIGHT_COMPLETE',flush=True)
