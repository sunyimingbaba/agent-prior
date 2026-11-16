"""Supplementary independent row identity, frozen-input and selected-prompt binding."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=read(r/'protocol_plan.json');cohort=read(r/'inputs/cohort.json');gallery=read(r/'evaluation/gallery.json');lock=read(r/'prompt_selection_lock_v2.json');records=read(r/'generation_v2/records.json')
assert len(cohort)==273 and len({x['key'] for x in cohort})==273 and len({x['image'] for x in cohort})==273
bound=0
for f in d['state_families']:
 m=read(r/'state_banks'/(f+'.json'))
 for split,n in [('train',63),('val',210)]:
  expected=[x for x in cohort if x['split']==split];assert len(expected)==len(m[split])==n
  for x,y in zip(m[split],expected):assert (x['key'],x['image'],x['row'])==(y['key'],y['image'],y['row']);bound+=1
val=[x for x in cohort if x['split']=='val'];assert [x['image'] for x in val]==gallery['names']==read(a.parent/'evaluation/gallery.json')['names']
assert [x['row'] for x in val]==list(range(210))
for n in ['quantity_cases.json','quantity_gallery.json','parent_selection_lock.json','frozen_path_assets.pt']:
 assert (r/'inputs'/n).read_bytes()==(a.parent/'inputs'/n).read_bytes(),n
assert (r/'inputs/cohort.json').read_bytes()==(a.parent/'inputs/cohort.json').read_bytes()
selected=[x for x in records if x['group']=='selected_validation'];assert len(selected)==210 and all(x['split']=='val' and x['prompt']==lock['chosen_prompt'] and x['policy_hash']==lock['policy_hash'] for x in selected)
train=[x for x in records if x['group']==lock['selected_train_group']];assert len(train)==63
reference=read(r/train[0]['trace_file'])['rounds'][0]
for x in selected:
 tr=read(r/x['trace_file']);first=tr['rounds'][0];assert first['prompt_text']==reference['prompt_text'] and first['message_roles']==reference['message_roles'] and tr['software_forced_tool'] is False
assert lock['chosen_prompt']=='contract_only' and reference['message_roles']==['system','user']
assert reference['prompt_text'].count('<|im_start|>assistant\n')==1 and '当前任务的交互顺序：第一回合只输出完整的 <tool_call>scene_classify</tool_call>。' in reference['prompt_text']
out={'ok':True,'state_bank_rows_bound_to_frozen_cohort':bound,'val_names_and_Arrow_rows_bound':210,'frozen_input_files_byte_identical_to_parent':5,'selected_validation_prompts_bound_to_train_lock':210,'software_forced_calls':0,'frozen_input_sha256':{n:sha(r/'inputs'/n) for n in ['cohort.json','quantity_cases.json','quantity_gallery.json','parent_selection_lock.json','frozen_path_assets.pt']}}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
