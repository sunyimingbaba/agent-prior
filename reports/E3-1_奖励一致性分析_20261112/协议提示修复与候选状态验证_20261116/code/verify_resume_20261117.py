"""Independently bind completed evidence to the immutable651-record paused snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paused',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
manifest=read(a.paused/'file_checksums.json');assert len(manifest)==1527 and read(r/'resume_20261117/before/file_checksums.json')==manifest
for n,v in manifest.items():
 assert sha(a.paused/n)==v['sha256'] and (a.paused/n).stat().st_size==v['bytes'],n
 if n!='generation_v2/records.jsonl':assert sha(r/n)==v['sha256'],n
before=(a.paused/'generation_v2/records.jsonl').read_bytes();full=(r/'generation_v2/records.jsonl').read_bytes();assert full.startswith(before) and (r/'resume_20261117/before/records.jsonl').read_bytes()==before
old=[json.loads(s) for s in before.splitlines()];new=[json.loads(s) for s in full[len(before):].splitlines()];allrows=read(r/'generation_v2/records.json');assert len(old)==651 and len(new)==21 and allrows==old+new
val=[c for c in read(r/'inputs/cohort.json') if c['split']=='val'];assert [x['key'] for x in new]==[c['key'] for c in val[189:]] and all(x['group']=='selected_validation' and x['prompt']=='contract_only' for x in new)
by={(x['group'],x['key']):x for x in allrows};warm=read(r/'resume_20261117/warm_replay.json');assert warm['ok'] and len(warm['replays'])==3
for x in warm['replays']:
 assert sha(r/x['tensor_file'])==x['tensor_sha256'] and sha(r/x['trace_file'])==x['trace_sha256'];source=by[x['record_group'],x['key']];one=torch.load(r/x['tensor_file'],map_location='cpu',weights_only=True);two=torch.load(r/source['tensor_file'],map_location='cpu',weights_only=True)
 assert len(one['step_states'])==len(two['step_states']) and all(torch.equal(u,v) for u,v in zip(one['step_states'],two['step_states'])) and all(torch.equal(v,two['candidates'][k]) for k,v in one['candidates'].items())
 first=read(r/x['trace_file']);second=read(r/source['trace_file']);assert [q['generated_ids'] for q in first['rounds']]==[q['generated_ids'] for q in second['rounds']]
g=read(r/'generation_result_v2.json');assert g['resumed_existing_records']==651 and g['new_generations_after_resume']==21 and g['supplemental_exact_warm_replays']==3 and g['Agent_optimizer_updates']==g['readout_optimizer_updates']==0 and g['initial_policy_hash']==g['final_policy_hash']
pre=read(r/'resume_20261117/preflight.json');assert pre['ok'] and pre['unchanged_environment'] and pre['packages']==read(a.paused/'runtime_environment.json')['packages'] and pre['seed0_cuda_loss']==35.21732711791992 and pre['finite_grad']
assert read(r/'resume_job_result_20261117.json')['status']=='complete'
out={'ok':True,'paused_manifest_files_verified':len(manifest),'paused_existing_files_preserved_exact':len(manifest)-1,'original_record_journal_prefix_preserved':651,'new_records_verified':21,'exact_warm_replays_verified':3,'selected_prompt_lock_preserved':True,'runtime_packages_unchanged':True,'Agent_optimizer_updates':0,'readout_optimizer_updates':0}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
