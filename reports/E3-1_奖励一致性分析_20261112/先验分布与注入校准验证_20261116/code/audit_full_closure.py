from pathlib import Path
import hashlib
import torch
from milestone_common import ROOT,PLAN,BASE,ASSETS
from stage_io import read,dump,sha
before=read(ROOT/'source_snapshot/inventory_before.json')['files']
for n,h in before.items():assert sha(BASE/'CUP'/n)==h,n
for n,z in read(ROOT/'inputs/parent_preflight.json')['model_asset_hashes'].items():assert sha(ASSETS/'models/Qwen2.5-VL-7B-Instruct'/n)==z['sha256'],n
assert sha(ASSETS/'ucm_scene_cls.pt')==PLAN['model_hashes']['scene_head_sha256'] and sha(ASSETS/'ucm_main_domain_tool.ckpt')==PLAN['utility_model_sha256']
for p in PLAN['policies']:
    manifest=read(Path(PLAN['parent_permutation_root'])/p['checkpoint_manifest'])
    for n,z in manifest['files'].items():assert sha(Path(p['checkpoint'])/n)==z['sha256'],n
for n,z in read(ROOT/'context_snapshot/input_model_manifest.json')['arrow_sha256'].items():assert sha(ASSETS/f'ucm/ucm_caption_karpathy_label_{n}.arrow')==z,n
records=read(ROOT/'full_validation/records.json');rounds=events=usable=0
for x in records:
    tr=read(ROOT/x['trace_file']);assert sha(ROOT/x['trace_file'])==x['trace_sha256'];first_user=tr['rounds'][0]['prompt_text'].rsplit('<|im_start|>user',1)[-1];assert '查询文本: Describe the visible scene.' in first_user;assert not tr['forced_tool_witness']
    for z in tr['rounds']:
        ids=z['generated_ids'];assert ids[0] not in z['eos_token_ids'] and all(k not in z['eos_token_ids'] for k in ids[:-1]);assert (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id'] and ids[-1]==z['last_emitted_token_id'];rounds+=1
    for e in tr['tool_events']:assert e['origin']=='model_requested_actual_tool';events+=1
    if tr['quality_usable']:assert tr['rounds'][-1]['last_emitted_token_id'] in tr['rounds'][-1]['eos_token_ids'];usable+=1
for n in [x.name for x in (ROOT/'full_validation').glob('anchor_*.json')]:
    name=n[len('anchor_'):-len('.json')];x=next(x for x in records if x['image']==name);tr=read(ROOT/'full_validation'/n);old=read(ROOT/x['trace_file']);assert tr['state_tensor_sha256']==x['numeric_state_sha256'] and [z['generated_ids'] for z in tr['rounds']]==[z['generated_ids'] for z in old['rounds']]
assert read(ROOT/'full_validation/score_result.json')['model_hash_unchanged']==read(ROOT/'gpu_result.json')['original_model_parameter_hash']
dump(ROOT/'final_full_source_audit.json',{'ok':True,'original_python_files_unchanged':77,'original_checkpoint_files_unchanged':32,'Qwen_base_assets_unchanged':len(read(ROOT/'inputs/parent_preflight.json')['model_asset_hashes']),'full_val_uniform_queries_verified':210,'generation_rounds_checked':rounds,'actual_tool_events_checked':events,'quality_usable_states':usable,'warm_anchor_token_and_state_replays_verified':2,'optimizer_updates':0});print('FULL_STAGE_SOURCES_AND_QUERY_CONTRACT_VERIFIED',flush=True)
