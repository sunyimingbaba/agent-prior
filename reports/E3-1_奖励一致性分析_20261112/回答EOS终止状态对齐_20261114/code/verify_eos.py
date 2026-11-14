import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import numpy as np
import torch

torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def tsha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
plan=read('plan.json');assert read('generation_pipeline_result.json')['status']=='complete'
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
for snap in ['generation_code_hashes.json','post_code_hashes.json']:
    for n,h in read(snap).items():assert sha(r/n)==h,n
archive=r/'parent_evidence'/plan['parent_archive_name'];assert sha(archive)==plan['parent_archive_sha256']
scratch=r/'scratch';scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='eos_parent_verify_',dir=scratch) as td:
    parent=Path(td)
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==2001 and len({x.name for x in members})==2001
        for m in members:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            q=parent/m.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read())
    manifest=json.loads((parent/'file_checksums.json').read_text(encoding='utf-8'));assert len(manifest)==2000
    for n,i in manifest.items():assert sha(parent/n)==i['sha256'],n
    check=subprocess.run([sys.executable,str(parent/'code/verify_supervision.py'),'--root',str(parent),'--out',str(parent/'parent_recheck.json')],capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    parent_check=json.loads((parent/'parent_recheck.json').read_text(encoding='utf-8'));assert parent_check['ok']
    for n,i in read('reference_file_checksums.json').items():
        assert sha(r/'reference_run'/n)==i['sha256'] and (r/'reference_run'/n).read_bytes()==(parent/n).read_bytes(),n
    for n in ['features/tensors.pt','features/gallery.json','inputs/cases.json','inputs/evaluation_cases.json']:
        assert (r/n).read_bytes()==(parent/n).read_bytes(),n
    old_original=json.loads((parent/'baseline_evidence/scores.json').read_text())
    old_original_traces={}
    with tarfile.open(parent/'baseline_evidence/quantity_counterfactual_evidence_20261114.tar.gz') as t:
        for x in old_original:old_original_traces[x['id']]=json.loads(t.extractfile(x['trace_file']).read())
ref_scores=read('reference_run/scoring/all_scores.json');ref={(x['image'],x['role'],x['seed']):x for x in ref_scores if x['arm']=='sft_strict'};assert len(ref)==425
cases=read('inputs/evaluation_cases.json');assert len(cases)==85 and not {x['image'] for x in cases}&set(plan['inherited_format_training_images'])
scores=read('scoring/all_scores.json');assert len(scores)==425 and len({x['id'] for x in scores})==425
f=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True);t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True)
assert t['record_ids']==[x['id'] for x in scores]
q64=t['image_features'].double()@f['quantity_text_features'].double().T
assert float((q64-t['quantity_similarities'].double()).abs().max())<1e-6
lookup={x:i for i,x in enumerate(t['record_ids'])};vl={x:i for i,x in enumerate(t['val_record_ids'])};assert len(vl)==110
v64=t['image_features'][[lookup[x] for x in t['val_record_ids']]].double()@f['original_val_text_features'].double().T
assert float((v64-t['val_similarities'].double()).abs().max())<1e-6
gallery=read('features/gallery.json');actual=read('replay_verification/actual_tool_replays.json')
answer=re.compile(r'<answer>(.*?)</answer>',re.S);tool=re.compile(r'<tool_call>\s*scene_classify\s*</tool_call>');any_tool=re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')
rounds=events=classification=ranks=unchanged=only_eos=same_text=eos_ended=0
for i,x in enumerate(scores):
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
    assert sha(r/x['state_file'])==x['state_file_sha256'] and tsha(h)==x['generation_state_sha256']==x['cache_state_sha256']==x['scorer_state_sha256']
    assert sha(r/x['trace_file'])==x['trace_sha256'];trace=read(x['trace_file']);old=ref[x['image'],x['role'],x['seed']];previous=read('reference_run/'+old['trace_file'])
    assert len(trace['rounds'])==len(previous['rounds']) and 1<=len(trace['rounds'])<=3 and not trace['forced_tool_witness']
    for ri,(z,o) in enumerate(zip(trace['rounds'],previous['rounds'])):
        ids=z['generated_ids'];assert z['prompt_token_ids']==o['prompt_token_ids'] and z['prompt_text']==o['prompt_text']
        assert ids[:len(o['generated_ids'])]==o['generated_ids']
        if ri<len(trace['rounds'])-1:assert ids==o['generated_ids'] and tool.fullmatch(z['generated_text'].strip())
        assert 1<=len(ids)==z['generated_token_count']<=256 and ids[0] not in z['eos_token_ids'] and all(v not in z['eos_token_ids'] for v in ids[:-1])
        assert (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id'] and ids[-1]==z['last_emitted_token_id']
        assert z['prompt_tokens']==len(z['prompt_token_ids']) and hashlib.sha256(z['prompt_text'].encode()).hexdigest()==z['prompt_text_sha256']
        kw=z['generation_kwargs'];assert kw['min_new_tokens']==1 and kw['top_k']==0 and kw['top_p']==1. and kw['temperature']==1. and kw['do_sample']==(x['role']=='candidate')
        assert z['message_roles']==['system','user','assistant','user','assistant','user']+['assistant','user']*ri
        rounds+=1
    assert len(trace['tool_events'])==len(previous['tool_events'])==x['tools_used']
    for e,o in zip(trace['tool_events'],previous['tool_events']):
        assert {k:v for k,v in e.items() if k!='confidence'}=={k:v for k,v in o.items() if k!='confidence'} and abs(e['confidence']-o['confidence'])<=1e-6
        obs=actual[x['image']];assert e['origin']=='model_requested_actual_tool' and e['text']==obs['text'] and abs(e['confidence']-obs['confidence'])<1e-6;events+=1
    final=trace['rounds'][-1];oldfinal=previous['rounds'][-1];text=final['generated_text'];matched=answer.search(text)
    usable=bool(matched and matched.group(1).strip()) and not any_tool.search(text) and '<result>' not in text and '</result>' not in text and trace['stop_reason']=='answer' \
      and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in trace['rounds'])
    aa=answer.fullmatch(text.strip());strict=bool(usable and aa and aa.group(1).strip() and all(tool.fullmatch(z['generated_text'].strip()) for z in trace['rounds'][:-1]))
    assert usable==x['quality_usable']==trace['quality_usable'] and strict==x['strict_protocol_usable']
    assert trace['state_tensor_sha256']==final['state_tensor_sha256']==x['generation_state_sha256']
    if usable:assert final['last_emitted_token_id'] in final['eos_token_ids']
    same_text+=text==oldfinal['generated_text'];eos_ended+=final['last_emitted_token_id'] in final['eos_token_ids']
    if final['generated_ids']==oldfinal['generated_ids']:
        assert x['generation_state_sha256']==old['generation_state_sha256'];unchanged+=1
    if len(final['generated_ids'])==len(oldfinal['generated_ids'])+1 and final['generated_ids'][:-1]==oldfinal['generated_ids'] and final['generated_ids'][-1] in final['eos_token_ids']:
        assert final['state_input_token_id']==oldfinal['last_emitted_token_id'];only_eos+=1
    assert x['reference_id']==old['id'] and x['reference_prefix_equal'] and x['reference_tools_equal']
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        cp,cn=[float(t['quantity_similarities'][i,c]) for c in cols]
        assert x[wording+'_positive']==cp and x[wording+'_negative']==cn and x['margin_'+wording]==cp-cn and x['correct_'+wording]==(cp>cn)
        classification+=bool(q64[i,cols[0]]>q64[i,cols[1]])==x['correct_'+wording]
    if x['split']=='val':
        vi=vl[x['id']];s=t['val_similarities'][vi];pos=gallery['val_positive_columns'][x['image']];rank=int((s[:391]>s[pos].max()).sum())
        assert rank==x['retrieval_rank0'] and x['retrieval_rr']==1/(rank+1);ranks+=int((v64[vi,:391]>v64[vi,pos].max()).sum())==rank
    else:assert x['retrieval_rr'] is None
result=read('generation/result.json');settings=read('generation/settings.json')
assert result['lora_tensor_hash_unchanged']==settings['policy_lora_tensor_hash']==plan['policy']['lora_tensor_hash'] and result['optimization_updates']==0
for item in cases:
    xs=[x for x in scores if x['image']==item['image']];assert len(xs)==5
    assert sum(x['role']=='greedy' for x in xs)==1
    assert sorted(x['seed'] for x in xs if x['role']=='candidate')==[800000+xs[0]['image_order']*4+k for k in range(4)]
historical=[]
for x in old_original:
    if x['image'] not in {z['image'] for z in cases}:continue
    z=dict(x,arm='historical_legacy');tr=old_original_traces[x['id']];aa=answer.fullmatch(tr['rounds'][-1]['generated_text'].strip())
    z['strict_protocol_usable']=bool(z['quality_usable'] and aa and aa.group(1).strip() and all(tool.fullmatch(v['generated_text'].strip()) for v in tr['rounds'][:-1]));historical.append(z)
all_records=scores+ref_scores+historical;analysis=read('analysis.json');metrics=['quality','strict_quality','tool_usage','generated_result_failure','margin_A','margin_B','pairacc_A','pairacc_B','retrieval_rr','retrieval_margin'];rebuilt={}
for arm in ['historical_legacy','c742_strict','sft_strict','sft_legacy_greedy','sft_fulltool_eos']:
    for item in cases:
        xs=[x for x in all_records if x['arm']==arm and x['image']==item['image']];greedy=next(x for x in xs if x['role']=='greedy');sampled=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);valid=[x for x in sampled if x['quality_usable']]
        mapping={'greedy':[greedy]}
        if arm!='sft_legacy_greedy':mapping.update(sample_mean=sampled,format_first=[valid[0] if valid else greedy],quantity_raw=[max(sampled,key=lambda x:x['margin_A'])],quantity_quality=[max(valid,key=lambda x:x['margin_A']) if valid else greedy],reverse_quality=[min(valid,key=lambda x:x['margin_A']) if valid else greedy])
        for selector,values in mapping.items():
            entry=next(x for x in analysis['per_image'] if x['arm']==arm and x['image']==item['image'] and x['selector']==selector);assert entry['selected_ids']==[x['id'] for x in values]
            for m in metrics:
                ns=[float(x['quality_usable']) if m=='quality' else float(x['strict_protocol_usable']) if m=='strict_quality' else float(x['tools_used']>0) if m=='tool_usage' else float(x['stop_reason']=='model_generated_tool_result') if m=='generated_result_failure' else float(x['correct_'+m[-1]]) if m.startswith('pairacc_') else x[m] for x in values]
                v=None if ns[0] is None else math.fsum(ns)/len(ns);rebuilt[arm,item['image'],selector,m]=v
                assert v is None and entry[m] is None or abs(v-entry[m])<1e-12
ci=0
for split,g in analysis['groups'].items():
    names=[x['image'] for x in cases if x['split']==split];n=len(names);draw=np.random.default_rng(2026111444).integers(0,n,size=(10000,n))
    def check(st,values):
        global ci
        d=np.asarray(values,dtype=np.float64);assert abs(float(d.mean())-st['mean'])<1e-12 and np.allclose(np.percentile(d[draw].mean(axis=1),[2.5,97.5]),st['interval95'],atol=1e-12,rtol=0);ci+=1
    for arm,selectors in g['means'].items():
        for selector,values in selectors.items():
            for m,st in values.items():check(st,[rebuilt[arm,name,selector,m] for name in names])
    for arm,comparisons in g['quantity_paired'].items():
        for contrast,values in comparisons.items():
            selected,base=contrast.split('_minus_')
            for m,st in values.items():check(st,[rebuilt[arm,name,selected,m]-rebuilt[arm,name,base,m] for name in names])
    for contrast,values in g['paired'].items():
        policies,selector=contrast.split('__');arm,base=policies.split('_minus_')
        for m,st in values.items():check(st,[rebuilt[arm,name,selector,m]-rebuilt[base,name,selector,m] for name in names])
g=analysis['groups']['train'];m=g['means'];threshold=plan['adoption_gate'];new=m['sft_fulltool_eos']
checks={'strict_sample_quality_at_least_90pct':new['sample_mean']['strict_quality']['mean']>=threshold['strict_sample_quality_min'],
 'strict_greedy_quality_at_least_95pct':new['greedy']['strict_quality']['mean']>=threshold['strict_greedy_quality_min'],
 'sample_actual_tool_use_at_least_80pct':new['sample_mean']['tool_usage']['mean']>=threshold['sample_actual_tool_use_min'],
 'generated_result_failure_at_most_2pct':new['sample_mean']['generated_result_failure']['mean']<=threshold['generated_result_failure_max'],
 'sample_B_pairacc_at_least_c742_strict':new['sample_mean']['pairacc_B']['mean']>=m['c742_strict']['sample_mean']['pairacc_B']['mean'],
 'greedy_B_pairacc_drop_at_most_5pp_vs_historical':new['greedy']['pairacc_B']['mean']>=m['historical_legacy']['greedy']['pairacc_B']['mean']-threshold['greedy_B_pairacc_drop_max_vs_historical_legacy'],
 'B_margin_gain_interval_above_zero':g['paired']['sft_fulltool_eos_minus_sft_strict__sample_mean']['margin_B']['interval95'][0]>0,
 'all_425_prefixes_match':all(x['reference_prefix_equal'] for x in scores),'policy_parameter_hash_unchanged':result['lora_tensor_hash_unchanged']==plan['policy']['lora_tensor_hash'] and result['optimization_updates']==0}
assert checks==analysis['adoption']['checks'] and all(checks.values())==analysis['adoption']['passed']
informative=0
for item in [x for x in cases if x['split']=='train']:
    valid=[x['margin_A'] for x in scores if x['image']==item['image'] and x['role']=='candidate' and x['quality_usable']];informative+=len(valid)>=2 and max(valid)-min(valid)>1e-6
th=plan['analysis']['next_training_gate'];gain=g['quantity_paired']['sft_fulltool_eos']['quantity_quality_minus_format_first']['margin_B']
qc={'sample_quality_at_least_90pct':new['sample_mean']['quality']['mean']>=th['train_generation_quality_min'],'at_least_38_informative_images':informative>=38,
 'B_margin_gain_at_least_1e_4':gain['mean']>=th['heldout_B_margin_gain_min'],'B_margin_gain_interval_above_zero':gain['interval95'][0]>0,
 'B_pairacc_at_least_format_first':new['quantity_quality']['pairacc_B']['mean']>=new['format_first']['pairacc_B']['mean'],
 'B_margin_vs_reverse_interval_above_zero':g['quantity_paired']['sft_fulltool_eos']['quantity_quality_minus_reverse_quality']['margin_B']['interval95'][0]>0,
 'selected_quality_at_least_95pct':new['quantity_quality']['quality']['mean']>=th['quality_selected_min']}
assert qc==analysis['quantity_readiness']['checks'] and all(qc.values())==analysis['quantity_readiness']['passed']
assert analysis['terminal_summary']=={'records':425,'same_decoded_text':same_text,'added_only_eos':only_eos,'unchanged_ids':unchanged,'new_last_is_eos':eos_ended}
assert read('replay_verification/result.json')['ok']
out={'ok':True,'new_states_and_traces_checked':425,'all_reference_token_prefixes_verified':425,'all_reference_tools_verified':425,'generation_rounds_checked':rounds,'real_tool_events_checked':events,
 'same_decoded_text_records':same_text,'only_eos_appended_records':only_eos,'unchanged_sequence_and_state_records':unchanged,
 'selectors_rebuilt':len(analysis['per_image']),'bootstrap_intervals_rebuilt':ci,'quantity_classifications_equal_float64':int(classification),'quantity_classification_checks':850,
 'val_ranks_equal_float64':int(ranks),'val_rank_checks':110,'parent_935_trajectories_and_30_updates_reverified':True,
 'adoption_and_quantity_gates_rebuilt':True,'optimizer_updates':0,'policy_lora_tensor_hash':plan['policy']['lora_tensor_hash'],'torch':torch.__version__,'numpy':np.__version__}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('ALL_EOS_CONTINUATIONS_AND_STATE_ALIGNMENT_DECISIONS_VERIFIED')
