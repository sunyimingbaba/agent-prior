"""Independent CPU reconstruction of evidence, selection and both decision gates."""
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
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();r=a.root;plan=json.loads((r/'plan.json').read_text(encoding='utf-8'))
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def tsha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
assert read('generation_pipeline_result_v3.json')['status']=='complete'
assert read('run_result.json')['status']=='complete'
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
for n,h in plan['shared_code_sha256'].items():assert sha(r/'code'/n)==h,n
for snap in ['generation_code_hashes.json','generation_retry_code_hashes.json','generation_v3_code_hashes.json','post_code_hashes.json']:
    for n,h in read(snap).items():assert sha(r/n)==h,n
for folder in ['failed_attempt_01','failed_attempt_02']:
    d=read(folder+'/diagnosis.json');assert d['status']=='failed'
    for n,info in d['preserved_files'].items():assert sha(r/folder/n)==info['sha256'],n
    for n,info in d.get('baseline_prefix_files',{}).items():assert sha(r/n)==info['sha256'],n
assert read('generation_pipeline_result.json')['phases'][-1]['exit_code']!=0
assert read('failed_attempt_01/diagnosis.json')['authoritative_original_status'].startswith('failed')

archive=r/'baseline_evidence/quantity_counterfactual_evidence_20261114.tar.gz'
assert sha(archive)==plan['baseline_archive_sha256']
scratch=r/'scratch';scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='independent_baseline_',dir=scratch) as td:
    br=Path(td)
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==1220 and len({x.name for x in members})==1220
        for m in members:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            out=br/m.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(t.extractfile(m).read())
    bm=json.loads((br/'file_checksums.json').read_text(encoding='utf-8'));assert len(bm)==1219
    for n,info in bm.items():assert sha(br/n)==info['sha256'],n
    baseline_output=br/'independent_output.json'
    result=subprocess.run([sys.executable,str(br/'code/verify_resumed_quantity.py'),'--root',str(br),'--out',str(baseline_output)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    baseline_verified=json.loads(baseline_output.read_text(encoding='utf-8'));assert baseline_verified['ok']
    baseline=read('baseline_evidence/scores.json')
    assert (br/'scoring/scores.json').read_bytes()==(r/'baseline_evidence/scores.json').read_bytes()
    old_trace={x['id']:json.loads((br/x['trace_file']).read_text(encoding='utf-8')) for x in baseline}

cases=read('inputs/cases.json');by={x['image']:x for x in cases};assert len(by)==100
cal=[]
for target in sorted({x['object'] for x in cases if x['split']=='train'}):
    xs=[x for x in cases if x['split']=='train' and x['object']==target]
    if len(xs)>=4:cal.extend(sorted(xs,key=lambda x:hashlib.sha256(('protocol_calibration_20261114:'+x['image']).encode()).hexdigest())[:3])
assert sorted(x['image'] for x in cal)==plan['calibration_images'] and len(cal)==15
cal=set(plan['calibration_images']);confirm=[x['image'] for x in cases if x['split']=='train' and x['image'] not in cal]
assert confirm==plan['confirmation_train_images'] and len(confirm)==63
assert not set(plan['policy']['inherited_optimization_images'])&set(by)
gallery=read('features/gallery.json');features=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True)
q=features['quantity_text_features'];v=features['original_val_text_features'];assert q.shape==(56,768) and v.shape==(419,768)
assert sha(r/'baseline_evidence/plan.json')==plan['baseline_plan_sha256']
assert sha(r/'baseline_evidence/scores.json')==plan['baseline_scoring_sha256']
records=[json.loads(s) for s in (r/'all_generation_records.jsonl').read_text(encoding='utf-8').splitlines()]
scores=read('scoring/all_scores.json');assert len(records)==len(scores)==626
assert len({x['id'] for x in scores})==626
assert [x['id'] for x in records]==[x['id'] for x in scores]
ts=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True)
assert ts['record_ids']==[x['id'] for x in scores]
q64=ts['image_features'].double()@q.double().T
assert float((q64-ts['quantity_similarities'].double()).abs().max())<1e-6
lookup={x:i for i,x in enumerate(ts['record_ids'])};vl={x:i for i,x in enumerate(ts['val_record_ids'])}
v64=ts['image_features'][[lookup[x] for x in ts['val_record_ids']]].double()@v.double().T
assert float((v64-ts['val_similarities'].double()).abs().max())<1e-6
tool_actual=read('replay_verification/actual_tool_replays.json');assert len(tool_actual)==100
answer=re.compile(r'<answer>(.*?)</answer>',re.S);call=re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')
strict_tool=re.compile(r'<tool_call>\s*scene_classify\s*</tool_call>')
prompts=read('inputs/protocol_prompts.json');rounds_checked=events_checked=classification_equal=ranks_equal=0
traces={};old_index={(x['image'],x['role'],x['seed']):x for x in baseline}
order={x['image']:i for i,x in enumerate(cases)}
for i,x in enumerate(scores):
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True)
    assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
    assert sha(r/x['state_file'])==x['state_file_sha256']
    digest=tsha(h);assert digest==x['generation_state_sha256']==x['cache_state_sha256']==x['scorer_state_sha256']
    assert sha(r/x['trace_file'])==x['trace_sha256']
    t=read(x['trace_file']);traces[x['id']]=t;rounds=t['rounds'];assert not t['forced_tool_witness'] and 1<=len(rounds)<=3
    assert digest==t['state_tensor_sha256']==rounds[-1]['state_tensor_sha256']
    final=rounds[-1]['generated_text'];matched=answer.search(final)
    usable=bool(matched and matched.group(1).strip()) and not call.search(final) and '<result>' not in final and '</result>' not in final \
        and t['stop_reason']=='answer' and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in rounds)
    strict_answer=answer.fullmatch(final.strip())
    strict=bool(usable and strict_answer and strict_answer.group(1).strip() and all(strict_tool.fullmatch(z['generated_text'].strip()) for z in rounds[:-1]))
    assert usable==x['quality_usable']==t['quality_usable'] and strict==x['strict_protocol_usable']
    assert x['tools_used']==t['tools_used']==len(t['tool_events'])
    for ri,z in enumerate(rounds):
        ids=z['generated_ids'];assert 1<=len(ids)==z['generated_token_count']<=256
        assert ids[0] not in z['eos_token_ids'] and all(a not in z['eos_token_ids'] for a in ids[:-1])
        assert len(z['prompt_token_ids'])==z['prompt_tokens'] and (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id']
        assert ids[-1]==z['last_emitted_token_id'] and hashlib.sha256(z['prompt_text'].encode()).hexdigest()==z['prompt_text_sha256']
        assert z['prompt_text'].endswith('<|im_start|>assistant\n') and z['message_roles']==['system','user','assistant','user','assistant','user']+['assistant','user']*ri
        kwargs=z['generation_kwargs'];assert kwargs['min_new_tokens']==1 and kwargs['top_k']==0 and kwargs['top_p']==1.
        assert kwargs['do_sample']==(x['role']=='candidate')
        assert kwargs['temperature']==(x['temperature'] if x['role']=='candidate' else 1.)
        if ri<len(rounds)-1:
            assert (call.fullmatch(z['generated_text'].strip()) if x['arm']=='legacy_reproduction' else strict_tool.fullmatch(z['generated_text'].strip()))
            event=next(e for e in t['tool_events'] if e['after_round']==ri+1)
            assert '<result>'+event['text']+'</result>' in rounds[ri+1]['prompt_text']
        rounds_checked+=1
    first=rounds[0]['prompt_text'];users=re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',first,re.S)
    assert users[-1].endswith('查询文本: '+plan['prompt'])
    if x['arm'].startswith('neutral_contract'):
        assert first.startswith('<|im_start|>system\n'+prompts['system']+'<|im_end|>')
        assert prompts['example_call'] in first and prompts['example_result'] in first and prompts['example_answer'] in first
    else:
        ref=old_index[x['image'],x['role'],x['seed']];assert first==old_trace[ref['id']]['rounds'][0]['prompt_text']
    for e in t['tool_events']:
        ref=tool_actual[x['image']];assert e['origin']=='model_requested_actual_tool' and e['text']==ref['text'] and abs(e['confidence']-ref['confidence'])<1e-6
        events_checked+=1
    if x['role']=='candidate':assert x['seed'] in [800000+order[x['image']]*4+k for k in range(4)]
    else:assert x['seed']==0 and x['temperature']==0
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        cp,cn=[float(ts['quantity_similarities'][i,c]) for c in cols]
        assert x[wording+'_positive']==cp and x[wording+'_negative']==cn and x['margin_'+wording]==cp-cn
        assert x['correct_'+wording]==(cp>cn)
        classification_equal+=bool(q64[i,cols[0]]>q64[i,cols[1]])==x['correct_'+wording]
    assert x['r_format']==(.2 if usable else 0.) and x['r_total_quantity']==x['r_format']+x['margin_A']
    if x['split']=='val':
        vi=vl[x['id']];sim=ts['val_similarities'][vi];pos=gallery['val_positive_columns'][x['image']]
        rank=int((sim[:391]>sim[pos].max()).sum());assert rank==x['retrieval_rank0'] and x['retrieval_rr']==1/(1+rank)
        ranks_equal+=int((v64[vi,:391]>v64[vi,pos].max()).sum())==rank
    else:assert x['retrieval_rr'] is None

repro=read('baseline_reproduction/result.json');assert repro['ok'] and len(repro['checks'])==6
for x in scores[:6]:
    ref=old_index[x['image'],x['role'],x['seed']];t=traces[x['id']];ot=old_trace[ref['id']]
    assert [z['generated_ids'] for z in t['rounds']]==[z['generated_ids'] for z in ot['rounds']]
    assert t['tool_events']==ot['tool_events'] and x['generation_state_sha256']==ref['generation_state_sha256']
stats=[]
for arm in plan['arms']:
    xs=[x for x in scores if x['arm']==arm['name'] and x['image'] in cal and x['role']=='candidate'];assert len(xs)==60
    quality=sum(x['quality_usable'] for x in xs)/60;strict=sum(x['strict_protocol_usable'] for x in xs)/60
    result_rate=sum(x['stop_reason']=='model_generated_tool_result' for x in xs)/60
    diverse=sum(len({x['generation_state_sha256'] for x in xs if x['image']==name and x['quality_usable']})>=2 for name in cal)
    g=plan['pilot_selection'];eligible=quality>=g['legacy_quality_min'] and strict>=g['strict_quality_min'] and result_rate<=g['generated_result_failure_rate_max'] and diverse>=g['images_with_2_valid_distinct_states_min']
    stats.append({'arm':arm['name'],'quality':quality,'strict_quality':strict,'generated_result_failure_rate':result_rate,'images_with_2_valid_distinct_states':diverse,'eligible':eligible})
selection=read('protocol_selection.json');assert stats==selection['pilot_stats'] and selection['used_calibration_train_only']
eligible=[x for x in stats if x['eligible']]
chosen=eligible[0] if eligible else sorted(stats,key=lambda x:(-x['quality'],x['generated_result_failure_rate'],[a['name'] for a in plan['arms']].index(x['arm'])))[0]
assert chosen['arm']==selection['selected_arm']['name'] and bool(eligible)==selection['pilot_criteria_satisfied']
full=read('scoring/scores.json');assert len(full)==500 and len({x['id'] for x in full})==500
all_by={x['id']:x for x in scores}
assert all(x==all_by[x['id']] and x['arm']==chosen['arm'] for x in full)
for item in cases:
    xs=[x for x in full if x['image']==item['image']];assert len(xs)==5 and sum(x['role']=='greedy' for x in xs)==1

quantity=read('analysis.json');rebuilt={}
metrics=['margin_A','margin_B','pairacc_A','pairacc_B','quality','tool_usage','retrieval_rr','retrieval_margin']
for item in cases:
    xs=[x for x in full if x['image']==item['image']];greedy=next(x for x in xs if x['role']=='greedy')
    sampled=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);valid=[x for x in sampled if x['quality_usable']]
    mapping={'greedy':[greedy],'sample_mean':sampled,'format_first':[valid[0] if valid else greedy],
      'quantity_raw':[max(sampled,key=lambda x:x['margin_A'])],'quantity_quality':[max(valid,key=lambda x:x['margin_A']) if valid else greedy],
      'reverse_quality':[min(valid,key=lambda x:x['margin_A']) if valid else greedy]}
    for selector,values in mapping.items():
        actual=next(x for x in quantity['per_image'] if x['image']==item['image'] and x['selector']==selector)
        assert actual['selected_ids']==[x['id'] for x in values]
        for m in metrics:
            ns=[float(x['correct_'+m[-1]]) if m.startswith('pairacc_') else float(x['quality_usable']) if m=='quality' else float(x['tools_used']>0) if m=='tool_usage' else x[m] for x in values]
            value=None if ns[0] is None else math.fsum(ns)/len(ns);rebuilt[item['image'],selector,m]=value
            assert value is None and actual[m] is None or abs(value-actual[m])<1e-12
ci_checks=0
for split in ['train','val']:
    names=[x['image'] for x in cases if x['split']==split];n=len(names);draw=np.random.default_rng(2026111418).integers(0,n,size=(10000,n))
    for selector,values in quantity['groups'][split]['selector_means'].items():
        for m,st in values.items():
            assert abs(float(np.mean([rebuilt[name,selector,m] for name in names]))-st['mean'])<1e-12
    for contrast,values in quantity['groups'][split]['paired'].items():
        s,b=contrast.split('_minus_')
        for m,st in values.items():
            ds=np.array([rebuilt[name,s,m]-rebuilt[name,b,m] for name in names]);assert abs(float(ds.mean())-st['mean'])<1e-12
            assert np.allclose(np.percentile(ds[draw].mean(axis=1),[2.5,97.5]),st['interval95'],atol=1e-12,rtol=0);ci_checks+=1
g=quantity['groups']['train'];th=plan['analysis']['next_training_gate'];gain=g['paired']['quantity_quality_minus_format_first']['margin_B']
informative=sum(len([x for x in full if x['image']==name and x['role']=='candidate' and x['quality_usable']])>=2 and
    max(x['margin_A'] for x in full if x['image']==name and x['role']=='candidate' and x['quality_usable'])-
    min(x['margin_A'] for x in full if x['image']==name and x['role']=='candidate' and x['quality_usable'])>1e-6 for name in [x['image'] for x in cases if x['split']=='train'])
checks={'training_candidate_protocol_quality':g['selector_means']['sample_mean']['quality']['mean']>=th['train_generation_quality_min'],
 'at_least_47_informative_training_images':informative>=th['train_images_with_2_usable_and_A_range_above_1e_6_min'],
 'B_margin_gain_at_least_1e_4':gain['mean']>=th['heldout_B_margin_gain_min'],'B_margin_gain_interval_above_zero':gain['interval95'][0]>0,
 'B_pairacc_at_least_format_first':g['selector_means']['quantity_quality']['pairacc_B']['mean']>=g['selector_means']['format_first']['pairacc_B']['mean'],
 'B_margin_vs_reverse_interval_above_zero':g['paired']['quantity_quality_minus_reverse_quality']['margin_B']['interval95'][0]>0,
 'selected_protocol_quality_at_least_95pct':g['selector_means']['quantity_quality']['quality']['mean']>=th['quality_selected_min']}
assert checks==quantity['decision']['checks'] and all(checks.values())==quantity['decision']['next_training_gate_passed']
protocol=read('protocol_analysis.json');assert protocol['selection']==selection
protocol_metrics=['quality','strict_quality','generated_result_failure','tool_usage','margin_B','pairacc_B','retrieval_rr']
strict_old={}
for x in baseline:
    t=old_trace[x['id']];aa=answer.fullmatch(t['rounds'][-1]['generated_text'].strip())
    strict_old[x['id']]=float(bool(x['quality_usable'] and aa and aa.group(1).strip() and all(strict_tool.fullmatch(z['generated_text'].strip()) for z in t['rounds'][:-1])))
for group,g in protocol['groups'].items():
    names=[x['image'] for x in cases if (x['image'] in cal if group=='calibration15' else x['split']=='train' and x['image'] not in cal if group=='confirmation_train63' else x['split']=='train' if group=='train78' else x['split']=='val')]
    n=len(names);draw=np.random.default_rng(2026111428).integers(0,n,size=(10000,n));pp={}
    for arm,source in [('baseline',baseline),('selected',full)]:
        for role in ['greedy','candidate']:
            for name in names:
                xs=[x for x in source if x['image']==name and x['role']==role]
                for m in protocol_metrics:
                    ns=[float(x['quality_usable']) if m=='quality' else strict_old[x['id']] if m=='strict_quality' and arm=='baseline'
                        else float(x['strict_protocol_usable']) if m=='strict_quality' else float(x['stop_reason']=='model_generated_tool_result') if m=='generated_result_failure'
                        else float(x['tools_used']>0) if m=='tool_usage' else float(x['correct_B']) if m=='pairacc_B' else x[m] for x in xs]
                    pp[name,arm,role,m]=None if ns[0] is None else math.fsum(ns)/len(ns)
            for m,st in g['means'][arm][role].items():
                ds=np.array([pp[name,arm,role,m] for name in names]);assert abs(float(ds.mean())-st['mean'])<1e-12
                assert np.allclose(np.percentile(ds[draw].mean(axis=1),[2.5,97.5]),st['interval95'],atol=1e-12,rtol=0);ci_checks+=1
    for role,values in g['paired'].items():
        for m,st in values.items():
            ds=np.array([pp[name,'selected',role,m]-pp[name,'baseline',role,m] for name in names]);assert abs(float(ds.mean())-st['mean'])<1e-12
            assert np.allclose(np.percentile(ds[draw].mean(axis=1),[2.5,97.5]),st['interval95'],atol=1e-12,rtol=0);ci_checks+=1
g=protocol['groups']['confirmation_train63'];th=plan['confirmation_gate'];candidate=g['means']['selected']['candidate']
selected_rate=float(np.mean([rebuilt[name,'quantity_quality','quality'] for name in confirm]))
checks={'sampled_quality_at_least_90pct':candidate['quality']['mean']>=th['legacy_quality_min'],
 'strict_sampled_quality_at_least_90pct':candidate['strict_quality']['mean']>=th['strict_quality_min'],
 'generated_result_failure_rate_at_most_2pct':candidate['generated_result_failure']['mean']<=th['generated_result_failure_rate_max'],
 'paired_quality_gain_interval_above_zero':g['paired']['candidate']['quality']['interval95'][0]>0,
 'selected_quality_at_least_95pct':selected_rate>=th['selected_quality_min']}
assert checks==protocol['confirmation_gate']['checks'] and all(checks.values())==protocol['confirmation_gate']['passed']
assert protocol['original_train78_quantity_gate']==quantity['decision'] and not protocol['confirmation_gate']['uses_calibration_and_val']
assert read('policy_settings.json')['lora_tensor_hash']==read('policy_settings_v3.json')['lora_tensor_hash']==plan['policy']['lora_tensor_hash']
assert read('generation_complete/result.json')['lora_tensor_hash_unchanged']==plan['policy']['lora_tensor_hash']
assert read('replay_verification/result.json')['ok']
out={'ok':True,'embedded_previous_1219_files_and_503_trajectories_verified':True,'exact_baseline_reproductions':6,
 'new_states_and_traces_checked':626,'generation_rounds_checked':rounds_checked,'real_tool_events_checked':events_checked,
 'neutral_actual_input_prompts_checked':626,'pilot_arms_rebuilt':3,'selected_main_records':500,'selectors_rebuilt':600,
 'bootstrap_intervals_rebuilt':ci_checks,'quantity_classifications_equal_float64':int(classification_equal),'quantity_classification_checks':1252,
 'val_ranks_equal_float64':int(ranks_equal),'val_rank_checks':len(vl),'confirmation_gate_rebuilt':True,'quantity_gate_rebuilt':True,
 'both_failed_attempts_preserved':True,'initial_misreported_status_identified':True,'optimization_updates':0,
 'torch':torch.__version__,'numpy':np.__version__}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('ALL_PROTOCOL_EVIDENCE_AND_DECISIONS_VERIFIED')
