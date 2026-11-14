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
from verify_training import verify_training

torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def tsha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
plan=read('plan.json');assert read('control_result.json')['status']==read('training_pipeline_result.json')['status']=='complete'
for n,h in plan['inputs_sha256'].items():assert sha(r/n)==h,n
for n,h in plan['shared_code_sha256'].items():assert sha(r/'code'/n)==h,n
for snap in ['control_code_hashes.json','training_code_hashes.json','post_code_hashes.json']:
    for n,h in read(snap).items():assert sha(r/n)==h,n
training=verify_training(r);assert training['ok']
cases=read('inputs/cases.json');evaluation=read('inputs/evaluation_cases.json');eval_ids={x['image'] for x in evaluation};by={x['image']:x for x in cases}
cal=[]
for target in sorted({x['object'] for x in cases if x['split']=='train'}):
    xs=[x for x in cases if x['split']=='train' and x['object']==target]
    if len(xs)>=4:cal.extend(sorted(xs,key=lambda x:hashlib.sha256(('protocol_calibration_20261114:'+x['image']).encode()).hexdigest())[:3])
assert sorted(x['image'] for x in cal)==plan['training_images'] and len(cal)==15
assert evaluation==[x for x in cases if x['image'] not in plan['training_images']] and len(evaluation)==85
assert sum(x['split']=='train' for x in evaluation)==63 and not set(plan['policy']['inherited_optimization_images'])&set(by)
archive=r/'baseline_evidence/quantity_counterfactual_evidence_20261114.tar.gz';assert sha(archive)==plan['baseline_archive_sha256']
scratch=r/'scratch';scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='sft_baseline_audit_',dir=scratch) as td:
    br=Path(td)
    with tarfile.open(archive) as t:
        ms=t.getmembers();assert len(ms)==1220 and len({m.name for m in ms})==1220
        for m in ms:
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            q=br/m.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read())
    manifest=json.loads((br/'file_checksums.json').read_text(encoding='utf-8'))
    for n,info in manifest.items():assert sha(br/n)==info['sha256'],n
    result=subprocess.run([sys.executable,str(br/'code/verify_resumed_quantity.py'),'--root',str(br),'--out',str(br/'independent_result.json')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    for src,dst in [('features/tensors.pt','features/tensors.pt'),('features/gallery.json','features/gallery.json'),('inputs/cases.json','inputs/cases.json'),
                    ('scoring/scores.json','baseline_evidence/scores.json'),('scoring/tensors.pt','baseline_evidence/scoring_tensors.pt')]:
        assert (br/src).read_bytes()==(r/dst).read_bytes(),src
    old=read('baseline_evidence/scores.json');old_trace={x['id']:json.loads((br/x['trace_file']).read_text(encoding='utf-8')) for x in old}
old_idx={(x['image'],x['role'],x['seed']):x for x in old}
for source in read('inputs/teacher_sources.json'):
    ref=next(x for x in old if x['id']==source['source_record_id']);assert ref['image']==source['image'] and ref['quality_usable'] and ref['trace_sha256']==source['source_trace_sha256']
    body=re.search(r'<answer>(.*?)</answer>',old_trace[ref['id']]['rounds'][-1]['generated_text'],re.S).group(1).strip();assert body==source['source_body']
    same=[x for x in old if x['image']==source['image']];greedy=next(x for x in same if x['role']=='greedy')
    chosen=greedy if greedy['quality_usable'] else next(x for x in sorted(same,key=lambda x:x['seed']) if x['role']=='candidate' and x['quality_usable'])
    assert chosen['id']==source['source_record_id']
teacher_tools=read('training/actual_teacher_tools.json');examples=read('training/teacher_examples.json')
for source in read('inputs/teacher_sources.json'):
    call=next(x for x in examples if x['image']==source['image'] and x['kind']=='call');answer=next(x for x in examples if x['image']==source['image'] and x['kind']=='answer')
    ref=next(x for x in old if x['image']==source['image'] and x['role']=='greedy')
    assert call['prompt_text']==old_trace[ref['id']]['rounds'][0]['prompt_text']
    expected=call['prompt_text']+'<tool_call>scene_classify</tool_call><|im_end|>\n<|im_start|>user\n<result>'+teacher_tools[source['image']]['text']+'</result><|im_end|>\n<|im_start|>assistant\n'
    assert answer['prompt_text']==expected

scores=read('scoring/all_scores.json');assert len(scores)==935 and len({x['id'] for x in scores})==935
t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True);assert t['record_ids']==[x['id'] for x in scores]
f=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True);gallery=read('features/gallery.json')
q64=t['image_features'].double()@f['quantity_text_features'].double().T
assert float((q64-t['quantity_similarities'].double()).abs().max())<1e-6
lookup={x:i for i,x in enumerate(t['record_ids'])};vl={x:i for i,x in enumerate(t['val_record_ids'])}
v64=t['image_features'][[lookup[x] for x in t['val_record_ids']]].double()@f['original_val_text_features'].double().T
assert len(vl)==242 and float((v64-t['val_similarities'].double()).abs().max())<1e-6
actual=read('replay_verification/actual_tool_replays.json');assert len(actual)==100
for name,tool in teacher_tools.items():assert tool['text']==actual[name]['text'] and abs(tool['confidence']-actual[name]['confidence'])<1e-6
strict_tool=re.compile(r'<tool_call>\s*scene_classify\s*</tool_call>');tool_pattern=re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?');answer_pattern=re.compile(r'<answer>(.*?)</answer>',re.S)
rounds_checked=events_checked=class_equal=ranks_equal=0
for i,x in enumerate(scores):
    assert x['image'] in eval_ids and x['image'] not in plan['training_images']
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True)
    assert h.dtype==torch.bfloat16 and h.shape==(3584,) and torch.isfinite(h).all()
    assert sha(r/x['state_file'])==x['state_file_sha256'] and tsha(h)==x['generation_state_sha256']==x['cache_state_sha256']==x['scorer_state_sha256']
    assert sha(r/x['trace_file'])==x['trace_sha256'];trace=read(x['trace_file']);rounds=trace['rounds']
    assert 1<=len(rounds)<=3 and not trace['forced_tool_witness']
    assert trace['state_tensor_sha256']==rounds[-1]['state_tensor_sha256']==x['generation_state_sha256']
    final=rounds[-1]['generated_text'];matched=answer_pattern.search(final)
    usable=bool(matched and matched.group(1).strip()) and not tool_pattern.search(final) and '<result>' not in final and '</result>' not in final \
      and trace['stop_reason']=='answer' and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in rounds)
    aa=answer_pattern.fullmatch(final.strip());strict=bool(usable and aa and aa.group(1).strip() and all(strict_tool.fullmatch(z['generated_text'].strip()) for z in rounds[:-1]))
    assert usable==x['quality_usable']==trace['quality_usable'] and strict==x['strict_protocol_usable']
    assert x['tools_used']==trace['tools_used']==len(trace['tool_events'])
    ref=old_idx[x['image'],x['role'],x['seed']];assert rounds[0]['prompt_text']==old_trace[ref['id']]['rounds'][0]['prompt_text']
    for ri,z in enumerate(rounds):
        ids=z['generated_ids'];assert 1<=len(ids)==z['generated_token_count']<=256
        assert ids[0] not in z['eos_token_ids'] and all(v not in z['eos_token_ids'] for v in ids[:-1])
        assert len(z['prompt_token_ids'])==z['prompt_tokens'] and (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id'] and ids[-1]==z['last_emitted_token_id']
        assert z['message_roles']==['system','user','assistant','user','assistant','user']+['assistant','user']*ri
        assert hashlib.sha256(z['prompt_text'].encode()).hexdigest()==z['prompt_text_sha256']
        kw=z['generation_kwargs'];assert kw['min_new_tokens']==1 and kw['top_k']==0 and kw['top_p']==1. and kw['temperature']==1. and kw['do_sample']==(x['role']=='candidate')
        if ri<len(rounds)-1:
            assert (strict_tool.fullmatch(z['generated_text'].strip()) if x['controller']=='strict' else tool_pattern.fullmatch(z['generated_text'].strip()))
            event=next(e for e in trace['tool_events'] if e['after_round']==ri+1);assert '<result>'+event['text']+'</result>' in rounds[ri+1]['prompt_text']
        rounds_checked+=1
    for event in trace['tool_events']:
        ref=actual[x['image']];assert event['origin']=='model_requested_actual_tool' and event['text']==ref['text'] and abs(event['confidence']-ref['confidence'])<1e-6
        events_checked+=1
    if x['role']=='candidate':assert x['seed'] in [800000+x['image_order']*4+k for k in range(4)] and x['temperature']==1.
    else:assert x['seed']==0 and x['temperature']==0.
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        cp,cn=[float(t['quantity_similarities'][i,c]) for c in cols]
        assert x[wording+'_positive']==cp and x[wording+'_negative']==cn and x['margin_'+wording]==cp-cn and x['correct_'+wording]==(cp>cn)
        class_equal+=bool(q64[i,cols[0]]>q64[i,cols[1]])==x['correct_'+wording]
    if x['split']=='val':
        vi=vl[x['id']];s=t['val_similarities'][vi];pos=gallery['val_positive_columns'][x['image']];rank=int((s[:391]>s[pos].max()).sum())
        assert x['retrieval_rank0']==rank and x['retrieval_rr']==1/(rank+1)
        ranks_equal+=int((v64[vi,:391]>v64[vi,pos].max()).sum())==rank
    else:assert x['retrieval_rr'] is None
for arm in plan['evaluation']['arms']:
    xs=[x for x in scores if x['arm']==arm['name']];assert len(xs)==85*(1+arm['samples_per_image'])
    settings=read('evaluation/'+arm['name']+'/settings.json');end=read('evaluation/'+arm['name']+'/result.json')
    expected=plan['policy']['lora_tensor_hash'] if arm['policy']=='inherited' else training['final_parameter_hash']
    assert settings['policy_frozen'] and settings['optimization_updates_during_evaluation']==0 and settings['policy_lora_tensor_hash']==end['lora_tensor_hash_unchanged']==expected
    for case in evaluation:
        chosen=[x for x in xs if x['image']==case['image']];assert len(chosen)==1+arm['samples_per_image'] and sum(x['role']=='greedy' for x in chosen)==1
historical=[dict(x,arm='historical_legacy') for x in old if x['image'] in eval_ids]
for x in historical:
    trace=old_trace[x['id']];aa=answer_pattern.fullmatch(trace['rounds'][-1]['generated_text'].strip())
    x['strict_protocol_usable']=bool(x['quality_usable'] and aa and aa.group(1).strip() and all(strict_tool.fullmatch(z['generated_text'].strip()) for z in trace['rounds'][:-1]))
all_records=scores+historical;analysis=read('analysis.json');rebuilt={};metrics=['quality','strict_quality','tool_usage','generated_result_failure','margin_A','margin_B','pairacc_A','pairacc_B','retrieval_rr','retrieval_margin']
for arm in ['historical_legacy','c742_strict','sft_strict','sft_legacy_greedy']:
    for item in evaluation:
        xs=[x for x in all_records if x['arm']==arm and x['image']==item['image']];greedy=next(x for x in xs if x['role']=='greedy')
        sampled=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);valid=[x for x in sampled if x['quality_usable']]
        mapping={'greedy':[greedy]}
        if arm!='sft_legacy_greedy':mapping.update(sample_mean=sampled,format_first=[valid[0] if valid else greedy],quantity_raw=[max(sampled,key=lambda x:x['margin_A'])],
           quantity_quality=[max(valid,key=lambda x:x['margin_A']) if valid else greedy],reverse_quality=[min(valid,key=lambda x:x['margin_A']) if valid else greedy])
        for selector,values in mapping.items():
            entry=next(x for x in analysis['per_image'] if x['arm']==arm and x['image']==item['image'] and x['selector']==selector)
            assert entry['selected_ids']==[x['id'] for x in values]
            for m in metrics:
                ns=[float(x['quality_usable']) if m=='quality' else float(x['strict_protocol_usable']) if m=='strict_quality' else float(x['tools_used']>0) if m=='tool_usage'
                  else float(x['stop_reason']=='model_generated_tool_result') if m=='generated_result_failure' else float(x['correct_'+m[-1]]) if m.startswith('pairacc_') else x[m] for x in values]
                v=None if ns[0] is None else math.fsum(ns)/len(ns);rebuilt[arm,item['image'],selector,m]=v
                assert v is None and entry[m] is None or abs(v-entry[m])<1e-12
ci_count=0
for split,g in analysis['groups'].items():
    names=[x['image'] for x in evaluation if x['split']==split];n=len(names);draw=np.random.default_rng(2026111438).integers(0,n,size=(10000,n))
    def check(st,values):
        global ci_count
        ds=np.asarray(values,dtype=np.float64);assert abs(float(ds.mean())-st['mean'])<1e-12
        assert np.allclose(np.percentile(ds[draw].mean(axis=1),[2.5,97.5]),st['interval95'],atol=1e-12,rtol=0);ci_count+=1
    for arm,selectors in g['means'].items():
        for selector,values in selectors.items():
            for m,st in values.items():check(st,[rebuilt[arm,name,selector,m] for name in names])
    for arm,comparisons in g['quantity_paired'].items():
        for contrast,values in comparisons.items():
            selector,base=contrast.split('_minus_')
            for m,st in values.items():check(st,[rebuilt[arm,name,selector,m]-rebuilt[arm,name,base,m] for name in names])
    for contrast,values in g['paired'].items():
        policy,selector=contrast.split('__');arm,base=policy.split('_minus_')
        for m,st in values.items():check(st,[rebuilt[arm,name,selector,m]-rebuilt[base,name,selector,m] for name in names])
g=analysis['groups']['train'];m=g['means'];th=plan['readiness_gate']
checks={'optimizer_updates_exactly_30':training['optimizer_updates_rebuilt']==30,
 'strict_sample_quality_at_least_90pct':m['sft_strict']['sample_mean']['strict_quality']['mean']>=th['strict_sample_quality_min'],
 'strict_greedy_quality_at_least_95pct':m['sft_strict']['greedy']['strict_quality']['mean']>=th['strict_greedy_quality_min'],
 'sample_actual_tool_use_at_least_80pct':m['sft_strict']['sample_mean']['tool_usage']['mean']>=th['sample_actual_tool_use_min'],
 'generated_result_failure_at_most_2pct':m['sft_strict']['sample_mean']['generated_result_failure']['mean']<=th['generated_result_failure_max'],
 'sample_B_pairacc_at_least_matched_control':m['sft_strict']['sample_mean']['pairacc_B']['mean']>=m['c742_strict']['sample_mean']['pairacc_B']['mean'],
 'legacy_greedy_B_pairacc_drop_at_most_5pp':m['sft_legacy_greedy']['greedy']['pairacc_B']['mean']>=m['historical_legacy']['greedy']['pairacc_B']['mean']-th['legacy_greedy_B_pairacc_drop_max']}
assert checks==analysis['format_readiness']['checks'] and all(checks.values())==analysis['format_readiness']['passed']
informative=0
for item in [x for x in evaluation if x['split']=='train']:
    valid=[x['margin_A'] for x in scores if x['arm']=='sft_strict' and x['image']==item['image'] and x['role']=='candidate' and x['quality_usable']]
    informative+=len(valid)>=2 and max(valid)-min(valid)>1e-6
qth=plan['analysis']['next_training_gate'];gain=g['quantity_paired']['sft_strict']['quantity_quality_minus_format_first']['margin_B']
qchecks={'sample_quality_at_least_90pct':m['sft_strict']['sample_mean']['quality']['mean']>=qth['train_generation_quality_min'],
 'at_least_38_informative_images':informative>=38,'B_margin_gain_at_least_1e_4':gain['mean']>=qth['heldout_B_margin_gain_min'],
 'B_margin_gain_interval_above_zero':gain['interval95'][0]>0,'B_pairacc_at_least_format_first':m['sft_strict']['quantity_quality']['pairacc_B']['mean']>=m['sft_strict']['format_first']['pairacc_B']['mean'],
 'B_margin_vs_reverse_interval_above_zero':g['quantity_paired']['sft_strict']['quantity_quality_minus_reverse_quality']['margin_B']['interval95'][0]>0,
 'selected_quality_at_least_95pct':m['sft_strict']['quantity_quality']['quality']['mean']>=qth['quality_selected_min']}
assert qchecks==analysis['quantity_readiness']['checks'] and all(qchecks.values())==analysis['quantity_readiness']['passed']
assert informative==analysis['quantity_readiness']['informative_train_images'] and read('teacher_replay.json')['ok'] and read('replay_verification/result.json')['ok']
out={'ok':True,'training':training,'optimizer_updates_verified':30,'new_states_and_traces_checked':935,'generation_rounds_checked':rounds_checked,
 'real_tool_events_checked':events_checked,'matched_neutral_prompts_checked':935,'evaluation_images_excluded_from_training':85,
 'selectors_rebuilt':len(analysis['per_image']),'bootstrap_intervals_rebuilt':ci_count,
 'quantity_classifications_equal_float64':int(class_equal),'quantity_classification_checks':1870,'val_ranks_equal_float64':int(ranks_equal),'val_rank_checks':242,
 'embedded_previous_503_trajectories_verified':True,'format_and_quantity_readiness_rebuilt':True,'torch':torch.__version__,'numpy':np.__version__}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('ALL_SUPERVISED_TRAINING_AND_FIXED_POLICY_EVIDENCE_VERIFIED')
