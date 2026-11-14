"""Independent CPU verification of cached states, rewards, tools and paired statistics."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
torch.set_num_threads(2)

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();r=a.root;plan=json.loads((r/'plan.json').read_text())
assert json.loads((r/'run_result.json').read_text())['status']=='complete'
assert json.loads((r/'gate_result.json').read_text())['status']=='complete'
assert not set(plan['training_rows'])&set(plan['validation_rows'])
answer=re.compile(r'<answer>(.*?)</answer>',re.S)
call=re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def tensor_sha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


state_checks=round_checks=tool_checks=0
tool_replays=json.loads((r/'training_reward_verification/actual_tool_replays.json').read_text())


def verify_record(x):
    global state_checks,round_checks,tool_checks
    state=torch.load(r/x['state_file'],map_location='cpu',weights_only=True)
    assert list(state.shape)==[3584] and state.dtype==torch.bfloat16 and torch.isfinite(state).all()
    assert sha(r/x['state_file'])==x['state_file_sha256']
    assert tensor_sha(state)==x['generation_state_sha256']
    assert x.get('cache_state_sha256',x.get('scorer_state_sha256'))==tensor_sha(state)
    trace=json.loads((r/x['trace_file']).read_text())
    if 'trace_sha256' in x:assert sha(r/x['trace_file'])==x['trace_sha256']
    assert not trace['forced_tool_witness']
    rounds=trace['rounds'];assert 1<=len(rounds)<=3
    assert tensor_sha(state)==trace['state_tensor_sha256']==rounds[-1]['state_tensor_sha256']
    assert trace['tools_used']==len(trace['tool_events'])==x['tools_used']
    final=rounds[-1]['generated_text'];matched=answer.search(final)
    usable=bool(matched and matched.group(1).strip()) and not call.search(final) \
        and '<result>' not in final and '</result>' not in final \
        and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in rounds) \
        and trace['stop_reason']=='answer'
    assert x['quality_usable']==trace['quality_usable']==usable
    for i,z in enumerate(rounds):
        ids=z['generated_ids'];assert len(ids)==z['generated_token_count'] and 1<=len(ids)<=256
        assert ids[0] not in z['eos_token_ids']
        assert all(t not in z['eos_token_ids'] for t in ids[:-1])
        assert z['prompt_tokens']==len(z['prompt_token_ids'])
        assert (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id']
        assert ids[-1]==z['last_emitted_token_id']
        assert hashlib.sha256(z['prompt_text'].encode()).hexdigest()==z['prompt_text_sha256']
        assert z['prompt_text'].endswith('<|im_start|>assistant\n')
        assert len(z['message_roles'])==6+2*i
        assert z['generation_kwargs']['min_new_tokens']==1 and z['generation_kwargs']['top_k']==0
        assert z['generation_kwargs']['repetition_penalty']==1.
        if i<len(rounds)-1:
            m=call.fullmatch(z['generated_text'].strip());assert m and m.group(1)=='scene_classify'
            event=next(e for e in trace['tool_events'] if e.get('after_round')==i+1)
            assert '<result>'+event['text']+'</result>' in rounds[i+1]['prompt_text']
            assert rounds[i+1]['message_roles'][-2:]==['assistant','user']
        round_checks+=1
    for e in trace['tool_events']:
        expected=tool_replays[str(x['row'])]
        assert e['origin']=='model_requested_actual_tool' and e['accepted']
        assert e['text']==expected['text'] and abs(e['confidence']-expected['confidence'])<1e-6
        tool_checks+=1
    state_checks+=1


training={}
for arm in ['smoke','retrieval','format_only']:
    result=json.loads((r/'training'/arm/'result.json').read_text());steps=json.loads((r/'training'/arm/'steps.json').read_text())
    records=[json.loads(x) for x in (r/'training'/arm/'samples.jsonl').read_text().splitlines()]
    G=2 if arm=='smoke' else 4
    assert len(records)==G*len(steps)==result['total_samples']
    if arm!='smoke':assert len(steps)==20
    lam=1. if arm=='smoke' else plan['arms'][arm]['lambda_retrieval']
    for x in records:
        assert x['row'] not in plan['validation_rows']
        assert x['row']==(plan['smoke_rows'] if arm=='smoke' else plan['training_rows'])[x['step']]
        verify_record(x)
        assert x['r_retrieval']==float(Fraction(1,1+x['proxy_rank0']))
        assert x['r_format']==(.2 if x['quality_usable'] else 0.)
        assert x['r_total']==x['r_format']+lam*x['r_retrieval']
    assert result['optimizer_updates']==sum(x['optimizer_update'] for x in steps)
    for step in steps:
        assert step['optimizer_update']==(step['reward_sample_std']>1e-8)
        assert math.isfinite(step['loss']) and math.isfinite(step['gradient_norm_before_clip'])
        if step['optimizer_update']:assert step['gradient_norm_before_clip']>0 and step['effective_loss_tokens']>0
    if not result['optimizer_updates']:assert result['initial_lora_tensor_hash']==result['final_lora_tensor_hash']
    training[arm]=result
witness=json.loads((r/'training/smoke/contract_witness.json').read_text())
assert witness['actual_forced_tool_executions']>=1 and witness['padding_logprob_exact_equal']
assert witness['original_logprob']==witness['padded_logprob'] and witness['added_padding_ignored']==17

scores=json.loads((r/'scoring/scores.json').read_text());assert len(scores)==330
for x in scores:verify_record(x)
assert training['smoke']['initial_lora_tensor_hash']==training['retrieval']['initial_lora_tensor_hash']==training['format_only']['initial_lora_tensor_hash']
for policy in ['baseline','retrieval','format_only']:
    settings=json.loads((r/'evaluation'/policy/'settings.json').read_text())
    expected=training['smoke']['initial_lora_tensor_hash'] if policy=='baseline' else training[policy]['final_lora_tensor_hash']
    assert settings['lora_tensor_hash']==settings['expected_tensor_hash']==expected
    if policy!='baseline':assert settings['checkpoint_restoration']['saved_float32_adapter_restored_exactly']

v=json.loads((r/'scoring/verification.json').read_text())
for n,h in v['verification_input_sha256'].items():assert sha(r/'verification_inputs'/n)==h
gallery=json.loads((r/'verification_inputs/text_gallery.json').read_text())
cache=torch.load(r/'verification_inputs/proxy_gallery.pt',map_location='cpu',weights_only=True)
original=torch.load(r/'verification_inputs/original_scoring_tensors.pt',map_location='cpu',weights_only=True)
t=torch.load(r/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)
assert torch.equal(t['text_features'],original['text_features'])
assert t['record_ids']==[x['id'] for x in scores]
proxy64=t['proxy_features'].double()@cache['image_features'].double().T
actual64=t['image_features'].double()@t['text_features'].double().T
proxy_error=float((proxy64-t['proxy_similarities'].double()).abs().max())
utility_error=float((actual64-t['actual_similarities'].double()).abs().max())
assert max(proxy_error,utility_error)<1e-6
proxy_equal=utility_equal=0
for i,x in enumerate(scores):
    ps=t['proxy_similarities'][i];rank=int((ps>ps[x['row']]).sum())
    assert rank==x['proxy_rank0'] and x['r_retrieval']==1/(1+rank)
    s=t['actual_similarities'][i];positive=gallery['positive_gallery_columns'][str(x['row'])]
    negatives=[j for j in range(gallery['unique_val_caption_count']) if j not in positive]
    best=float(s[positive].max());actual_rank=int((s[:gallery['unique_val_caption_count']]>best).sum())
    assert actual_rank==x['utility_rank0'] and x['utility_rr']==1/(1+actual_rank)
    assert x['utility_margin']==best-float(s[negatives].max())
    for label,cols in gallery['quantity_text_columns'][str(x['row'])].items():assert x['E1_margin_'+label]==float(s[cols[0]]-s[cols[1]])
    proxy_equal+=int((proxy64[i]>proxy64[i,x['row']]).sum())==rank
    utility_equal+=int((actual64[i,:gallery['unique_val_caption_count']]>actual64[i,positive].max()).sum())==actual_rank
tr=torch.load(r/'training_reward_verification/tensors.pt',map_location='cpu',weights_only=True)
training_scores=json.loads((r/'training_reward_verification/records.json').read_text())
recomputed=tr['proxy_features'].double()@cache['image_features'].double().T
assert float((recomputed-tr['proxy_similarities'].double()).abs().max())<1e-6
for i,x in enumerate(training_scores):
    ps=tr['proxy_similarities'][i];assert int((ps>ps[x['row']]).sum())==x['proxy_rank0']
assert json.loads((r/'training_reward_verification/result.json').read_text())['all_training_proxy_ranks_rewards_exact']

f=json.loads((r/'analysis.json').read_text());assert f['plan_sha256']==sha(r/'plan.json')
rebuilt={}
for item in f['per_image']:
    selected=[x for x in scores if x['id'] in item['selected_ids']]
    assert selected and len(selected)==len(item['selected_ids'])
    candidates=[x for x in scores if x['row']==item['row'] and x['policy']==item['policy'] and x['role']=='candidate']
    if item['mode']=='greedy':assert len(selected)==1 and selected[0]['role']=='greedy'
    elif item['mode']=='sample_mean':assert {x['id'] for x in selected}=={x['id'] for x in candidates}
    else:
        field='r_retrieval' if item['mode']=='retrieval_select' else 'r_total'
        ranked=sorted(candidates,key=lambda x:x[field],reverse=True)
        assert {x['id'] for x in selected}=={x['id'] for x in ranked if x[field]==ranked[0][field]}
    values={metric:sum(float(x['quality_usable']) if metric=='quality' else float(x['tools_used']>0)
        if metric=='tool_usage' else x[metric] for x in selected)/len(selected) for metric in
        ['utility_rr','utility_margin','E1_margin_A','E1_margin_B','quality','tool_usage','r_retrieval']}
    for k,value in values.items():assert abs(value-item[k])<1e-12
    rebuilt[item['policy'],item['row'],item['mode']]=values
ci_checks=0
for group,rows in plan['validation_groups'].items():
    draw=np.random.default_rng(plan['evaluation']['bootstrap']['seed']).integers(0,len(rows),size=(10000,len(rows)))
    for name,policies in f['groups'][group]['paired'].items():
        policy,baseline=name.split('_minus_')
        for mode,metrics in policies.items():
            for metric,declared in metrics.items():
                delta=np.array([rebuilt[policy,row,mode][metric]-rebuilt[baseline,row,mode][metric] for row in rows])
                assert abs(float(delta.mean())-declared['mean'])<1e-12
                assert np.allclose(np.percentile(delta[draw].mean(axis=1),[2.5,97.5]),declared['interval95'],atol=1e-12,rtol=0)
                ci_checks+=1
main=f['groups']['all22']['paired']
checks={'rr_gain_interval_above_baseline':main['retrieval_minus_baseline']['greedy']['utility_rr']['interval95'][0]>0,
    'rr_gain_interval_above_format_control':main['retrieval_minus_format_only']['greedy']['utility_rr']['interval95'][0]>0,
    'greedy_quality_within_5_percentage_points':main['retrieval_minus_baseline']['greedy']['quality']['mean']>=-.05,
    'retrieval_optimizer_updated':training['retrieval']['optimizer_updates']>0}
assert checks==f['decision']['checks'] and all(checks.values())==f['decision']['expand_criterion_passed']
for n,h in json.loads((r/'pipeline_code_hashes.json').read_text()).items():assert sha(r/n)==h
audit={'ok':True,'state_and_trace_checks':state_checks,'rounds_checked':round_checks,'actual_tool_events_checked':tool_checks,
    'evaluation_scores_checked':330,'training_rewards_checked':len(training_scores),'paired_bootstrap_intervals_checked':ci_checks,
    'proxy_rank64_equal_count':int(proxy_equal),'utility_rank64_equal_count':int(utility_equal),
    'proxy_cpu64_max_abs_dot_error':proxy_error,'utility_cpu64_max_abs_dot_error':utility_error,
    'all_three_points_share_initial_adapter':True,'trained_checkpoint_restoration_exact':True,
    'torch_cpu_version':torch.__version__,'numpy_version':np.__version__}
a.out.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit))
