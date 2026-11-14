import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
torch.set_num_threads(2)

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();r=a.root;plan=json.loads((r/'plan.json').read_text())
assert json.loads((r/'run_result.json').read_text())['status']=='complete'
metadata=json.loads((r/'inputs/dataset_metadata.json').read_text())
assert hashlib.sha256((r/'inputs/dataset_metadata.json').read_bytes()).hexdigest()==plan['new_validation_selection']['metadata_sha256']
old=plan['validation_groups']['legacy22'];train=plan['training_rows'];eligible=[i for i in range(210) if i not in set(old)|set(train)]
assert sorted(random.Random(20261114).sample(eligible,50))==plan['validation_groups']['new50']
assert not set(plan['validation_rows'])&set(train)
answer=re.compile(r'<answer>(.*?)</answer>',re.S);call=re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')
actual_tools=json.loads((r/'training_reward_verification/actual_tool_replays.json').read_text())
state_checks=round_checks=tool_checks=0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def tensor_sha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def check_trace(x):
    global state_checks,round_checks,tool_checks
    h=torch.load(r/x['state_file'],map_location='cpu',weights_only=True)
    assert h.dtype==torch.bfloat16 and list(h.shape)==[3584] and torch.isfinite(h).all()
    assert sha(r/x['state_file'])==x['state_file_sha256']
    digest=tensor_sha(h);assert digest==x['generation_state_sha256']==x.get('cache_state_sha256',x.get('scorer_state_sha256'))
    trace=json.loads((r/x['trace_file']).read_text());rounds=trace['rounds']
    if 'trace_sha256' in x:assert sha(r/x['trace_file'])==x['trace_sha256']
    assert not trace['forced_tool_witness'] and 1<=len(rounds)<=3
    assert digest==trace['state_tensor_sha256']==rounds[-1]['state_tensor_sha256']
    assert x['tools_used']==trace['tools_used']==len(trace['tool_events'])
    final=rounds[-1]['generated_text'];matched=answer.search(final)
    usable=bool(matched and matched.group(1).strip()) and not call.search(final) and '<result>' not in final \
        and '</result>' not in final and trace['stop_reason']=='answer' \
        and all(not z['token_limit'] and 'addCriterion' not in z['generated_text'] for z in rounds)
    assert usable==x['quality_usable']==trace['quality_usable']
    for i,z in enumerate(rounds):
        ids=z['generated_ids'];assert len(ids)==z['generated_token_count'] and 1<=len(ids)<=256
        assert ids[0] not in z['eos_token_ids'] and all(t not in z['eos_token_ids'] for t in ids[:-1])
        assert z['prompt_tokens']==len(z['prompt_token_ids']) and (z['prompt_token_ids']+ids)[-2]==z['state_input_token_id']
        assert ids[-1]==z['last_emitted_token_id']
        assert hashlib.sha256(z['prompt_text'].encode()).hexdigest()==z['prompt_text_sha256']
        assert z['prompt_text'].endswith('<|im_start|>assistant\n') and len(z['message_roles'])==6+2*i
        assert z['generation_kwargs']['min_new_tokens']==1 and z['generation_kwargs']['top_k']==0
        if i<len(rounds)-1:
            m=call.fullmatch(z['generated_text'].strip());assert m and m.group(1)=='scene_classify'
            e=next(e for e in trace['tool_events'] if e.get('after_round')==i+1)
            assert '<result>'+e['text']+'</result>' in rounds[i+1]['prompt_text']
        round_checks+=1
    for e in trace['tool_events']:
        ref=actual_tools[str(x['row'])]
        assert e['origin']=='model_requested_actual_tool' and e['text']==ref['text'] and abs(e['confidence']-ref['confidence'])<1e-6
        tool_checks+=1
    state_checks+=1


gallery=json.loads((r/'verification_inputs/text_gallery.json').read_text())
positive=json.loads((r/'scoring/positive_columns.json').read_text());n=gallery['unique_val_caption_count'];assert n==391
lookup={text:i for i,text in enumerate(gallery['text_order'][:n])}
for row in metadata['rows']:assert positive[str(row['row'])]==sorted({lookup[c] for c in row['captions']})
negative={row:[i for i in range(n) if i not in pos] for row,pos in positive.items()}
training_count=0
for arm in ['smoke','aligned_margin']:
    samples=[json.loads(x) for x in (r/'training'/arm/'samples.jsonl').read_text().splitlines()]
    t=torch.load(r/'training'/arm/'reward_tensors.pt',map_location='cpu',weights_only=True)
    assert t['record_ids']==[x['id'] for x in samples]
    sim64=t['image_features'].double()@t['text_features'].double().T
    assert float((sim64-t['similarities'].double()).abs().max())<1e-6
    for i,x in enumerate(samples):
        check_trace(x);training_count+=1;assert x['row'] in train and x['row'] not in plan['validation_rows']
        sim=t['similarities'][i];row=str(x['row']);best=float(sim[positive[row]].max());neg=float(sim[negative[row]].max())
        assert best-neg==x['r_aligned']
        rank=int((sim[:n]>best).sum());assert rank==x['actual_rank0'] and x['r_actual_rr']==1/(1+rank)
        assert x['r_format']==(.2 if x['quality_usable'] else 0.) and x['r_total']==x['r_format']+x['r_aligned']
    result=json.loads((r/'training'/arm/'result.json').read_text());steps=json.loads((r/'training'/arm/'steps.json').read_text())
    assert result['optimizer_updates']==sum(x['optimizer_update'] for x in steps)
    assert result['initial_lora_tensor_hash']==plan['inherited_initial_hash']
    if arm=='aligned_margin':assert len(samples)==80 and len(steps)==20
    for x in steps:
        assert x['optimizer_update']==(x['reward_sample_std']>1e-8)
        assert math.isfinite(x['loss']) and math.isfinite(x['gradient_norm_before_clip'])
        if x['optimizer_update']:assert x['gradient_norm_before_clip']>0
assert json.loads((r/'training_reward_verification/result.json').read_text())['all_training_ranks_equal']
scores=json.loads((r/'scoring/scores.json').read_text());assert len(scores)==len({x['id'] for x in scores})==864
assert sum(x['reused_previous'] for x in scores)==198
t=torch.load(r/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)
original=torch.load(r/'verification_inputs/original_scoring_tensors.pt',map_location='cpu',weights_only=True)
assert torch.equal(t['text_features'],original['text_features']) and t['record_ids']==[x['id'] for x in scores]
cache=torch.load(r/'verification_inputs/proxy_gallery.pt',map_location='cpu',weights_only=True)
proxy64=t['proxy_features'].double()@cache['image_features'].double().T
sim64=t['image_features'].double()@t['text_features'].double().T
assert float((proxy64-t['proxy_similarities'].double()).abs().max())<1e-6
assert float((sim64-t['actual_similarities'].double()).abs().max())<1e-6
old_scores={x['id']:x for x in json.loads((r/'inputs/previous_eval_scores.json').read_text())}
rank64_proxy=rank64_actual=0
for i,x in enumerate(scores):
    check_trace(x)
    ps=t['proxy_similarities'][i];pr=int((ps>ps[x['row']]).sum())
    assert pr==x['proxy_rank0'] and x['r_proxy']==1/(1+pr)
    sim=t['actual_similarities'][i];row=str(x['row']);best=float(sim[positive[row]].max());neg=float(sim[negative[row]].max())
    rank=int((sim[:n]>best).sum())
    assert rank==x['utility_rank0'] and x['utility_rr']==float(Fraction(1,rank+1))
    assert x['utility_margin']==x['r_aligned']==best-neg
    assert x['r_format']==(.2 if x['quality_usable'] else 0.)
    assert x['r_total_proxy']==x['r_format']+x['r_proxy'] and x['r_total_aligned']==x['r_format']+x['r_aligned']
    if x['reused_previous']:
        before=old_scores[x['id']];assert before['generation_state_sha256']==x['generation_state_sha256']
        assert before['proxy_rank0']==pr and before['utility_rank0']==rank
        assert abs(before['utility_margin']-x['utility_margin'])<1e-6
    rank64_proxy+=int((proxy64[i]>proxy64[i,x['row']]).sum())==pr
    rank64_actual+=int((sim64[i,:n]>sim64[i,positive[row]].max()).sum())==rank
for policy in plan['evaluation']['policy_points']:
    j=json.loads((r/'evaluation'/policy/'settings.json').read_text())
    expected=plan['inherited_initial_hash'] if policy=='baseline' else \
        json.loads((r/'training/aligned_margin/result.json').read_text())['final_lora_tensor_hash'] if policy=='aligned_margin' else plan['inherited_policies'][policy]['lora_tensor_hash']
    assert j['lora_tensor_hash']==expected
f=json.loads((r/'analysis.json').read_text());assert f['plan_sha256']==sha(r/'plan.json')
rebuilt={}
for item in f['per_image']:
    selected=[x for x in scores if x['policy']==item['policy'] and x['row']==item['row']
              and x['role']==('greedy' if item['mode']=='greedy' else 'candidate')]
    assert {x['id'] for x in selected}==set(item['selected_ids'])
    for metric in ['utility_rr','utility_margin','quality','tool_usage','r_proxy']:
        vals=[float(x['quality_usable']) if metric=='quality' else float(x['tools_used']>0) if metric=='tool_usage' else x[metric] for x in selected]
        actual=sum(vals)/len(vals);assert abs(actual-item[metric])<1e-12
        rebuilt[item['policy'],item['row'],item['mode'],metric]=actual
ci_checks=0
for group,rows in plan['validation_groups'].items():
    draw=np.random.default_rng(20261114).integers(0,len(rows),size=(10000,len(rows)))
    for name,modes in f['groups'][group]['paired'].items():
        policy,base=name.split('_minus_')
        for mode,metrics in modes.items():
            for metric,declared in metrics.items():
                d=np.array([rebuilt[policy,row,mode,metric]-rebuilt[base,row,mode,metric] for row in rows])
                assert abs(float(d.mean())-declared['mean'])<1e-12
                assert np.allclose(np.percentile(d[draw].mean(axis=1),[2.5,97.5]),declared['interval95'],atol=1e-12,rtol=0)
                ci_checks+=1
g=f['groups']['new50'];m=g['policy_means'];d=g['paired'];training=f['aligned_training']
checks={'new50_rr_interval_above_proxy':d['aligned_margin_minus_retrieval']['greedy']['utility_rr']['interval95'][0]>0,
        'new50_rr_interval_above_format':d['aligned_margin_minus_format_only']['greedy']['utility_rr']['interval95'][0]>0,
        'new50_greedy_rr_at_least_initial':m['aligned_margin']['greedy']['utility_rr']['mean']>=m['baseline']['greedy']['utility_rr']['mean'],
        'new50_greedy_quality_within_5pp':d['aligned_margin_minus_baseline']['greedy']['quality']['mean']>=-.05,
        'optimizer_updated':training['optimizer_updates']>0}
assert checks==f['decision']['checks'] and all(checks.values())==f['decision']['expand_criterion_passed']
for n,h in json.loads((r/'pipeline_code_hashes.json').read_text()).items():assert sha(r/n)==h
for n,h in json.loads((r/'inputs/shared_code_hashes.json').read_text()).items():assert sha(r/'code'/n)==h
audit={'ok':True,'states_and_traces_checked':state_checks,'rounds_checked':round_checks,'real_tool_events_checked':tool_checks,
       'training_rewards_checked':training_count,'evaluation_scores_checked':864,'inherited_legacy_states_checked':198,
       'new50_split_reproduced':True,'all_210_positive_sets_rebuilt_from_dataset':True,'paired_bootstrap_intervals_checked':ci_checks,
       'proxy_rank64_equal_count':int(rank64_proxy),'actual_rank64_equal_count':int(rank64_actual),
       'shared_rollout_probability_checkpoint_code_unchanged':True,'torch':torch.__version__,'numpy':np.__version__}
a.out.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit))
