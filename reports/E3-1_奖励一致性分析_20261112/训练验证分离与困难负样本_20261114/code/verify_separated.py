import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
import torch
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();r=a.root;plan=json.loads((r/'plan.json').read_text())
assert json.loads((r/'run_result.json').read_text())['status']=='complete'
actual_tools=json.loads((r/'replay_verification/actual_tool_replays.json').read_text())
answer=re.compile(r'<answer>(.*?)</answer>',re.S);call=re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')
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
        ref=actual_tools[x['split']+':'+str(x['row'])]
        assert e['origin']=='model_requested_actual_tool' and e['text']==ref['text'] and abs(e['confidence']-ref['confidence'])<1e-6
        tool_checks+=1
    state_checks+=1


def canon(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip().rstrip(' .!?')
for name,h in plan['inputs_sha256'].items():assert sha(r/name)==h
for name,h in plan['shared_code_sha256'].items():assert sha(r/'code'/name)==h
audit=json.loads((r/'split_audit.json').read_text())
assert sha(r/'inputs/scene_label_source.py')==audit['scene_label_source_sha256']
scene_ast=ast.parse((r/'inputs/scene_label_source.py').read_text())
classes=next(ast.literal_eval(n.value) for n in scene_ast.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='UCM_CLASSES' for t in n.targets))
metadata={s:json.loads((r/'inputs'/(s+'_metadata.json')).read_text())['rows'] for s in ['train','val']}
for split,xs in metadata.items():
    for x in xs:
        label=(int(Path(x['image']).stem)-1)//100
        assert x['split']==split and x['scene_id']==label and x['scene']==classes[label]
val_hashes={x['image_bytes_sha256'] for x in metadata['val']}
assert not val_hashes&{x['image_bytes_sha256'] for x in metadata['train']}
val_texts={canon(c) for x in metadata['val'] for c in x['captions']}
gallery=json.loads((r/'inputs/training_text_gallery.json').read_text());texts=gallery['text_order'];lookup={t:i for i,t in enumerate(texts)}
assert not {canon(t) for t in texts}&val_texts and len(texts)==1437
expected=list(dict.fromkeys(c for x in metadata['train'] for c in x['captions'] if canon(c) not in val_texts))
assert texts==expected
eligible={s:[] for s in range(21)}
for x in metadata['train']:
    pos=sorted({lookup[c] for c in x['captions'] if c in lookup})
    neg=[i for i,t in enumerate(gallery['entries']) if x['scene_id'] in t['scene_ids'] and i not in pos]
    if pos and neg:
        m=gallery['maps'][str(x['row'])];eligible[x['scene_id']].append(x['row'])
        assert m['positive']==pos and m['same_scene_negative']==neg
        assert m['global_negative']==[i for i in range(len(texts)) if i not in pos]
        assert m['prompt_caption']==next(c for c in x['captions'] if c in lookup)
        assert canon(m['prompt_caption']) not in val_texts
for i,t in enumerate(gallery['entries']):
    src=[x for x in metadata['train'] if t['text'] in x['captions']]
    assert t['source_rows']==[x['row'] for x in src]
    assert set(t['scene_ids'])=={x['scene_id'] for x in src} and t['text']==texts[i]
rng=random.Random(2026111403);selected=[rng.choice(eligible[s]) for s in range(21) if eligible[s]];rng.shuffle(selected)
assert selected==plan['training_rows'] and len(selected)==19
previous=json.loads((r/'inputs/previous_plan.json').read_text());seen=set(previous['training_rows'])|set(previous['validation_rows'])
assert sorted(random.Random(2026111404).sample([i for i in range(210) if i not in seen],50))==plan['validation_rows']
assert not seen&set(plan['validation_rows'])
branch=json.loads((r/'pilot/branch_decision.json').read_text());training_rewards=0;rank64_equal=0;total_rank_checks=0
pilot_records=[]
sources=[('pilot',r/'pilot/records.jsonl')]+[(arm,r/'training'/arm/'samples.jsonl') for arm in ['smoke']+branch['arms']]
train_features=torch.load(r/'features/train.pt',map_location='cpu',weights_only=True)['text_features']
for arm,path in sources:
    xs=[json.loads(s) for s in path.read_text().splitlines()]
    t=torch.load(path.parent/'reward_tensors.pt',map_location='cpu',weights_only=True)
    assert t['record_ids']==[x['id'] for x in xs] and torch.equal(t['text_features'],train_features)
    sim64=t['image_features'].double()@t['text_features'].double().T
    assert float((sim64-t['similarities'].double()).abs().max())<1e-6
    for i,x in enumerate(xs):
        assert x['split']=='train' and x['row'] in plan['training_rows'];check_trace(x);training_rewards+=1
        sim=t['similarities'][i];m=gallery['maps'][str(x['row'])]
        best=float(sim[m['positive']].max());negative=float(sim[m['global_negative']].max());scene=float(sim[m['same_scene_negative']].max())
        assert best-negative==x['r_global'] and best-scene==x['r_scene']
        assert float(sim[x['global_negative_index']])==negative and x['global_negative_index'] in m['global_negative']
        assert float(sim[x['scene_negative_index']])==scene and x['scene_negative_index'] in m['same_scene_negative']
        assert x['global_negative_is_same_scene']==(x['global_negative_index'] in m['same_scene_negative'])
        rank=int((sim>best).sum());assert x['rank0']==rank and x['rr']==1/(1+rank)
        assert x['r_format']==(.2 if x['quality_usable'] else 0.)
        if arm!='pilot':assert x['r_total']==x['r_format']+(0. if arm=='format_only' else x['r_scene'] if arm=='scene_margin' else x['r_global'])
        rank64_equal+=int((sim64[i]>sim64[i,m['positive']].max()).sum())==rank;total_rank_checks+=1
    if arm=='pilot':pilot_records=xs;assert len(xs)==76
    else:
        result=json.loads((path.parent/'result.json').read_text());steps=json.loads((path.parent/'steps.json').read_text())
        assert result['optimizer_updates']==sum(x['optimizer_update'] for x in steps)
        assert result['initial_lora_tensor_hash']==plan['inherited_initial_hash']
        if arm!='smoke':assert len(xs)==76 and len(steps)==19
        for step in steps:
            assert step['optimizer_update']==(step['reward_sample_std']>1e-8)
            assert math.isfinite(step['loss']) and math.isfinite(step['gradient_norm_before_clip'])
            if step['optimizer_update']:assert step['gradient_norm_before_clip']>0
groups=[]
for group in branch['groups']:
    xs=[x for x in pilot_records if x['row']==group['row']];assert len(xs)==4
    advantages=[]
    for metric in ['r_global','r_scene']:
        v=torch.tensor([x['r_format']+x[metric] for x in xs],dtype=torch.float32);sd=v.std()
        advantages.append((v-v.mean())/(sd+1e-8) if sd>1e-8 else torch.zeros_like(v))
    maximum=float((advantages[0]-advantages[1]).abs().max())
    assert abs(maximum-group['max_standardized_advantage_difference'])<1e-6
    assert group['additional_signal']==(maximum>.05);groups.append(maximum>.05)
assert sum(groups)==branch['actual_groups'] and branch['required_groups']==5
assert branch['scene_arm_enabled']==(sum(groups)>=5)
assert branch['global_negative_same_scene_count']==sum(x['global_negative_is_same_scene'] for x in pilot_records)
scores=json.loads((r/'scoring/scores.json').read_text());assert len(scores)==150*len(branch['policies'])==len({x['id'] for x in scores})
t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True)
assert t['record_ids']==[x['id'] for x in scores]
assert torch.equal(t['text_features'],torch.load(r/'features/val.pt',map_location='cpu',weights_only=True)['text_features'])
vg=json.loads((r/'features/val_gallery.json').read_text());vl={v:i for i,v in enumerate(vg['text_order'][:391])}
positive=json.loads((r/'features/val_positive_columns.json').read_text())
for x in metadata['val']:assert positive[str(x['row'])]==sorted({vl[c] for c in x['captions']})
sim64=t['image_features'].double()@t['text_features'].double().T
assert float((sim64-t['similarities'].double()).abs().max())<1e-6
for i,x in enumerate(scores):
    assert x['split']=='val' and x['row'] in plan['validation_rows'];check_trace(x)
    pos=positive[str(x['row'])];neg=[c for c in range(391) if c not in pos];sim=t['similarities'][i]
    best=float(sim[pos].max());rank=int((sim[:391]>best).sum())
    assert rank==x['utility_rank0'] and 1/(1+rank)==x['utility_rr'] and best-float(sim[neg].max())==x['utility_margin']
    rank64_equal+=int((sim64[i,:391]>sim64[i,pos].max()).sum())==rank;total_rank_checks+=1
for policy in branch['policies']:
    expected=plan['inherited_initial_hash'] if policy=='baseline' else json.loads((r/'training'/policy/'result.json').read_text())['final_lora_tensor_hash']
    assert json.loads((r/'evaluation'/policy/'settings.json').read_text())['lora_tensor_hash']==expected
f=json.loads((r/'analysis.json').read_text());assert f['plan_sha256']==sha(r/'plan.json')
rebuilt={}
for item in f['per_image']:
    xs=[x for x in scores if x['policy']==item['policy'] and x['row']==item['row'] and x['role']==('greedy' if item['mode']=='greedy' else 'candidate')]
    assert set(item['selected_ids'])=={x['id'] for x in xs}
    for metric in ['utility_rr','utility_margin','quality','tool_usage']:
        values=[float(x['quality_usable']) if metric=='quality' else float(x['tools_used']>0) if metric=='tool_usage' else x[metric] for x in xs]
        value=sum(values)/len(values);assert abs(value-item[metric])<1e-12;rebuilt[item['policy'],item['row'],item['mode'],metric]=value
draw=np.random.default_rng(2026111406).integers(0,50,size=(10000,50));ci_count=0
for contrast,modes in f['paired'].items():
    policy,base=contrast.split('_minus_')
    for mode,metrics in modes.items():
        for metric,declared in metrics.items():
            d=np.array([rebuilt[policy,row,mode,metric]-rebuilt[base,row,mode,metric] for row in plan['validation_rows']])
            assert abs(float(d.mean())-declared['mean'])<1e-12
            assert np.allclose(np.percentile(d[draw].mean(axis=1),[2.5,97.5]),declared['interval95'],atol=1e-12,rtol=0);ci_count+=1
chosen='scene_margin' if branch['scene_arm_enabled'] else 'global_margin'
comparators=['global_margin','format_only'] if branch['scene_arm_enabled'] else ['baseline','format_only']
checks={f'greedy_rr_ci_above_{b}':f['paired'][chosen+'_minus_'+b]['greedy']['utility_rr']['interval95'][0]>0 for b in comparators}
means=f['policy_means']
checks.update(greedy_rr_at_least_initial=means[chosen]['greedy']['utility_rr']['mean']>=means['baseline']['greedy']['utility_rr']['mean'],
              greedy_quality_within_5pp=means[chosen]['greedy']['quality']['mean']-means['baseline']['greedy']['quality']['mean']>=-.05,
              optimizer_updated=f['training'][chosen]['optimizer_updates']>0)
assert f['decision']['chosen_by_training_only_gate']==chosen and f['decision']['checks']==checks
assert f['decision']['expand_criterion_passed']==all(checks.values())
for name in ['gate_code_hashes.json','pipeline_code_hashes.json','post_code_hashes.json']:
    for file,h in json.loads((r/name).read_text()).items():assert sha(r/file)==h
assert json.loads((r/'replay_verification/result.json').read_text())['ok']
output={'ok':True,'train_val_images_and_canonical_text_disjoint':True,'fresh50_selection_reproduced':True,
        'all_training_positive_and_scene_negative_sets_rebuilt':True,'scene_branch_gate_reproduced':True,
        'states_and_traces_checked':state_checks,'rounds_checked':round_checks,'real_tool_events_checked':tool_checks,
        'training_and_pilot_rewards_checked':training_rewards,'evaluation_scores_checked':len(scores),
        'rank64_equal_count':int(rank64_equal),'total_rank_checks':total_rank_checks,'paired_bootstrap_intervals_checked':ci_count,
        'torch':torch.__version__,'numpy':np.__version__}
a.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output))
