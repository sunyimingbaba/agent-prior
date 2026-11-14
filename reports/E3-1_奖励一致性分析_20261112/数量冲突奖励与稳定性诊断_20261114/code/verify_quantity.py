import argparse
import csv
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
        ref=actual_tools[x['image']]
        assert e['origin']=='model_requested_actual_tool' and e['text']==ref['text'] and abs(e['confidence']-ref['confidence'])<1e-6
        tool_checks+=1
    state_checks+=1


for name,h in plan['inputs_sha256'].items():assert sha(r/name)==h
for name,h in plan['shared_code_sha256'].items():assert sha(r/'code'/name)==h
def read(name):return list(csv.DictReader((r/'inputs'/name).open(encoding='utf-8-sig')))
work={x['图像编号']:x for x in read('e1_final_100.csv')};aligned={x['图像编号']:x for x in read('e1_alignment_audit.csv')}
old={(x['image'],x['wording'][0]):x for x in read('e1_pair_scores.csv') if x['model']=='AGSE'}
cases=json.loads((r/'inputs/cases.json').read_text());by={x['image']:x for x in cases}
assert len(work)==len(cases)==len(by)==100 and sum(x['split']=='train' for x in cases)==78
assert not {x['暂不纳入图像'] for x in read('e1_overlap_removals.csv')}&set(by)
assert len({x['image_rgb_sha256'] for x in cases})==100
words={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
for item in cases:
    w=work[item['image']];ref=aligned[item['image']]
    assert w['状态']=='已确认数量' and int(w['用户确认数量'])==item['actual_count']==int(ref['人工确认数量'])
    assert item['split']==ref['来源']==w['数据来源'].split()[-1] and item['row']==int(ref['Arrow行号'])
    assert item['image_rgb_sha256']==ref['图像RGB_SHA256'] and item['original_first_caption']==ref['第一条描述']
    assert item['historical_prior_sha256']==ref['先验SHA256']
    for wording,pair in item['pairs'].items():
        row=old[item['image'],wording];pos=int(row['correct_text_position'])
        assert pair['positive']==[row['text1'],row['text2']][pos-1] and pair['negative']==[row['text1'],row['text2']][2-pos]
        assert item['actual_count']==int(row['actual_count']) and pair['wrong_count']==int(row['incorrect_count'])!=item['actual_count']
        assert re.sub(r'\b'+words[item['actual_count']]+r'\b',words[pair['wrong_count']],pair['positive'],count=1)==pair['negative']
assert not set(plan['policy']['inherited_optimization_images'])&set(by)
gallery=json.loads((r/'features/gallery.json').read_text());features=torch.load(r/'features/tensors.pt',map_location='cpu',weights_only=True)
qfeatures=features['quantity_text_features'];vfeatures=features['original_val_text_features']
assert qfeatures.shape==(56,768) and vfeatures.shape==(419,768)
for item in cases:
    for wording,pair in item['pairs'].items():
        assert [gallery['quantity_text_order'][i] for i in gallery['quantity_columns'][item['image']][wording]]==[pair['positive'],pair['negative']]
alignment=json.loads((r/'dataset_alignment.json').read_text());assert alignment['all_rgb_hashes_and_original_first_captions_match']
vl={text:i for i,text in enumerate(gallery['original_val_text_order'][:391])}
for item in alignment['cases']:
    assert item['image_rgb_sha256']==by[item['image']]['image_rgb_sha256']
    if item['split']=='val':assert gallery['val_positive_columns'][item['image']]==sorted({vl[c] for c in item['captions']})
anchors=json.loads((r/'historical_anchors/records.json').read_text());at=torch.load(r/'historical_anchors/tensors.pt',map_location='cpu',weights_only=True)
assert at['images']==[x['image'] for x in cases] and len(anchors)==100
asim64=at['image_features'].double()@qfeatures.double().T
assert float((asim64-at['quantity_similarities'].double()).abs().max())<1e-6
for i,x in enumerate(anchors):
    item=by[x['image']];assert sha(r/x['state_file'])==x['state_file_sha256']==item['historical_prior_sha256']
    hidden=torch.load(r/x['state_file'],map_location='cpu',weights_only=True);assert hidden.shape==(3584,) and hidden.dtype==torch.bfloat16
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        for index,sign in enumerate(['positive','negative']):
            assert x[wording+'_'+sign]==float(at['quantity_similarities'][i,cols[index]])
            assert abs(x[wording+'_'+sign]-item['pairs'][wording]['historical_'+sign+'_score'])<1e-6
scores=json.loads((r/'scoring/smoke_scores.json').read_text())+json.loads((r/'scoring/scores.json').read_text())
assert len(scores)==503 and len({x['id'] for x in scores})==503
t=torch.load(r/'scoring/tensors.pt',map_location='cpu',weights_only=True)
assert t['record_ids']==[x['id'] for x in scores]
q64=t['image_features'].double()@qfeatures.double().T
assert float((q64-t['quantity_similarities'].double()).abs().max())<1e-6
vlookup={name:i for i,name in enumerate(t['val_record_ids'])};assert len(vlookup)==110
all_index={name:i for i,name in enumerate(t['record_ids'])}
v64=t['image_features'][[all_index[name] for name in t['val_record_ids']]].double()@vfeatures.double().T
assert float((v64-t['val_similarities'].double()).abs().max())<1e-6
rank64=0;classification64=0
for i,x in enumerate(scores):
    check_trace(x);item=by[x['image']];assert x['row']==item['row'] and x['split']==item['split']
    trace=json.loads((r/x['trace_file']).read_text());first=trace['rounds'][0]['prompt_text']
    users=re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',first,re.S)
    assert users[-1].endswith('查询文本: '+plan['prompt'])
    for wording,cols in gallery['quantity_columns'][x['image']].items():
        cp,cn=[float(t['quantity_similarities'][i,c]) for c in cols]
        assert x[wording+'_positive']==cp and x[wording+'_negative']==cn and x['margin_'+wording]==cp-cn
        assert x['correct_'+wording]==(cp>cn)
        classification64+=bool(q64[i,cols[0]]>q64[i,cols[1]])==x['correct_'+wording]
    assert x['r_format']==(.2 if x['quality_usable'] else 0.)
    assert x['r_total_quantity']==x['r_format']+x['margin_A']
    assert x['r_total_quality_gated']==x['r_format']+(x['margin_A'] if x['quality_usable'] else 0.)
    if x['split']=='val':
        vi=vlookup[x['id']];sim=t['val_similarities'][vi];pos=gallery['val_positive_columns'][x['image']]
        best=float(sim[pos].max());rank=int((sim[:391]>best).sum());neg=[c for c in range(391) if c not in pos]
        assert rank==x['retrieval_rank0'] and x['retrieval_rr']==1/(1+rank) and x['retrieval_margin']==best-float(sim[neg].max())
        rank64+=int((v64[vi,:391]>v64[vi,pos].max()).sum())==rank
    else:assert x['retrieval_rr'] is None
full=[x for x in scores if x['id'].startswith('all__')];assert len(full)==500
for i,item in enumerate(cases):
    xs=[x for x in full if x['image']==item['image']];assert len(xs)==5
    sampled=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);assert len(sampled)==4
    assert [x['seed'] for x in sampled]==[800000+i*4+g for g in range(4)]
for mode in ['smoke','generation']:
    result=json.loads((r/mode/'result.json').read_text());settings=json.loads((r/mode/'settings.json').read_text())
    assert result['lora_tensor_hash_unchanged']==settings['policy_lora_tensor_hash']==plan['policy']['lora_tensor_hash']
    assert settings['optimization_updates']==0 and settings['policy_frozen']
f=json.loads((r/'analysis.json').read_text());assert f['plan_sha256']==sha(r/'plan.json') and f['optimization_updates']==0
rebuilt={}
for item in cases:
    xs=[x for x in full if x['image']==item['image']];greedy=next(x for x in xs if x['role']=='greedy')
    sampled=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);valid=[x for x in sampled if x['quality_usable']]
    mapping={'greedy':[greedy],'sample_mean':sampled,'format_first':[valid[0] if valid else greedy],
             'quantity_raw':[max(sampled,key=lambda x:x['margin_A'])],
             'quantity_quality':[max(valid,key=lambda x:x['margin_A']) if valid else greedy],
             'reverse_quality':[min(valid,key=lambda x:x['margin_A']) if valid else greedy]}
    for selector,values in mapping.items():
        actual=next(x for x in f['per_image'] if x['image']==item['image'] and x['selector']==selector)
        assert actual['selected_ids']==[x['id'] for x in values]
        for metric in ['margin_A','margin_B','pairacc_A','pairacc_B','quality','tool_usage','retrieval_rr','retrieval_margin']:
            numbers=[float(x['correct_'+metric[-1]]) if metric.startswith('pairacc_') else float(x['quality_usable']) if metric=='quality'
                else float(x['tools_used']>0) if metric=='tool_usage' else x[metric] for x in values]
            value=None if numbers[0] is None else sum(numbers)/len(numbers)
            assert value is None and actual[metric] is None or abs(value-actual[metric])<1e-12
            rebuilt[item['image'],selector,metric]=value
    diag=next(x for x in f['candidate_diagnostics'] if x['image']==item['image'])
    informative=len(valid)>=2 and max(x['margin_A'] for x in valid)-min(x['margin_A'] for x in valid)>1e-6
    assert diag['informative']==informative and diag['usable_candidates']==len(valid)
ci_count=0
for split in ['train','val']:
    images=[x['image'] for x in cases if x['split']==split];n=len(images);draw=np.random.default_rng(2026111418).integers(0,n,size=(10000,n))
    for contrast,metrics in f['groups'][split]['paired'].items():
        selector,base=contrast.split('_minus_')
        for metric,declared in metrics.items():
            d=np.array([rebuilt[name,selector,metric]-rebuilt[name,base,metric] for name in images])
            assert abs(float(d.mean())-declared['mean'])<1e-12
            assert np.allclose(np.percentile(d[draw].mean(axis=1),[2.5,97.5]),declared['interval95'],atol=1e-12,rtol=0);ci_count+=1
    for selector,metrics in f['groups'][split]['selector_means'].items():
        for metric,s in metrics.items():assert abs(float(np.mean([rebuilt[name,selector,metric] for name in images]))-s['mean'])<1e-12
g=f['groups']['train'];threshold=plan['analysis']['next_training_gate'];gain=g['paired']['quantity_quality_minus_format_first']['margin_B']
informative=sum(x['informative'] for x in f['candidate_diagnostics'] if x['split']=='train')
assert informative==g['candidate_diagnostics']['informative_images']
checks={'training_candidate_protocol_quality':g['selector_means']['sample_mean']['quality']['mean']>=threshold['train_generation_quality_min'],
 'at_least_47_informative_training_images':informative>=threshold['train_images_with_2_usable_and_A_range_above_1e_6_min'],
 'B_margin_gain_at_least_1e_4':gain['mean']>=threshold['heldout_B_margin_gain_min'],'B_margin_gain_interval_above_zero':gain['interval95'][0]>0,
 'B_pairacc_at_least_format_first':g['selector_means']['quantity_quality']['pairacc_B']['mean']>=g['selector_means']['format_first']['pairacc_B']['mean'],
 'B_margin_vs_reverse_interval_above_zero':g['paired']['quantity_quality_minus_reverse_quality']['margin_B']['interval95'][0]>0,
 'selected_protocol_quality_at_least_95pct':g['selector_means']['quantity_quality']['quality']['mean']>=threshold['quality_selected_min']}
assert checks==f['decision']['checks'] and all(checks.values())==f['decision']['next_training_gate_passed'] and f['decision']['gate_uses_train_only']
for name in ['gate_code_hashes.json','pipeline_code_hashes.json','post_code_hashes.json']:
    for file,h in json.loads((r/name).read_text()).items():assert sha(r/file)==h
assert json.loads((r/'replay_verification/result.json').read_text())['ok']
output={'ok':True,'confirmed_images_checked':100,'count_only_counterfactual_pairs_checked':200,'historical_scalar_anchors_checked':400,
        'states_and_traces_checked':state_checks,'generation_rounds_checked':round_checks,'real_tool_events_checked':tool_checks,
        'quantity_score_records_checked':503,'actual_neutral_input_prompts_checked':503,'selectors_rebuilt':600,
        'paired_bootstrap_intervals_checked':ci_count,'float64_quantity_classifications_equal':int(classification64),
        'quantity_classification_checks':1006,'float64_val_retrieval_ranks_equal':int(rank64),'val_retrieval_rank_checks':110,
        'all_policy_parameter_hashes_frozen':True,'training_only_gate_rebuilt':True,'torch':torch.__version__,'numpy':np.__version__}
a.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output))
