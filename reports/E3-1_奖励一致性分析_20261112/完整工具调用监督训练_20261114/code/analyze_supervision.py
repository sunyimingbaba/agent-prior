import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
plan=json.loads((r/'plan.json').read_text());cases=json.loads((r/'inputs/evaluation_cases.json').read_text());case_ids={x['image'] for x in cases}
records=json.loads((r/'scoring/all_scores.json').read_text());historical=[x for x in json.loads((r/'baseline_evidence/scores.json').read_text()) if x['image'] in case_ids]
needed={x['trace_file'] for x in historical};traces={}
with tarfile.open(r/'baseline_evidence/quantity_counterfactual_evidence_20261114.tar.gz') as t:
    for m in t:
        if m.name in needed:traces[m.name]=json.loads(t.extractfile(m).read())
assert len(traces)==425
for x in historical:
    t=traces[x['trace_file']];answer=re.fullmatch(r'<answer>(.*?)</answer>',t['rounds'][-1]['generated_text'].strip(),re.S)
    x['strict_protocol_usable']=bool(x['quality_usable'] and answer and answer.group(1).strip() and all(re.fullmatch(r'<tool_call>\s*scene_classify\s*</tool_call>',z['generated_text'].strip()) for z in t['rounds'][:-1]))
    x['arm']='historical_legacy'
all_records=records+historical
arms=['historical_legacy','c742_strict','sft_strict','sft_legacy_greedy']
metrics=['quality','strict_quality','tool_usage','generated_result_failure','margin_A','margin_B','pairacc_A','pairacc_B','retrieval_rr','retrieval_margin']
per=[];diagnostics=[]
for arm in arms:
    for item in cases:
        xs=[x for x in all_records if x['arm']==arm and x['image']==item['image']];greedy=next(x for x in xs if x['role']=='greedy')
        sampled=sorted([x for x in xs if x['role']=='candidate'],key=lambda x:x['seed']);valid=[x for x in sampled if x['quality_usable']]
        selected={'greedy':[greedy]}
        if arm!='sft_legacy_greedy':
            assert len(sampled)==4
            selected.update(sample_mean=sampled,format_first=[valid[0] if valid else greedy],
                quantity_raw=[max(sampled,key=lambda x:x['margin_A'])],quantity_quality=[max(valid,key=lambda x:x['margin_A']) if valid else greedy],
                reverse_quality=[min(valid,key=lambda x:x['margin_A']) if valid else greedy])
            av=[x['margin_A'] for x in valid]
            diagnostics.append({'arm':arm,'image':item['image'],'split':item['split'],'valid_samples':len(valid),
                'informative':len(valid)>=2 and max(av)-min(av)>1e-6,'A_range_valid':float(np.ptp(av)) if av else 0.,
                'unique_valid_states':len({x['generation_state_sha256'] for x in valid}),'fallback':not valid})
        for selector,values in selected.items():
            entry={'arm':arm,'image':item['image'],'split':item['split'],'selector':selector,'selected_ids':[x['id'] for x in values]}
            for m in metrics:
                ns=[float(x['quality_usable']) if m=='quality' else float(x['strict_protocol_usable']) if m=='strict_quality'
                    else float(x['tools_used']>0) if m=='tool_usage' else float(x['stop_reason']=='model_generated_tool_result') if m=='generated_result_failure'
                    else float(x['correct_'+m[-1]]) if m.startswith('pairacc_') else x[m] for x in values]
                entry[m]=None if ns[0] is None else math.fsum(ns)/len(ns)
            per.append(entry)
index={(x['arm'],x['image'],x['selector']):x for x in per};groups={};summary=[];paired_rows=[]
for split in ['train','val']:
    images=[x['image'] for x in cases if x['split']==split];n=len(images)
    draw=np.random.default_rng(plan['analysis']['bootstrap']['seed']).integers(0,n,size=(10000,n))
    def stats(ns):
        v=np.asarray(ns,dtype=np.float64);ci=np.percentile(v[draw].mean(axis=1),[2.5,97.5])
        return {'mean':float(v.mean()),'interval95':[float(x) for x in ci],'images':n}
    g={'images':n,'means':{},'paired':{},'quantity_paired':{}}
    for arm in arms:
        selectors=['greedy'] if arm=='sft_legacy_greedy' else list(plan['selectors']);g['means'][arm]={}
        for selector in selectors:
            g['means'][arm][selector]={}
            for m in metrics:
                ns=[index[arm,name,selector][m] for name in images]
                if ns[0] is None:continue
                st=stats(ns);g['means'][arm][selector][m]=st
                summary.append({'split':split,'arm':arm,'selector':selector,'metric':m,'mean':st['mean'],'lo':st['interval95'][0],'hi':st['interval95'][1]})
        if arm!='sft_legacy_greedy':
            g['quantity_paired'][arm]={}
            for selector,base in [('quantity_quality','format_first'),('quantity_quality','reverse_quality'),('quantity_quality','greedy')]:
                key=selector+'_minus_'+base;g['quantity_paired'][arm][key]={}
                for m in g['means'][arm]['greedy']:
                    st=stats([index[arm,name,selector][m]-index[arm,name,base][m] for name in images]);g['quantity_paired'][arm][key][m]=st
                    paired_rows.append({'split':split,'comparison':arm+'__'+key,'metric':m,'mean':st['mean'],'lo':st['interval95'][0],'hi':st['interval95'][1]})
    comparisons=[('sft_strict','c742_strict',s) for s in ['greedy','sample_mean','format_first','quantity_quality']]+[('sft_legacy_greedy','historical_legacy','greedy')]
    for arm,base,selector in comparisons:
        key=arm+'_minus_'+base+'__'+selector;g['paired'][key]={}
        for m in g['means'][arm][selector]:
            st=stats([index[arm,name,selector][m]-index[base,name,selector][m] for name in images]);g['paired'][key][m]=st
            paired_rows.append({'split':split,'comparison':key,'metric':m,'mean':st['mean'],'lo':st['interval95'][0],'hi':st['interval95'][1]})
    groups[split]=g
g=groups['train'];m=g['means'];threshold=plan['readiness_gate'];training=json.loads((r/'training/result.json').read_text())
checks={'optimizer_updates_exactly_30':training['optimizer_updates']==threshold['optimizer_updates_required'],
 'strict_sample_quality_at_least_90pct':m['sft_strict']['sample_mean']['strict_quality']['mean']>=threshold['strict_sample_quality_min'],
 'strict_greedy_quality_at_least_95pct':m['sft_strict']['greedy']['strict_quality']['mean']>=threshold['strict_greedy_quality_min'],
 'sample_actual_tool_use_at_least_80pct':m['sft_strict']['sample_mean']['tool_usage']['mean']>=threshold['sample_actual_tool_use_min'],
 'generated_result_failure_at_most_2pct':m['sft_strict']['sample_mean']['generated_result_failure']['mean']<=threshold['generated_result_failure_max'],
 'sample_B_pairacc_at_least_matched_control':m['sft_strict']['sample_mean']['pairacc_B']['mean']>=m['c742_strict']['sample_mean']['pairacc_B']['mean'],
 'legacy_greedy_B_pairacc_drop_at_most_5pp':m['sft_legacy_greedy']['greedy']['pairacc_B']['mean']>=m['historical_legacy']['greedy']['pairacc_B']['mean']-threshold['legacy_greedy_B_pairacc_drop_max']}
qth=plan['analysis']['next_training_gate'];gain=g['quantity_paired']['sft_strict']['quantity_quality_minus_format_first']['margin_B']
informative=sum(x['informative'] for x in diagnostics if x['arm']=='sft_strict' and x['split']=='train')
qchecks={'sample_quality_at_least_90pct':m['sft_strict']['sample_mean']['quality']['mean']>=qth['train_generation_quality_min'],
 'at_least_38_informative_images':informative>=qth['train_images_with_2_usable_and_A_range_above_1e_6_min'],
 'B_margin_gain_at_least_1e_4':gain['mean']>=qth['heldout_B_margin_gain_min'],'B_margin_gain_interval_above_zero':gain['interval95'][0]>0,
 'B_pairacc_at_least_format_first':m['sft_strict']['quantity_quality']['pairacc_B']['mean']>=m['sft_strict']['format_first']['pairacc_B']['mean'],
 'B_margin_vs_reverse_interval_above_zero':g['quantity_paired']['sft_strict']['quantity_quality_minus_reverse_quality']['margin_B']['interval95'][0]>0,
 'selected_quality_at_least_95pct':m['sft_strict']['quantity_quality']['quality']['mean']>=qth['quality_selected_min']}
result={'ok':True,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'optimizer_updates':training['optimizer_updates'],
 'groups':groups,'per_image':per,'candidate_diagnostics':diagnostics,
 'format_readiness':{'passed':all(checks.values()),'checks':checks,'uses_confirmation_train_only':True},
 'quantity_readiness':{'passed':all(qchecks.values()),'checks':qchecks,'informative_train_images':informative,'uses_confirmation_train_only':True,
    'label_assisted_offline_selection':True},'training':training}
(r/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name,rows in [('selector_summary.csv',summary),('paired_deltas.csv',paired_rows),('per_image_selectors.csv',per),('candidate_diagnostics.csv',diagnostics),('quantity_scores.csv',records)]:
    with (r/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        for x in rows:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()})
print(json.dumps({'format_readiness':result['format_readiness'],'quantity_readiness':result['quantity_readiness'],
    'matched_strict_sample_quality_before':m['c742_strict']['sample_mean']['strict_quality']['mean'],
    'matched_strict_sample_quality_after':m['sft_strict']['sample_mean']['strict_quality']['mean'],
    'actual_tool_use_after':m['sft_strict']['sample_mean']['tool_usage']['mean']}))
