import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
plan=json.loads((r/'plan.json').read_text());scores=json.loads((r/'scoring/scores.json').read_text())
old=json.loads((r/'baseline_evidence/scores.json').read_text());quantity=json.loads((r/'analysis.json').read_text())
old_quantity=json.loads((r/'baseline_evidence/analysis.json').read_text())
cases=json.loads((r/'inputs/cases.json').read_text());cal=set(plan['calibration_images'])
groups={};rows=[];paired_rows=[];per_image=[];failure_rows=[]
metrics=['quality','strict_quality','generated_result_failure','tool_usage','margin_B','pairacc_B','retrieval_rr']
import tarfile
needed={x['trace_file'] for x in old};baseline_traces={}
with tarfile.open(r/'baseline_evidence/quantity_counterfactual_evidence_20261114.tar.gz') as archive:
    for member in archive:
        if member.name in needed:baseline_traces[member.name]=json.loads(archive.extractfile(member).read())
assert len(baseline_traces)==500

def strict_quality(x,baseline=False):
    if not baseline:return float(x['strict_protocol_usable'])
    import re
    trace=baseline_traces[x['trace_file']]
    answer=re.fullmatch(r'<answer>(.*?)</answer>',trace['rounds'][-1]['generated_text'].strip(),re.S)
    return float(bool(x['quality_usable'] and answer and answer.group(1).strip() and
        all(re.fullmatch(r'<tool_call>\s*scene_classify\s*</tool_call>',z['generated_text'].strip()) for z in trace['rounds'][:-1])))

strict_old={x['id']:strict_quality(x,True) for x in old}
for split in ['calibration15','confirmation_train63','train78','val22']:
    images=[x['image'] for x in cases if (x['image'] in cal if split=='calibration15' else
       x['split']=='train' and x['image'] not in cal if split=='confirmation_train63' else
       x['split']=='train' if split=='train78' else x['split']=='val')]
    n=len(images);draw=np.random.default_rng(plan['protocol_analysis']['bootstrap']['seed']).integers(0,n,size=(10000,n))
    def stats(v):
        v=np.asarray(v,dtype=np.float64)
        return {'mean':float(v.mean()),'interval95':[float(x) for x in np.percentile(v[draw].mean(axis=1),[2.5,97.5])],'images':n}
    g={'images':n,'means':{},'paired':{}}
    for arm,source in [('baseline',old),('selected',scores)]:
        g['means'][arm]={}
        for role in ['greedy','candidate']:
            g['means'][arm][role]={}
            selected=[x for x in source if x['image'] in images and x['role']==role]
            failure_rows.append({'group':split,'arm':arm,'role':role,'records':len(selected),
                'quality_valid':sum(x['quality_usable'] for x in selected),'stop_reasons':dict(Counter(x['stop_reason'] for x in selected))})
            for name in images:
                xs=[x for x in selected if x['image']==name];assert len(xs)==(4 if role=='candidate' else 1)
                item={'group':split,'image':name,'arm':arm,'role':role}
                for m in metrics:
                    values=[float(x['quality_usable']) if m=='quality' else strict_old[x['id']] if m=='strict_quality' and arm=='baseline'
                      else float(x['strict_protocol_usable']) if m=='strict_quality'
                      else float(x['stop_reason']=='model_generated_tool_result') if m=='generated_result_failure'
                      else float(x['tools_used']>0) if m=='tool_usage' else float(x['correct_B']) if m=='pairacc_B' else x[m] for x in xs]
                    item[m]=None if values[0] is None else math.fsum(values)/len(values)
                item['valid_distinct_hidden_states']=len({x['generation_state_sha256'] for x in xs if x['quality_usable']})
                per_image.append(item)
            for m in metrics:
                values=[x[m] for x in per_image if x['group']==split and x['arm']==arm and x['role']==role]
                if values[0] is None:continue
                st=stats(values);g['means'][arm][role][m]=st
                rows.append({'group':split,'arm':arm,'role':role,'metric':m,'mean':st['mean'],'lo':st['interval95'][0],'hi':st['interval95'][1]})
    for role in ['greedy','candidate']:
        g['paired'][role]={}
        for m in metrics:
            if m not in g['means']['selected'][role]:continue
            lookup={(x['image'],x['arm']):x[m] for x in per_image if x['group']==split and x['role']==role}
            st=stats([lookup[name,'selected']-lookup[name,'baseline'] for name in images]);g['paired'][role][m]=st
            paired_rows.append({'group':split,'role':role,'metric':m,'mean':st['mean'],'lo':st['interval95'][0],'hi':st['interval95'][1]})
    groups[split]=g

g=groups['confirmation_train63'];threshold=plan['confirmation_gate']
selected_quality=[x for x in quantity['per_image'] if x['image'] in plan['confirmation_train_images'] and x['selector']=='quantity_quality']
selected_rate=float(np.mean([x['quality'] for x in selected_quality]));candidate=g['means']['selected']['candidate']
checks={'sampled_quality_at_least_90pct':candidate['quality']['mean']>=threshold['legacy_quality_min'],
 'strict_sampled_quality_at_least_90pct':candidate['strict_quality']['mean']>=threshold['strict_quality_min'],
 'generated_result_failure_rate_at_most_2pct':candidate['generated_result_failure']['mean']<=threshold['generated_result_failure_rate_max'],
 'paired_quality_gain_interval_above_zero':g['paired']['candidate']['quality']['interval95'][0]>0,
 'selected_quality_at_least_95pct':selected_rate>=threshold['selected_quality_min']}
out={'ok':True,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'optimization_updates':0,
 'groups':groups,'per_image':per_image,'failure_breakdown':failure_rows,
 'confirmation_gate':{'passed':all(checks.values()),'checks':checks,'uses_calibration_and_val':False,
    'selected_quantity_protocol_quality':selected_rate},
 'original_train78_quantity_gate':quantity['decision'],'selection':json.loads((r/'protocol_selection.json').read_text())}
(r/'protocol_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for name,values in [('protocol_summary.csv',rows),('protocol_paired_deltas.csv',paired_rows),('protocol_per_image.csv',per_image),('protocol_failures.csv',failure_rows)]:
    with (r/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader()
        for x in values:w.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in x.items()})
print(json.dumps({'confirmation_gate':out['confirmation_gate'],'original_train78_quantity_gate':out['original_train78_quantity_gate'],
                 'confirmation_candidate_means':g['means'],'paired_quality_gain':g['paired']['candidate']['quality']}))
