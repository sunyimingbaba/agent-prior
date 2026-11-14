"""Describe protocol failure modes without changing scores or selection."""
import argparse
from collections import Counter
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
assert json.loads((r/'post_result.json').read_text())['status']=='complete'
scores=json.loads((r/'scoring/scores.json').read_text());analysis=json.loads((r/'analysis.json').read_text())
assert len(scores)==500
by={x['id']:x for x in scores};groups={}
for split in ['train','val']:
    g={}
    for role in ['greedy','candidate']:
        xs=[x for x in scores if x['split']==split and x['role']==role]
        bad=[x for x in xs if not x['quality_usable']]
        g[role]={'records':len(xs),'protocol_valid':len(xs)-len(bad),'protocol_valid_rate':(len(xs)-len(bad))/len(xs),
                 'all_stop_reasons':dict(Counter(x['stop_reason'] for x in xs)),
                 'invalid_stop_reasons':dict(Counter(x['stop_reason'] for x in bad)),
                 'invalid_ids':[x['id'] for x in bad]}
    selected=[by[x['selected_ids'][0]] for x in analysis['per_image'] if x['split']==split and x['selector']=='quantity_raw']
    g['raw_quantity_selection']={'images':len(selected),'protocol_valid':sum(x['quality_usable'] for x in selected),
        'invalid_stop_reasons':dict(Counter(x['stop_reason'] for x in selected if not x['quality_usable']))}
    groups[split]=g
out={'ok':True,'scope':'descriptive breakdown of existing 500 frozen-policy trajectories; no new generation or optimization',
     'source_scores':'scoring/scores.json','source_analysis':'analysis.json','groups':groups}
(r/'generation_failure_diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({s:{k:{z:v for z,v in x.items() if z!='invalid_ids'} for k,x in g.items()} for s,g in groups.items()}))
