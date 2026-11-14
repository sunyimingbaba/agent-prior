import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'protocol_analysis.json').read_text());cases={x['image']:x for x in json.loads((r/'inputs/cases.json').read_text())}
rows=[]
for group in ['confirmation_train63','val22']:
    for target in sorted({x['object'] for x in cases.values()}):
        for arm in ['baseline','selected']:
            for role in ['greedy','candidate']:
                xs=[x for x in j['per_image'] if x['group']==group and x['arm']==arm and x['role']==role and cases[x['image']]['object']==target]
                if not xs:continue
                rows.append({'图像组':group,'目标':target,'流程':arm,'轨迹类型':role,'图像数':len(xs),
                    '协议合格率':sum(x['quality'] for x in xs)/len(xs),'严格协议合格率':sum(x['strict_quality'] for x in xs)/len(xs),
                    '自产工具结果率':sum(x['generated_result_failure'] for x in xs)/len(xs),
                    'B句对判对率':sum(x['pairacc_B'] for x in xs)/len(xs),'B句对平均间隔':sum(x['margin_B'] for x in xs)/len(xs)})
with (r/'分目标协议与数量结果.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('OBJECT_BREAKDOWN_ROWS',len(rows))
