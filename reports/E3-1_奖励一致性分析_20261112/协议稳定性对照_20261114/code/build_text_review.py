"""Prepare a human-review table, preserving text and avoiding invented text-accuracy scores."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;raw=r/'原始运行记录'
cases={x['image']:x for x in json.loads((r/'inputs/cases.json').read_text())}
analysis=json.loads((r/'analysis.json').read_text());scores={x['id']:x for x in json.loads((raw/'scoring/scores.json').read_text())}
rows=[]
for choice in analysis['per_image']:
    if choice['selector'] not in ['greedy','format_first','quantity_quality']:continue
    x=scores[choice['selected_ids'][0]];item=cases[x['image']];trace=json.loads((raw/x['trace_file']).read_text())
    rows.append({'图像':x['image'],'原数据划分':x['split'],'本轮用途':'校准' if x['image'] in json.loads((r/'plan.json').read_text())['calibration_images'] else '确认' if x['split']=='train' else '补充诊断',
      '对象':item['object'],'人工确认数量':item['actual_count'],'选择方式':choice['selector'],'轨迹ID':x['id'],
      '协议合格':x['quality_usable'],'严格协议合格':x['strict_protocol_usable'],'真实工具调用次数':x['tools_used'],
      'B句对判对':x['correct_B'],'B句对分数间隔':x['margin_B'],
      '最终原始文本':trace['rounds'][-1]['generated_text'],'完整助手轨迹':trace['trajectory_text'],
      '文本事实准确性':'[待人工核对]','原始轨迹文件':x['trace_file'],'轨迹SHA256':x['trace_sha256']})
assert len(rows)==300
with (r/'文本事实复核清单.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('TEXT_REVIEW_ROWS',len(rows))
