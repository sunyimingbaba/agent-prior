import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;raw=r/'原始运行记录'
j=json.loads((r/'analysis.json').read_text());scores={x['id']:x for x in json.loads((raw/'scoring/all_scores.json').read_text())}
cases={x['image']:x for x in json.loads((r/'inputs/cases.json').read_text())};rows=[]
for entry in j['per_image']:
    if entry['arm']!='sft_strict' or entry['selector'] not in ['greedy','format_first','quantity_quality']:continue
    x=scores[entry['selected_ids'][0]];t=json.loads((raw/x['trace_file']).read_text());item=cases[x['image']]
    rows.append({'图像':x['image'],'来源':x['split'],'对象':item['object'],'人工数量':item['actual_count'],'选择方式':entry['selector'],
      '协议合格':x['quality_usable'],'严格协议合格':x['strict_protocol_usable'],'实际工具调用数':x['tools_used'],
      '首轮实际输出':t['rounds'][0]['generated_text'],'最终实际输出':t['rounds'][-1]['generated_text'],
      'B句对判对':x['correct_B'],'B句对间隔':x['margin_B'],'文本事实准确性':'[待人工核对]',
      '轨迹ID':x['id'],'原始轨迹文件':x['trace_file'],'轨迹SHA256':x['trace_sha256']})
assert len(rows)==255
with (r/'文本事实复核清单.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('SFT_TEXT_REVIEW_ROWS',len(rows))
