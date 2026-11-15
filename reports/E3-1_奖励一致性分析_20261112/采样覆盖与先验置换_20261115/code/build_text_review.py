import csv
from pathlib import Path
from stage_io import read
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;raw=r/'原始运行记录';plan=read(r/'plan.json');by={x['image']:x for x in read(r/'inputs/cases.json')};rows=[]
for policy in plan['policies']:
    for line in (raw/'generation'/policy['name']/'records.jsonl').read_text().splitlines():
        import json
        x=json.loads(line);tr=read(raw/x['trace_file']);c=by[x['image']]
        rows.append({'策略':x['policy'],'图像':x['image'],'人工数量':c['actual_count'],'对象':c['object'],'采样序号':x['sample'],'种子':x['seed'],'质量合格':x['quality_usable'],'严格格式合格':x['strict_protocol_usable'],'实际工具数':x['tools_used'],'首轮输出':tr['rounds'][0]['generated_text'],'最终输出':tr['rounds'][-1]['generated_text'],'文本事实准确性':'[待确认]','状态哈希':x['generation_state_sha256'],'轨迹文件':x['trace_file'],'轨迹哈希':x['trace_sha256']})
assert len(rows)==256
with (r/'256条文本事实复核清单.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('ALL_256_TEXTS_INDEXED')
