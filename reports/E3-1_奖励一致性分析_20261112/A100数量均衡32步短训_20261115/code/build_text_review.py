import argparse
import csv
from pathlib import Path
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;raw=r/'原始运行记录';cases={x['image']:x for x in read(r/'inputs/cases.json')};rows=[]
for x in read(raw/'scoring/all_scores.json'):
    if x['arm']!='after':continue
    tr=read(raw/x['trace_file']);c=cases[x['image']]
    rows.append({'图像':x['image'],'集合':x['split'],'角色':x['role'],'种子':x['seed'],'对象':c['object'],'人工数量':c['actual_count'],'质量合格':x['quality_usable'],'严格格式合格':x['strict_protocol_usable'],
      '真实工具数':x['tools_used'],'首轮实际输出':tr['rounds'][0]['generated_text'],'最终实际输出':tr['rounds'][-1]['generated_text'],'B句对判对':x['correct_B'],'B间隔':x['margin_B'],
      '文本事实准确性':'[待确认]','原始轨迹':x['trace_file'],'轨迹SHA256':x['trace_sha256']})
assert len(rows)==425
with (r/'训练后文本事实复核清单.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('ALL_425_FINAL_TEXTS_INDEXED')
