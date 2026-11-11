#!/usr/bin/env python3
"""Summarize already-scored E1 images without changing scores or predictions."""
import argparse, collections, csv, json, math
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--results',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args()
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
results=read(a.results); audits=read(a.audit)
assert len(results)==400 and len(audits)==100
idx={(x['model'],x['image'],x['wording']):x for x in results}
assert len(idx)==400
ann={x['图像编号']:x for x in audits};assert len(ann)==100
rows=[]
for name in sorted(ann,key=lambda n:int(n.split('.')[0])):
    v={}
    for method in ('CLIP','AGSE'):
        for key,wording in [('A','A_原句式'),('B','B_改写句式')]:
            r=idx[(method,name,wording)];actual=int(r['actual_count']);wrong=int(r['incorrect_count'])
            cp=float(r['correct_score']);cn=float(r['wrong_score']);assert math.isfinite(cp) and math.isfinite(cn) and cp!=cn
            assert (r['correct']=='True')==(cp>cn)
            v[method+'_'+key]={'correct_score':cp,'wrong_score':cn,'margin':cp-cn,'preferred_count':actual if cp>cn else wrong,'correct':cp>cn}
    agA=v['AGSE_A']['correct'];agB=v['AGSE_B']['correct']
    category={(True,True):'两种句式均正确',(False,False):'两种句式均错误',(True,False):'A正确_B错误',(False,True):'A错误_B正确'}[(agA,agB)]
    r=idx[('AGSE',name,'A_原句式')];audit=ann[name]
    row={'图像编号':name,'数据来源':r['split'],'对象':r['object'],'真实数量':int(r['actual_count']),
         '对照错误数量':int(r['incorrect_count']),'AGSE判断类型':category,
         '第一句含对应数词':audit['第一条描述出现对应英文数词'],'原始第一条描述':r['original_first_caption']}
    for key in ('AGSE_A','AGSE_B','CLIP_A','CLIP_B'):
        for f,val in v[key].items():row[key+'_'+f]=val
    rows.append(row)
assert len(rows)==100
by=collections.Counter(r['AGSE判断类型'] for r in rows)
assert by=={'两种句式均正确':76,'两种句式均错误':12,'A正确_B错误':5,'A错误_B正确':7}
summary={'图片总数':len(rows),'AGSE判断类型':dict(by),
         '两种句式均错误的对象':dict(collections.Counter(r['对象'] for r in rows if r['AGSE判断类型']=='两种句式均错误')),
         '两种句式均错误的第一句含对应数词':dict(collections.Counter(r['第一句含对应数词'] for r in rows if r['AGSE判断类型']=='两种句式均错误')),
         '错误或变化图片总数':sum(r['AGSE判断类型']!='两种句式均正确' for r in rows),
         '筛选用于说明的案例':['983.tif','1937.tif','2014.tif','184.tif']}
a.output_dir.mkdir(parents=True,exist_ok=True)
with (a.output_dir/'100张图逐图判断.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output_dir/'分析统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
