#!/usr/bin/env python3
"""Create six reviewable caption interventions from original recorded captions."""
import argparse,csv,json,re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args()
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return {r['图像编号']:r for r in csv.DictReader(f)}
rows=read(a.analysis);audit=read(a.audit)
names=['101.tif','1918.tif','1937.tif','2014.tif','2058.tif','2086.tif']
labels={'101.tif':'两种句式均正确','1918.tif':'两种句式均正确','1937.tif':'两种句式均错误',
        '2014.tif':'A正确_B错误','2058.tif':'A错误_B正确','2086.tif':'两种句式均错误'}
words={2:'Two',3:'Three',4:'Four',6:'Six'}
translations={
    '101.tif':('四架飞机停在机场。','飞机停在机场。'),
    '1918.tif':('两个储罐位于长方形框内，旁边停着一些车。','储罐位于长方形框内，旁边停着一些车。'),
    '1937.tif':('四个储罐整齐排列在地面上。','储罐整齐排列在地面上。'),
    '2014.tif':('六个网球场排成两排，周围有植物。','网球场排成两排，周围有植物。'),
    '2058.tif':('三个网球场整齐排列，周围有植物。','网球场整齐排列，周围有植物。'),
    '2086.tif':('四个网球场整齐排列在草坪上。','网球场整齐排列在草坪上。'),
}
results=[]
for name in names:
    r=rows[name];check=audit[name];original=r['原始第一条描述'];n=int(r['真实数量'])
    assert original==check['第一条描述'] and r['AGSE判断类型']==labels[name]
    assert int(r['对照错误数量'])>1 and n>1
    prefix=words[n]+' '
    assert original.startswith(prefix),(name,original)
    neutral=original[len(prefix):];neutral=neutral[0].upper()+neutral[1:]
    strict=''
    if name=='2014.tif':
        assert neutral.count('two lines')==1
        strict=neutral.replace('two lines','rows')
    results.append({'图像编号':name,'数据来源':r['数据来源'],'对象':r['对象'],'用户确认数量':n,
                    '检索时正确数量':n,'检索时错误数量':r['对照错误数量'],
                    '原先AGSE判断':r['AGSE判断类型'],'原始第一条描述':original,'原始描述中文':translations[name][0],
                    '去掉目标数量_最小改写':neutral,'最小改写中文':translations[name][1],
                    '进一步去除行数_候选':strict,'无数字候选中文':'网球场成排排列，周围有植物。' if name=='2014.tif' else '',
                    '审阅状态':'待用户确认'})
a.output_dir.mkdir(parents=True,exist_ok=True)
with (a.output_dir/'六图先验输入候选.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
summary={'selected_images':names,'n':len(results),'selection':'post-hoc exploratory: 2 both correct, 2 both wrong, 2 wording-dependent',
         'original_captions_verified':True,'candidate_status':'pending user review',
         'image_2014_note':'minimal rewrite retains two lines; second option replaces two lines with rows',
         'new_priors_generated':False,'new_model_scores_computed':False}
(a.output_dir/'设计核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
