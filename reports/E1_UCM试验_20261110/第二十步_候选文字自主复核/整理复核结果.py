#!/usr/bin/env python3
"""Record assistant visual review decisions without overwriting the draft."""
import csv,json,re
from collections import Counter
from pathlib import Path
base=Path(__file__).resolve().parent;src=base.parent/'第十九步_扩充先验文字对照草案'/'64图候选描述_待确认.csv';prev=base.parent/'第十六步_先验生成环境核查'/'六图对照输入_准备清单.csv'
with src.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
with prev.open(encoding='utf-8-sig',newline='') as f:old={x['图像编号']:x['数量去除候选'] for x in csv.DictReader(f)}
# These decisions followed visual inspection of all 64 local images in 8 contact sheets.
revisions={
 '1923.tif':('The image shows a storage tank area with a building beside it .','图中有储罐区域，旁边有建筑。','图中可见相邻建筑，是否为住宅难以确认；将 house 改为 building。原句的整齐排列细节已在区域化时弱化。'),
 '1947.tif':('The image shows a storage tank site on the ground .','地面上有储罐场地。','去掉有歧义的 small 修饰；原句中小型及互相邻近的细节有所减少。'),
 '1974.tif':('The image shows a storage tank site on the ground .','地面上有储罐场地。','去掉有歧义的 small 修饰，小型这一细节有所减少。'),
 '1981.tif':('The image shows a storage tank site on the ground .','地面上有储罐场地。','去掉有歧义的 small 修饰，小型这一细节有所减少。'),
 '1952.tif':('Storage tanks are on the ground .','地面上有储罐。','去掉不够自然的 same 修饰，地面描述作保守处理；原句的相似性和草坪细节有所减少。'),
 '1955.tif':('Similar storage tanks are on sandy ground .','相似的储罐位于沙土地面上。','修正原候选 Same storage tanks 的英语表达，并保留相似性和沙土地面。'),
 '2061.tif':('Tennis courts arranged neatly with some plants surrounded .','网球场排列整齐，周围有植物。','修正原始标注沿用的拼写 plamts → plants；其余句式沿用原标注。'),
}
assert len(rows)==64 and len(revisions)==7 and not (set(revisions)&set(old))
for r in rows:
 name=r['图像编号'];r['草案候选']=r['候选描述']
 if name in revisions:
  en,zh,reason=revisions[name];r['候选描述']=en;r['候选中文释义']=zh;r['复核处理']='调整';r['复核说明']=reason
 elif name in old:
  assert r['候选描述']==old[name];r['复核处理']='沿用六图探索';r['复核说明']='保留前次实验的已用输入，便于比较。'
 else:
  r['复核处理']='沿用草案';r['复核说明']='已对照本地图像和原始描述检查；改写的数量与语义线索见核查提示。'
 r['复核人']='助手视觉复核';r['复核用途']='后续对照候选，按分组和数据来源单独统计'
 assert not re.search(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b',r['候选描述'],re.I),name
 if name in old:assert r['候选描述']==old[name]
 assert r['原始描述']!=r['候选描述']
with (base/'64图复核候选.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'复核图数':len(rows),'评审方式':'助手逐图视觉复核，用户此前确认了图中数量','延用六图探索':sum(r['复核处理']=='沿用六图探索' for r in rows),'调整候选':sum(r['复核处理']=='调整' for r in rows),'沿用新增草案':sum(r['复核处理']=='沿用草案' for r in rows),'数量对含1':sum(r['错误对照数量']=='1' for r in rows),'数据来源':dict(Counter(r['数据来源'] for r in rows)),'状态':'复核完成，尚未进行扩充生成与检索评分'}
(base/'复核统计.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
