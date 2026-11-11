#!/usr/bin/env python3
"""Prepare auditable human-review drafts for the target-number-explicit subset."""
import base64,csv,hashlib,html,io,json,re
from collections import Counter
from pathlib import Path
import pyarrow as pa
from PIL import Image
root=Path('/Users/botlv/CUP-code');p=root/'reports/E1_UCM试验_20261110';out=p/'第十九步_扩充先验文字对照草案';imgs=out/'审阅图片';imgs.mkdir(exist_ok=True)
def read(q):
 with q.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
audit=read(p/'第十一步_模型与缓存核查/100图像与先验核查.csv');cases=read(p/'第十步_统一句式与对照方案/100组两种句式_对照清单.csv');by={x['图像编号']:x for x in cases if x['句式']=='A_原句式'};prev={x['图像编号']:x for x in read(p/'第十六步_先验生成环境核查/六图对照输入_准备清单.csv')}
# For 2-versus-1 cases, use an area or an invariant noun; reviews must check lost details.
manual={
'1981':('The image shows a small storage tank area on the ground .','图中地面上有小型储罐区域。','改用区域表述，弱化单复数线索。'),
'180':('The image shows aircraft stopped at the airport .','机场停有飞机。','aircraft 的单复数形式相同。'),
'1923':('The image shows a storage tank area with a house beside .','图中有储罐区域，旁边有一栋房屋。','改用区域表述；原句的整齐排列细节有弱化。'),
'143':('The image shows aircraft in the airport .','机场有飞机。','aircraft 的单复数形式相同。'),
'1936':('The image shows a storage tank area on the ground .','图中地面上有储罐区域。','原句中的不同大小有多对象线索，此版暂时省略。'),
'1946':('The image shows a storage tank area on the ground .','图中地面上有储罐区域。','原句的互相邻近关系可能暗示多个，此版暂时省略。'),
'2031':('The image shows a tennis court area with plants and lawn around it .','图中有网球场区域，周围有植物和草坪。','改用区域表述；原句的整齐排列细节有弱化。'),
'1826':('Housing is surrounded by lawn in the sparse residential area .','稀疏住宅区中的房屋被草坪环绕。','用 housing 表示房屋整体，具体数量留待判断。'),
'1841':('Housing is surrounded by withered plants and lawn in the sparse residential area .','稀疏住宅区中的房屋被枯萎植物和草坪环绕。','用 housing 表示房屋整体，具体数量留待判断。'),
'1947':('The image shows a small storage tank area on the ground .','图中地面上有小型储罐区域。','原句的互相邻近关系可能暗示多个，此版暂时省略。'),
'1974':('The image shows a small storage tank area on the ground .','图中地面上有小型储罐区域。','改用区域表述，弱化单复数线索。'),
'2009':('The image shows a tennis court area on red ground .','图中有红色地面上的网球场区域。','改用区域表述；原句的整齐排列细节有弱化。'),
'2029':('The image shows a tennis court area on green ground with buildings around it .','图中有绿色地面上的网球场区域，周围有建筑。','改用区域表述，弱化单复数线索。'),
'2032':('The image shows a tennis court area with buildings around it .','图中有网球场区域，周围有建筑。','改用区域表述；原句的整齐排列细节有弱化。'),
'2047':('The image shows a tennis court area with buildings and plants around it .','图中有网球场区域，周围有建筑和植物。','改用区域表述；原句的整齐排列细节有弱化。'),
'2055':('The image shows a tennis court area with a road beside it .','图中有网球场区域，旁边有道路。','改用区域表述；原句的整齐排列细节有弱化。'),
'149':('The image shows aircraft at the airport .','机场有飞机。','aircraft 的单复数形式相同。'),
'2046':('The image shows a tennis court area with buildings and plants around it .','图中有网球场区域，周围有建筑和植物。','改用区域表述；原句的整齐排列细节有弱化。'),
'190':('The image shows aircraft at the airport .','机场有飞机。','aircraft 的单复数形式相同。'),
}
nums={'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six'}
tables={s:pa.ipc.open_file(str(Path('/Users/botlv/cup-deploy-assets/arrow/ucm')/f'ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ('train','val')}

def zh_caption(text):
 s=text.lower();kind='飞机' if 'airplane' in s else '储罐' if 'storage tank' in s else '网球场' if 'tennis court' in s else '棒球场' if 'baseball diamond' in s else '房屋'
 parts=['图中有'+kind]
 if 'small' in s:parts.append('体型较小')
 if 'black' in s:parts.append('为黑色')
 if 'different' in s:parts.append('彼此有差异')
 if 'same' in s:parts.append('样式相同')
 if 'neatly' in s:parts.append('排列整齐')
 if 'in rows' in s:parts.append('成排排列')
 elif 'in two lines' in s:parts.append('排成两排')
 elif 'in a line' in s:parts.append('排列成一排')
 if 'airport' in s:parts.append('位于机场')
 if 'on the red ground' in s:parts.append('位于红色地面')
 elif 'on the green ground' in s:parts.append('位于绿色地面')
 elif 'on the ground' in s:parts.append('位于地面')
 elif 'in the lawn' in s or 'on the lawn' in s:parts.append('位于草坪')
 elif 'on the sands' in s:parts.append('位于沙地')
 if 'rectangular box' in s:parts.append('位于长方形框内')
 if 'house beside' in s:parts.append('旁边有房屋')
 if 'cars parked beside' in s:parts.append('旁边有车辆')
 if 'road beside' in s:parts.append('旁边有道路')
 if 'next to each other' in s:parts.append('彼此相邻')
 if 'next to some buildings' in s:parts.append('靠近一些建筑')
 if 'trees surrounded' in s:parts.append('周围有树')
 if 'withered plants' in s:parts.append('周围有枯萎植物')
 elif 'plants' in s or 'plamts' in s:parts.append('周围有植物')
 if 'buildings' in s and 'next to some buildings' not in s:parts.append('附近有建筑')
 if 'and lawn' in s or 'surrounded by lawn' in s:parts.append('附近有草坪')
 if 'sparse residential area' in s:parts.append('位于稀疏住宅区')
 return '，'.join(parts)+'。'

drafts=[]
for x in audit:
 if x['第一条描述出现对应英文数词']!='True':continue
 name=x['图像编号'];stem=Path(name).stem;case=by[name];original=x['第一条描述'];n=case['实际数量'];wrong=case['错误数量'];word=nums[n]
 assert case['原始第一条描述']==original and re.search(r'\b'+word+r'\b',original,re.I)
 if stem in manual:
  assert wrong=='1';neutral,cn,note=manual[stem];group='B_包含1的数量对_需重点确认'
 else:
  assert wrong!='1';neutral=re.sub(r'\b'+word+r'\s+', '',original,count=1,flags=re.I)
  neutral=re.sub(r'^There is (?=[a-z]+s\b)','There are ',neutral,flags=re.I)
  neutral=neutral[0].upper()+neutral[1:]
  note='保留原句其余措辞；复数与本组两个候选数量均相容。';group='A_两种数量均大于1'
  if name in prev:neutral=prev[name]['数量去除候选'];note+='沿用六图试验已使用的描述。'
  elif name=='2024.tif':neutral=neutral.replace('in two lines','in rows');note+='“两排”调整为“成排”，排列细节有所减少。'
  cn=zh_caption(neutral)
  if name in ('2014.tif','2024.tif'):cn=cn.replace('排成两排','成排排列');note+='排列描述由“两排”改为“成排”。' if name=='2014.tif' else ''
 assert neutral and neutral!=original
 assert not re.search(r'\b'+word+r'\b',neutral,re.I)
 if stem in manual:assert not re.search(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b',neutral,re.I)
 table=tables[x['来源']];idx=int(x['Arrow行号']);assert table['path'][idx].as_py()==name and table['caption'][idx].as_py()[0]==original
 im=Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==x['图像RGB_SHA256'];im.save(imgs/(stem+'.png'))
 status='已用于六图探索' if name in prev else '待用户确认'
 drafts.append({'图像编号':name,'数据来源':case['数据来源'],'目标':case['目标'],'图中数量':n,'错误对照数量':wrong,'分组':group,'原始描述':original,'候选描述':neutral,'候选中文释义':cn,'核查提示':note,'当前状态':status})
assert len(drafts)==64 and len(manual)==19 and sum(x['分组'].startswith('B_') for x in drafts)==19 and len(prev)==6
with (out/'64图候选描述_待确认.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(drafts[0]));w.writeheader();w.writerows(drafts)
for x in drafts:
 if x['图像编号'] in prev:assert x['候选描述']==prev[x['图像编号']]['数量去除候选']
counts={'原E1图像数':len(audit),'第一条描述含对应数字词':64,'余下图像待单独分类':36,'错误数量为1的需重点确认':19,'其余两数量均大于1':45,'先前六图':6,'新增待确认':58,'来源':dict(Counter(x['数据来源'] for x in drafts)),'人工状态':'除先前六图外，候选均等待用户确认'}
(out/'准备统计.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
styles='body{font:17px system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;background:#f4f6f9;color:#182337}h1,h2{color:#173a62}.card{background:#fff;border-radius:12px;padding:16px;margin:14px 0;display:flex;gap:22px;box-shadow:0 1px 5px #ccd3dc}.card img{width:230px;height:230px;object-fit:contain;image-rendering:auto}.info{flex:1}p{line-height:1.55;margin:.5em 0}.flag{color:#a64010;font-weight:700}.small{color:#516077}@media(max-width:700px){.card{display:block}.card img{max-width:100%;width:256px;height:auto}}'
page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E1先验输入候选审阅</title><style>'+styles+'</style><body><h1>E1：先验输入候选审阅</h1><p>请对照图片，检查候选描述的意思和图中内容。前 19 张的对照数量包含 1，采用飞机、区域或房屋整体等表达，重点关注被简化的细节。其余 45 张采用与本组两个数量都相容的复数表达。状态标为“待用户确认”的条目需要你的判断。</p><p>提示：中文为便于阅读的释义，正式输入以英文为准；图片只供审阅。审阅结果请在对话中告诉我。</p>']
for group,title in [('B_包含1的数量对_需重点确认','第一组：19 张需要重点确认'),('A_两种数量均大于1','第二组：45 张复数数量对（含已完成的六图）')]:
 page.append('<h2>'+title+'</h2>')
 for x in drafts:
  if x['分组']!=group:continue
  esc=html.escape;stem=Path(x['图像编号']).stem
  page.append(f'<div class="card"><img src="审阅图片/{stem}.png" alt="{esc(x["图像编号"])}"><div class="info"><strong>{esc(x["图像编号"])} · {esc(x["数据来源"])} · 图中{x["图中数量"]}，对照{x["错误对照数量"]}</strong><p>原始英文：{esc(x["原始描述"])}</p><p>候选英文：{esc(x["候选描述"])}</p><p>候选中文：{esc(x["候选中文释义"])}</p><p class="flag">核查提示：{esc(x["核查提示"])}</p><p class="small">状态：{esc(x["当前状态"])}</p></div></div>')
page.append('</body></html>');(out/'图文审阅页面.html').write_text('\n'.join(page),encoding='utf-8');print(json.dumps(counts,ensure_ascii=False,indent=2))
