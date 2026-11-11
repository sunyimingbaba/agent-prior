#!/usr/bin/env python3
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
base=Path(__file__).resolve().parent;prev=base.parent/'第二十一步_64图先验输入对照';images=base.parent/'第十九步_扩充先验文字对照草案/审阅图片';names=['147.tif','1841.tif','1974.tif','257.tif']
with (prev/'64图已复核输入.csv').open(encoding='utf-8-sig',newline='') as f:inputs={x['图像编号']:x for x in csv.DictReader(f)}
with (prev/'逐图配对结果.csv').open(encoding='utf-8-sig',newline='') as f:cases=[x for x in csv.DictReader(f) if x['图像'] in names]
gen={(x['image'],x['condition']):x for x in json.loads((prev/'58图生成记录.json').read_text()) if x['image'] in names};scores=[x for x in json.loads((prev/'64图评分.json').read_text()) if x['image'] in names]
assert len(cases)==8 and len(gen)==8 and len(scores)==24
out=[]
for name in names:
 x=inputs[name];a=gen[name,'original'];b=gen[name,'neutral'];out.append({'image':name,'split':x['数据来源'],'object':x['目标'],'true_count':int(x['图中数量']),'wrong_count':int(x['错误对照数量']),'original_caption':x['原始描述'],'neutral_caption':x['候选描述'],'original_tool':a['tool_events'],'neutral_tool':b['tool_events'],'original_trace':(prev/'58图生成轨迹'/f'{name[:-4]}_original.txt').read_text(),'neutral_trace':(prev/'58图生成轨迹'/f'{name[:-4]}_neutral.txt').read_text(),'historical_similarity_original':a['historical_cosine'],'scores':[s for s in scores if s['image']==name]})
(base/'四图证据.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (base/'四图配对分差.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
font=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',25);small=ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc',19)
im=Image.new('RGB',(720,792),'#f4f7fa');draw=ImageDraw.Draw(im)
for i,name in enumerate(names):
 x=(i%2)*360;y=(i//2)*396;r=inputs[name];pic=Image.open(images/f'{name[:-4]}.png').convert('RGB').resize((320,320));im.paste(pic,(x+20,y+52))
 draw.text((x+20,y+12),f"{name[:-4]}：{r['目标']}（{r['图中数量']} 对 {r['错误对照数量']}）",font=font,fill='#14283f')
 draw.text((x+20,y+370),'图像来自 UCM；数量已人工确认',font=small,fill='#4c5d6c')
im.save(base/'四张图总览.png');print('已保存四图证据、评分与本地图像总览')
