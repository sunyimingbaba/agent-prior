#!/usr/bin/env python3
"""Render four selected examples from the checked source images."""
import argparse,csv,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser()
p.add_argument('--images-dir',type=Path,required=True)
p.add_argument('--analysis',type=Path,required=True)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--font',type=Path,default=Path('/System/Library/Fonts/Hiragino Sans GB.ttc'))
a=p.parse_args()
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return {r['图像编号']:r for r in csv.DictReader(f)}
rows=read(a.analysis);audit=read(a.audit)
font=ImageFont.truetype(str(a.font),26);title=ImageFont.truetype(str(a.font),29)
output=Image.new('RGB',(1600,1000),(245,247,250));d=ImageDraw.Draw(output)
examples=[('983.tif','原描述：some bunkers'),('1937.tif','原描述：four storage tanks'),('2014.tif','原描述：six tennis courts'),('184.tif','原描述：a white airplane')]
obj={'沙坑':'沙坑','高尔夫沙坑':'沙坑','储罐':'储罐','网球场':'网球场','飞机':'飞机'}
for i,(name,note) in enumerate(examples):
    r=rows[name];ox=(i%2)*800;oy=(i//2)*500
    d.rounded_rectangle((ox+8,oy+8,ox+792,oy+490),radius=16,fill='white',outline=(213,219,227),width=2)
    split='训练集' if r['数据来源']=='UCM train' else '验证集'
    d.text((ox+25,oy+20),f"{name}  ·  {obj[r['对象']]}  ·  {split}",font=title,fill=(28,40,55))
    im=Image.open(a.images_dir/name).convert('RGB')
    assert hashlib.sha256(im.tobytes()).hexdigest()==audit[name]['图像RGB_SHA256']
    output.paste(im.resize((350,350),Image.Resampling.NEAREST),(ox+25,oy+85))
    x=ox+400;y=oy+99
    d.text((x,y),f"实际：{r['真实数量']}　对照：{r['对照错误数量']}",font=font,fill=(28,40,55))
    for n,key in enumerate(('A','B')):
        pred=r[f'AGSE_{key}_preferred_count'];isright=r[f'AGSE_{key}_correct']=='True';color=(26,119,76) if isright else (185,46,43)
        d.text((x,y+64+n*64),f"AGSE {key} 选 {pred}  ·  {'正确' if isright else '错误'}",font=font,fill=color)
    d.text((x,y+205),note,font=font,fill=(72,83,97))
    d.text((x,y+252),'A、B 是两种句式',font=font,fill=(72,83,97))
output.save(a.output)
print('created',a.output)
