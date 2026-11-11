#!/usr/bin/env python3
"""Make a local preview of the six reviewed UCM images."""
import argparse,csv,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser()
for k in ['candidates','audit','images-dir','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--font',type=Path,default=Path('/System/Library/Fonts/Hiragino Sans GB.ttc'))
a=p.parse_args()
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(a.candidates);audit={r['图像编号']:r for r in read(a.audit)}
assert len(rows)==6
font=ImageFont.truetype(str(a.font),23);large=ImageFont.truetype(str(a.font),26)
out=Image.new('RGB',(1110,820),(245,247,250));draw=ImageDraw.Draw(out)
for i,r in enumerate(rows):
    x=(i%3)*370;y=(i//3)*410;name=r['图像编号'];im=Image.open(a.images_dir/name).convert('RGB')
    assert hashlib.sha256(im.tobytes()).hexdigest()==audit[name]['图像RGB_SHA256']
    draw.rounded_rectangle((x+5,y+5,x+365,y+405),radius=12,fill='white',outline=(210,218,227),width=2)
    draw.text((x+17,y+14),f"{name} · {r['对象']}",font=large,fill=(22,39,55))
    out.paste(im.resize((312,312),Image.Resampling.NEAREST),(x+28,y+55))
    draw.text((x+20,y+370),f"实际 {r['用户确认数量']}  /  对照 {r['检索时错误数量']}",font=font,fill=(55,70,85))
out.save(a.output)
print('created',a.output)
