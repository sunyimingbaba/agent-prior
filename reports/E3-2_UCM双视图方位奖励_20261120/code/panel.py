from PIL import Image,ImageDraw
from shared import ROOT,read,table,rgb
t=table()
rows=read(ROOT/'inputs/candidates.json')
for start in range(0,len(rows),8):
    panel=Image.new('RGB',(960,4*145),'white')
    draw=ImageDraw.Draw(panel)
    for j,r in enumerate(rows[start:start+8]):
        im=rgb(t,r);x=j//4*480;y=j%4*145
        for k,v in enumerate([im,im.transpose(Image.Transpose.ROTATE_180)]):
            v.thumbnail((128,128));panel.paste(v,(x+128*k,y+16))
        draw.text((x,y),r['image']+' '+r['pair']+'  | original / 180',fill='black')
        from textwrap import wrap
        draw.multiline_text((x+259,y+25),'\n'.join(wrap(r['A'][0],30)),fill='black')
    panel.save(ROOT/'review'/f'candidates_{start:02d}.jpg',quality=95)
