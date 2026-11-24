from pathlib import Path
import sys,hashlib
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parents[2]/'E3-2_UCM有效图组定向短训_20261124';sys.path.insert(0,str(BASE/'code'))
from panel_assets import samples,picture,read,blocked
SCENES=[0,2,3,4,6,8,10,11,12,13,14,15,16,17,18,20]
def main(dest):
    dest.mkdir(exist_ok=True);t,ix=samples();used=blocked()|{r['image'] for r in read(BASE/'inputs/panel.json')}
    for scene in SCENES:
        names=sorted([n for n in ix if (int(n[:-4])-1)//100==scene and n not in used],key=lambda n:hashlib.sha256(('pair-confirm-'+n).encode()).hexdigest())[:54]
        canvas=Image.new('RGB',(9*100,((len(names)+8)//9)*117),'white');d=ImageDraw.Draw(canvas)
        for i,n in enumerate(names):
            x=i%9*100;y=i//9*117;canvas.paste(picture(t,ix[n]).resize((98,98)),(x,y));d.text((x+2,y+98),n,fill='black')
        canvas.save(dest/f'{scene}.jpg',quality=91)
if __name__=='__main__':main(Path(sys.argv[1]))
