"""Pool of UCM training photographs reserved for new visual labels before scoring."""
from pathlib import Path
import hashlib
import sys
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parents[1];REP=OUT.parent
sys.path.insert(0,str(REP/'E3-2_UCM有效图组定向短训_20261124/code'))
from panel_assets import samples,blocked,picture,read

def candidates():
    table,ix=samples();used=blocked()
    for p in REP.glob('E3*/**/inputs/panel.json'):
        rows=read(p)
        if isinstance(rows,list):used.update(r['image'] for r in rows if isinstance(r,dict) and 'image' in r)
    for p in REP.glob('E3*/**/inputs/lock.json'):
        lock=read(p)
        if isinstance(lock,dict):
            for r in lock.get('removed',[]):
                if isinstance(r,dict):used.update(r.get('images',[]))
    return table,ix,sorted(set(ix)-used,key=lambda n:(int(n[:-4])-1)//100)
def main(folder):
    folder.mkdir(exist_ok=True);t,ix,ids=candidates()
    for scene in range(21):
        ns=sorted([n for n in ids if (int(n[:-4])-1)//100==scene],key=lambda n:int(n[:-4]))
        if not ns:continue
        canvas=Image.new('RGB',(8*111,((len(ns)+7)//8)*127),'white');draw=ImageDraw.Draw(canvas)
        for j,n in enumerate(ns):
            x=j%8*111;y=j//8*127;canvas.paste(picture(t,ix[n]).resize((108,108)),(x,y));draw.text((x+2,y+109),n,fill='black')
        canvas.save(folder/(str(scene)+'.jpg'),quality=89)
    print({i:sum((int(n[:-4])-1)//100==i for n in ids) for i in range(21)})
if __name__=='__main__':main(Path(sys.argv[1]))
