import csv,hashlib,io,json,sys
from pathlib import Path
import pyarrow as pa
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parent;OLD=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
ARROW=Path('/Users/botlv/cup-deploy-assets/arrow/ucm/ucm_caption_karpathy_label_train.arrow')
def read(p):return json.loads(p.read_text())
def blocked():
    used={r['image'] for k in ('inputs','confirm/inputs') for r in read(OLD/k/'panel.json')}
    used.update(r['image'] for r in read(REP/'E3-2_UCM同起点三奖励短训_20261120/inputs/train.json'))
    used.update(r['image'] for r in read(REP/'E3-2_UCM双视图方位奖励_20261120/inputs/e1/cases.json') if r['split']=='train')
    with (REP/'E3_UCM候选奖励与跨图复验_20261123/开发逐图.csv').open(encoding='utf-8-sig') as f:
        used.update(r['image'] for r in csv.DictReader(f))
    return used
def samples():
    t=pa.ipc.open_file(pa.memory_map(str(ARROW))).read_all();names=t['path'].to_pylist()
    return t,{s:i for i,s in enumerate(names)}
def picture(t,j):return Image.open(io.BytesIO(t['image'][j].as_py())).convert('RGB')
def sheets(out):
    out.mkdir(exist_ok=True);t,ix=samples();ban=blocked()
    for scene in (0,2,3,4,6,12,13,15,16,18,20):
        names=sorted([n for n in ix if (int(n[:-4])-1)//100==scene and n not in ban],key=lambda n:hashlib.sha256(('active-panel-'+n).encode()).hexdigest())[:48]
        canvas=Image.new('RGB',(8*110,6*127),'white');d=ImageDraw.Draw(canvas)
        for i,n in enumerate(names):
            x=i%8*110;y=i//8*127;canvas.paste(picture(t,ix[n]).resize((108,108)),(x,y));d.text((x+2,y+109),n,fill='black')
        canvas.save(out/f'{scene}.jpg',quality=90)
if __name__=='__main__':sheets(Path(sys.argv[1]))
