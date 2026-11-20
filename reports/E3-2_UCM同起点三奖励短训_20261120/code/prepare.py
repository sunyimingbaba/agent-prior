"""Lock the reviewed train contrasts, all validation photos and the subset never previously scored for fine detail."""
from pathlib import Path
import json,random,hashlib,io,sys
import pyarrow as pa
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1];inputs=root/'inputs'
previous=root.parent/'E3-1_奖励一致性分析_20261112'
last=previous/'跨图关系奖励与留出评估_20261120'
def read(p):return json.loads(p.read_bytes())
def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
rows=read(previous/'空间关系候选奖励核验_20261120/inputs/selected.json')
train=[r for r in rows if r['pair'] not in ['harbor_channel','parking_rows','housing_street','river_banks']]
assert len(train)==24
random.Random(50811).shuffle(train)
dump(inputs/'train.json',train)
seen=set(read(last/'inputs/excluded.json'))
for p in [last/'inputs/selected.json',last/'object_inputs/cases.json']:seen.update(r['image'] for r in read(p))
t=pa.ipc.open_file(str(Path(sys.argv[1])/'ucm_caption_karpathy_label_val.arrow')).read_all()
val=[]
for i,name in enumerate(t['path'].to_pylist()):
    im=Image.open(io.BytesIO(t['image'][i].as_py())).convert('RGB')
    val.append(dict(row=i,image=name,orig_query=t['caption'][i].as_py()[0],rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),fresh=name not in seen,block=(int(Path(name).stem)-1)//100))
dump(inputs/'validation.json',val)
fresh=[r for r in val if r['fresh']]
for j in range(0,len(fresh),24):
    page=Image.new('RGB',(6*180,4*199),'white');d=ImageDraw.Draw(page)
    for k,r in enumerate(fresh[j:j+24]):
        x=k%6*180;y=k//6*199
        im=Image.open(io.BytesIO(t['image'][r['row']].as_py())).convert('RGB')
        page.paste(im.resize((178,178)),(x,y));d.text((x+2,y+178),r['image'],fill='black')
    page.save(root/'review'/f'new_{j//24}.jpg',quality=92)
print(len(train),len(val),len(fresh))
