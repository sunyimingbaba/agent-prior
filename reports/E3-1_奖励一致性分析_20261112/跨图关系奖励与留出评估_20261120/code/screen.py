"""Render unseen train and validation tiles before writing relation labels."""
import random,io,sys,json,hashlib,csv
from pathlib import Path
from PIL import Image,ImageDraw
import pyarrow as pa
root=Path(__file__).resolve().parents[1]
prev=root.parent/'空间关系候选奖励核验_20261120'
excluded=set(json.loads((prev/'inputs/excluded_e1.json').read_bytes()))
excluded.update(r['image'] for r in json.loads((prev/'inputs/pool.json').read_bytes()))
with (prev/'画面重叠候选.csv').open(encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):excluded.update((r['image_a'],r['image_b']))
(root/'inputs/excluded.json').write_text(json.dumps(sorted(excluded)))
cases=[]
for split in ('train','val'):
    table=pa.ipc.open_file(str(Path(sys.argv[1])/f'ucm_caption_karpathy_label_{split}.arrow')).read_all()
    names=table['path'].to_pylist()
    blocks=[1,2,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for i in range(0,len(blocks),3):
        pool=[]
        for block in blocks[i:i+3]:
            avail=[k for k,n in enumerate(names) if (int(Path(n).stem)-1)//100==block and n not in excluded]
            chosen=sorted(random.Random(20268119+block).sample(avail,min(16 if split=='train' else 10,len(avail))))
            pool.extend(chosen)
        w=8;sheet=Image.new('RGB',(w*155,((len(pool)+w-1)//w)*173),'white');draw=ImageDraw.Draw(sheet)
        for j,k in enumerate(pool):
            name=names[k];im=Image.open(io.BytesIO(table['image'][k].as_py())).convert('RGB')
            cases.append(dict(split=split,row=k,image=name,block=(int(Path(name).stem)-1)//100,rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
            x=j%w*155;y=j//w*173;sheet.paste(im.resize((153,153)),(x,y));draw.text((x+2,y+154),name,fill='black')
        sheet.save(root/'review'/f'{split}_{i//3}.jpg',quality=92)
(root/'inputs/pool.json').write_text(json.dumps(cases,indent=2))
