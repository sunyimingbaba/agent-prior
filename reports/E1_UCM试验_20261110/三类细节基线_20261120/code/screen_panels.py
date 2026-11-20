"""Display a score-blind candidate pool by scene and split."""
from pathlib import Path
import hashlib, io, json, random, sys
from PIL import Image, ImageDraw
import pyarrow as pa
root=Path(__file__).resolve().parents[1]
rows=[]
for block in [2,3,4,9,10,14,15,19,20]:
    sheet=Image.new('RGB',(5*188,4*208),'white');draw=ImageDraw.Draw(sheet)
    for j,split in enumerate(('train','val')):
        table=pa.ipc.open_file(str(Path(sys.argv[1])/f'ucm_caption_karpathy_label_{split}.arrow')).read_all()
        pool=[i for i,n in enumerate(table['path'].to_pylist()) if (int(Path(n).stem)-1)//100==block]
        chosen=sorted(random.Random(20261120+block+100*j).sample(pool,min(10,len(pool))))
        for k,i in enumerate(chosen):
            name=table['path'][i].as_py()
            img=Image.open(io.BytesIO(table['image'][i].as_py())).convert('RGB')
            rows.append(dict(image=name,split=split,row=i,rgb_sha256=hashlib.sha256(img.tobytes()).hexdigest()))
            x=k%5*188;y=(k//5+2*j)*208
            sheet.paste(img.resize((186,186)),(x,y));draw.text((x+2,y+187),f'{name} {split}',fill='black')
    sheet.save(root/'review'/f'pool_{block}.jpg',quality=94)
(root/'inputs/screen_pool.json').write_text(json.dumps(rows,indent=2))
