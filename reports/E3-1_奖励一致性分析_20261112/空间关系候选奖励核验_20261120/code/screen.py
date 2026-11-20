"""Draw score-blind natural UCM train tiles, leaving previous E1 scored IDs out."""
from pathlib import Path
import hashlib,io,json,random,sys
import pyarrow as pa
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
previous=Path(sys.argv[2])
excluded=set()
for p in (next(previous.glob('三类*/inputs/scored_images.json')),next(previous.glob('UCM验证*/inputs/cases.json'))):
    excluded.update(r['image'] for r in json.loads(p.read_bytes()))
src=Path(sys.argv[1])
tab=pa.ipc.open_file(str(src)).read_all()
paths=tab['path'].to_pylist()
records=[]
blocks=[2,3,9,10,11,12,14,15,16,18,19,20]
for i in range(0,len(blocks),2):
    sheet=Image.new('RGB',(8*150,4*169),'white')
    draw=ImageDraw.Draw(sheet)
    for j,b in enumerate(blocks[i:i+2]):
        pool=[k for k,n in enumerate(paths) if (int(Path(n).stem)-1)//100==b and n not in excluded]
        for z,k in enumerate(sorted(random.Random(11201+b).sample(pool,16))):
            n=paths[k];im=Image.open(io.BytesIO(tab['image'][k].as_py())).convert('RGB')
            x=z%8*150;y=(z//8+2*j)*169
            sheet.paste(im.resize((148,148)),(x,y));draw.text((x+2,y+149),n,fill='black')
            records.append(dict(image=n,row=k,split='train',block=b,rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    sheet.save(root/'review'/f'screen_{i//2}.jpg',quality=92)
(root/'inputs/pool.json').write_text(json.dumps(records,indent=2))
(root/'inputs/excluded_e1.json').write_text(json.dumps(sorted(excluded)))
