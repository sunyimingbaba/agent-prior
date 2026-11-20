"""Lock two paraphrasings of reciprocal natural train contrasts before inference."""
from pathlib import Path
import csv,hashlib,io,json,sys
from PIL import Image,ImageDraw
import pyarrow as pa
root=Path(__file__).resolve().parents[1]
tab=pa.ipc.open_file(sys.argv[1]).read_all()
pool={r['image']:r for r in json.loads((root/'inputs/pool.json').read_bytes())}
excluded=json.loads((root/'inputs/excluded_e1.json').read_bytes())
papers=list(csv.DictReader((root/'inputs/pairs.tsv').open(),delimiter='\t'))
rows=[]
for i,p in enumerate(papers):
    if i%4==0:
        img=Image.new('RGB',(1024,570),'white');draw=ImageDraw.Draw(img)
    for j,(side,other) in enumerate([('left','right'),('right','left')]):
        n=p[side]+'.tif';r=pool[n]
        assert n not in excluded and tab['path'][r['row']].as_py()==n
        picture=Image.open(io.BytesIO(tab['image'][r['row']].as_py())).convert('RGB')
        assert hashlib.sha256(picture.tobytes()).hexdigest()==r['rgb_sha256']
        x=i%2*512+j*256;y=i%4//2*285
        img.paste(picture,(x,y));draw.text((x+2,y+258),n+' '+p['id'],fill='black')
        rows.append(dict(**r,pair=p['id'],side=side,scene_query='Scene: '+p['scene']+'.',
                         orig_query=tab['caption'][r['row']].as_py()[0],
                         A=[p['A_'+side],p['A_'+other]],B=[p['B_'+side],p['B_'+other]]))
    if i%4==3 or i==len(papers)-1:img.save(root/'review'/f'pairs_{i//4}.jpg',quality=93)
assert len(rows)==len({r['image'] for r in rows}) and all(r['split']=='train' for r in rows)
(root/'inputs/cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
