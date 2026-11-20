"""Separate two-object intervention on real validation photos not scored in the frame-axis study."""
import json,sys,io,hashlib
from pathlib import Path
import pyarrow as pa
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
spec=[
(384,'beach','the sea','the sandy beach','below'),
(490,'industrial area','the row of parked cars','the largest warehouse roof','below'),
(989,'golf course','the curved path','the sand bunkers','left'),
(1187,'intersection','the white car','the center of the road junction','left'),
(1287,'residential area','the rectangular turquoise pool','the large dark-roofed house directly beside it','above'),
(1487,'overpass','the dark shadow of the overpass','the elevated road','above'),
(1882,'residential area','the blue-green water feature','the large light-roofed house','above'),
(1890,'residential area','the blue swimming pool','the isolated house','above'),
(1988,'storage tank area','the dark shadow','the circular tank','right'),
(2087,'tennis court area','the tennis courts','the straight road beside the courts','right'),
(2090,'tennis court area','the tennis courts','the group of trees beside the courts','left'),
(1182,'intersection','the tan-roofed building','the crossing road beside its long side','below')]
# Each word bank is reciprocal. Original and rotated views use identical sentences with swapped signs.
opposite=dict(above='below',below='above',left='right',right='left')
def sentence(a,d,b):
    loc={'above':'above','below':'below','left':'to the left of','right':'to the right of'}[d]
    loc2={'above':'higher','below':'lower','left':'farther left','right':'farther right'}[d]
    verb='are' if a in ['the tennis courts','the sand bunkers'] else 'is'
    v2='are' if b in ['the tennis courts','the sand bunkers'] else 'is'
    inv={'above':'below','below':'above','left':'to the right of','right':'to the left of'}[d]
    res=[f'{a} {verb} {loc} {b}.',f'Relative to {b}, {a} {verb} positioned {loc2} in the view.',
         f'Comparing positions in the picture, {a} lies {loc2} than {b}.',f'{b} {v2} {inv} {a}.']
    return [x[0].upper()+x[1:] for x in res]
t=pa.ipc.open_file(str(Path(sys.argv[1])/'ucm_caption_karpathy_label_val.arrow')).read_all()
excluded=set(json.loads((root/'inputs/excluded.json').read_bytes()))|{r['image'] for r in json.loads((root/'inputs/selected.json').read_bytes())}
cases=[];panel=Image.new('RGB',(4*242,3*262),'white');d=ImageDraw.Draw(panel)
for j,(num,scene,a,b,direction) in enumerate(spec):
    name=f'{num}.tif';assert name not in excluded
    i=t['path'].to_pylist().index(name)
    im=Image.open(io.BytesIO(t['image'][i].as_py())).convert('RGB')
    pos=sentence(a,direction,b);neg=sentence(a,opposite[direction],b)
    r=dict(split='val',image=name,row=i,scene=scene,pair=f'object_{j+1:02}',subject=a,reference=b,side=0,relation=direction,contrast='object',
           scene_query=f'An aerial view of a {scene}.',orig_query=t['caption'][i].as_py()[0],rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest())
    r.update({w:[x.replace('courts lies','courts lie'),y.replace('courts lies','courts lie')] for w,x,y in zip('ABCD',pos,neg)})
    cases.append(r);x=j%4*242;y=j//4*262;panel.paste(im.resize((240,240)),(x,y));d.text((x,y+240),name,fill='black')
(root/'object_inputs/cases.json').write_text(json.dumps(cases,indent=2));panel.save(root/'object_inputs/selected.jpg',quality=95)
