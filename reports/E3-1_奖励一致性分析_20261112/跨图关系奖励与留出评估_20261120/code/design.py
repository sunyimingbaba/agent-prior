"""Build matched reciprocal pairs from the independently screened UCM images."""
from pathlib import Path
import hashlib,json,sys,io
from PIL import Image,ImageDraw
import pyarrow as pa
root=Path(__file__).resolve().parents[1]
# First image carries the first side of the named contrast. Coordinates are in the displayed image.
pairs=[
(141,167,'airplane','nose','the largest airplane'),(218,238,'baseball field','lr','the dirt infield'),
(252,228,'baseball field','tb','the dirt infield'),(317,308,'beach','lr','the sandy beach'),
(313,306,'beach','hv','the shoreline'),(815,845,'freeway','hv','the freeway'),
(827,819,'freeway','diag','the freeway'),(1039,1002,'harbor','hv','the long docks lined with boats'),
(1074,1030,'harbor','diag','the rows of moored boats'),(1152,1128,'intersection','grid','the crossing streets'),
(1135,1103,'intersection','grid','the crossing streets'),(1366,1314,'mobile home park','hv','the street between the homes'),
(1423,1419,'overpass','hv','the elevated road'),(1456,1440,'overpass','diag','the elevated road'),
(1513,1502,'parking lot','hv','the rows of parked cars'),(1552,1528,'parking lot','diag','the rows of parked cars'),
(1677,1621,'river','diag','the river'),(1763,1731,'runway','hv','the runway'),
(1762,1773,'runway','diag','the runway'),(1905,1904,'storage tank area','tb','the pair of circular tanks'),
(1916,1921,'storage tank area','lr','the large circular tank'),(2054,2066,'tennis court area','tb','the tennis courts'),
(388,385,'beach','lr','the sandy beach'),(616,690,'residential area','hv','the central corridor between the buildings'),
(882,890,'freeway','hv','the freeway'),(1081,1090,'harbor','tilt','the rows of moored boats'),
(1181,1186,'intersection','grid','the crossing streets'),(1382,1386,'mobile home park','hv','the central street between the homes'),
(1485,1482,'overpass','hv','the elevated road'),(1590,1585,'parking lot','hv','the rows of parked cars'),
(1684,1687,'river','hv','the river')]
# Four separately worded contrast banks. A and B select candidates. C and D are evaluation only.
templates={
'lr':[
('X is on the left side.','X is on the right side.'),('The left half of this view contains X.','The right half of this view contains X.'),
('X lies farther west than the middle of the picture.','X lies farther east than the middle of the picture.'),
('The image center is to the right of X.','The image center is to the left of X.')],
'tb':[
('X is near the top.','X is near the bottom.'),('The upper half of this view contains X.','The lower half of this view contains X.'),
('X lies above the middle of the picture.','X lies below the middle of the picture.'),
('The image center is below X.','The image center is above X.')],
'hv':[
('X runs mostly horizontally.','X runs mostly vertically.'),('X extends from left to right.','X extends from top to bottom.'),
('The main direction of X is across the width of the frame.','The main direction of X is along the height of the frame.'),
('The side edges of the picture are linked by the direction of X.','The upper and lower edges of the picture are linked by the direction of X.')],
'diag':[
('X slopes down toward the right.','X slopes up toward the right.'),('X runs from upper left to lower right.','X runs from lower left to upper right.'),
('Following X from the left edge takes one lower in the picture.','Following X from the left edge takes one higher in the picture.'),
('The right end of X stands below its left end.','The right end of X stands above its left end.')],
'grid':[
('X follow the sides of the picture.','X run diagonally across the picture.'),('X form an upright cross.','X form a tilted cross.'),
('X are aligned with the horizontal and vertical frame axes.','X are at an oblique angle to the horizontal and vertical frame axes.'),
('The picture borders are nearly parallel to one or the other of X.','The picture borders cut obliquely across both of X.')],
'tilt':[
('X are slanted.','X are upright.'),('X run diagonally.','X run from top to bottom.'),
('X make a visible angle with the vertical edges.','X follow the vertical edges.'),
('The side borders are oblique to X.','The side borders are parallel to X.')],
'nose':[
('The nose of X points up.','The nose of X points down.'),('X is headed towards the top edge.','X is headed towards the bottom edge.'),
('X faces the upper border of the picture.','X faces the lower border of the picture.'),
('The tail is below the cockpit of X.','The tail is above the cockpit of X.')]}
# Swap the order for the two infields to match sides in the picture.
pairs[1]=(238,218,'baseball field','lr','the dirt infield')
def sentence(template,obj):
    s=template.replace('X',obj)
    if obj in ('the long docks lined with boats','the rows of moored boats','the rows of parked cars','the tennis courts'):
        for a,b in [(' runs ',' run '),(' extends ',' extend '),(' slopes ',' slope '),(' lies ',' lie '),(' is near',' are near')]:s=s.replace(a,b)
    return s[0].upper()+s[1:]
pool={r['image']:r for r in json.loads((root/'inputs/pool.json').read_bytes())}
tables={s:pa.ipc.open_file(str(Path(sys.argv[1])/f'ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ['train','val']}
cases=[]; panels=[]
for j,(n,m,scene,typ,obj) in enumerate(pairs):
    entries=[]
    for side,idx in enumerate([n,m]):
        name=f'{idx}.tif';r=dict(pool[name]);t=tables[r['split']]
        assert r['split']==('train' if j<22 else 'val')
        r.update(pair=f"{r['split']}_{j+1:02}",side=side,scene=scene,contrast=typ,subject=obj,orig_query=t['caption'][r['row']].as_py()[0],
                 scene_query=f'An aerial view of a {scene}.')
        for key,wordpair in zip('ABCD',templates[typ]):
            r[key]=[sentence(wordpair[ix],obj) for ix in (side,1-side)]
        entries.append(Image.open(io.BytesIO(t['image'][r['row']].as_py())).convert('RGB'));cases.append(r)
    panels.append((j,entries))
assert len(set(r['image'] for r in cases))==len(cases) and len(set(r['rgb_sha256'] for r in cases))==len(cases)
for start in range(0,len(panels),6):
    sheet=Image.new('RGB',(1024,((min(6,len(panels)-start)+1)//2)*284),'white');d=ImageDraw.Draw(sheet)
    for jj,ims in panels[start:start+6]:
        i=jj-start;x=i%2*512;y=i//2*284;r=cases[jj*2];s=cases[jj*2+1]
        for k,im in enumerate(ims):sheet.paste(im.resize((255,255)),(x+256*k,y))
        d.text((x,y+256),f"{r['image']} | {s['image']}       {r['subject']}  {r['contrast']}",fill='black')
    sheet.save(root/'review'/f'pairs_{start//6}.jpg',quality=94)
(root/'inputs/selected.json').write_text(json.dumps(cases,indent=2))
