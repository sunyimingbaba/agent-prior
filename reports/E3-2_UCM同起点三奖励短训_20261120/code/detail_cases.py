from shared import ROOT,read,dump
from pathlib import Path
import io,sys
import pyarrow as pa
from PIL import Image,ImageDraw
# Pixel assessment from separate UCM validation images. Arrows are not added to the inputs.
trial=[
(183,'airport apron','the large light hangar roof','the two fully visible airplanes','below'),
(383,'beach','the sandy beach','the sea','above'),
(389,'beach','the sea','the sandy beach','right'),
(484,'industrial area','the gray roof in the top row','the white roof beside it in the same row','right'),
(489,'industrial area','the pale cream roof section','the adjoining band of rust-colored roof panels','below'),
(681,'residential area','the red-roofed houses','the street beside the housing rows','right'),
(687,'residential area','the striped parking bays along the uppermost building','the roof of that building','below'),
(883,'freeway','the two red cars','the dark divider between the carriageways','above'),
(889,'freeway','the long truck','the two passenger cars','below'),
(981,'golf course','the largest group of trees','the striped green','right'),
(986,'golf course','the larger sand bunker','the smaller sand bunker','above'),
(1184,'road intersection','the tan-roofed building','the center of the intersection','left'),
(1188,'road intersection','the white truck','the center of the intersection','above'),
(1281,'residential area','the small red car','the house next to its driveway','above'),
(1289,'residential area','the small blue pool','the straight street that runs through the neighborhood','right'),
(1390,'mobile home park','the pale blue-roofed rectangular home inside the road bend','the curved street around it','right'),
(1784,'runway','the curving taxiway entrance','the dark rubber marks along the runway center','above')]
def loc(d):return {'above':'above','below':'below','left':'to the left of','right':'to the right of'}[d]
opp=dict(above='below',below='above',right='left',left='right')
def verb(s):return 'are' if s in ('the two fully visible airplanes','the red-roofed houses','the striped parking bays along the uppermost building','the two red cars','the two passenger cars','the dark rubber marks along the runway center') else 'is'
val={r['image']:r for r in read(ROOT/'inputs/validation.json')}
rows=[];panels=[]
t=pa.ipc.open_file(str(Path(sys.argv[1])/'ucm_caption_karpathy_label_val.arrow')).read_all()
for n,scene,a,b,d in trial:
    r=dict(val[f'{n}.tif']);assert r['fresh'];r.update(scene_query=f'An aerial view of a {scene}.',subject=a,reference=b,direction=d)
    r['C']=[f'{a.capitalize()} {verb(a)} {loc(q)} {b}.' for q in (d,opp[d])]
    r['D']=[f'{b.capitalize()} {verb(b)} {loc(q)} {a}.' for q in (opp[d],d)]
    rows.append(r);panels.append(Image.open(io.BytesIO(t['image'][r['row']].as_py())).convert('RGB'))
for j in range(0,len(rows),6):
    page=Image.new('RGB',(768,((len(rows[j:j+6])+2)//3)*278),'white');dr=ImageDraw.Draw(page)
    for k,r in enumerate(rows[j:j+6]):
        x=k%3*256;y=k//3*278;page.paste(panels[j+k].resize((256,256)),(x,y));dr.text((x+2,y+256),r['image'],fill='black')
    page.save(ROOT/'review'/f'detail_{j//6}.jpg',quality=96)
dump(ROOT/'inputs/detail_proposals.json',rows)
