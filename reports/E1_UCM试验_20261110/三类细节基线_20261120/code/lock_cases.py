"""Lock image-verified text reversals and the existing quantity labels before scoring."""
from pathlib import Path
import csv, json, hashlib
r=Path(__file__).resolve().parents[1];d=r/'inputs'
def read(n):return json.loads((d/n).read_bytes())
def write(n,x):(d/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
original=read('images.json')
images={v['image']:v for v in original+read('screen_pool.json')}
cases=read('quantity.json')
with (d/'paired_visual_labels.tsv').open() as f:
    for tpl in csv.DictReader(f,delimiter='\t'):
        for side,other in [('a','b'),('b','a')]:
            img=tpl['image_'+side]+'.tif'
            assert images[img]['split']==tpl['split']
            cases.append(dict(dimension=tpl['dimension'],split=tpl['split'],image=img,wording=tpl['template'],
                              positive=tpl['text_'+side],negative=tpl['text_'+other],side=side,
                              matched_image=tpl['image_'+other]+'.tif',source='assistant_pixel_review'))
assert len({(x['image'],x['dimension'],x['wording']) for x in cases})==len(cases)
used=sorted({x['image'] for x in cases})
write('cases.json',cases);write('scored_images.json',[images[k] for k in used])
seen={v['image'] for v in original};write('extra_images.json',[images[k] for k in used if k not in seen])
print(len(used),len(cases),len(set(used)-seen))
