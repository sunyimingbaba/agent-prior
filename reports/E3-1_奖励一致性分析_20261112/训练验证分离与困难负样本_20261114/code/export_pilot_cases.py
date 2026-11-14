"""Exploratory review: first candidate per image, with all pilot scores exported."""
import csv
import io
import json
from pathlib import Path
import textwrap

import pyarrow as pa
import torch
from PIL import Image,ImageDraw,ImageFont

r=Path(__file__).resolve().parent.parent;out=r/'negative_review';out.mkdir(exist_ok=False)
g=json.loads((r/'inputs/training_text_gallery.json').read_text());metadata=json.loads((r/'inputs/train_metadata.json').read_text())['rows']
xs=[json.loads(s) for s in (r/'pilot/records.jsonl').read_text().splitlines()]
t=torch.load(r/'pilot/reward_tensors.pt',map_location='cpu',weights_only=True);rows=[];selected=[]
for i,x in enumerate(xs):
    pos=g['maps'][str(x['row'])]['positive'];pi=max(pos,key=lambda c:float(t['similarities'][i,c]));info=metadata[x['row']]
    entry={'id':x['id'],'row':x['row'],'image':x['image'],'scene':info['scene'],'sample':x['sample'],
        'prompt':g['maps'][str(x['row'])]['prompt_caption'],'best_positive':g['text_order'][pi],
        'global_negative':g['text_order'][x['global_negative_index']],
        'scene_negative':g['text_order'][x['scene_negative_index']],
        'r_global':x['r_global'],'r_scene':x['r_scene'],'quality_usable':x['quality_usable']}
    rows.append(entry)
    if x['sample']==0:selected.append(entry)
with (out/'pilot_negative_candidates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
table=pa.ipc.open_file('/private/AGSE_E1_20261110/legacy/assets/ucm/ucm_caption_karpathy_label_train.arrow').read_all()
font=ImageFont.load_default(size=17);bold=ImageFont.load_default(size=19)
for page,start in enumerate(range(0,len(selected),4),1):
    canvas=Image.new('RGB',(1100,1200),'white');draw=ImageDraw.Draw(canvas)
    for k,x in enumerate(selected[start:start+4]):
        left=(k%2)*550;top=(k//2)*600
        draw.text((left+12,top+8),f"{x['image']} | {x['scene']}",font=bold,fill='black')
        image=Image.open(io.BytesIO(table['image'][x['row']].as_py())).convert('RGB');image.thumbnail((280,280))
        canvas.paste(image,(left+130,top+38))
        y=top+327
        for label,text in [('Positive',x['best_positive']),('Global negative',x['global_negative']),('Scene negative',x['scene_negative'])]:
            for line in textwrap.wrap(label+': '+text,width=56):
                draw.text((left+12,y),line,font=font,fill='black');y+=22
    canvas.save(out/f'first_candidate_review_{page}.png')
(out/'selection.json').write_text(json.dumps({'selection':'sample index 0 for every training image; exploratory semantic review, no effect on branch gate or training',
    'images':selected,'reviewer':'assistant visual inspection; provisional semantic compatibility, separate from original dataset labels'},ensure_ascii=False,indent=2)+'\n')
print('NEGATIVE_CANDIDATES_AND_FIVE_CONTACT_SHEETS_EXPORTED')
