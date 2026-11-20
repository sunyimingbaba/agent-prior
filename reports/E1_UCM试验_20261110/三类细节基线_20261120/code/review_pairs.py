from pathlib import Path
import csv,io
import pyarrow as pa
from PIL import Image, ImageDraw
import sys
r=Path(__file__).resolve().parents[1]
images={}
for split in ('train','val'):
 t=pa.ipc.open_file(str(Path(sys.argv[1])/f'ucm_caption_karpathy_label_{split}.arrow')).read_all()
 for i,n in enumerate(t['path'].to_pylist()):images[n]=(t,i)
pairs=list(csv.DictReader((r/'inputs/paired_visual_labels.tsv').open(),delimiter='\t'))
for start in range(0,len(pairs),4):
 sheet=Image.new('RGB',(1024,2*315),'white');draw=ImageDraw.Draw(sheet)
 for i,row in enumerate(pairs[start:start+4]):
  for k,label in enumerate(('a','b')):
   name=row['image_'+label]+'.tif';t,idx=images[name];x=i%2*512+k*256;y=i//2*315
   img=Image.open(io.BytesIO(t['image'][idx].as_py())).convert('RGB')
   sheet.paste(img,(x,y));draw.text((x+3,y+257),name+' '+row['split'],fill='black')
   import textwrap
   draw.multiline_text((x+3,y+271),'\n'.join(textwrap.wrap(row['text_'+label],40)),fill='black',spacing=0)
 sheet.save(r/'review'/f'pairs_{start//4+1:02d}.jpg',quality=92)
