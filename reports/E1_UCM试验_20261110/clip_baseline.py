import os, sys, json, time, io
from pathlib import Path
import torch, pyarrow as pa
from PIL import Image
BASE=Path('/public/cup-legacy-smoke-20261026')
OUT=Path('/public/agse-e1-pilot-20261110')
sys.path.insert(0,str(BASE/'CUP'))
from prompt_clip import clip
counts={190:(2,3,'airplanes'),2081:(3,4,'tennis courts'),2082:(4,3,'tennis courts'),2084:(2,3,'tennis courts'),2085:(2,3,'tennis courts'),2086:(4,3,'tennis courts'),2087:(2,3,'tennis courts'),2088:(4,3,'tennis courts'),2089:(4,3,'tennis courts'),2090:(2,3,'tennis courts')}
words={2:'two',3:'three',4:'four'}
torch.set_num_threads(min(8,os.cpu_count() or 1))
t=pa.ipc.open_file(str(BASE/'assets/ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
indices={str(t['path'][i].as_py()):i for i in range(t.num_rows)}
model,preprocess=clip.load(str(BASE/'assets/models/ViT-L-14.pt'),device='cpu',jit=False)
model.float().eval()
rows=[]
with torch.inference_mode():
 for n,(pos,neg,obj) in counts.items():
  idx=indices[f'{n}.tif']; im=Image.open(io.BytesIO(t['image'][idx].as_py())).convert('RGB')
  text=[f'There are {words[pos]} {obj} in the image.',f'There are {words[neg]} {obj} in the image.']
  image=preprocess(im).unsqueeze(0); tokens=clip.tokenize(text)
  vi=model.encode_image(image,torch.zeros((1,16,1024))); tx=model.encode_text(tokens)
  vi=vi/vi.norm(dim=-1,keepdim=True); tx=tx/tx.norm(dim=-1,keepdim=True)
  scores=(vi@tx.T).flatten().tolist(); row={'image':f'{n}.tif','arrow_row':idx,'positive':text[0],'negative':text[1],'positive_score':scores[0],'negative_score':scores[1],'margin':scores[0]-scores[1]}
  rows.append(row); print(json.dumps(row),flush=True)
result={'dataset':'UCM val','model':'CLIP ViT-L/14 original backbone','task':'image-text minimal pair, number','review':'assistant visual review; independent human review pending','n':len(rows),'pairacc':sum(r['margin']>0 for r in rows)/len(rows),'mean_margin':sum(r['margin'] for r in rows)/len(rows),'rows':rows}
(OUT/'clip_baseline_results.json').write_text(json.dumps(result,indent=2))
print('FINISHED',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
