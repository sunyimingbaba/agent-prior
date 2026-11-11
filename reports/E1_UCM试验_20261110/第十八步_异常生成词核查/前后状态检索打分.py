#!/usr/bin/env python3
"""Probe impact of choosing adjacent generation states for two images."""
import csv,hashlib,io,json,os,sys
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image
root=Path('/private/AGSE_E1_20261110');base=root/'legacy';old=root/'e1_wording100_smoke_20261110';parent=root/'e1_prior_compat_20261111';out=parent/'token_position_20261111'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
names={'101.tif','1937.tif'};cases=[x for x in read(old/'cases.csv') if x['图像编号'] in names];audit={x['图像编号']:x for x in read(old/'audit.csv') if x['图像编号'] in names}
assert len(cases)==4 and len(audit)==2
ref={(r['image'],r['wording']):r for r in json.loads((parent/'paired_six_20261111/scores.json').read_text()) if r['image'] in names and r['condition']=='original'};assert len(ref)==4
ck=base/'assets/ucm_main_domain_tool.ckpt';clip_path=base/'assets/models/ViT-L-14.pt'
assert sha(ck)=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc';assert sha(clip_path)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
table=pa.ipc.open_file(str(base/'assets/ucm/ucm_caption_karpathy_label_train.arrow')).read_all();data={}
for name in names:
 x=audit[name];i=int(x['Arrow行号']);assert x['来源']=='train' and table['path'][i].as_py()==name
 image=Image.open(io.BytesIO(table['image'][i].as_py())).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()==x['图像RGB_SHA256']
 state={n:torch.load(out/f'{Path(name).stem}_{n}.pt',map_location='cpu',weights_only=True).float().reshape(1,-1) for n in ('previous','actual')}
 assert all(tuple(v.shape)==(1,3584) and torch.isfinite(v).all() for v in state.values());data[name]=(image,state)
torch.manual_seed(0);torch.set_num_threads(min(8,os.cpu_count() or 1));sys.path.insert(0,str(base/'CUP'))
from prompt_clip import clip
load=clip.load
def cpu_load(name,device='cpu',jit=False,download_root=None):
 if name=='ViT-L/14':name=str(clip_path)
 return load(name,device='cpu',jit=jit,download_root=download_root)
clip.load=cpu_load
from vilt.modules.vilt_module import ViLTransformerSS
class MyBackboneFinetuning:
 def __init__(self,*args,**kwargs):pass
import __main__
__main__.MyBackboneFinetuning=MyBackboneFinetuning
ckpt=torch.load(ck,map_location='cpu',weights_only=False);cfg=dict(ckpt['hyper_parameters']['config']);cfg['load_path']='';cfg['test_only']=False;assert cfg['use_agent_prior']
model=ViLTransformerSS(cfg).cpu().eval();model.load_state_dict(ckpt['state_dict'],strict=True);del ckpt
rows=[]
with torch.inference_mode():
 for name in sorted(names):
  image,states=data[name];im=model.preprocess(image).unsqueeze(0)
  for condition,hidden in states.items():
   visual=model.img_embeds(im,agent_hidden=hidden)['image_feats']
   for case in [r for r in cases if r['图像编号']==name]:
    scores=(visual@model.txt_embeds(clip.tokenize([case['文本1'],case['文本2']],context_length=77-model.prompt_length))['text_feats'].T).flatten().tolist();pos=int(case['正确文本位置'])-1
    x={'image':name,'condition':condition,'wording':case['句式'],'correct_score':scores[pos],'wrong_score':scores[1-pos],'margin':scores[pos]-scores[1-pos],'correct':scores[pos]>scores[1-pos]}
    if condition=='actual':
     r=ref[name,case['句式']];diff=max(abs(x['correct_score']-r['correct_score']),abs(x['wrong_score']-r['wrong_score']));x['paired_original_max_diff']=diff;assert diff<1e-6
    rows.append(x)
assert len(rows)==8
(out/'position_scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('COMPLETE',json.dumps(rows,ensure_ascii=False),flush=True)
