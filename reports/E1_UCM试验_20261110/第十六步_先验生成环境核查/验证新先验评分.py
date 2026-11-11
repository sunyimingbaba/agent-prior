#!/usr/bin/env python3
"""Check one original-caption prior against the frozen historical AGSE scorer."""
import csv,hashlib,io,json,sys
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image
root=Path('/private/AGSE_E1_20261110');base=root/'legacy';out=root/'e1_prior_compat_20261111/smoke_1937_original'
inputs=root/'e1_wording100_smoke_20261110';ckpt_path=base/'assets/ucm_main_domain_tool.ckpt';clip_path=base/'assets/models/ViT-L-14.pt'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert digest(ckpt_path)=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
assert digest(clip_path)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
cases=[x for x in read(inputs/'cases.csv') if x['图像编号']=='1937.tif'];assert len(cases)==2
historic=[x for x in read(inputs/'full_run/results.csv') if x['model']=='AGSE' and x['image']=='1937.tif'];assert len(historic)==2
historic={x['wording']:x for x in historic}
t=pa.ipc.open_file(str(base/'assets/ucm/ucm_caption_karpathy_label_train.arrow')).read_all();assert t['path'][1556].as_py()=='1937.tif'
image=Image.open(io.BytesIO(t['image'][1556].as_py())).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()=='d34589a2207d5463d7ebf83cb973be5b0d5e8e85dabe68b67876591d2af50d8a'
old=root/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_train/1556.pt';new=out/'smoke_original_prior.pt'
assert digest(old)=='a1ddbc360cdd708dd5104a27973f74797f66bfbd2a5eff0190197616ef2ce02c'
prior={k:torch.load(p,map_location='cpu',weights_only=True).float().reshape(1,-1) for k,p in [('historic',old),('new',new)]}
assert all(tuple(p.shape)==(1,3584) and torch.isfinite(p).all() for p in prior.values())
torch.manual_seed(0);torch.set_num_threads(min(8,__import__('os').cpu_count() or 1));sys.path.insert(0,str(base/'CUP'))
from prompt_clip import clip
real_load=clip.load
def cpu_load(name,device='cpu',jit=False,download_root=None):
 if name=='ViT-L/14':name=str(clip_path)
 return real_load(name,device='cpu',jit=jit,download_root=download_root)
clip.load=cpu_load
from vilt.modules.vilt_module import ViLTransformerSS
class MyBackboneFinetuning:
 def __init__(self,*args,**kwargs):pass
import __main__
__main__.MyBackboneFinetuning=MyBackboneFinetuning
ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False);config=dict(ckpt['hyper_parameters']['config']);config['load_path']='';config['test_only']=False
assert config.get('use_agent_prior') is True
model=ViLTransformerSS(config).cpu().eval();model.load_state_dict(ckpt['state_dict'],strict=True);del ckpt
results=[]
with torch.inference_mode():
 im=model.preprocess(image).unsqueeze(0)
 for label,hidden in prior.items():
  visual=model.img_embeds(im,agent_hidden=hidden)['image_feats']
  for case in cases:
   text=model.txt_embeds(clip.tokenize([case['文本1'],case['文本2']],context_length=77-model.prompt_length))['text_feats'];scores=(visual@text.T).flatten().tolist();pos=int(case['正确文本位置'])-1
   entry={'prior':label,'wording':case['句式'],'score1':scores[0],'score2':scores[1],'margin':scores[pos]-scores[1-pos],'correct':scores[pos]>scores[1-pos]}
   if label=='historic':
    ref=historic[case['句式']];entry['max_difference_from_full_run']=max(abs(scores[i]-float(ref[f'score{i+1}'])) for i in range(2));assert entry['max_difference_from_full_run']<1e-6
   results.append(entry)
(out/'smoke_scoring.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('SCORING_RESULT',json.dumps(results,ensure_ascii=False),flush=True)
