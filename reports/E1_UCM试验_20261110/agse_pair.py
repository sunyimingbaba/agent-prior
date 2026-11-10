import os, sys, json, io, random
from pathlib import Path
import torch, pyarrow as pa
from PIL import Image
BASE=Path('/public/cup-legacy-smoke-20261026')
OUT=Path('/public/agse-e1-pilot-20261110')
sys.path.insert(0,str(BASE/'CUP'))
from prompt_clip import clip
real_load=clip.load
def cpu_load(name, device='cpu', jit=False, download_root=None):
 return real_load(str(BASE/'assets/models/ViT-L-14.pt') if name == 'ViT-L/14' else name, device='cpu', jit=jit, download_root=download_root)
clip.load=cpu_load
from vilt.modules.vilt_module import ViLTransformerSS
class MyBackboneFinetuning:
 def __init__(self,*args,**kwargs):pass
import __main__
__main__.MyBackboneFinetuning=MyBackboneFinetuning
random.seed(0); torch.manual_seed(0); torch.set_num_threads(min(8,os.cpu_count() or 1))
ckpt=torch.load(BASE/'assets/ucm_main_domain_tool.ckpt',map_location='cpu',weights_only=False)
config=dict(ckpt['hyper_parameters']['config']); config['load_path']=''; config['test_only']=False
print('use_agent_prior',config.get('use_agent_prior'),flush=True)
assert config.get('use_agent_prior') is True
model=ViLTransformerSS(config).cpu().eval()
print('load',model.load_state_dict(ckpt['state_dict'],strict=True),flush=True)
del ckpt
_,prep=real_load(str(BASE/'assets/models/ViT-L-14.pt'),device='cpu',jit=False)
T=pa.ipc.open_file(str(BASE/'assets/ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
indices={str(T['path'][i].as_py()):i for i in range(T.num_rows)}
counts={190:(2,3,'airplanes'),2081:(3,4,'tennis courts'),2082:(4,3,'tennis courts'),2084:(2,3,'tennis courts'),2085:(2,3,'tennis courts'),2086:(4,3,'tennis courts'),2087:(2,3,'tennis courts'),2088:(4,3,'tennis courts'),2089:(4,3,'tennis courts'),2090:(2,3,'tennis courts')}
words={2:'two',3:'three',4:'four'};rows=[]
with torch.inference_mode():
 for n,(pos,neg,obj) in counts.items():
  idx=indices[f'{n}.tif']; im=Image.open(io.BytesIO(T['image'][idx].as_py())).convert('RGB')
  text=[f'There are {words[pos]} {obj} in the image.',f'There are {words[neg]} {obj} in the image.']
  hidden=torch.load(OUT/f'prior/ucm_caption_karpathy_label_val/{idx}.pt',map_location='cpu',weights_only=True).float().reshape(1,-1)
  assert hidden.shape==(1,3584) and torch.isfinite(hidden).all()
  vi=model.img_embeds(prep(im).unsqueeze(0),agent_hidden=hidden)['image_feats']
  tx=model.txt_embeds(clip.tokenize(text, context_length=77-model.prompt_length))['text_feats']
  scores=(vi@tx.T).flatten().tolist()
  row={'image':f'{n}.tif','arrow_row':idx,'positive':text[0],'negative':text[1],'positive_score':scores[0],'negative_score':scores[1],'margin':scores[0]-scores[1]};rows.append(row);print(json.dumps(row),flush=True)
result={'dataset':'UCM val','model':'archived UCM main AGSE domain tool checkpoint','task':'image-text minimal pair, number','review':'assistant visual review; independent human review pending','n':len(rows),'pairacc':sum(r['margin']>0 for r in rows)/len(rows),'mean_margin':sum(r['margin'] for r in rows)/len(rows),'rows':rows}
(OUT/'agse_pair_results.json').write_text(json.dumps(result,indent=2))
print('FINISHED',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
