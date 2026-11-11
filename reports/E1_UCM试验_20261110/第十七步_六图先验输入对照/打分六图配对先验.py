#!/usr/bin/env python3
"""Score historical and paired fresh priors with frozen AGSE; verify historical scores."""
import csv,hashlib,io,json,os,sys
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image
root=Path('/private/AGSE_E1_20261110');base=root/'legacy';inputs=root/'e1_wording100_smoke_20261110';out=root/'e1_prior_compat_20261111/paired_six_20261111'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
meta=json.loads((out/'metadata.json').read_text());records=json.loads((out/'records.json').read_text());assert meta['status']=='complete' and len(records)==12
names={r['image'] for r in records};assert len(names)==6 and {(r['image'],r['condition']) for r in records}=={(name,c) for name in names for c in ('original','neutral')}
cases=[r for r in read(inputs/'cases.csv') if r['图像编号'] in names];audit={r['图像编号']:r for r in read(inputs/'audit.csv') if r['图像编号'] in names};historic={(r['image'],r['wording']):r for r in read(inputs/'full_run/results.csv') if r['model']=='AGSE' and r['image'] in names}
assert len(cases)==12 and len(audit)==6 and len(historic)==12
ckpt_path=base/'assets/ucm_main_domain_tool.ckpt';clip_path=base/'assets/models/ViT-L-14.pt'
assert sha(ckpt_path)=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
assert sha(clip_path)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
tables={s:pa.ipc.open_file(str(base/f'assets/ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ('train','val')};data={};recordmap={(r['image'],r['condition']):r for r in records}
for name in names:
 ref=audit[name];idx=int(ref['Arrow行号']);table=tables[ref['来源']];assert table['path'][idx].as_py()==name and table['caption'][idx].as_py()[0]==ref['第一条描述']
 image=Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()==ref['图像RGB_SHA256']
 old=root/'ucm_prior_archive/agent_cache_domain_tool'/ref['先验文件'];assert sha(old)==ref['先验SHA256'];paths={'historic':old}
 for cond in ('original','neutral'):
  rec=recordmap[name,cond];p=out/rec['prior_file'];assert sha(p)==rec['prior_sha256'];paths[cond]=p
 priors={label:torch.load(p,map_location='cpu',weights_only=True).float().reshape(1,-1) for label,p in paths.items()}
 assert all(tuple(x.shape)==(1,3584) and torch.isfinite(x).all() for x in priors.values());data[name]=(image,priors)
torch.manual_seed(0);torch.set_num_threads(min(8,os.cpu_count() or 1));sys.path.insert(0,str(base/'CUP'))
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
ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False);cfg=dict(ckpt['hyper_parameters']['config']);cfg['load_path']='';cfg['test_only']=False;assert cfg['use_agent_prior'] is True
model=ViLTransformerSS(cfg).cpu().eval();print('CHECKPOINT',model.load_state_dict(ckpt['state_dict'],strict=True),flush=True);del ckpt
results=[]
with torch.inference_mode():
 for name in sorted(names):
  image,priors=data[name];im=model.preprocess(image).unsqueeze(0)
  for label,hidden in priors.items():
   visual=model.img_embeds(im,agent_hidden=hidden)['image_feats']
   for case in [r for r in cases if r['图像编号']==name]:
    texts=[case['文本1'],case['文本2']];text=model.txt_embeds(clip.tokenize(texts,context_length=77-model.prompt_length))['text_feats'];scores=(visual@text.T).flatten().tolist();pos=int(case['正确文本位置'])-1
    rec={'image':name,'split':case['数据来源'],'condition':label,'wording':case['句式'],'correct_text':case['正确描述'],'wrong_text':case['错误描述'],'correct_score':scores[pos],'wrong_score':scores[1-pos],'margin':scores[pos]-scores[1-pos],'correct':scores[pos]>scores[1-pos],'tie':scores[pos]==scores[1-pos]}
    if label=='historic':
     ref=historic[name,case['句式']];diff=max(abs(scores[i]-float(ref[f'score{i+1}'])) for i in (0,1));rec['historical_score_max_abs_diff']=diff;assert diff<1e-6
    results.append(rec);print('SCORED',name,label,case['句式'],flush=True)
assert len(results)==36
(out/'scores.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['image','split','condition','wording','correct_text','wrong_text','correct_score','wrong_score','margin','correct','tie','historical_score_max_abs_diff'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
print('COMPLETE',len(results),'scored pairs',flush=True)
