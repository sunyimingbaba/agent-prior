#!/usr/bin/env python3
"""Generate fresh paired priors for six audited UCM images, retaining the tool trace."""
import argparse,contextlib,csv,hashlib,io,json,os,random,sys,time
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path('/private/AGSE_E1_20261110');base=root/'legacy';assets=base/'assets';out=a.out.resolve()
if root not in out.parents or out.exists():raise ValueError('must use a new directory inside the private root')
inputs=rows(a.inputs);audit=rows(root/'e1_wording100_smoke_20261110/audit.csv');audit={r['图像编号']:r for r in audit}
expected={'101.tif','1918.tif','1937.tif','2014.tif','2058.tif','2086.tif'}
assert len(inputs)==6 and {r['图像编号'] for r in inputs}==expected
assert next(r for r in inputs if r['图像编号']=='2014.tif')['数量去除候选']=='Tennis courts arranged in rows with some plants surrounded .'
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':str(out/'huggingface'),'TMPDIR':str(out),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
out.mkdir(parents=True);(out/'priors').mkdir();(out/'traces').mkdir()
random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(min(8,os.cpu_count() or 1));torch.cuda.reset_peak_memory_stats()
sys.path.insert(0,str(base/'CUP'));sys.path.insert(0,str(base/'CUP/agent'))
from rollout import load_mllm,rollout_one
from tools import scene_classify_with_conf
from prompt_clip import clip
print('LOADING_QWEN',flush=True);start=time.monotonic()
model,processor=load_mllm(str(assets/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(assets/'ucm_main_lora'))
clipmodel,preprocess=clip.load(str(assets/'models/ViT-L-14.pt'),device='cuda',jit=False);clipmodel=clipmodel.float().cuda().eval()
head_ckpt=torch.load(assets/'ucm_scene_cls.pt',map_location='cpu',weights_only=True);head=torch.nn.Linear(head_ckpt['feat_dim'],len(head_ckpt['classes']));head.load_state_dict(head_ckpt['head'],strict=True);head=head.float().cuda().eval();classes=list(head_ckpt['classes'])
tables={s:pa.ipc.open_file(str(assets/f'ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ('train','val')}
records=[]
for item in inputs:
 name=item['图像编号'];ref=audit[name];split=ref['来源'];idx=int(ref['Arrow行号']);table=tables[split]
 assert table['path'][idx].as_py()==name and table['caption'][idx].as_py()[0]==ref['第一条描述']==item['原始先验输入']
 image=Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()==ref['图像RGB_SHA256']
 old=root/'ucm_prior_archive/agent_cache_domain_tool'/ref['先验文件'];assert sha(old)==ref['先验SHA256'];oldhidden=torch.load(old,map_location='cpu',weights_only=True).float().reshape(1,-1);assert tuple(oldhidden.shape)==(1,3584)
 for condition,key in [('original','原始先验输入'),('neutral','数量去除候选')]:
  events=[]
  def scene_tool(im):
   text,confidence=scene_classify_with_conf(im,model=clipmodel,preprocess=preprocess,cls_head=head,cls_names=classes);accepted=confidence>=0.0
   events.append({'text':text,'confidence':float(confidence),'accepted':accepted});return text if accepted else None
  print('GENERATING',name,condition,flush=True);trace=io.StringIO();t0=time.monotonic()
  with contextlib.redirect_stdout(trace):
   hidden,used,filtered=rollout_one(model,processor,image,item[key],{'scene_classify':scene_tool},max_rounds=3,max_new_tokens=64,device='cuda',verbose=True,injected_h_end=False,temperature=0.0)
  assert tuple(hidden.shape)==(3584,) and bool(torch.isfinite(hidden).all()) and used==len(events)
  stem=Path(name).stem+'_'+condition;prior=out/'priors'/f'{stem}.pt';torch.save(hidden,prior);(out/'traces'/f'{stem}.txt').write_text(trace.getvalue(),encoding='utf-8')
  cosine=float(torch.nn.functional.cosine_similarity(hidden.float().reshape(1,-1),oldhidden).item())
  rec={'image':name,'split':split,'row':idx,'condition':condition,'caption':item[key],'image_sha256':ref['图像RGB_SHA256'],'historical_prior_sha256':sha(old),'prior_file':str(prior.relative_to(out)),'prior_sha256':sha(prior),'historical_cosine':cosine,'tool_used':used,'tool_filtered':filtered,'tool_events':events,'trace_file':f'traces/{stem}.txt','contains_addCriterion':'addCriterion' in trace.getvalue(),'elapsed_seconds':round(time.monotonic()-t0,3)}
  records.append(rec);(out/'records.partial.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('DONE',name,condition,'tool',used,'cos',round(cosine,6),flush=True)
assert len(records)==12
meta={'status':'complete','input_sha256':sha(a.inputs),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'temperature':0.0,'max_rounds':3,'max_new_tokens':64,'threshold':0.0,'injected_h_end':False,'count':len(records),'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),'elapsed_seconds':round(time.monotonic()-start,2)}
(out/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('COMPLETE',json.dumps(meta),flush=True)
