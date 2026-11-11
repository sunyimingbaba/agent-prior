#!/usr/bin/env python3
"""Score 58 new image pairs and merge six previously scored pairs."""
import csv,hashlib,io,json,os,sys
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image
root=Path('/private/AGSE_E1_20261110');base=root/'legacy';old=root/'e1_wording100_smoke_20261110';parent=root/'e1_prior_compat_20261111';out=parent/'paired_58_20261111';six=parent/'paired_six_20261111'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
inputs=read(parent/'paired64_reviewed.csv');assert len(inputs)==64 and len({x['图像编号'] for x in inputs})==64
by={x['图像编号']:x for x in inputs};six_names={x['图像编号'] for x in inputs if x['复核处理']=='沿用六图探索'};names=set(by)-six_names;assert len(six_names)==6 and len(names)==58
meta=json.loads((out/'metadata.json').read_text());six_meta=json.loads((six/'metadata.json').read_text());records=json.loads((out/'records.json').read_text());six_records=json.loads((six/'records.json').read_text());assert meta['status']=='complete' and len(records)==116 and len(six_records)==12
assert meta['input_sha256']==sha(parent/'paired64_reviewed.csv')
for key in ('device','torch','cuda','temperature','max_rounds','max_new_tokens','threshold','injected_h_end'):assert meta[key]==six_meta[key],key
assert {(r['image'],r['condition']) for r in records}=={(n,c) for n in names for c in ('original','neutral')}
assert {(r['image'],r['condition']) for r in six_records}=={(n,c) for n in six_names for c in ('original','neutral')}
for r in records+six_records:
 x=by[r['image']];assert r['caption']==x['原始描述' if r['condition']=='original' else '候选描述']
 directory=out if r['image'] in names else six;assert sha(directory/r['prior_file'])==r['prior_sha256']
cases=[r for r in read(old/'cases.csv') if r['图像编号'] in by];audit={r['图像编号']:r for r in read(old/'audit.csv') if r['图像编号'] in by};historic={(r['image'],r['wording']):r for r in read(old/'full_run/results.csv') if r['model']=='AGSE' and r['image'] in by}
assert len(cases)==128 and len(audit)==64 and len(historic)==128
ckpt_path=base/'assets/ucm_main_domain_tool.ckpt';clip_path=base/'assets/models/ViT-L-14.pt'
assert sha(ckpt_path)=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc';assert sha(clip_path)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
tables={s:pa.ipc.open_file(str(base/f'assets/ucm/ucm_caption_karpathy_label_{s}.arrow')).read_all() for s in ('train','val')};data={};recordmap={(r['image'],r['condition']):r for r in records}
for name in names:
 ref=audit[name];idx=int(ref['Arrow行号']);table=tables[ref['来源']];assert table['path'][idx].as_py()==name and table['caption'][idx].as_py()[0]==ref['第一条描述']==by[name]['原始描述']
 image=Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()==ref['图像RGB_SHA256']
 prior=root/'ucm_prior_archive/agent_cache_domain_tool'/ref['先验文件'];assert sha(prior)==ref['先验SHA256'];paths={'historic':prior}
 for condition in ('original','neutral'):
  rec=recordmap[name,condition];paths[condition]=out/rec['prior_file'];assert sha(paths[condition])==rec['prior_sha256']
 priors={label:torch.load(p,map_location='cpu',weights_only=True).float().reshape(1,-1) for label,p in paths.items()};assert all(tuple(x.shape)==(1,3584) and torch.isfinite(x).all() for x in priors.values());data[name]=(image,priors)
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
ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False);cfg=dict(ckpt['hyper_parameters']['config']);cfg['load_path']='';cfg['test_only']=False;assert cfg.get('use_agent_prior') is True
model=ViLTransformerSS(cfg).cpu().eval();print('CHECKPOINT',model.load_state_dict(ckpt['state_dict'],strict=True),flush=True);del ckpt
results=[];text_cache={}
with torch.inference_mode():
 for num,name in enumerate(sorted(names),1):
  image,priors=data[name];im=model.preprocess(image).unsqueeze(0)
  for label,hidden in priors.items():
   visual=model.img_embeds(im,agent_hidden=hidden)['image_feats']
   for case in [r for r in cases if r['图像编号']==name]:
    texts=(case['文本1'],case['文本2'])
    if texts not in text_cache:text_cache[texts]=model.txt_embeds(clip.tokenize(list(texts),context_length=77-model.prompt_length))['text_feats']
    scores=(visual@text_cache[texts].T).flatten().tolist();pos=int(case['正确文本位置'])-1
    rec={'image':name,'split':case['数据来源'],'condition':label,'wording':case['句式'],'correct_text':case['正确描述'],'wrong_text':case['错误描述'],'correct_score':scores[pos],'wrong_score':scores[1-pos],'margin':scores[pos]-scores[1-pos],'correct':scores[pos]>scores[1-pos],'tie':scores[pos]==scores[1-pos]}
    if label=='historic':
     ref=historic[name,case['句式']];diff=max(abs(scores[i]-float(ref[f'score{i+1}'])) for i in (0,1));rec['historical_score_max_abs_diff']=diff;assert diff<1e-6
    results.append(rec)
  print('SCORED_NEW_IMAGE',num,'/ 58',name,flush=True)
assert len(results)==348
six_scores=json.loads((six/'scores.json').read_text());assert len(six_scores)==36 and {x['image'] for x in six_scores}==six_names
for row in six_scores:
 if row['condition']=='historic':
  ref=historic[row['image'],row['wording']];assert abs(row['correct_score']-float(ref['correct_score']))<1e-6 and abs(row['wrong_score']-float(ref['wrong_score']))<1e-6
results.extend(six_scores)
for r in results:
 info=by[r['image']];r['cohort']='exploratory_6' if r['image'] in six_names else 'new_58';r['actual_count']=int(info['图中数量']);r['wrong_count']=int(info['错误对照数量']);r['group']='includes_one' if r['wrong_count']==1 else 'both_above_one'
assert len(results)==384 and len({(r['image'],r['condition'],r['wording']) for r in results})==384
(out/'scores64.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
fields=['image','split','cohort','actual_count','wrong_count','group','condition','wording','correct_text','wrong_text','correct_score','wrong_score','margin','correct','tie','historical_score_max_abs_diff']
with (out/'scores64.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
print('COMPLETE',len(results),'scores',len(text_cache),'unique text pairs',flush=True)
