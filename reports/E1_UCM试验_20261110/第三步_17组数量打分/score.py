import csv,hashlib,io,json,os,random,sys
from pathlib import Path
import torch,pyarrow as pa
from PIL import Image
R=Path('/private/AGSE_E1_20261110'); B=R/'legacy'; O=R/'e1_quantity_17_20261110'
sys.path.insert(0,str(B/'CUP'))
from prompt_clip import clip
real_load=clip.load
def cpu_load(name,device='cpu',jit=False,download_root=None):
 return real_load(str(B/'assets/models/ViT-L-14.pt') if name=='ViT-L/14' else name,device='cpu',jit=jit,download_root=download_root)
clip.load=cpu_load
from vilt.modules.vilt_module import ViLTransformerSS
class MyBackboneFinetuning:
 def __init__(self,*args,**kwargs):pass
import __main__
__main__.MyBackboneFinetuning=MyBackboneFinetuning
random.seed(0);torch.manual_seed(0);torch.set_num_threads(min(8,os.cpu_count() or 1))
samples=json.loads((O/'samples.json').read_text());assert len(samples)==17 and len({x['图像编号'] for x in samples})==17
T=pa.ipc.open_file(str(B/'assets/ucm/ucm_caption_karpathy_label_val.arrow')).read_all(); P=R/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val'
items=[]
for x in samples:
 idx=int(x['验证集行号']);assert T['path'][idx].as_py()==x['图像编号']
 im=Image.open(io.BytesIO(T['image'][idx].as_py())).convert('RGB')
 assert hashlib.sha256(im.tobytes()).hexdigest()==x['图像RGB_SHA256']
 assert T['caption'][idx].as_py()[0]==x['原始第一条描述']
 hidden=torch.load(P/f'{idx}.pt',map_location='cpu',weights_only=True).float().reshape(1,-1)
 assert hidden.shape==(1,3584) and torch.isfinite(hidden).all()
 items.append((x,im,hidden))
print('INPUTS_VERIFIED',len(items),flush=True)
results={}
for method in ('clip','agse'):
 if method=='clip':model,pre=real_load(str(B/'assets/models/ViT-L-14.pt'),device='cpu',jit=False);model.float().eval()
 else:
  ck=torch.load(B/'assets/ucm_main_domain_tool.ckpt',map_location='cpu',weights_only=False); cfg=dict(ck['hyper_parameters']['config']);cfg['load_path']='';cfg['test_only']=False;assert cfg.get('use_agent_prior') is True
  model=ViLTransformerSS(cfg).cpu().eval();print('AGSE_CHECKPOINT',model.load_state_dict(ck['state_dict'],strict=True),flush=True);del ck;pre=model.preprocess
 rows=[]
 with torch.inference_mode():
  for x,im,h in items:
   texts=[x['正确描述_AI构造'],x['错误描述_AI构造']]; image=pre(im).unsqueeze(0)
   if method=='clip':
    vi=model.encode_image(image,torch.zeros((1,16,1024)));tx=model.encode_text(clip.tokenize(texts));vi=vi/vi.norm(dim=-1,keepdim=True);tx=tx/tx.norm(dim=-1,keepdim=True)
   else:
    vi=model.img_embeds(image,agent_hidden=h)['image_feats'];tx=model.txt_embeds(clip.tokenize(texts,context_length=77-model.prompt_length))['text_feats']
   q=(vi@tx.T).flatten();assert torch.isfinite(q).all();a,b=q.tolist();row={'image':x['图像编号'],'arrow_row':int(x['验证集行号']),'scene':x['场景'],'object':x['对象'],'count':x['可见对象数量_AI初判'],'positive':texts[0],'negative':texts[1],'positive_score':a,'negative_score':b,'margin':a-b,'correct':a>b,'original_first_caption':x['原始第一条描述']};rows.append(row);print(method,json.dumps(row,ensure_ascii=False),flush=True)
 results[method]={'model':method,'n':len(rows),'correct':sum(x['correct'] for x in rows),'tie':sum(x['margin']==0 for x in rows),'pairacc':sum(x['correct'] for x in rows)/len(rows),'mean_margin':sum(x['margin'] for x in rows)/len(rows),'rows':rows};del model
 (O/f'{method}_results.json').write_text(json.dumps(results[method],ensure_ascii=False,indent=2))
 print('COMPLETE',method,results[method]['correct'],flush=True)
by={x['image']:x for x in results['agse']['rows']};cols=['image','arrow_row','scene','object','count','positive','negative','clip_positive','clip_negative','clip_margin','clip_correct','agse_positive','agse_negative','agse_margin','agse_correct','original_first_caption']
with (O/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for c in results['clip']['rows']:
  a=by[c['image']];w.writerow({k:c[k] for k in ['image','arrow_row','scene','object','count','positive','negative','original_first_caption']} | {f'clip_{k}':c[v] for k,v in [('positive','positive_score'),('negative','negative_score'),('margin','margin'),('correct','correct')]} | {f'agse_{k}':a[v] for k,v in [('positive','positive_score'),('negative','negative_score'),('margin','margin'),('correct','correct')]})
print('FINISHED',flush=True)
