#!/usr/bin/env python3
"""Generate one original-caption prior to validate an isolated GPU environment."""
import argparse,hashlib,io,json,os,random,sys,time
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/private/AGSE_E1_20261110'))
parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args();root=a.root.resolve();out=a.out.resolve()
if root not in out.parents:raise ValueError('output must be inside private experiment root')
out.mkdir(parents=True,exist_ok=False)
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':str(out/'huggingface'),
                   'TMPDIR':str(out),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
base=root/'legacy';assets=base/'assets';sys.path.insert(0,str(base/'CUP'));sys.path.insert(0,str(base/'CUP/agent'))
from rollout import load_mllm,rollout_one
from tools import scene_classify_with_conf
from prompt_clip import clip

random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
torch.set_num_threads(min(8,os.cpu_count() or 1));torch.cuda.reset_peak_memory_stats()
t=pa.ipc.open_file(str(assets/'ucm/ucm_caption_karpathy_label_val.arrow')).read_all();idx=192
assert t['path'][idx].as_py()=='1983.tif'
caption=t['caption'][idx].as_py()[0]
assert caption=='A storage tank with some houses and plants beside .'
image=Image.open(io.BytesIO(t['image'][idx].as_py())).convert('RGB')
img_sha=hashlib.sha256(image.tobytes()).hexdigest()
assert img_sha=='7ca60533f6615e71e0bc77df4d50f12823fe76a88f63cf7abbf818ca001cbf54'
historical=root/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val/192.pt'
assert hashlib.sha256(historical.read_bytes()).hexdigest()=='e3daf7e838fdea114fb3304e3355e8a6754a182a82d7f5c6c29b255e9df3383b'
started=time.monotonic();print('LOADING_QWEN',flush=True)
model,processor=load_mllm(str(assets/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(assets/'ucm_main_lora'))
print('LOADING_TOOL_CLIP',flush=True)
clip_model,preprocess=clip.load(str(assets/'models/ViT-L-14.pt'),device='cuda',jit=False)
clip_model=clip_model.float().eval().cuda()
head_ckpt=torch.load(assets/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
head=torch.nn.Linear(head_ckpt['feat_dim'],len(head_ckpt['classes']))
head.load_state_dict(head_ckpt['head'],strict=True);head=head.float().cuda().eval();classes=list(head_ckpt['classes'])
probe_text,probe_conf=scene_classify_with_conf(image,model=clip_model,preprocess=preprocess,cls_head=head,cls_names=classes)
assert isinstance(probe_text,str) and 0<=probe_conf<=1
print('TOOL_PROBE',probe_text,round(probe_conf,6),flush=True)
tool_events=[]
def scene_tool(im):
    text,conf=scene_classify_with_conf(im,model=clip_model,preprocess=preprocess,cls_head=head,cls_names=classes)
    accepted=conf>=0.0
    tool_events.append({'text':text,'confidence':float(conf),'accepted':accepted})
    return text if accepted else None
print('GENERATING_ONE_PRIOR',flush=True)
hidden,used,filtered=rollout_one(model,processor,image,caption,{'scene_classify':scene_tool},
                                max_rounds=3,max_new_tokens=64,device='cuda',verbose=True,
                                injected_h_end=False,temperature=0.0)
assert tuple(hidden.shape)==(3584,) and bool(torch.isfinite(hidden).all())
prior=torch.load(historical,map_location='cpu',weights_only=True)
assert tuple(prior.shape)==(3584,) and bool(torch.isfinite(prior).all())
new=hidden.float();old=prior.float();cos=float(torch.nn.functional.cosine_similarity(new[None],old[None]).item())
newfile=out/'smoke_original_prior.pt';torch.save(hidden,newfile)
result={'purpose':'environment smoke; original caption only; not six-image intervention',
        'image':'1983.tif','split':'val','arrow_row':idx,'caption':caption,'image_rgb_sha256':img_sha,
        'model':'Qwen2.5-VL-7B-Instruct','lora':'ucm_main_lora','device':torch.cuda.get_device_name(0),
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'temperature':0.0,'max_new_tokens':64,'max_rounds':3,'threshold':0.0,
        'manual_tool_probe':{'text':probe_text,'confidence':float(probe_conf)},
        'actual_tool_events':tool_events,'tool_used':used,'tool_filtered':filtered,
        'hidden_shape':list(hidden.shape),'hidden_dtype':str(hidden.dtype),'finite':True,
        'historical_prior_sha256':hashlib.sha256(historical.read_bytes()).hexdigest(),
        'new_prior_sha256':hashlib.sha256(newfile.read_bytes()).hexdigest(),
        'cosine_similarity_to_historical':cos,'max_absolute_difference':float((new-old).abs().max().item()),
        'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
        'elapsed_seconds':round(time.monotonic()-started,2)}
(out/'smoke_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('SMOKE_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
