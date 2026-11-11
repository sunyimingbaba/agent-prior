#!/usr/bin/env python3
"""Inspect an environment without generating priors or changing existing packages."""
import argparse, importlib.metadata, json, os, platform, sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=('legacy','modern','compat'),required=True)
parser.add_argument('--assets',type=Path,required=True)
parser.add_argument('--scratch',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
for path in (a.scratch.resolve(),a.output.resolve()):
    if Path('/private') not in path.parents:
        raise ValueError('scratch and output must be inside /private')
a.scratch.mkdir(parents=True,exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':str(a.scratch/'hf'),
                   'TMPDIR':str(a.scratch),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
pkgs=['torch','transformers','peft','accelerate','pyarrow','torchvision','pillow','numpy','tqdm','ftfy','regex','safetensors','tokenizers']
versions={}
for name in pkgs:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
r={'mode':a.mode,'python':platform.python_version(),'python_executable':sys.executable,'python_prefix':sys.prefix,
   'packages':versions,'asset_paths':{},'processor_check':'not attempted', 'gpu':{}}
for name,rel in {'base_model':'models/Qwen2.5-VL-7B-Instruct','lora':'ucm_main_lora',
                 'scene_head':'ucm_scene_cls.pt','clip':'models/ViT-L-14.pt'}.items():
    path=(a.assets/rel).resolve();r['asset_paths'][name]={'path':str(path),'exists':path.exists(),'under_private':Path('/private') in path.parents}
if a.mode in ('legacy','compat'):
    try:
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
        from safetensors import safe_open
        import torch
        proc=AutoProcessor.from_pretrained(str(a.assets/'models/Qwen2.5-VL-7B-Instruct'),local_files_only=True)
        with safe_open(a.assets/'ucm_main_lora/adapter_model.safetensors',framework='pt',device='cpu') as f:
            count=len(f.keys())
        head=torch.load(a.assets/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
        r['processor_check']={'ok':True,'processor':type(proc).__name__,'qwen_class_imported':Qwen2_5_VLForConditionalGeneration.__name__,
                              'peft_class_imported':PeftModel.__name__,'lora_tensor_headers':count,
                              'scene_head_classes':len(head['classes']),'scene_head_feature_dim':head['feat_dim']}
    except Exception as e:r['processor_check']={'ok':False,'error':f'{type(e).__name__}: {e}'}
try:
    import torch
    gpu={'cuda_version':torch.version.cuda,'available':torch.cuda.is_available(),'arch_list':torch.cuda.get_arch_list()}
    if gpu['available']:
        gpu.update({'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))})
        try:
            x=torch.ones((4,),device='cuda');gpu['tiny_compute_result']=float((x*x).sum().cpu());gpu['tiny_compute_ok']=True
        except Exception as e:gpu.update({'tiny_compute_ok':False,'tiny_compute_error':f'{type(e).__name__}: {str(e).splitlines()[0]}'})
    r['gpu']=gpu
except Exception as e:r['gpu']={'error':f'{type(e).__name__}: {e}'}
required=['torch','transformers','peft','accelerate','pyarrow','torchvision','pillow','numpy','tqdm','ftfy','regex','safetensors','tokenizers']
r['missing_required_packages']=[n for n in required if versions[n] is None]
r['full_generation_tested']=False
r['priors_generated']=False
r['score_evaluated']=False
a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r,ensure_ascii=False),flush=True)
