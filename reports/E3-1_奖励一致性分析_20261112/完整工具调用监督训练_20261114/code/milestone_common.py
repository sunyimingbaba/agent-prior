import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
assert str(ROOT).startswith('/private/AGSE_E3_1_rebuild_20261112/full_tool_supervision_20261114')
PLAN = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
BASE = Path(PLAN['base_root'])
ASSETS = BASE/'assets'
SOURCE = Path(PLAN['source_run_root'])
for name in ['scratch', 'scratch/cache', 'scratch/hf', 'scratch/matplotlib']:
    (ROOT/name).mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(ROOT/'scratch/hf'),
    TMPDIR=str(ROOT/'scratch'), XDG_CACHE_HOME=str(ROOT/'scratch/cache'),
    MPLCONFIGDIR=str(ROOT/'scratch/matplotlib'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
sys.path.insert(0,str(BASE/'CUP'))
sys.path.insert(0,str(BASE/'CUP/agent'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


def setup():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch


def seed(value):
    import torch
    import numpy as np
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def case_images(cases):
    import pyarrow as pa
    from PIL import Image
    tables={split:pa.ipc.open_file(str(ASSETS/f'ucm/ucm_caption_karpathy_label_{split}.arrow')).read_all() for split in {x['split'] for x in cases}}
    result={}
    for item in cases:
        table=tables[item['split']];row=item['row'];assert table['path'][row].as_py()==item['image']
        image=Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')
        assert hashlib.sha256(image.tobytes()).hexdigest()==item['image_rgb_sha256']
        captions=table['caption'][row].as_py();assert captions[0]==item['original_first_caption']
        result[item['image']]={'image':image,'name':item['image'],'caption':PLAN['prompt'],'captions':captions}
    return result


def load_policy(lora_path=None,trainable=False):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'),
             torch_dtype=torch.bfloat16,device_map='cuda')
    model=PeftModel.from_pretrained(model,str(lora_path or ASSETS/'ucm_main_lora'),is_trainable=trainable)
    model.eval()
    for name,p in model.named_parameters():
        p.requires_grad_(trainable and 'lora_' in name)
        if 'lora_' in name:p.data=p.data.float()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    return model,AutoProcessor.from_pretrained(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'))


def private_clip():
    from prompt_clip import clip
    original=clip.load
    if getattr(original,'_milestone_private',False):return clip
    def load(name,device='cpu',jit=False,download_root=None):
        return original(str(ASSETS/'models/ViT-L-14.pt') if name=='ViT-L/14' else name,
                        device='cpu',jit=jit,download_root=download_root)
    load._milestone_private=True
    clip.load=load
    return clip


def real_tools():
    import torch
    from tools import scene_classify_with_conf
    clip=private_clip()
    model,preprocess=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    model=model.float().eval().requires_grad_(False).cuda()
    ckpt=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(ckpt['feat_dim'],len(ckpt['classes']))
    head.load_state_dict(ckpt['head'],strict=True);head=head.float().eval().requires_grad_(False).cuda()
    @torch.no_grad()
    def scene(image):
        text,confidence=scene_classify_with_conf(image,model=model,preprocess=preprocess,
                                               cls_head=head,cls_names=list(ckpt['classes']))
        assert float(confidence)>=PLAN['rollout']['tool_threshold']
        return {'text':text,'confidence':float(confidence),'accepted':True}
    return {'scene_classify':scene}


def lora_hash(model):
    import torch
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if 'lora_' in name:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
