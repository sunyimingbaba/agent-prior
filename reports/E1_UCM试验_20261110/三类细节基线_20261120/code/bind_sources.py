from common import ROOT, BASE, ASSETS, PLAIN_CKPT, read, dump, digest
import importlib.metadata
import subprocess

paths=[ASSETS/'ucm_main_domain_tool.ckpt',ROOT/'models/ucm_ccb_clustering_prior.ckpt',PLAIN_CKPT,
       ASSETS/'models/ViT-L-14.pt',ASSETS/'ucm_scene_cls.pt',BASE/'CUP/agent/rollout.py',
       BASE/'CUP/agent/tools.py',BASE/'CUP/vilt/modules/vilt_module.py',BASE/'CUP/prompt_clip/model.py']
paths += [ASSETS/f'ucm/ucm_caption_karpathy_label_{s}.arrow' for s in ('train','val')]
paths += [p for p in (ASSETS/'ucm_main_lora').glob('*') if p.is_file() and p.suffix in ('.safetensors','.json')]
paths += [p for p in (ASSETS/'models/Qwen2.5-VL-7B-Instruct').glob('*') if p.is_file() and (p.suffix in ('.safetensors','.json','.txt') and not p.name.endswith('.original.json'))]
archive=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool'
for item in read(ROOT/'inputs/scored_images.json'):
    paths.append(archive/('ucm_caption_karpathy_label_'+item['split'])/(str(item['row'])+'.pt'))
dump(ROOT/'asset_bindings.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),mount=subprocess.check_output(['findmnt','-n','-T','/private','-o','SOURCE,TARGET,FSTYPE'],text=True).strip(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','pyarrow']},files={str(p):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(paths)}))
