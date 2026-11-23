import platform
from pathlib import Path
import torch, peft, transformers
from shared import ROOT, BASE, REBUILD, ASSETS, SCORER, sha, dump

if __name__=='__main__':
    model=ASSETS/'models/Qwen2.5-VL-7B-Instruct'
    files=list(sorted(model.glob('*.safetensors')))+list(sorted(model.glob('*.json')))
    files += [ASSETS/'models/ViT-L-14.pt', ASSETS/'ucm_scene_cls.pt', ASSETS/'ucm_main_domain_tool.ckpt', SCORER,
              REBUILD/'relation_candidates_20261120/gallery/train.pt', REBUILD/'relation_candidates_20261120/native/features.pt',
              REBUILD/'relation_candidates_20261120/native/index.json',BASE/'CUP/agent/rollout.py',BASE/'CUP/agent/tools.py',
              BASE/'CUP/agent/grpo/scorer.py',BASE/'CUP/agent/grpo/reward.py']
    for split in ('train','val'):
        files.append(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow')
    dump(ROOT/'provenance.json',dict(python=platform.python_version(),pytorch=str(torch.__version__),cuda=torch.version.cuda,
        transformers=transformers.__version__,peft=peft.__version__,gpu=torch.cuda.get_device_name(0),
        sha256={str(p):sha(p) for p in files}))
