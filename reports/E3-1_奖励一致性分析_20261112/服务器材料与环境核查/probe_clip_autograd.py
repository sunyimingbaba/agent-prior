#!/usr/bin/env python3
"""Check real frozen CLIP towers with temporary prompt gradients; no optimizer."""
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

OUT = Path('/private/AGSE_E3_1_rebuild_20261112/audit_20261112')
ROOT = Path('/private/AGSE_E1_20261110/legacy')
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'MPLCONFIGDIR': str(OUT/'scratch_compat/matplotlib'),
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
sys.path.insert(0, str(ROOT/'CUP'))
start = time.monotonic()
result = {'scope': 'frozen CLIP image/text towers and temporary prompt tensors',
          'optimizer_used': False, 'model_parameters_updated': False,
          'formal_training_started': False, 'full_retrieval_model_tested': False}
try:
    import pyarrow as pa
    import torch
    from PIL import Image
    from prompt_clip import clip

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model, preprocess = clip.load(str(ROOT/'assets/models/ViT-L-14.pt'),
                                  device='cuda', jit=False)
    model = model.float().eval()
    model.requires_grad_(False)
    table = pa.ipc.open_file(str(ROOT/'assets/ucm/ucm_caption_karpathy_label_train.arrow')).read_all()
    image = Image.open(io.BytesIO(table['image'][0].as_py())).convert('RGB')
    caption = table['caption'][0].as_py()[0]
    image_tensor = preprocess(image).unsqueeze(0).cuda()
    tokens = clip.tokenize([caption, 'A harbor with ships beside the water.'], context_length=61).cuda()
    visual_prompt = torch.zeros((1, 16, 1024), device='cuda', requires_grad=True)
    text_prompt = torch.zeros((2, 16, 768), device='cuda', requires_grad=True)
    image_feature = model.encode_image(image_tensor, visual_prompt)
    text_feature = model.encode_text(tokens, text_prompt)
    image_feature = image_feature/image_feature.norm(dim=-1, keepdim=True)
    text_feature = text_feature/text_feature.norm(dim=-1, keepdim=True)
    similarities = image_feature @ text_feature.T
    loss = similarities[0, 1] - similarities[0, 0]
    loss.backward()
    torch.cuda.synchronize()
    gradient_checks = {}
    for name, prompt in [('visual_prompt', visual_prompt), ('text_prompt', text_prompt)]:
        grad = prompt.grad
        gradient_checks[name] = {'present': grad is not None,
            'finite': bool(torch.isfinite(grad).all()) if grad is not None else False,
            'norm': float(grad.norm().cpu()) if grad is not None else None}
    result.update({'ok': bool(torch.isfinite(loss).item() and
        all(g['present'] and g['finite'] and g['norm'] > 0 for g in gradient_checks.values())),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'image_path': table['path'][0].as_py(), 'arrow_row': 0,
        'image_feature_shape': list(image_feature.shape),
        'text_feature_shape': list(text_feature.shape),
        'gradient_checks': gradient_checks,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'note': 'Similarity values are used only for an autograd probe; no retrieval metric is reported.'})
except Exception as error:
    result.update({'ok': False, 'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()})
result['elapsed_seconds'] = time.monotonic()-start
(OUT/'clip_autograd.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
