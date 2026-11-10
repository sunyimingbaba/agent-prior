"""Evaluate eight pre-specified text pairs with two frozen retrieval models."""
import csv
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import torch
from PIL import Image

ROOT = Path('/private/AGSE_E1_20261110')
BASE = ROOT / 'legacy'
OUT = ROOT / 'e1_balanced_wording_20261110'
PRIOR = ROOT / 'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val'
sys.path.insert(0, str(BASE / 'CUP'))
from prompt_clip import clip

real_load = clip.load


def cpu_load(name, device='cpu', jit=False, download_root=None):
    if name == 'ViT-L/14':
        name = str(BASE / 'assets/models/ViT-L-14.pt')
    return real_load(name, device='cpu', jit=jit, download_root=download_root)


clip.load = cpu_load
from vilt.modules.vilt_module import ViLTransformerSS


class MyBackboneFinetuning:
    def __init__(self, *args, **kwargs):
        pass


import __main__
__main__.MyBackboneFinetuning = MyBackboneFinetuning
random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count() or 1))

with (OUT / '对照方案明细.csv').open(encoding='utf-8-sig', newline='') as f:
    cases = list(csv.DictReader(f))
samples = json.loads((ROOT / 'e1_quantity_17_20261110/samples.json').read_text())
by_name = {x['图像编号']: x for x in samples}
table = pa.ipc.open_file(str(BASE / 'assets/ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
assert len(cases) == 8 and len({x['图像编号'] for x in cases}) == 4

data = []
for case in cases:
    sample = by_name[case['图像编号']]
    idx = int(sample['验证集行号'])
    assert table['path'][idx].as_py() == case['图像编号']
    img = Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB')
    assert hashlib.sha256(img.tobytes()).hexdigest() == sample['图像RGB_SHA256']
    assert table['caption'][idx].as_py()[0] == case['原始第一条标注']
    hidden = torch.load(PRIOR / f'{idx}.pt', map_location='cpu', weights_only=True).float().reshape(1, -1)
    assert hidden.shape == (1, 3584) and torch.isfinite(hidden).all()
    data.append((case, idx, img, hidden))
print('INPUTS_VERIFIED', len(data), flush=True)

results = []
for method in ('CLIP', 'AGSE'):
    if method == 'CLIP':
        model, preprocess = real_load(str(BASE / 'assets/models/ViT-L-14.pt'), device='cpu', jit=False)
        model.float().eval()
    else:
        checkpoint = torch.load(BASE / 'assets/ucm_main_domain_tool.ckpt', map_location='cpu', weights_only=False)
        config = dict(checkpoint['hyper_parameters']['config'])
        config['load_path'] = ''
        config['test_only'] = False
        assert config.get('use_agent_prior') is True
        model = ViLTransformerSS(config).cpu().eval()
        print('CHECKPOINT', model.load_state_dict(checkpoint['state_dict'], strict=True), flush=True)
        del checkpoint
        preprocess = model.preprocess
    with torch.inference_mode():
        for case, idx, img, hidden in data:
            image = preprocess(img).unsqueeze(0)
            texts = [case['文本1'], case['文本2']]
            if method == 'CLIP':
                visual = model.encode_image(image, torch.zeros((1, 16, 1024)))
                textual = model.encode_text(clip.tokenize(texts))
                visual = visual / visual.norm(dim=-1, keepdim=True)
                textual = textual / textual.norm(dim=-1, keepdim=True)
            else:
                visual = model.img_embeds(image, agent_hidden=hidden)['image_feats']
                textual = model.txt_embeds(clip.tokenize(texts, context_length=77-model.prompt_length))['text_feats']
            scores = (visual @ textual.T).flatten()
            assert torch.isfinite(scores).all()
            score1, score2 = scores.tolist()
            correct_idx = 0 if case['正确文本位置'] == '文本1' else 1
            correct_score = (score1, score2)[correct_idx]
            wrong_score = (score1, score2)[1-correct_idx]
            row = {
                'model': method, 'image': case['图像编号'], 'arrow_row': idx,
                'scene': case['场景'], 'actual_count': case['用户确认实际数量'],
                'wording': case['句式'], 'text1': texts[0], 'text2': texts[1],
                'correct_text_position': case['正确文本位置'],
                'score1': score1, 'score2': score2,
                'correct_score': correct_score, 'wrong_score': wrong_score,
                'margin': correct_score-wrong_score, 'correct': correct_score > wrong_score,
                'original_first_caption': case['原始第一条标注'],
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    del model

assert len(results) == 16
with (OUT / 'result.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
(OUT / 'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
print('FINISHED', flush=True)
