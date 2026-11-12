#!/usr/bin/env python3
"""Strict checkpoint/optimizer audit and bounded inference of the frozen last scorer."""
import hashlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02')
BASE = Path('/private/AGSE_E1_20261110/legacy')
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'TMPDIR': str(ROOT/'run_01/scratch'), 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
    'HF_HOME': str(ROOT/'run_01/scratch/hf'),
    'MPLCONFIGDIR': str(ROOT/'run_01/scratch/matplotlib'),
    'XDG_CACHE_HOME': str(ROOT/'run_01/scratch/cache')})
sys.path.insert(0, str(BASE/'CUP'))
import torch
import pyarrow as pa
from PIL import Image
from vilt.modules.vilt_module import ViLTransformerSS
from run_frozen import MyBackboneFinetuning
from prompt_clip import clip

torch.set_num_threads(4)
result = json.loads((ROOT/'training_result.json').read_text(encoding='utf-8'))
assert result['ok'] and result['status'] == 'complete' and result['completed_epochs'] == 50
assert result['optimizer_steps'] == 1600
settings = json.loads((ROOT/'resolved_settings.json').read_text(encoding='utf-8'))
clip_load = clip.load


def private_clip_load(name, device='cpu', jit=False, download_root=None):
    if name == 'ViT-L/14':
        name = str(BASE/'assets/models/ViT-L-14.pt')
    return clip_load(name, device='cpu', jit=jit, download_root=download_root)


clip.load = private_clip_load


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def parameter_hash(named):
    h = hashlib.sha256()
    for name, tensor in named:
        tensor = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


audits = {}
last_model = None
for label in ['validation_best', 'last']:
    registration = result['checkpoints'][label]
    path = Path(registration['path'])
    assert path.stat().st_size == registration['bytes'] and sha(path) == registration['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = dict(saved['hyper_parameters']['config'])
    for key in ['agent_cache_dir', 'data_root', 'max_epoch', 'batch_size', 'per_gpu_batchsize',
                'seed', 'precision', 'prompt_length', 'learning_rate', 'use_agent_prior']:
        assert config[key] == settings['config'][key], key
    assert config['load_path'] == '' and not config['test_only']
    model = ViLTransformerSS(config)
    loaded = model.load_state_dict(saved['state_dict'], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(bool(torch.isfinite(value).all()) for value in saved['state_dict'].values()
               if isinstance(value, torch.Tensor))
    optimizer_states = saved.get('optimizer_states', [])
    assert len(optimizer_states) == 1 and optimizer_states[0]['state']
    tensor_count = 0
    for state in optimizer_states[0]['state'].values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all())
                tensor_count += 1
    MyBackboneFinetuning().freeze_clip(model.model)
    backbone_sha = parameter_hash(model.model.state_dict().items())
    assert backbone_sha == result['initial_backbone_sha256']
    if label == 'last':
        assert saved['epoch'] == 49
        assert abs(saved['global_step']-result['optimizer_steps']) <= 1
        assert parameter_hash((n, p) for n, p in model.named_parameters() if p.requires_grad) == result['final_trainable_sha256']
        last_model = model
    audits[label] = {**registration, 'epoch': saved['epoch'], 'global_step': saved['global_step'],
        'strict_load_ok': True, 'all_state_tensors_finite': True,
        'optimizer_state_tensors_finite': tensor_count,
        'optimizer_parameter_state_count': len(optimizer_states[0]['state']),
        'lr_scheduler_state_present': bool(saved.get('lr_schedulers')),
        'frozen_backbone_matches_initial': True}
    if label != 'last':
        del model
    del saved

last_model.requires_grad_(False)
last_model.eval().cuda()
assert not last_model.training and all(not p.requires_grad for p in last_model.parameters())
table = pa.ipc.open_file(str(BASE/'assets/ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
images, priors, captions, inputs = [], [], [], []
for row in [0, 105, 209]:
    image = Image.open(io.BytesIO(table['image'][row].as_py())).convert('RGB')
    caption = table['caption'][row].as_py()[0]
    prior = torch.load(Path(settings['config']['agent_cache_dir'])/'ucm_caption_karpathy_label_val'/f'{row}.pt', map_location='cpu', weights_only=True)
    images.append(last_model.preprocess(image))
    priors.append(prior)
    captions.append(caption)
    inputs.append({'row': row, 'image': table['path'][row].as_py(), 'caption': caption})
with torch.inference_mode():
    image_batch = torch.stack(images).cuda()
    prior_batch = torch.stack(priors).cuda()
    text_tokens = clip.tokenize(captions, context_length=61).cuda()
    image_features = last_model.img_embeds(image_batch, agent_hidden=prior_batch)['image_feats']
    text_features = last_model.txt_embeds(text_tokens)['text_feats']
    proxy_features = last_model.img_embeds(torch.zeros_like(image_batch), agent_hidden=prior_batch)['image_feats']
    for features in [image_features, text_features, proxy_features]:
        assert bool(torch.isfinite(features).all()) and bool((features.norm(dim=1)>0).all())
    scores = last_model.model.logit_scale.exp() * torch.nn.functional.normalize(image_features, dim=1) @ torch.nn.functional.normalize(text_features, dim=1).T
    assert bool(torch.isfinite(scores).all())
inference = {'frozen_eval': True, 'inputs': inputs, 'image_feature_shape': list(image_features.shape),
    'text_feature_shape': list(text_features.shape), 'proxy_feature_shape': list(proxy_features.shape),
    'all_features_and_scores_finite': True, 'image_text_scores': scores.cpu().tolist()}
output = {'verified': True, 'checkpoints': audits, 'frozen_last_inference': inference,
          'trainer_optimizer_steps': result['optimizer_steps'], 'completed_epochs': 50}
(ROOT/'checkpoint_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
registry = {'status': 'frozen', 'selected_checkpoint': audits['last'],
    'selection_policy': settings['checkpoint_policy'], 'completed_epochs': 50,
    'optimizer_steps': result['optimizer_steps'], 'test_evaluation_executed': False,
    'resolved_settings_sha256': sha(ROOT/'resolved_settings.json'),
    'training_input_manifest_sha256': settings['input_manifest_sha256'],
    'verification_sha256': sha(ROOT/'checkpoint_verification.json'),
    'runtime_requirement': 'strict load, eval(), requires_grad_(False)',
    'inference_finite': True}
(ROOT/'frozen_scorer_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
print(json.dumps(output, ensure_ascii=False))
