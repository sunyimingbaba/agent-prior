"""CPU audit and exact schedule/sample order before any readout optimizer step."""
import hashlib
import inspect
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
import torch
import pytorch_lightning as pl
from milestone_common import ROOT, BASE, setup, seed
from stage_io import read, dump, sha
from matched_common_v2 import TRAIN_PLAN, CACHE_PLAN, PARENT, load_initial, install_arm, original_train_dataset, state_hash

setup()
for n, h in read(ROOT/'training_code_hashes_v2.json').items():
    assert sha(ROOT/n) == h, n
sources = read(PARENT/'source_snapshot/inventory_before.json')['files']
for n, h in sources.items():
    assert sha(BASE/'CUP'/n) == h, n
for n in ['run_frozen.py', 'vilt/config.py', 'vilt/modules/vilt_module.py',
          'vilt/modules/vilt_utils.py', 'vilt/modules/objectives.py',
          'vilt/datasets/base_dataset.py', 'vilt/datasets/ucm_caption_karpathy_dataset.py']:
    target = ROOT/'source_snapshot'/n
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == (BASE/'CUP'/n).read_bytes()
    else:
        shutil.copy2(BASE/'CUP'/n, target)
dump(ROOT/'source_snapshot/inventory_before.json', {'files': sources, 'original_root': str(BASE/'CUP')})
model, config = load_initial()
dataset = original_train_dataset(config)
mapping = [[i, row, cap, dataset.all_texts[row][cap]] for i, (row, cap) in dataset.index_mapper.items()]
assert len(dataset) == TRAIN_PLAN['train_caption_samples']
assert len(dataset.table) == 1680
batches = math.ceil(len(dataset)/32)
steps = math.ceil(batches/8)
assert steps*TRAIN_PLAN['epochs'] == TRAIN_PLAN['optimizer_steps_per_arm']
orders = torch.stack([torch.randperm(len(dataset), generator=torch.Generator().manual_seed(2026111600+e))
                      for e in range(TRAIN_PLAN['epochs'])])
assert all(torch.equal(x.sort().values, torch.arange(len(dataset))) for x in orders)
torch.save(orders, ROOT/'training_orders.pt')
trainable = {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
assert all(not p.requires_grad for p in model.model.parameters())
model.trainer = SimpleNamespace(max_steps=TRAIN_PLAN['optimizer_steps_per_arm'])
optimizers, schedulers = model.configure_optimizers()
optimizer = optimizers[0]
names = {id(p): n for n, p in model.named_parameters()}
optimizer_groups = [{'lr': g['lr'], 'weight_decay': g['weight_decay'],
                     'names': [names[id(p)] for p in g['params']],
                     'trainable_names': [names[id(p)] for p in g['params'] if p.requires_grad]}
                    for g in optimizer.param_groups]
covered = [n for g in optimizer_groups for n in g['trainable_names']]
assert len(covered) == len(set(covered)) and set(covered) == set(trainable)
probe = torch.randn(2, 3584)
properties = {}
for arm in TRAIN_PLAN['arms']:
    hooks = install_arm(model, arm)
    with torch.no_grad():
        output = model.agent_proj(probe)
        gate = model.cluster_mapping(output[:, None].repeat(1,16,1)).clamp(0,1)
    distance = float((output[0]-output[1]).abs().max())
    if arm in ['calibrated_shared', 'learned_visual_only']:
        assert distance == 0.
    else:
        assert distance > 0.
    if arm == 'learned_visual_only':
        assert torch.equal(gate, torch.ones_like(gate))
    properties[arm] = {'projected_input_specificity_max_difference': distance,
                       'gate_min': float(gate.min()), 'gate_max': float(gate.max())}
    for h in hooks:
        h.remove()
dump(ROOT/'training_preflight.json', {'ok': True, 'cpu_only': True,
     'initial_model_state_hash': state_hash(model.state_dict().items()),
     'train_images': 1680, 'train_caption_samples': len(dataset), 'batches_per_epoch': batches,
     'optimizer_steps_per_epoch': steps, 'optimizer_steps_per_arm': steps*TRAIN_PLAN['epochs'],
     'total_four_arm_optimizer_steps': steps*TRAIN_PLAN['epochs']*4,
     'schedule_total_steps': TRAIN_PLAN['optimizer_steps_per_arm'],
     'schedule_warmup_steps': int(TRAIN_PLAN['warmup_fraction']*TRAIN_PLAN['optimizer_steps_per_arm']),
     'trainable': trainable, 'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
     'optimizer_groups': optimizer_groups, 'sampler_order_sha256': sha(ROOT/'training_orders.pt'),
     'caption_index_order_sha256': hashlib.sha256(json.dumps(mapping,ensure_ascii=False).encode()).hexdigest(),
     'arm_source_independence_witnesses': properties,
     'original_python_files_unchanged': len(sources), 'test_evaluations': 0,
     'Trainer_fit_signature': str(inspect.signature(pl.Trainer.fit))})
dump(ROOT/'training_caption_index.json', mapping)
print('MATCHED_TRAINING_PREFLIGHT_COMPLETE', len(dataset), steps*TRAIN_PLAN['epochs'], flush=True)
