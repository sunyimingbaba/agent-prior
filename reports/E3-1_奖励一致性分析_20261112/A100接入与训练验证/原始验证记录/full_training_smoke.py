#!/usr/bin/env python3
"""One native Lightning training batch on a fresh model; all outputs are new."""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()
OUT = Path('/private/AGSE_E3_1_rebuild_20261112/a100_audit_20261112')
BASE = Path('/private/AGSE_E1_20261110/legacy')
OUT.mkdir(parents=True, exist_ok=True)
scratch = OUT/'scratch_training'
scratch.mkdir(exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_HOME': str(scratch/'hf'), 'TMPDIR': str(scratch),
                   'MPLCONFIGDIR': str(scratch/'matplotlib'),
                   'XDG_CACHE_HOME': str(scratch/'cache'),
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
sys.path.insert(0, str(BASE/'CUP'))
result = {'scope': 'one-batch environment and checkpoint smoke check',
          'formal_scorer_training_started': False,
          'existing_source_files_modified': False,
          'existing_model_weights_modified': False,
          'model_initialization': 'fresh trainable modules with local frozen CLIP backbone',
          'configuration_reference': 'archived AGSE checkpoint metadata',
          'smoke_prior_source': 'existing domain-tool train priors, for interface/gradient checks',
          'batch_size': args.batch_size}
started = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


try:
    import pyarrow as pa
    import pytorch_lightning as pl
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from prompt_clip import clip
    from vilt.modules.vilt_module import ViLTransformerSS
    from run_frozen import MyBackboneFinetuning

    torch.set_num_threads(4)
    pl.seed_everything(0)
    original = torch.load(BASE/'assets/ucm_main_domain_tool.ckpt',
                          map_location='cpu', weights_only=False, mmap=True)
    config = dict(original['hyper_parameters']['config'])
    del original
    config.update({'load_path': '', 'test_only': False, 'use_agent_prior': True,
                   'data_root': str(BASE/'assets/ucm'),
                   'agent_cache_dir': '/private/AGSE_E1_20261110/ucm_prior_archive/agent_cache_domain_tool',
                   'log_dir': str(OUT/'lightning_logs'), 'save_image_path': str(OUT/'images'),
                   'per_gpu_batchsize': args.batch_size, 'batch_size': args.batch_size,
                   'num_gpus': 1, 'num_nodes': 1, 'num_workers': 0,
                   'max_epoch': 1, 'max_steps': 1, 'get_recall_metric': False})
    (OUT/'smoke_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    clip_load = clip.load

    def local_clip_load(name, device='cpu', jit=False, download_root=None):
        if name == 'ViT-L/14':
            name = str(BASE/'assets/models/ViT-L-14.pt')
        return clip_load(name, device='cpu', jit=jit, download_root=download_root)

    clip.load = local_clip_load
    model = ViLTransformerSS(config)
    table = pa.ipc.open_file(str(BASE/'assets/ucm/ucm_caption_karpathy_label_train.arrow')).read_all()
    examples = []
    image_names = []
    for index in range(args.batch_size):
        image = Image.open(io.BytesIO(table['image'][index].as_py())).convert('RGB')
        prior_path = Path(config['agent_cache_dir'])/'ucm_caption_karpathy_label_train'/f'{index}.pt'
        prior = torch.load(prior_path, map_location='cpu', weights_only=True).float()
        examples.append({'clip_img': model.preprocess(image),
                         'clip_text_token': clip.tokenize([table['caption'][index].as_py()[0]], context_length=61)[0],
                         'agent_hidden': prior})
        image_names.append(table['path'][index].as_py())
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gradient_records = []

    class GradientAudit(pl.Callback):
        def on_train_start(self, trainer, module):
            result['frozen_backbone'] = all(not p.requires_grad for p in module.model.parameters())
            result['trainable_parameter_count'] = sum(p.numel() for p in module.parameters() if p.requires_grad)
            result['tracked_before'] = {}
            self.before = {name: p.detach().cpu().clone() for name, p in module.named_parameters()
                           if name in ['prompt_embeddings', 'visual_prompt_embeddings', 'agent_proj.0.weight']}

        def on_after_backward(self, trainer, module):
            grads = [(name, p.grad) for name, p in module.named_parameters() if p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for _, g in grads)
            gradient_records.append({'tensors_with_gradient': len(grads), 'all_finite': finite,
                'agent_projection_gradient_present': any(n.startswith('agent_proj.') for n, _ in grads),
                'backbone_has_gradients': any(n.startswith('model.') for n, _ in grads)})
            if not finite:
                raise RuntimeError('Non-finite gradient in training smoke')

        def on_train_end(self, trainer, module):
            named = dict(module.named_parameters())
            result['parameter_max_changes'] = {
                name: float((named[name].detach().cpu()-value).abs().max())
                for name, value in self.before.items()}

    audit = GradientAudit()
    trainer = pl.Trainer(gpus=1, logger=False, checkpoint_callback=False,
                         callbacks=[MyBackboneFinetuning(), audit],
                         default_root_dir=str(OUT), max_epochs=1, max_steps=1,
                         limit_train_batches=1, limit_val_batches=0,
                         num_sanity_val_steps=0, progress_bar_refresh_rate=0,
                         weights_summary=None, precision=32, accumulate_grad_batches=1)
    torch.cuda.reset_peak_memory_stats()
    train_started = time.monotonic()
    trainer.fit(model, train_dataloaders=loader)
    torch.cuda.synchronize()
    result['one_batch_training_seconds'] = time.monotonic()-train_started
    result['global_step'] = trainer.global_step
    result['gradient_records'] = gradient_records
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    result['image_names'] = image_names
    result['training_metrics'] = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                                  if isinstance(v, torch.Tensor) and v.numel() == 1}
    checkpoint_path = OUT/f'environment_smoke_bs{args.batch_size}.ckpt'
    trainer.save_checkpoint(str(checkpoint_path))
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    loaded = model.load_state_dict(saved['state_dict'], strict=True)
    result['checkpoint_strict_load_ok'] = not loaded.missing_keys and not loaded.unexpected_keys
    optimizer_saved = saved.get('optimizer_states', [])
    result['optimizer_state_saved'] = bool(optimizer_saved)
    if optimizer_saved:
        trainer.optimizers[0].load_state_dict(optimizer_saved[0])
    result['checkpoint'] = {'path': str(checkpoint_path), 'bytes': checkpoint_path.stat().st_size,
                             'sha256': sha(checkpoint_path), 'global_step': saved.get('global_step')}
    result['ok'] = bool(result['frozen_backbone'] and result['checkpoint_strict_load_ok']
                        and result['optimizer_state_saved'] and gradient_records
                        and all(g['all_finite'] and not g['backbone_has_gradients'] for g in gradient_records)
                        and any(v > 0 for v in result['parameter_max_changes'].values())
                        and trainer.global_step == 1)
except Exception as error:
    result.update({'ok': False, 'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()})
result['elapsed_seconds'] = time.monotonic()-started
(OUT/'full_training_smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
sys.exit(0 if result['ok'] else 1)
