"""Native CUP loss, optimizer, FP32 Trainer; compact resumable readout checkpoints."""
import argparse
import datetime
import functools
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
from milestone_common import ROOT, setup, seed
from stage_io import read, dump, sha
from matched_common import TRAIN_PLAN, CACHE_PLAN, load_initial, install_arm, original_train_dataset, prepare_gallery, evaluate, state_hash
from vilt.modules.vilt_module import ViLTransformerSS

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=TRAIN_PLAN['arms'], required=True)
a = p.parse_args()
setup()
assert os.environ.get('PYTHONHASHSEED') == '0'
assert read(ROOT/'cache_result.json')['ok'] and read(ROOT/'training_preflight.json')['ok']
for n, h in read(ROOT/'training_code_hashes.json').items():
    assert sha(ROOT/n) == h, n
arm = a.arm
out = ROOT/'training'/arm
out.mkdir(parents=True, exist_ok=True)
assert not (out/'training_result.json').exists()
started = time.monotonic()


def progress(phase, **extra):
    dump(out/'progress.json', {'phase': phase, 'arm': arm, 'elapsed_seconds': time.monotonic()-started,
         'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra})


class Readout(ViLTransformerSS):
    def validation_step(self, batch, batch_idx):
        return None

    def validation_epoch_end(self, outputs):
        metrics = evaluate(self, self._gallery, out/f'validation/epoch_{self.current_epoch+1:02d}')
        self.log('val/mR', metrics['mR'], logger=True, prog_bar=False)
        progress('validated', epoch=self.current_epoch+1, optimizer_step=self.global_step, metrics=metrics)
        print('MATCHED_VAL', arm, self.current_epoch+1, json.dumps(metrics), flush=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {n: x for n, x in checkpoint['state_dict'].items() if not n.startswith('model.')}
        checkpoint['matched_readout'] = {'arm': arm, 'base_sha256': CACHE_PLAN['utility_model_sha256'],
             'training_plan_sha256': sha(ROOT/'training_plan.json'), 'sample_order_sha256': sha(ROOT/'training_orders.pt'),
             'backbone_omitted': True}
        checkpoint['matched_rng'] = {'python': random.getstate(), 'numpy': np.random.get_state(),
                                     'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}

    def on_load_checkpoint(self, checkpoint):
        info = checkpoint['matched_readout']
        assert info['arm'] == arm and info['base_sha256'] == CACHE_PLAN['utility_model_sha256']
        assert info['training_plan_sha256'] == sha(ROOT/'training_plan.json')
        assert info['sample_order_sha256'] == sha(ROOT/'training_orders.pt')
        assert not any(n.startswith('model.') for n in checkpoint['state_dict'])
        checkpoint['state_dict'].update({'model.'+n: x for n, x in self.model.state_dict().items()})
        rng = checkpoint['matched_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state_all(rng['cuda'])


model, config = load_initial(Readout)
initial_backbone = state_hash(model.model.state_dict().items())
initial_readout = state_hash((n, x) for n, x in model.state_dict().items() if not n.startswith('model.'))
hooks = install_arm(model, arm)
dataset = original_train_dataset(config)
assert len(dataset) == TRAIN_PLAN['train_caption_samples']
orders = torch.load(ROOT/'training_orders.pt', map_location='cpu', weights_only=True)
assert orders.shape == (TRAIN_PLAN['epochs'], len(dataset))


class OrderedSampler(Sampler):
    epoch = 0
    def __iter__(self):
        return iter(orders[self.epoch].tolist())
    def __len__(self):
        return len(dataset)


sampler = OrderedSampler()
loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4,
           collate_fn=functools.partial(dataset.collate, mlm_collator=None), pin_memory=True,
           generator=torch.Generator().manual_seed(0))
model._gallery = prepare_gallery(model, 'val')
dump(out/'resolved_config.json', config)
dump(out/'initialization.json', {'initial_backbone_hash': initial_backbone,
     'initial_readout_hash': initial_readout, 'initial_full_state_hash': TRAIN_PLAN['initial_model_state_hash'],
     'trainable': {n: list(x.shape) for n, x in model.named_parameters() if x.requires_grad},
     'sample_order_sha256': sha(ROOT/'training_orders.pt')})
losslog = (out/'batch_losses.jsonl').open('a', encoding='utf-8')
auditlog = (out/'gradient_audit.jsonl').open('a', encoding='utf-8')


class Audit(pl.Callback):
    def on_train_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx):
        self.current_batch = batch_idx

    def on_train_epoch_start(self, trainer, module):
        sampler.epoch = trainer.current_epoch
        assert 0 <= sampler.epoch < TRAIN_PLAN['epochs']
        progress('training', epoch=sampler.epoch+1, optimizer_step=trainer.global_step)

    def on_after_backward(self, trainer, module):
        if self.current_batch % 25 == 0:
            gradients = [(n, x.grad) for n, x in module.named_parameters() if x.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for _, g in gradients)
            assert not any(n.startswith('model.') for n, _ in gradients)
            auditlog.write(json.dumps({'epoch': trainer.current_epoch+1, 'batch': self.current_batch,
                   'tensors': len(gradients), 'finite': True, 'backbone_has_gradient': False,
                   'gradient_norms': {n: float(g.norm()) for n, g in gradients}})+'\n')
            auditlog.flush()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        loss = float(loss.detach())
        assert math.isfinite(loss)
        losslog.write(json.dumps({'epoch': trainer.current_epoch+1, 'batch': batch_idx,
            'optimizer_step': trainer.global_step, 'loss': loss,
            'lr': trainer.optimizers[0].param_groups[0]['lr'], 'raw_indices': batch['raw_index']})+'\n')
        if batch_idx % 25 == 0:
            losslog.flush()
            progress('training', epoch=trainer.current_epoch+1, batch=batch_idx+1,
                     optimizer_step=trainer.global_step, loss=loss)


checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(out/'checkpoints'),
     filename='epoch={epoch:02d}-step={step}', auto_insert_metric_name=False,
     save_top_k=1, monitor='val/mR', mode='max', save_last=True, verbose=True)
resume = out/'checkpoints/last.ckpt'
trainer = pl.Trainer(gpus=1, precision=32, accelerator=None, max_epochs=TRAIN_PLAN['epochs'],
     max_steps=TRAIN_PLAN['optimizer_steps_per_arm'], accumulate_grad_batches=8,
     num_sanity_val_steps=0, val_check_interval=1.0, progress_bar_refresh_rate=0,
     weights_summary=None, log_every_n_steps=25, terminate_on_nan=True,
     resume_from_checkpoint=str(resume) if resume.exists() else None,
     callbacks=[Audit(), checkpoint], logger=pl.loggers.TensorBoardLogger(str(out/'logs'), name='readout', version=0))
torch.cuda.reset_peak_memory_stats()
model.cuda()
if not resume.exists():
    evaluate(model, model._gallery, out/'initial_validation')
seed(0)
trainer.fit(model, train_dataloader=loader, val_dataloaders=[torch.utils.data.DataLoader([0])])
losslog.flush()
auditlog.flush()
assert trainer.global_step == TRAIN_PLAN['optimizer_steps_per_arm']
assert state_hash(model.model.state_dict().items()) == initial_backbone
paths = {}
for role, path in [('last', checkpoint.last_model_path), ('validation_best', checkpoint.best_model_path)]:
    pth = Path(path)
    saved = torch.load(pth, map_location='cpu', weights_only=False)
    assert saved['matched_readout']['backbone_omitted']
    assert not any(n.startswith('model.') for n in saved['state_dict'])
    paths[role] = {'path': str(pth.relative_to(ROOT)), 'sha256': sha(pth), 'bytes': pth.stat().st_size,
                   'global_step': saved['global_step'], 'epoch': saved['epoch']+1}
result = {'ok': True, 'arm': arm, 'epochs': TRAIN_PLAN['epochs'], 'optimizer_steps': trainer.global_step,
          'backbone_hash_unchanged': initial_backbone, 'initial_readout_hash': initial_readout,
          'final_readout_hash': state_hash((n, x) for n, x in model.state_dict().items() if not n.startswith('model.')),
          'checkpoints': paths, 'best_val_mR': float(checkpoint.best_model_score),
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic()-started}
assert result['final_readout_hash'] != initial_readout
dump(out/'training_result.json', result)
progress('complete', optimizer_steps=trainer.global_step)
print('MATCHED_ARM_TRAINING_COMPLETE', json.dumps(result), flush=True)
