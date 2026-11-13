#!/usr/bin/env python3
"""Dedicated fit-only entry, using the preserved CUP modules and native PL training."""
import argparse
import copy
import datetime
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--prepare-only', action='store_true')
args = parser.parse_args()
ROOT = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_resume_20261113')
PREVIOUS = Path('/private/AGSE_E3_1_rebuild_20261112/scorer_training_20261112/attempt_02')
RESUME = PREVIOUS/'snapshots/epoch_30/last.ckpt'
RESUME_SHA = '0c0fe707dac3db1d59a9dedca024977b26a27f7a394d8e7becf84e5efc158583'
BASE = Path('/private/AGSE_E1_20261110/legacy')
PRIOR = Path('/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_20261112')
RUN = ROOT/'run_01'
ROOT.mkdir(parents=True, exist_ok=True)
RUN.mkdir(exist_ok=True)
lock = (ROOT/'training.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for name in ['scratch/hf', 'scratch/matplotlib', 'scratch/cache/torch/kernels', 'images']:
    (RUN/name).mkdir(parents=True, exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_HOME': str(RUN/'scratch/hf'), 'TMPDIR': str(RUN/'scratch'),
                   'MPLCONFIGDIR': str(RUN/'scratch/matplotlib'),
                   'XDG_CACHE_HOME': str(RUN/'scratch/cache'),
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                   'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '16417'})
assert os.environ.get('PYTHONHASHSEED') == '0'
sys.path.insert(0, str(BASE/'CUP'))
sys.stdout.reconfigure(encoding='utf-8')
started = time.monotonic()
status = {'status': 'preflight', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checkpoint_policy': 'last checkpoint after 50 epochs', 'test_evaluation_executed': False,
          'completed_epochs': 30, 'resume_from': str(RESUME), 'resume_sha256': RESUME_SHA}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def progress(phase, **extra):
    status.update(extra)
    status.update(status=phase, elapsed_seconds=time.monotonic()-started,
                  last_update_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(ROOT/('prepare_progress.json' if args.prepare_only else 'progress.json'), status)


def parameter_hash(named):
    h = hashlib.sha256()
    for name, tensor in named:
        tensor = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(value):
    """Canonical tensor/state digest that remains identical across CPU/GPU placement."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(b'T')
            h.update(parameter_hash([('tensor', item)]).encode())
        elif isinstance(item, dict):
            h.update(b'D')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b'L')
            for element in item:
                visit(element)
        else:
            h.update(json.dumps([type(item).__name__, item], sort_keys=True).encode())
    visit(value)
    return h.hexdigest()


try:
    import torch
    import pytorch_lightning as pl
    from vilt.config import ex
    from vilt.datamodules.multitask_datamodule import MTDataModule
    from vilt.modules.vilt_module import ViLTransformerSS
    from run_frozen import MyBackboneFinetuning
    from pytorch_lightning.trainer.states import TrainerFn
    from pytorch_lightning.utilities.model_helpers import is_overridden
    import prompt_clip.clip as prompt_clip
    import model.clip as dataset_clip

    torch.set_num_threads(4)
    assert sha(RESUME) == RESUME_SHA
    saved_resume = torch.load(RESUME, map_location='cpu', weights_only=False, mmap=True)
    assert saved_resume['epoch'] == 30 and saved_resume['global_step'] == 960
    assert len(saved_resume['optimizer_states']) == len(saved_resume['lr_schedulers']) == 1
    resume_optimizer_hash = state_hash(saved_resume['optimizer_states'][0])
    resume_scheduler_hash = state_hash(saved_resume['lr_schedulers'][0])
    old_best = saved_resume['callbacks'][pl.callbacks.ModelCheckpoint]
    old_best_path = Path(old_best['best_model_path'])
    assert old_best_path.parent == PREVIOUS/'run_01/checkpoints'
    assert float(old_best['best_model_score']) == 80.25
    previous_start = json.loads((PREVIOUS/'run_01/training_events.jsonl').read_text().splitlines()[0])
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cuda_witness = torch.randn(16, 16, device='cuda', requires_grad=True)
    witness_loss = (cuda_witness @ cuda_witness.T).square().mean()
    witness_loss.backward()
    assert bool(torch.isfinite(witness_loss)) and bool(torch.isfinite(cuda_witness.grad).all())
    del cuda_witness, witness_loss
    manifest = json.loads((ROOT/'training_input_manifest_v2.json').read_text(encoding='utf-8'))
    for package, version in manifest['expected_versions'].items():
        assert importlib.metadata.version(package) == version, package
    for name, expected in manifest['source_sha256'].items():
        assert sha(BASE/'CUP'/name) == expected, name
    assert sha(BASE/'assets/models/ViT-L-14.pt') == manifest['clip_sha256']
    for split, expected in manifest['arrow_sha256'].items():
        assert sha(BASE/'assets/ucm'/f'ucm_caption_karpathy_label_{split}.arrow') == expected, split
    for name, key in [('input_model_manifest.json', 'prior_manifest_sha256'),
                      ('run_01/settings.json', 'prior_settings_sha256'),
                      ('run_01/records.jsonl', 'prior_records_sha256')]:
        assert sha(PRIOR/name) == manifest[key], name
    prior_meta = json.loads((PRIOR/'run_01/metadata.json').read_text(encoding='utf-8'))
    assert prior_meta['ok'] and prior_meta['completed'] == 1890
    prior_records = [json.loads(line) for line in (PRIOR/'run_01/records.jsonl').read_text(encoding='utf-8').splitlines()]
    for record in prior_records:
        path = PRIOR/'run_01'/record['prior_file']
        assert sha(path) == record['prior_file_sha256']
        vector = torch.load(path, map_location='cpu', weights_only=True)
        assert tuple(vector.shape) == (3584,) and bool(torch.isfinite(vector).all())

    @ex.command
    def resolve_training_config(_config):
        return dict(_config)

    updates = {'seed': 0, 'load_path': '', 'resume_from': str(RESUME), 'test_only': False,
               'use_agent_prior': True, 'agent_cache_dir': str(PRIOR/'run_01/priors'),
               'data_root': str(BASE/'assets/ucm'), 'log_dir': str(RUN/'logs'),
               'save_image_path': str(RUN/'images'), 'json': str(RUN/'unused_legacy_annotation.json'),
               'batch_size': 256, 'per_gpu_batchsize': 32, 'num_gpus': 1, 'num_nodes': 1,
               'num_workers': 8, 'precision': 32}
    config = ex.run(command_name='resolve_training_config',
                    named_configs=['task_finetune_irtr_ucm_randaug_ViTL14'],
                    config_updates=updates,
                    options={'--unobserved': True, '--capture': 'no'}).result
    assert config['max_epoch'] == 50 and config['max_steps'] is None
    assert config['learning_rate'] == 1e-4 and config['warmup_steps'] == 0.1
    assert config['get_recall_metric'] and config['prompt_length'] == 16
    for module in [prompt_clip, dataset_clip]:
        original_load = module.load

        def private_clip_load(name, device='cpu', jit=False, download_root=None, _load=original_load):
            if name == 'ViT-L/14':
                name = str(BASE/'assets/models/ViT-L-14.pt')
            return _load(name, device='cpu', jit=jit, download_root=download_root)

        module.load = private_clip_load

    trainer_settings = {'gpus': 1, 'num_nodes': 1, 'accelerator': 'ddp', 'precision': 32,
                        'max_epochs': 50, 'accumulate_grad_batches': 8,
                        'num_sanity_val_steps': 0, 'val_check_interval': 1.0,
                        'progress_bar_refresh_rate': 0, 'weights_summary': None,
                        'log_every_n_steps': 10, 'terminate_on_nan': True,
                        'resume_from_checkpoint': str(RESUME)}
    fingerprint = {'config': config, 'trainer': trainer_settings,
                   'input_manifest_sha256': sha(ROOT/'training_input_manifest_v2.json'),
                   'training_script_sha256': sha(Path(__file__)),
                   'test_execution': manifest['test_execution'], 'python_hash_seed': '0',
                   'checkpoint_policy': manifest['checkpoint_policy'],
                   'resume': {'path': str(RESUME), 'sha256': RESUME_SHA, 'completed_epochs': 30,
                              'global_step': 960, 'optimizer_state_sha256': resume_optimizer_hash,
                              'scheduler_state_sha256': resume_scheduler_hash,
                              'checkpoint_keys': list(saved_resume),
                              'process_restart_seed': 0}}
    fingerprint_path = ROOT/'resolved_settings.json'
    if fingerprint_path.exists():
        assert json.loads(fingerprint_path.read_text(encoding='utf-8')) == fingerprint
    else:
        write_json(fingerprint_path, fingerprint)
    pl.seed_everything(0)
    dm = MTDataModule(config, dist=False)
    original_overrides = {hook: is_overridden(hook, dm) for hook in ['setup', 'teardown']}
    # Python 3.12 formats this legacy str Enum as TrainerFn.FITTING.
    # Normalize before PL 1.4's existing hook tracker builds attribute names.
    for hook in ['setup', 'teardown']:
        tracked = getattr(dm, hook)

        @functools.wraps(getattr(tracked, '__wrapped__', tracked))
        def normalized_stage(*positional, _tracked=tracked, **keywords):
            if positional:
                return _tracked(getattr(positional[0], 'value', positional[0]),
                                *positional[1:], **keywords)
            if 'stage' in keywords:
                keywords['stage'] = getattr(keywords['stage'], 'value', keywords['stage'])
            return _tracked(**keywords)

        setattr(dm, hook, normalized_stage)
    assert {hook: is_overridden(hook, dm) for hook in ['setup', 'teardown']} == original_overrides
    if args.prepare_only:
        dm.setup(stage=TrainerFn.FITTING)
        assert dm._has_setup_fit
        datasets = {}
        for split in ['train', 'val', 'test']:
            dataset = getattr(dm.dms[0], split+'_dataset')
            index_content = [[i, row, cap, dataset.all_texts[row][cap]]
                             for i, (row, cap) in dataset.index_mapper.items()]
            datasets[split] = {'images': len(dataset.table), 'caption_samples': len(dataset),
                'caption_index_order_sha256': hashlib.sha256(json.dumps(index_content, ensure_ascii=False).encode()).hexdigest(),
                'batches': math.ceil(len(dataset)/32)}
        assert datasets['train']['images'] == 1680 and datasets['val']['images'] == 210
        train_batches = datasets['train']['batches']
        dm.teardown(stage=TrainerFn.FITTING)
        assert dm._has_teardown_fit
        result = {'ok': True, 'datasets': datasets, 'effective_batch_size': 256,
                  'enum_setup_teardown_tracking_passed': True,
                  'hook_override_flags_preserved': original_overrides,
                  'accumulate_grad_batches': 8,
                  'expected_optimizer_steps_per_epoch': math.ceil(train_batches/8),
                  'expected_optimizer_steps_total': math.ceil(train_batches/8)*50,
                  'remaining_epochs': 20, 'remaining_batches': 5000, 'remaining_optimizer_steps': 640,
                  'resume_epoch': 30, 'resume_global_step': 960,
                  'resume_optimizer_hash': resume_optimizer_hash,
                  'resume_scheduler_hash': resume_scheduler_hash,
                  'resume_lr': [g['lr'] for g in saved_resume['optimizer_states'][0]['param_groups']],
                  'legacy_scheduler_total_steps_formula': train_batches*50//8,
                  'checkpoint_policy': manifest['checkpoint_policy'],
                  'actual_precision': 32, 'test_evaluation_executed': False}
        write_json(ROOT/'preflight_result.json', result)
        progress('prepared', **result)
        print('PREPARED', json.dumps(result), flush=True)
        sys.exit(0)

    assert not (RUN/'logs').exists(), 'Use the existing run only through a separately audited resume entry'
    assert json.loads((ROOT/'preflight_result.json').read_text())['ok']
    model = ViLTransformerSS(config)
    initial_backbone_sha = parameter_hash(model.model.state_dict().items())
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(RUN/'checkpoints'),
        filename='epoch={epoch:02d}-step={step}', auto_insert_metric_name=False,
        save_top_k=1, monitor='val/the_metric', mode='max', save_last=True, verbose=True)
    # Native PL 1.4 restores only best score/path; seed its full top-k bookkeeping.
    checkpoint_dir = RUN/'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    isolated_best = checkpoint_dir/old_best_path.name
    assert not isolated_best.exists()
    shutil.copy2(old_best_path, isolated_best)
    assert sha(isolated_best) == sha(old_best_path)
    original_checkpoint_load = checkpoint.on_load_checkpoint
    def restore_checkpoint_callback(trainer, module, callback_state):
        remapped = copy.deepcopy(callback_state)
        remapped['best_model_path'] = str(isolated_best)
        original_checkpoint_load(trainer, module, remapped)
        checkpoint.best_k_models = {str(isolated_best): checkpoint.best_model_score}
        checkpoint.kth_best_model_path = str(isolated_best)
        checkpoint.kth_value = checkpoint.best_model_score
        checkpoint.last_model_path = str(checkpoint_dir/'last.ckpt')
    checkpoint.on_load_checkpoint = restore_checkpoint_callback
    logger = pl.loggers.TensorBoardLogger(str(RUN/'logs'), name='plain_prior_scorer', version='run_01')
    events = (RUN/'training_events.jsonl').open('a', encoding='utf-8')
    losses = (RUN/'batch_losses.jsonl').open('a', encoding='utf-8')

    def event(kind, **values):
        row = {'event': kind, 'elapsed_seconds': time.monotonic()-started, **values}
        events.write(json.dumps(row, ensure_ascii=False)+'\n')
        events.flush()
        os.fsync(events.fileno())

    class TrainingAudit(pl.Callback):
        def on_train_start(self, trainer, module):
            assert all(not p.requires_grad for p in module.model.parameters())
            self.trainable = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
            self.initial_trainable_sha = parameter_hash(self.trainable)
            assert trainer.current_epoch == 30 and trainer.global_step == 960
            expected_trainable_hash = parameter_hash((name, saved_resume['state_dict'][name])
                                                       for name, _ in self.trainable)
            assert self.initial_trainable_sha == expected_trainable_hash
            assert initial_backbone_sha == previous_start['initial_backbone_sha256']
            assert parameter_hash(module.model.state_dict().items()) == initial_backbone_sha
            restored_optimizer_hash = state_hash(trainer.optimizers[0].state_dict())
            restored_scheduler_hash = state_hash(trainer.lr_schedulers[0]['scheduler'].state_dict())
            assert restored_optimizer_hash == resume_optimizer_hash
            assert restored_scheduler_hash == resume_scheduler_hash
            assert float(checkpoint.best_model_score) == 80.25
            assert set(checkpoint.best_k_models) == {str(isolated_best)}
            restore_audit = {'verified': True, 'epoch': trainer.current_epoch,
                'global_step': trainer.global_step, 'trainable_parameters_match_checkpoint': True,
                'optimizer_state_sha256': restored_optimizer_hash,
                'scheduler_state_sha256': restored_scheduler_hash,
                'lr': [g['lr'] for g in trainer.optimizers[0].param_groups],
                'scheduler_last_epoch': trainer.lr_schedulers[0]['scheduler'].last_epoch,
                'retained_previous_best_score': float(checkpoint.best_model_score)}
            write_json(ROOT/'restore_audit.json', restore_audit)
            self.backwards_checked = 0
            self.batches_seen = 0
            self.train_started = time.monotonic()
            train_loader = trainer.train_dataloader
            sampler = getattr(train_loader, 'sampler', None)
            event('train_start', trainable_parameter_count=sum(p.numel() for _, p in self.trainable),
                  initial_backbone_sha256=initial_backbone_sha,
                  initial_trainable_sha256=self.initial_trainable_sha,
                  resume_restore_audit=restore_audit,
                  sampler_class=type(sampler).__name__,
                  sampler_shuffle=getattr(sampler, 'shuffle', None),
                  train_batches=trainer.num_training_batches,
                  scheduler_state=trainer.lr_schedulers[0]['scheduler'].state_dict())
            progress('training', epoch=trainer.current_epoch, batch=0, optimizer_steps=trainer.global_step)

        def on_after_backward(self, trainer, module):
            if self.batches_seen % 25 == 0:
                grads = [(n, p.grad) for n, p in module.named_parameters() if p.grad is not None]
                assert grads and all(bool(torch.isfinite(g).all()) for _, g in grads)
                assert not any(n.startswith('model.') for n, _ in grads)
                assert any(n.startswith('agent_proj.') for n, _ in grads)
                self.backwards_checked += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
            loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
            if loss is not None:
                loss = float(loss.detach().cpu())
                assert math.isfinite(loss)
            self.batches_seen += 1
            losses.write(json.dumps({'epoch': trainer.current_epoch, 'batch': batch_idx,
                'optimizer_step': trainer.global_step, 'loss': loss,
                'lr': trainer.optimizers[0].param_groups[0]['lr']})+'\n')
            if self.batches_seen % 25 == 0 or batch_idx == 0:
                losses.flush()
                os.fsync(losses.fileno())
                progress('training', epoch=trainer.current_epoch, batch=batch_idx+1,
                         optimizer_steps=trainer.global_step, batches_seen=self.batches_seen,
                         latest_loss=loss, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                print('TRAIN_PROGRESS', trainer.current_epoch, batch_idx+1,
                      'step', trainer.global_step, 'loss', loss, flush=True)

        def on_validation_start(self, trainer, module):
            progress('validating', epoch=trainer.current_epoch, optimizer_steps=trainer.global_step)

        def on_validation_end(self, trainer, module):
            metrics = {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
                       if isinstance(v, torch.Tensor) and v.numel() == 1}
            assert all(math.isfinite(v) for v in metrics.values())
            event('validation_end', epoch=trainer.current_epoch, optimizer_step=trainer.global_step,
                  metrics=metrics, checkpoint_snapshot_stage='before_epoch_checkpoint_save',
                  observed_best_model_path=checkpoint.best_model_path,
                  observed_last_model_path=checkpoint.last_model_path)
            progress('epoch_complete', completed_epochs=trainer.current_epoch+1,
                     optimizer_steps=trainer.global_step, metrics=metrics)
            print('EPOCH_COMPLETE', trainer.current_epoch+1, json.dumps(metrics), flush=True)

        def on_train_end(self, trainer, module):
            final_backbone_sha = parameter_hash(module.model.state_dict().items())
            assert final_backbone_sha == initial_backbone_sha
            final_trainable_sha = parameter_hash(self.trainable)
            assert final_trainable_sha != self.initial_trainable_sha
            status.update(backbone_unchanged=True, initial_backbone_sha256=initial_backbone_sha,
                          final_backbone_sha256=final_backbone_sha,
                          initial_trainable_sha256=self.initial_trainable_sha,
                          final_trainable_sha256=final_trainable_sha,
                          gradient_checks_passed=self.backwards_checked,
                          batches_seen=self.batches_seen)

    audit = TrainingAudit()
    trainer = pl.Trainer(**trainer_settings, logger=logger,
        default_root_dir=str(RUN), callbacks=[MyBackboneFinetuning(), checkpoint,
            pl.callbacks.LearningRateMonitor(logging_interval='step'), audit])
    torch.cuda.reset_peak_memory_stats()
    progress('initializing_trainer')
    trainer.fit(model, datamodule=dm)
    torch.cuda.synchronize()
    losses.flush()
    events.flush()
    assert status['completed_epochs'] == 50, status.get('completed_epochs')
    assert audit.batches_seen == 5000 and audit.backwards_checked == 200
    assert trainer.global_step == 1600
    assert checkpoint.last_model_path and checkpoint.best_model_path
    paths = {}
    for label, value in [('last', checkpoint.last_model_path), ('validation_best', checkpoint.best_model_path)]:
        path = Path(value)
        paths[label] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    status.update(ok=True, completed_epochs=50, trainer_current_epoch=trainer.current_epoch,
                  optimizer_steps=trainer.global_step,
                  resumed_epochs=20, resumed_optimizer_steps=640,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), checkpoints=paths,
                  best_monitor_value=float(checkpoint.best_model_score.detach().cpu()))
    progress('complete')
    write_json(ROOT/'training_result.json', status)
    print('TRAINING_COMPLETE', json.dumps(status, ensure_ascii=False), flush=True)
except Exception as error:
    status.update(ok=False, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    progress('failed')
    write_json(ROOT/('prepare_failure.json' if args.prepare_only else 'training_failure.json'), status)
    print('FAILED', json.dumps(status, ensure_ascii=False), flush=True)
    sys.exit(1)
