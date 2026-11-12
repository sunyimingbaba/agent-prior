#!/usr/bin/env python3
"""Generate and verify the fixed train/val plain-prior cache, with durable records."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
out = args.out.resolve()
if not str(out).startswith('/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_'):
    raise ValueError('Use the dedicated new private full-cache directory')
out.mkdir(parents=True, exist_ok=True)
lock = (out/'generation.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
for relative in ['priors', 'traces', 'incomplete', 'failures', 'scratch/huggingface',
                 'scratch/cache/torch/kernels', 'scratch/matplotlib']:
    (out/relative).mkdir(parents=True, exist_ok=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'HF_HOME': str(out/'scratch/huggingface'), 'TMPDIR': str(out/'scratch'),
                   'MPLCONFIGDIR': str(out/'scratch/matplotlib'),
                   'XDG_CACHE_HOME': str(out/'scratch/cache'),
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
BASE = Path('/private/AGSE_E1_20261110/legacy')
ASSETS = BASE/'assets'
MODEL = ASSETS/'models/Qwen2.5-VL-7B-Instruct'
sys.path.insert(0, str(BASE/'CUP'))
sys.path.insert(0, str(BASE/'CUP/agent'))
started = time.monotonic()
manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
rows = manifest['rows']
assert len(rows) == 1890 and manifest['split_counts'] == {'train': 1680, 'val': 210}
completed = {}
records_path = out/'records.jsonl'
status = {'status': 'starting', 'total': len(rows), 'completed': 0,
          'formal_scorer_training_started': False, 'existing_sources_modified': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def progress(phase):
    status.update({'status': phase, 'completed': len(completed),
                   'elapsed_seconds_this_attempt': time.monotonic()-started,
                   'last_update_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    atomic_json(out/'progress.json', status)


def validate_record(record, expected):
    for name in ['split', 'row', 'image', 'caption', 'image_rgb_sha256']:
        assert record[name] == expected[name], (name, record['split'], record['row'])
    prior_path = out/record['prior_file']
    assert sha_file(prior_path) == record['prior_file_sha256']
    hidden = torch.load(prior_path, map_location='cpu', weights_only=True)
    assert list(hidden.shape) == [3584] and str(hidden.dtype) == 'torch.bfloat16'
    assert bool(torch.isfinite(hidden).all()) and tensor_sha(hidden) == record['tensor_sha256']
    trace_path = out/record['trace_file']
    assert sha_file(trace_path) == record['trace_file_sha256']
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    assert len(trace) == 1 and trace[-1]['final_state_sha256'] == record['tensor_sha256']
    assert trace[-1]['generation_kwargs']['do_sample'] is False and record['tools_used'] == 0
    return hidden


try:
    import pyarrow as pa
    import torch
    from PIL import Image
    from rollout import load_mllm, rollout_one, PLAIN_PROMPT

    torch.set_num_threads(4)
    assert torch.__version__ == manifest['expected_torch']
    assert metadata.version('transformers') == manifest['expected_transformers']
    assert sha_file(BASE/'CUP/agent/rollout.py') == manifest['rollout_source_sha256']
    fingerprint = {'manifest_sha256': sha_file(args.manifest),
        'generator_sha256': sha_file(Path(__file__)),
        'rollout_source_sha256': manifest['rollout_source_sha256'],
        'model_asset_hashes': manifest['model_asset_hashes'],
        'torch': torch.__version__, 'transformers': metadata.version('transformers'),
        'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'system_prompt': PLAIN_PROMPT, 'lora_path': '', 'tools_available': [],
        'seed_per_image': 0, 'temperature': 0.0, 'max_new_tokens': 64,
        'max_rounds': 3, 'injected_h_end': False, 'generation_batch_size': 1,
        'state_rule': 'last generation call, final decoding step, final layer, final position'}
    settings_path = out/'settings.json'
    if settings_path.exists():
        assert json.loads(settings_path.read_text(encoding='utf-8')) == fingerprint
    else:
        atomic_json(settings_path, fingerprint)
    tables = {}
    progress('verifying_source_assets')
    for split in ['train', 'val']:
        p = ASSETS/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow'
        assert sha_file(p) == manifest['arrow_sha256'][split]
        tables[split] = pa.ipc.open_file(str(p)).read_all()
        assert len(tables[split]) == manifest['split_counts'][split]
    for name, expected in manifest['model_asset_hashes'].items():
        p = MODEL/name
        assert p.stat().st_size == expected['bytes'] and sha_file(p) == expected['sha256'], name
        if name.endswith('.safetensors'):
            print('VERIFIED_MODEL_SHARD', name, flush=True)
    expected_rows = {(r['split'], r['row']): r for r in rows}
    if records_path.exists():
        for line in records_path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            key = (r['split'], r['row'])
            assert key not in completed
            validate_record(r, expected_rows[key])
            completed[key] = r
    status['reused_validated_records'] = len(completed)
    progress('loading_model')
    print('LOADING_BASE_QWEN', 'already_completed', len(completed), flush=True)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, processor = load_mllm(str(MODEL), 'cuda', lora_path='')
    assert not getattr(model, 'peft_config', None)
    status.update({'model_class': type(model).__name__,
                   'model_dtype': str(next(model.parameters()).dtype),
                   'processor_class': type(processor).__name__})
    original_generate = model.generate
    active_calls = []

    def traced_generate(*positional, **kwargs):
        outputs = original_generate(*positional, **kwargs)
        prompt_length = kwargs['input_ids'].shape[1]
        new_ids = outputs.sequences[0, prompt_length:]
        steps = len(outputs.hidden_states)
        selected = outputs.hidden_states[-1][-1][:, -1, :].detach().cpu()
        r = {'prompt_tokens': prompt_length, 'generated_tokens': new_ids.numel(),
             'decoding_steps': steps, 'generated_token_ids': new_ids.detach().cpu().tolist(),
             'generated_text': processor.decode(new_ids, skip_special_tokens=True),
             'final_state_shape_before_squeeze': list(selected.shape),
             'final_state_sha256': tensor_sha(selected.squeeze(0)),
             'final_hidden_step_shape': list(outputs.hidden_states[-1][-1].shape),
             'generation_kwargs': {k: kwargs[k] for k in ['max_new_tokens', 'do_sample',
                 'temperature', 'output_hidden_states', 'return_dict_in_generate']}}
        if steps == new_ids.numel() and steps > 0:
            position = prompt_length+steps-2
            token_id = int(outputs.sequences[0, position].item())
            r.update({'state_input_sequence_position': position, 'state_input_token_id': token_id,
                      'state_input_token': processor.decode([token_id], skip_special_tokens=False)})
        r['last_emitted_token_id'] = int(new_ids[-1].item())
        r['last_emitted_token'] = processor.decode([r['last_emitted_token_id']], skip_special_tokens=False)
        active_calls.append(r)
        return outputs

    model.generate = traced_generate
    progress('generating')
    with records_path.open('a', encoding='utf-8') as record_file:
        for item in rows:
            key = (item['split'], item['row'])
            if key in completed:
                continue
            split, index = key
            table = tables[split]
            assert table['path'][index].as_py() == item['image']
            assert table['caption'][index].as_py()[0] == item['caption']
            raw_image = table['image'][index].as_py()
            assert hashlib.sha256(raw_image).hexdigest() == item['image_bytes_sha256']
            image = Image.open(io.BytesIO(raw_image)).convert('RGB')
            assert list(image.size) == item['image_size']
            assert hashlib.sha256(image.tobytes()).hexdigest() == item['image_rgb_sha256']
            directory = f'ucm_caption_karpathy_label_{split}'
            prior_path = out/'priors'/directory/f'{index}.pt'
            trace_path = out/'traces'/split/f'{index}.json'
            text_path = out/'traces'/split/f'{index}.txt'
            for p in [prior_path, trace_path, text_path]:
                p.parent.mkdir(parents=True, exist_ok=True)
            existing = [p for p in [prior_path, trace_path, text_path] if p.exists()]
            if existing:
                recovery = out/'incomplete'/f'{split}_{index}_{time.time_ns()}'
                recovery.mkdir()
                for p in existing:
                    shutil.move(str(p), str(recovery/p.name))
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.cuda.reset_peak_memory_stats()
            active_calls = []
            trace_text = io.StringIO()
            tick = time.monotonic()
            status['current_input'] = {'split': split, 'row': index, 'image': item['image']}
            try:
                with contextlib.redirect_stdout(trace_text):
                    hidden, used, filtered = rollout_one(model, processor, image, item['caption'], {},
                        max_rounds=3, max_new_tokens=64, device='cuda', verbose=True,
                        injected_h_end=False, system_prompt=PLAIN_PROMPT, temperature=0.0)
            finally:
                text_path.write_text(trace_text.getvalue(), encoding='utf-8')
                atomic_json(trace_path, active_calls)
            torch.cuda.synchronize()
            assert list(hidden.shape) == [3584] and bool(torch.isfinite(hidden).all())
            assert str(hidden.dtype) == 'torch.bfloat16' and used == filtered == 0
            assert len(active_calls) == 1 and tensor_sha(hidden) == active_calls[-1]['final_state_sha256']
            anchor = manifest['smoke_tensor_anchors'].get(split+':'+str(index))
            if anchor:
                assert tensor_sha(hidden) == anchor, 'Small-run control vector changed'
            temporary = prior_path.with_name(prior_path.name+'.partial')
            torch.save(hidden, temporary)
            os.replace(temporary, prior_path)
            assert torch.equal(hidden, torch.load(prior_path, map_location='cpu', weights_only=True))
            r = {**item, 'prior_file': str(prior_path.relative_to(out)),
                 'prior_file_sha256': sha_file(prior_path), 'tensor_sha256': tensor_sha(hidden),
                 'shape': list(hidden.shape), 'dtype': str(hidden.dtype), 'finite': True,
                 'trace_file': str(trace_path.relative_to(out)), 'trace_file_sha256': sha_file(trace_path),
                 'generated_text': active_calls[-1]['generated_text'],
                 'generated_tokens': active_calls[-1]['generated_tokens'],
                 'last_emitted_token': active_calls[-1]['last_emitted_token'],
                 'state_input_token': active_calls[-1].get('state_input_token'),
                 'reached_token_limit': active_calls[-1]['generated_tokens'] == 64,
                 'tools_used': used, 'tools_filtered': filtered,
                 'smoke_anchor': bool(anchor), 'smoke_anchor_match': True if anchor else None,
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                 'elapsed_seconds': time.monotonic()-tick}
            record_file.write(json.dumps(r, ensure_ascii=False)+'\n')
            record_file.flush()
            os.fsync(record_file.fileno())
            completed[key] = r
            if len(completed) % 25 == 0 or len(completed) == 1:
                progress('generating')
                print('PROGRESS', len(completed), '/', len(rows), split, index,
                      'gpu_gib', round(r['peak_allocated_bytes']/1024**3, 3), flush=True)
    progress('verifying_all_outputs')
    for item in rows:
        validate_record(completed[(item['split'], item['row'])], item)
    for split, expected_count in manifest['split_counts'].items():
        directory = out/'priors'/f'ucm_caption_karpathy_label_{split}'
        assert {p.name for p in directory.glob('*.pt')} == {f'{i}.pt' for i in range(expected_count)}
    values = list(completed.values())
    status.update({'ok': True, 'split_counts': manifest['split_counts'],
                   'all_1890_records_verified': True,
                   'smoke_anchor_matches': sum(r['smoke_anchor'] and r['smoke_anchor_match'] for r in values),
                   'reached_token_limit_count': sum(r['reached_token_limit'] for r in values),
                   'end_token_counts': {token: sum(r['last_emitted_token'] == token for r in values)
                                        for token in sorted({r['last_emitted_token'] for r in values})},
                   'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in values)})
    progress('complete')
    atomic_json(out/'metadata.json', status)
    print('COMPLETE', json.dumps(status, ensure_ascii=False), flush=True)
except Exception as error:
    status.update({'ok': False, 'error': f'{type(error).__name__}: {error}',
                   'traceback': traceback.format_exc()})
    progress('failed')
    atomic_json(out/'failures'/f'{time.time_ns()}.json', status)
    atomic_json(out/'metadata.json', status)
    print('FAILED', json.dumps(status, ensure_ascii=False), flush=True)
    sys.exit(1)
