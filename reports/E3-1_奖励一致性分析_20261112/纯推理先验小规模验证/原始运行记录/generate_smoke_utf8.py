#!/usr/bin/env python3
"""Trace the existing plain rollout on four fixed images plus one repeat."""
import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
out = args.out.resolve()
if not str(out).startswith('/private/AGSE_E3_1_rebuild_20261112/') or out.exists():
    raise ValueError('Use a new output directory inside the private E3-1 root')
out.mkdir(parents=True)
for relative in ['priors', 'repeats', 'traces', 'scratch']:
    (out / relative).mkdir()
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HOME': str(out / 'scratch/huggingface'), 'TMPDIR': str(out / 'scratch'), 'MPLCONFIGDIR': str(out / 'scratch/matplotlib'), 'XDG_CACHE_HOME': str(out / 'scratch/cache'), 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
BASE = Path('/private/AGSE_E1_20261110/legacy')
ASSETS = BASE / 'assets'
sys.path.insert(0, str(BASE / 'CUP'))
sys.path.insert(0, str(BASE / 'CUP/agent'))
SELECTION = [('train', 0, '1.tif'), ('train', 840, '1043.tif'), ('train', 1679, '2080.tif'), ('val', 105, '1084.tif')]

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def tensor_sha(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()
started = time.monotonic()
records = []
status = {'purpose': 'small plain-prior pipeline validation', 'formal_training_started': False, 'existing_sources_modified': False, 'selection': SELECTION, 'generation_calls': 5, 'unique_images': 4, 'lora_path': '', 'tools_available': [], 'seed': 0, 'temperature': 0.0, 'max_new_tokens': 64, 'max_rounds': 3, 'injected_h_end': False, 'state_rule': 'last generation call, final decoding step, final layer, final position'}
try:
    import pyarrow as pa
    import torch
    from PIL import Image
    from rollout import load_mllm, rollout_one, PLAIN_PROMPT
    torch.set_num_threads(4)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    status.update({'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': metadata.version('transformers'), 'python': sys.version, 'system_prompt': PLAIN_PROMPT, 'rollout_source_sha256': sha_file(BASE / 'CUP/agent/rollout.py')})
    tables = {}
    inputs = []
    for split, index, expected_name in SELECTION:
        if split not in tables:
            arrow_path = ASSETS / 'ucm' / f'ucm_caption_karpathy_label_{split}.arrow'
            tables[split] = pa.ipc.open_file(str(arrow_path)).read_all()
            status.setdefault('arrow_sha256', {})[split] = sha_file(arrow_path)
        table = tables[split]
        assert table['path'][index].as_py() == expected_name
        image = Image.open(io.BytesIO(table['image'][index].as_py())).convert('RGB')
        inputs.append({'split': split, 'row': index, 'image': expected_name, 'caption': table['caption'][index].as_py()[0], 'image_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
    (out / 'inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'settings.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('LOADING_BASE_QWEN', flush=True)
    model, processor = load_mllm(str(ASSETS / 'models/Qwen2.5-VL-7B-Instruct'), 'cuda', lora_path='')
    assert not getattr(model, 'peft_config', None)
    status['model_class'] = type(model).__name__
    status['model_dtype'] = str(next(model.parameters()).dtype)
    status['processor_class'] = type(processor).__name__
    generate = model.generate
    active_calls = []

    def traced_generate(*positional, **kwargs):
        outputs = generate(*positional, **kwargs)
        prompt_length = kwargs['input_ids'].shape[1]
        new_ids = outputs.sequences[0, prompt_length:]
        steps = len(outputs.hidden_states)
        selected = outputs.hidden_states[-1][-1][:, -1, :].detach().cpu()
        record = {'prompt_tokens': prompt_length, 'generated_tokens': new_ids.numel(), 'decoding_steps': steps, 'generated_token_ids': new_ids.detach().cpu().tolist(), 'generated_text': processor.decode(new_ids, skip_special_tokens=True), 'final_state_shape_before_squeeze': list(selected.shape), 'final_state_sha256': tensor_sha(selected.squeeze(0)), 'final_hidden_step_shape': list(outputs.hidden_states[-1][-1].shape), 'generation_kwargs': {k: kwargs[k] for k in ['max_new_tokens', 'do_sample', 'temperature', 'output_hidden_states', 'return_dict_in_generate']}}
        if steps == new_ids.numel() and steps > 0:
            position = prompt_length + steps - 2
            state_token_id = int(outputs.sequences[0, position].item())
            record.update({'state_input_sequence_position': position, 'state_input_token_id': state_token_id, 'state_input_token': processor.decode([state_token_id], skip_special_tokens=False)})
        record['last_emitted_token_id'] = int(new_ids[-1].item())
        record['last_emitted_token'] = processor.decode([record['last_emitted_token_id']], skip_special_tokens=False)
        active_calls.append(record)
        return outputs
    model.generate = traced_generate
    original = None
    for call_index, item in enumerate(inputs + [inputs[0]]):
        is_repeat = call_index == len(inputs)
        split, index = (item['split'], item['row'])
        table = tables[split]
        image = Image.open(io.BytesIO(table['image'][index].as_py())).convert('RGB')
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()
        active_calls = []
        trace = io.StringIO()
        tick = time.monotonic()
        print('GENERATING', call_index, item['image'], 'repeat' if is_repeat else split, flush=True)
        try:
            with contextlib.redirect_stdout(trace):
                hidden, tools_used, tools_filtered = rollout_one(model, processor, image, item['caption'], {}, max_rounds=3, max_new_tokens=64, device='cuda', verbose=True, injected_h_end=False, system_prompt=PLAIN_PROMPT, temperature=0.0)
        finally:
            (out / 'traces' / f'{call_index:02d}.txt').write_text(trace.getvalue(), encoding='utf-8')
            (out / 'traces' / f'{call_index:02d}.json').write_text(json.dumps(active_calls, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        torch.cuda.synchronize()
        assert tuple(hidden.shape) == (3584,) and bool(torch.isfinite(hidden).all())
        assert tools_used == tools_filtered == 0 and len(active_calls) == 1
        assert tensor_sha(hidden) == active_calls[-1]['final_state_sha256']
        if is_repeat:
            prior_path = out / 'repeats/train_0_repeat.pt'
        else:
            prior_path = out / 'priors' / f'ucm_caption_karpathy_label_{split}' / f'{index}.pt'
            prior_path.parent.mkdir(exist_ok=True)
        torch.save(hidden, prior_path)
        reloaded = torch.load(prior_path, map_location='cpu', weights_only=True)
        assert torch.equal(hidden, reloaded)
        record = {**item, 'repeat': is_repeat, 'shape': list(hidden.shape), 'dtype': str(hidden.dtype), 'finite': True, 'l2_norm': float(hidden.float().norm()), 'tools_used': tools_used, 'prior_file': str(prior_path.relative_to(out)), 'prior_file_sha256': sha_file(prior_path), 'tensor_sha256': tensor_sha(hidden), 'trace_file': f'traces/{call_index:02d}.json', 'generated_text': active_calls[-1]['generated_text'], 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'elapsed_seconds': time.monotonic() - tick, 'matches_selected_generation_state': True, 'reload_equal': True}
        if call_index == 0:
            original = hidden.clone()
        if is_repeat:
            record.update({'repeat_tensor_equal': torch.equal(original, hidden), 'repeat_max_abs_difference': float((original.float() - hidden.float()).abs().max()), 'repeat_cosine_similarity': float(torch.nn.functional.cosine_similarity(original.float()[None], hidden.float()[None]).item())})
        records.append(record)
        (out / 'records.partial.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('DONE', call_index, item['image'], 'tokens', active_calls[-1]['generated_tokens'], 'gpu_gib', round(record['peak_allocated_bytes'] / 1024 ** 3, 3), flush=True)
    status.update({'ok': True, 'completed_calls': len(records), 'repeat_tensor_equal': records[-1]['repeat_tensor_equal'], 'repeat_max_abs_difference': records[-1]['repeat_max_abs_difference'], 'peak_allocated_bytes': max((r['peak_allocated_bytes'] for r in records))})
except Exception as error:
    status.update({'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(), 'completed_calls': len(records)})
status['elapsed_seconds'] = time.monotonic() - started
(out / 'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(out / 'metadata.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('RESULT', json.dumps(status, ensure_ascii=False), flush=True)
sys.exit(0 if status['ok'] else 1)
