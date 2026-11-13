#!/usr/bin/env python3
"""Matched adapter/prompt/feedback diagnostics, preserving all original assets."""
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
assert ROOT.parent == Path('/private/AGSE_E3_1_rebuild_20261112')
BASE = Path('/private/AGSE_E1_20261110/legacy')
ASSETS = BASE/'assets'
PLAN = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
OUT = ROOT/'outputs'
OUT.mkdir(exist_ok=False)
for folder in ['states', 'traces', 'scratch/hf', 'scratch/cache', 'scratch/matplotlib']:
    (OUT/folder).mkdir(parents=True)
os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HOME': str(OUT/'scratch/hf'), 'TMPDIR': str(OUT/'scratch'),
    'MPLCONFIGDIR': str(OUT/'scratch/matplotlib'), 'XDG_CACHE_HOME': str(OUT/'scratch/cache'),
    'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
sys.path.insert(0, str(BASE/'CUP'))
sys.path.insert(0, str(BASE/'CUP/agent'))
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from rollout import load_mllm, rollout_one, SYSTEM_PROMPT, PLAIN_PROMPT
from tools import scene_classify_with_conf
from prompt_clip import clip


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


started = time.monotonic()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
seed()
x = torch.randn(16, 16, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
dump(ROOT/'environment_witness.json', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
    'transformers': metadata.version('transformers'), 'peft': metadata.version('peft'),
    'seed0_kernel_loss': float(loss.detach()), 'finite_loss_and_gradient': True,
    'sitecustomize_sha256': sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py'),
    'warm_reuse': True, 'environment_packages_modified': False})
del x, loss
manifest = json.loads((ROOT/'base_input_manifest.json').read_text())
model_dir = ASSETS/'models/Qwen2.5-VL-7B-Instruct'
source_paths = [BASE/'CUP/agent/rollout.py', BASE/'CUP/agent/tools.py']
source_hashes = {str(p): sha(p) for p in source_paths}
assert sha(source_paths[0]) == PLAN['rollout_sha256']
asset_hashes = {}
for name, expected in manifest['model_asset_hashes'].items():
    path = model_dir/name
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    asset_hashes[str(path)] = expected['sha256']
for path, digest in [(ASSETS/'ucm_main_lora/adapter_model.safetensors', PLAN['lora_weights_sha256']),
    (ASSETS/'models/ViT-L-14.pt', PLAN['clip_sha256']), (ASSETS/'ucm_scene_cls.pt', PLAN['scene_head_sha256'])]:
    assert sha(path) == digest
    asset_hashes[str(path)] = digest
arrow = ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow'
assert sha(arrow) == manifest['arrow_sha256']['val']
table = pa.ipc.open_file(str(arrow)).read_all()
images = {}
for item in PLAN['images']:
    raw = table['image'][item['row']].as_py()
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    assert table['path'][item['row']].as_py() == item['image']
    assert table['caption'][item['row']].as_py()[0] == item['caption']
    assert hashlib.sha256(raw).hexdigest() == item['image_bytes_sha256']
    assert hashlib.sha256(image.tobytes()).hexdigest() == item['image_rgb_sha256']
    images[item['row']] = image
seed()
print('LOADING_MATCHED_BASE_AND_ADAPTER', flush=True)
model, processor = load_mllm(str(model_dir), 'cuda', str(ASSETS/'ucm_main_lora'))
model.eval().requires_grad_(False)
loaded = get_peft_model_state_dict(model)
saved = load_file(str(ASSETS/'ucm_main_lora/adapter_model.safetensors'), device='cpu')
assert set(loaded) == set(saved)
adapter_audit = []
for name, value in saved.items():
    actual = loaded[name].detach().cpu()
    assert value.shape == actual.shape and torch.isfinite(value).all() and torch.isfinite(actual).all()
    match = torch.equal(actual, value.to(actual.dtype))
    assert match, name
    adapter_audit.append({'name': name, 'shape': list(value.shape), 'saved_dtype': str(value.dtype),
        'loaded_dtype': str(actual.dtype), 'loaded_matches_saved_after_dtype_cast': match,
        'rms': float(value.float().square().mean().sqrt()), 'max_abs': float(value.float().abs().max())})
dump(ROOT/'adapter_loading_audit.json', {'ok': True, 'tensor_count': len(adapter_audit),
    'all_keys_shapes_and_values_match': True, 'active_adapters': model.active_adapters,
    'adapter_config': json.loads((ASSETS/'ucm_main_lora/adapter_config.json').read_text()), 'tensors': adapter_audit})
del loaded, saved
tool_clip, preprocess = clip.load(str(ASSETS/'models/ViT-L-14.pt'), device='cuda', jit=False)
tool_clip = tool_clip.float().eval().requires_grad_(False)
head_saved = torch.load(ASSETS/'ucm_scene_cls.pt', map_location='cpu', weights_only=True)
head = torch.nn.Linear(head_saved['feat_dim'], len(head_saved['classes']))
head.load_state_dict(head_saved['head'], strict=True)
head = head.cuda().float().eval().requires_grad_(False)
classes = list(head_saved['classes'])
tool_responses = {}
for row, image in images.items():
    text, confidence = scene_classify_with_conf(image, model=tool_clip, preprocess=preprocess,
        cls_head=head, cls_names=classes)
    tool_responses[row] = {'text': text, 'confidence': float(confidence)}
dump(ROOT/'tool_responses.json', tool_responses)


def messages_for(item, prompt='tool'):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT if prompt == 'tool' else PLAIN_PROMPT}]
    if prompt == 'tool':
        messages += [
            {'role': 'user', 'content': [{'type': 'text', 'text': '查询文本: There is a harbor with several ships.'}]},
            {'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'},
            {'role': 'user', 'content': '<result>harbor (0.61), port (0.24), ship (0.08)</result>'},
            {'role': 'assistant', 'content': '<answer>图中是一座港口，与查询文本相符。</answer>'}]
    messages.append({'role': 'user', 'content': [{'type': 'image'},
        {'type': 'text', 'text': '查询文本: '+item['caption']}]})
    return messages


def logit_summary(logits):
    logits = logits.detach().float()
    probabilities = logits.softmax(-1)
    values, ids = logits.topk(10)
    return {'top10': [{'id': int(i), 'text': processor.decode([int(i)], skip_special_tokens=False),
                       'logit': float(v) if math.isfinite(float(v)) else None,
                       'probability': float(probabilities[i])} for v, i in zip(values, ids)],
        'selected': {str(i): {'token': processor.decode([i], skip_special_tokens=False),
            'logit': float(logits[i]) if math.isfinite(float(logits[i])) else None,
            'probability': float(probabilities[i]),
            'rank1': 1+int((logits > logits[i]).sum())} for i in [151645, 151643, 151644, 18724, 151657]},
        'entropy': float(-(probabilities*probabilities.clamp_min(1e-40).log()).sum())}


active_calls, initial_inputs, full_logits = [], {}, {}
current_case, current_row, keep_full = '', None, False
original_generate = model.generate


def traced_generate(*args, **kwargs):
    if not active_calls and current_row not in initial_inputs and '_tool_' in current_case:
        initial_inputs[current_row] = {k: v.detach().clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
    outputs = original_generate(*args, **kwargs, output_scores=True, output_logits=True)
    prompt_length = kwargs['input_ids'].shape[1]
    new_ids = outputs.sequences[0, prompt_length:].cpu()
    state = outputs.hidden_states[-1][-1][0, -1, :].detach().cpu()
    assert list(state.shape) == [3584] and torch.isfinite(state).all()
    positions = sorted({0, len(new_ids)-1} | {i for i, t in enumerate(new_ids.tolist()) if t == 18724})
    if len(positions) > 3:
        positions = [positions[0], positions[1], positions[-1]]
    step_records = [{'step0': step, 'emitted_id': int(new_ids[step]),
        'raw': logit_summary(outputs.logits[step][0]),
        'processed': logit_summary(outputs.scores[step][0])} for step in positions]
    if keep_full and not active_calls:
        full_logits[current_case] = outputs.logits[0][0].detach().cpu().float()
    active_calls.append({'prompt_tokens': prompt_length, 'prompt_tail_ids': kwargs['input_ids'][0,-40:].cpu().tolist(),
        'generated_token_ids': new_ids.tolist(), 'generated_text': processor.decode(new_ids, skip_special_tokens=True),
        'raw_decoded_text': processor.decode(new_ids, skip_special_tokens=False),
        'state_sha256': tensor_sha(state), 'first_and_selected_steps': step_records})
    return outputs


model.generate = traced_generate
records = []


def save_case(role, adapter, item, variant, temperature, hidden, events, elapsed):
    path = OUT/'states'/f'{current_case}.pt'
    torch.save(hidden, path)
    assert torch.equal(hidden, torch.load(path, map_location='cpu', weights_only=True))
    trace_path = OUT/'traces'/f'{current_case}.json'
    dump(trace_path, active_calls)
    trajectory = '\n'.join(c['generated_text'] for c in active_calls)
    first = active_calls[0]['first_and_selected_steps'][0]
    record = {'id': current_case, 'role': role, 'adapter': adapter, 'row': item['row'], 'image': item['image'],
        'variant': variant, 'temperature': temperature, 'seed': 0, 'trajectory': trajectory,
        'empty': not trajectory.strip(), 'contains_addCriterion': 'addCriterion' in trajectory,
        'has_complete_answer': __import__('re').search(r'<answer>.*?</answer>', trajectory, __import__('re').DOTALL) is not None,
        'first_emitted_id': active_calls[0]['generated_token_ids'][0],
        'first_step_raw_eos_probability': first['raw']['selected']['151645']['probability'],
        'first_step_processed_eos_probability': first['processed']['selected']['151645']['probability'],
        'rounds': len(active_calls), 'tools': events, 'state_file': str(path.relative_to(ROOT)),
        'state_sha256': tensor_sha(hidden), 'trace_file': str(trace_path.relative_to(ROOT)),
        'trace_sha256': sha(trace_path), 'elapsed_seconds': elapsed}
    records.append(record)
    with (OUT/'records.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    dump(ROOT/'progress.json', {'completed': len(records), 'planned': 42, 'last_case': current_case})
    print('DIAGNOSED', len(records), '/42', current_case, 'empty', record['empty'], 'answer', record['has_complete_answer'], flush=True)


for adapter in ['enabled', 'disabled']:
    context = contextlib.nullcontext() if adapter == 'enabled' else model.disable_adapter()
    with context:
        for item in PLAN['images']:
            current_row = item['row']
            for prompt in ['tool', 'plain']:
                for temperature in [0.0, 1.0]:
                    current_case = f'{current_row}_{adapter}_{prompt}_t{temperature:g}'
                    keep_full = prompt == 'tool' and temperature == 0
                    seed()
                    active_calls, events = [], []

                    def scene_tool(image):
                        text, confidence = scene_classify_with_conf(image, model=tool_clip, preprocess=preprocess,
                            cls_head=head, cls_names=classes)
                        events.append({'text': text, 'confidence': float(confidence)})
                        return text if confidence >= 0 else None

                    start = time.monotonic()
                    with contextlib.redirect_stdout(io.StringIO()):
                        hidden, used, filtered = rollout_one(model, processor, images[current_row], item['caption'],
                            {'scene_classify': scene_tool} if prompt == 'tool' else {},
                            max_rounds=3, max_new_tokens=64, device='cuda', verbose=True,
                            injected_h_end=False, system_prompt=SYSTEM_PROMPT if prompt == 'tool' else PLAIN_PROMPT,
                            temperature=temperature)
                    assert used+filtered == len(events) and tensor_sha(hidden) == active_calls[-1]['state_sha256']
                    save_case('native', adapter, item, prompt, temperature, hidden, events, time.monotonic()-start)

for adapter in ['enabled', 'disabled']:
    context = contextlib.nullcontext() if adapter == 'enabled' else model.disable_adapter()
    with context:
        for item in PLAN['images']:
            current_row = item['row']
            base_inputs = initial_inputs[current_row]
            for variant in ['raw_unclosed', 'raw_closed', 'chat_closed']:
                current_case = f'{current_row}_{adapter}_feedback_{variant}'
                keep_full = variant == 'raw_unclosed'
                result = '<result>'+tool_responses[current_row]['text']+'</result>'
                if variant.startswith('raw_'):
                    call = '<tool_call>scene_classify'+('</tool_call>' if variant == 'raw_closed' else '')
                    feedback = processor(text=[call+'\n'+result+'\n'], return_tensors='pt').input_ids.cuda()
                    inputs = dict(base_inputs)
                    inputs['input_ids'] = torch.cat([base_inputs['input_ids'], feedback], dim=1)
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                else:
                    messages = messages_for(item)+[{'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'},
                        {'role': 'user', 'content': result}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[images[current_row]], return_tensors='pt').to('cuda')
                seed()
                active_calls = []
                start = time.monotonic()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False, temperature=1.0,
                        output_hidden_states=True, return_dict_in_generate=True)
                hidden = outputs.hidden_states[-1][-1][0,-1,:].detach().cpu()
                save_case('fixed_feedback', adapter, item, variant, 0.0, hidden, [tool_responses[current_row]], time.monotonic()-start)
                del outputs

anchor = PLAN['images'][0]
text = processor.apply_chat_template(messages_for(anchor), tokenize=False, add_generation_prompt=True)
expected_inputs = processor(text=[text], images=[images[anchor['row']]], return_tensors='pt')
processor_checks = []
for name, kwargs in [('default', {}), ('use_fast=False', {'use_fast': False}),
                     ('fix_mistral_regex=True', {'fix_mistral_regex': True})]:
    try:
        alternative = processor if name == 'default' else AutoProcessor.from_pretrained(str(model_dir), **kwargs)
        rendered = alternative.apply_chat_template(messages_for(anchor), tokenize=False, add_generation_prompt=True)
        values = alternative(text=[rendered], images=[images[anchor['row']]], return_tensors='pt')
        comparisons = {}
        for key in expected_inputs:
            a, b = expected_inputs[key], values[key]
            comparisons[key] = {'shape_default': list(a.shape), 'shape_variant': list(b.shape),
                'bitwise_equal': torch.equal(a,b), 'max_abs_delta': float((a.float()-b.float()).abs().max()) if a.shape == b.shape else None}
        processor_checks.append({'variant': name, 'ok': True, 'rendered_text_equal': rendered == text,
            'tokenizer_class': type(alternative.tokenizer).__name__, 'image_processor_class': type(alternative.image_processor).__name__,
            'comparisons': comparisons})
    except Exception as error:
        processor_checks.append({'variant': name, 'ok': False, 'error': repr(error)})
dump(ROOT/'processor_input_checks.json', processor_checks)
torch.save(full_logits, OUT/'selected_first_step_logits.pt')
assert len(records) == 42 and all(p.grad is None and not p.requires_grad for p in model.parameters())
assert all(sha(path) == digest for path, digest in source_hashes.items())
dump(ROOT/'verification.json', {'ok': True, 'records': len(records), 'all_states_finite_3584': True,
    'all_state_save_reload_bitwise_equal': True, 'adapter_load_verified': True,
    'source_files_unchanged': source_hashes, 'input_asset_hashes': asset_hashes,
    'plan_sha256': sha(ROOT/'plan.json'), 'script_sha256': sha(__file__),
    'frozen_eval_no_parameter_updates': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'elapsed_seconds': time.monotonic()-started})
print('DIAGNOSIS_COMPLETE', flush=True)
