#!/usr/bin/env python3
"""Original tool rollout, serial seeded candidates, greedy reference and repeats."""
import contextlib
import datetime
import hashlib
import importlib.metadata as metadata
import io
import json
import shutil
import time

from pilot_common import ROOT, BASE, ASSETS, REBUILD, PLAN, sha, tensor_sha, dump, setup_torch, seed

torch = setup_torch()
import pyarrow as pa
from PIL import Image
from rollout import load_mllm, SYSTEM_PROMPT
from messages_rollout import rollout_messages
from tools import scene_classify_with_conf
from prompt_clip import clip

out = ROOT/'generation'
out.mkdir(exist_ok=False)
for folder in ['priors', 'traces', 'historical_priors']:
    (out/folder).mkdir()
started = time.monotonic()
source_paths = [BASE/'CUP/agent/rollout.py', BASE/'CUP/agent/tools.py']
source_hashes = {str(p): sha(p) for p in source_paths}
assert source_hashes[str(source_paths[0])] == '76db78e4730e22be5f75c4d8f74b4371ddb4ad81d11e1d2eb810a1b3d35c2365'
manifest = json.loads((ROOT/'base_input_manifest.json').read_text(encoding='utf-8'))
assert torch.__version__ == manifest['expected_torch'] == '2.4.1+cu124'
assert metadata.version('transformers') == manifest['expected_transformers']
assert sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py') == '5dae4359905bc6d44c11eaaa957df8ec85f3975fe6173c35ad54164f055e6026'
packages = {key: metadata.version(key) for key in ['torch', 'transformers', 'peft', 'pyarrow', 'numpy', 'pytorch-lightning']}
seed(0)
kernel_input = torch.randn(16, 16, device='cuda', requires_grad=True)
kernel_loss = (kernel_input @ kernel_input.T).square().mean()
kernel_loss.backward()
assert torch.isfinite(kernel_loss) and torch.isfinite(kernel_input.grad).all()
dump(ROOT/'environment_witness.json', {'packages': packages, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(), 'seed0_kernel_loss': float(kernel_loss),
    'finite_loss_and_gradient': True, 'warm_reuse': True, 'environment_modified': False,
    'sitecustomize_sha256': sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')})
del kernel_input, kernel_loss
for name, info in manifest['model_asset_hashes'].items():
    p = ASSETS/'models/Qwen2.5-VL-7B-Instruct'/name
    assert p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], name
for path, digest in [(ASSETS/'models/ViT-L-14.pt', PLAN['clip_sha256']),
    (ASSETS/'ucm_main_lora/adapter_model.safetensors', PLAN['lora_weights_sha256']),
    (ASSETS/'ucm_scene_cls.pt', PLAN['scene_head_sha256'])]:
    assert sha(path) == digest, str(path)
arrow = ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow'
assert sha(arrow) == manifest['arrow_sha256']['val']
table = pa.ipc.open_file(str(arrow)).read_all()
assert len(table) == 210
images = {}
for item in PLAN['images']:
    row = item['row']
    raw = table['image'][row].as_py()
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    assert table['path'][row].as_py() == item['image']
    assert table['caption'][row].as_py()[0] == item['caption']
    assert hashlib.sha256(raw).hexdigest() == item['image_bytes_sha256']
    assert hashlib.sha256(image.tobytes()).hexdigest() == item['image_rgb_sha256']
    historical = BASE.parent/'ucm_prior_archive/agent_cache_domain_tool'/item['historical_prior_file']
    assert sha(historical) == item['historical_prior_sha256']
    shutil.copyfile(historical, out/'historical_priors'/f'{row}.pt')
    images[row] = image
print('VERIFIED_ASSETS_LOADING_MODELS', flush=True)
seed(0)
model, processor = load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'), 'cuda', str(ASSETS/'ucm_main_lora'))
model.eval().requires_grad_(False)
tool_clip, preprocess = clip.load(str(ASSETS/'models/ViT-L-14.pt'), device='cuda', jit=False)
tool_clip = tool_clip.float().eval().requires_grad_(False)
head_saved = torch.load(ASSETS/'ucm_scene_cls.pt', map_location='cpu', weights_only=True)
head = torch.nn.Linear(head_saved['feat_dim'], len(head_saved['classes']))
head.load_state_dict(head_saved['head'], strict=True)
head = head.float().cuda().eval().requires_grad_(False)
classes = list(head_saved['classes'])
dump(out/'settings.json', {'plan_sha256': sha(ROOT/'plan.json'), 'source_hashes': source_hashes,
    'system_prompt': SYSTEM_PROMPT, 'model_generation_defaults': model.generation_config.to_dict(),
    'lora_config_sha256': sha(ASSETS/'ucm_main_lora/adapter_config.json'),
    'scene_classes': classes, 'state_rule': PLAN['state_rule'],
    'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})

active_calls = []
original_generate = model.generate


def traced_generate(*args, **kwargs):
    outputs = original_generate(*args, **kwargs)
    prompt_length = kwargs['input_ids'].shape[1]
    new_ids = outputs.sequences[0, prompt_length:]
    steps = len(outputs.hidden_states)
    assert steps == new_ids.numel() and steps > 0
    selected = outputs.hidden_states[-1][-1][0, -1, :].detach().cpu()
    assert list(selected.shape) == [3584] and torch.isfinite(selected).all()
    position = prompt_length+steps-2
    token = int(outputs.sequences[0, position])
    active_calls.append({'prompt_tokens': prompt_length, 'generated_token_ids': new_ids.cpu().tolist(),
        'generated_tokens': len(new_ids), 'generated_text': processor.decode(new_ids, skip_special_tokens=True),
        'last_emitted_token': processor.decode([int(new_ids[-1])], skip_special_tokens=False),
        'state_input_token_id': token, 'state_input_token': processor.decode([token], skip_special_tokens=False),
        'state_input_sequence_position': position, 'decoding_steps': steps,
        'final_state_sha256': tensor_sha(selected), 'final_state_shape': list(selected.shape),
        'generation_kwargs': {k: kwargs[k] for k in ['max_new_tokens', 'do_sample', 'temperature',
            'output_hidden_states', 'return_dict_in_generate', 'min_new_tokens']}})
    return outputs


model.generate = traced_generate
records, repeats = [], []
schedule = [(item, 'reference', 0, 0.0) for item in PLAN['images']]
schedule += [(item, 'candidate', s, PLAN['temperature']) for item in PLAN['images'] for s in PLAN['seeds']]
schedule += [(item, 'repeat', 0, PLAN['temperature']) for item in PLAN['images']]
for item, role, sample_seed, temperature in schedule:
    row = item['row']
    seed(sample_seed)
    active_calls = []
    events = []

    def scene_tool(image):
        text, confidence = scene_classify_with_conf(image, model=tool_clip, preprocess=preprocess,
            cls_head=head, cls_names=classes)
        accepted = confidence >= PLAN['tool_conf_threshold']
        events.append({'text': text, 'confidence': float(confidence), 'accepted': accepted})
        return text if accepted else None

    stem = f'{row}_{role}_{sample_seed}'
    trace_text = io.StringIO()
    tick = time.monotonic()
    try:
        with contextlib.redirect_stdout(trace_text):
            hidden, used, filtered, message_audit = rollout_messages(model, processor, images[row], item['caption'],
                {'scene_classify': scene_tool}, max_rounds=PLAN['max_rounds'],
                max_new_tokens=PLAN['max_new_tokens_per_round'], device='cuda', verbose=True,
                injected_h_end=False, system_prompt=SYSTEM_PROMPT, temperature=temperature)
    finally:
        (out/'traces'/f'{stem}.txt').write_text(trace_text.getvalue(), encoding='utf-8')
        dump(out/'traces'/f'{stem}.json', active_calls)
    assert list(hidden.shape) == [3584] and str(hidden.dtype) == 'torch.bfloat16'
    assert torch.isfinite(hidden).all() and used+filtered == len(events)
    assert tensor_sha(hidden) == active_calls[-1]['final_state_sha256']
    prior_file = out/'priors'/f'{stem}.pt'
    torch.save(hidden, prior_file)
    assert torch.equal(hidden, torch.load(prior_file, map_location='cpu', weights_only=True))
    trajectory = '\n'.join(call['generated_text'] for call in active_calls)
    record = {'id': stem, 'role': role, 'seed': sample_seed, 'temperature': temperature,
        'row': row, 'image': item['image'], 'caption': item['caption'],
        'prior_file': str(prior_file.relative_to(ROOT)), 'prior_sha256': sha(prior_file),
        'tensor_sha256': tensor_sha(hidden), 'trace_file': f'generation/traces/{stem}.json',
        'trace_sha256': sha(out/'traces'/f'{stem}.json'), 'trajectory_text': trajectory,
        'trajectory_text_sha256': hashlib.sha256(trajectory.encode()).hexdigest(),
        'tools_used': used, 'tools_filtered': filtered, 'tool_events': events,
        'rounds': len(active_calls), 'state_input_token': active_calls[-1]['state_input_token'],
        'last_emitted_token': active_calls[-1]['last_emitted_token'],
        'any_token_limit': any(c['generated_tokens'] == PLAN['max_new_tokens_per_round'] for c in active_calls),
        'contains_addCriterion': 'addCriterion' in trajectory, 'message_audit': message_audit,
        'quality_usable': message_audit['usable'], 'quality_stop_reason': message_audit['stop_reason'],
        'elapsed_seconds': time.monotonic()-tick}
    if role == 'repeat':
        original = next(r for r in records if r['row'] == row and r['role'] == 'candidate' and r['seed'] == sample_seed)
        match = {'row': row, 'seed': sample_seed, 'state_hash_equal': record['tensor_sha256'] == original['tensor_sha256'],
            'generated_token_ids_equal': [c['generated_token_ids'] for c in active_calls] == [c['generated_token_ids'] for c in json.loads((ROOT/original['trace_file']).read_text())],
            'tool_events_equal': events == original['tool_events']}
        assert all(match[k] for k in ['state_hash_equal', 'generated_token_ids_equal', 'tool_events_equal'])
        repeats.append(match)
    records.append(record)
    with (out/'records.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    dump(ROOT/'progress.json', {'phase': 'generation', 'completed': len(records), 'total': 30,
        'last': record['id'], 'elapsed_seconds': time.monotonic()-started})
    print('GENERATED', len(records), '/30', stem, 'tools', used, 'rounds', len(active_calls), flush=True)
    if len(records) == 3:
        gate = {'ok': all(r['quality_usable'] for r in records), 'references': [r['id'] for r in records], 'usable': [r['quality_usable'] for r in records]}
        dump(out/'quality_gate.json', gate)
        assert gate['ok'], 'Greedy message-repair gate failed; preserve evidence and stop before sampling.'
assert len(records) == 30 and len(repeats) == 3
assert all(sha(path) == digest for path, digest in source_hashes.items())
assert all(p.grad is None and not p.requires_grad for p in model.parameters())
dump(out/'verification.json', {'ok': True, 'count': len(records), 'candidates': 24, 'references': 3,
    'repeats': repeats, 'all_states_finite_bfloat16_3584': True, 'save_reload_bitwise_equal': True,
    'source_files_unchanged': source_hashes, 'plan_sha256': sha(ROOT/'plan.json'),
    'generator_sha256': sha(__file__), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'elapsed_seconds': time.monotonic()-started})
print('GENERATION_COMPLETE', flush=True)
