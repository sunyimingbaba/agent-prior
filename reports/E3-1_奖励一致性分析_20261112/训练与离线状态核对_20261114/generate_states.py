"""Freeze weights; replay actual GRPO forward on archived complete sequences."""
import datetime
import importlib.metadata as metadata
import io
import json
import re
import shutil
import time

from audit_common import ROOT, BASE, ASSETS, PLAN, SOURCE, dump, sha, tensor_sha, setup_torch, seed

torch = setup_torch()
import pyarrow as pa
from PIL import Image
from agent.rollout import load_mllm
from agent.grpo import train_grpo

started = time.monotonic()
train_path = BASE/'CUP/agent/grpo/train_grpo.py'
assert sha(train_path) == PLAN['train_source_sha256']
source_plan = json.loads((SOURCE/'plan.json').read_text())
assert sha(SOURCE/'plan.json') == PLAN['source_plan_sha256']
assert sha(SOURCE/'scoring/scores.json') == PLAN['source_scores_sha256']
assert sha(ASSETS/'ucm_main_lora/adapter_model.safetensors') == source_plan['lora_weights_sha256']
manifest = json.loads((SOURCE/'base_input_manifest.json').read_text())
for name, info in manifest['model_asset_hashes'].items():
    path = ASSETS/'models/Qwen2.5-VL-7B-Instruct'/name
    assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256']
assert sha(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow') == manifest['arrow_sha256']['val']
assert torch.__version__ == manifest['expected_torch']
assert metadata.version('transformers') == manifest['expected_transformers']
site_hash = sha(BASE/'env/lib/python3.12/site-packages/sitecustomize.py')
assert site_hash == '5dae4359905bc6d44c11eaaa957df8ec85f3975fe6173c35ad54164f055e6026'
states = ROOT/'states'
states.mkdir(exist_ok=False)
(ROOT/'traces').mkdir()
(ROOT/'source_inputs').mkdir()
(ROOT/'source_snapshots').mkdir(exist_ok=True)
sources = {}
for name in ['agent/grpo/train_grpo.py', 'agent/rollout.py', 'agent/grpo/scorer.py', 'agent/grpo/reward.py']:
    path = BASE/'CUP'/name
    sources[str(path)] = sha(path)
    shutil.copy2(path, ROOT/'source_snapshots'/path.name)

# A new seeded CUDA calculation confirms the reused environment executes kernels.
seed(0)
x = torch.randn(16, 16, device='cuda', requires_grad=True)
loss = (x@x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
dump(ROOT/'environment_witness.json', {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
    'seed0_kernel_loss': float(loss), 'finite_loss_and_gradient': True,
    'warm_reuse': True, 'environment_modified': False, 'sitecustomize_sha256': site_hash,
    'packages': {name: metadata.version(name) for name in ['transformers', 'peft', 'numpy', 'pyarrow']}})
del x, loss

table = pa.ipc.open_file(str(ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow')).read_all()
source_records = {x['id']: x for x in [json.loads(z) for z in (SOURCE/'generation/records.jsonl').read_text().splitlines()]}
images = {}
for row in PLAN['rows']:
    raw = table['image'][row].as_py()
    item = next(x for x in source_plan['images'] if x['row'] == row)
    assert __import__('hashlib').sha256(raw).hexdigest() == item['image_bytes_sha256']
    images[row] = Image.open(io.BytesIO(raw)).convert('RGB')

seed(0)
model, processor = load_mllm(str(ASSETS/'models/Qwen2.5-VL-7B-Instruct'), 'cuda', str(ASSETS/'ucm_main_lora'))
model.eval().requires_grad_(False)
records, comparisons, extension = [], [], []
trace_cache = {}


def save_state(source, variant, value, quality, extra=None):
    value = value.detach().cpu().to(torch.bfloat16).contiguous()
    assert list(value.shape) == [3584] and torch.isfinite(value).all()
    name = source['source_id']+'__'+variant
    path = states/(name+'.pt')
    torch.save(value, path)
    assert torch.equal(value, torch.load(path, map_location='cpu', weights_only=True))
    result = {'id': name, 'source_id': source['source_id'], 'source_role': source['role'],
        'row': source['row'], 'image': source['image'], 'variant': variant,
        'state_file': str(path.relative_to(ROOT)), 'state_file_sha256': sha(path),
        'state_tensor_sha256': tensor_sha(value), 'quality_usable': quality, **(extra or {})}
    records.append(result)
    return value


def pair(source_id, label, left, right):
    left, right = left.float(), right.float()
    comparisons.append({'source_id': source_id, 'comparison': label,
        'cosine': float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        'max_abs_difference': float((left-right).abs().max()),
        'relative_l2': float((left-right).norm()/left.norm()),
        'bitwise_equal': bool(torch.equal(left, right))})


@torch.no_grad()
def replay(source, rendered, generated_ids, padded_to=None):
    inputs = processor(text=[rendered], images=[images[source['row']]], return_tensors='pt').to('cuda')
    prompt_len = inputs.input_ids.shape[1]
    generated = torch.tensor(generated_ids, dtype=torch.long, device='cuda')
    seq = torch.cat([inputs.input_ids[0], generated])
    pad_count = 0 if padded_to is None else padded_to-len(generated_ids)
    assert pad_count >= 0
    if pad_count:
        seq = torch.nn.functional.pad(seq, (0, pad_count), value=processor.tokenizer.pad_token_id)
    captured = {}
    original_forward = model.forward

    def observed_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        captured['last'] = output.hidden_states[-1][0, -1, :].detach().cpu()
        captured['penultimate'] = output.hidden_states[-1][0, -2, :].detach().cpu()
        captured['attention_all_ones'] = bool((kwargs['attention_mask'] == 1).all())
        captured['input_length'] = kwargs['input_ids'].shape[1]
        return output

    model.forward = observed_forward
    try:
        logp, h = train_grpo.forward_logprob(model, processor, images[source['row']], rendered, seq, prompt_len)
    finally:
        model.forward = original_forward
    assert torch.equal(h.to(torch.bfloat16), captured['last'])
    assert captured['attention_all_ones'] and torch.isfinite(logp)
    return h, captured['penultimate'], {'prompt_tokens': prompt_len, 'generated_token_ids': generated_ids,
        'right_pad_count': pad_count, 'input_length': captured['input_length'],
        'last_input_token_id': int(seq[-1]), 'penultimate_input_token_id': int(seq[-2]),
        'attention_all_ones': True, 'logprob_sum_including_padding': float(logp),
        'summed_logprob_token_count': len(generated_ids)+pad_count}


for n, source in enumerate(PLAN['records'], 1):
    original = source_records[source['source_id']]
    assert sha(SOURCE/original['prior_file']) == source['source_prior_sha256']
    assert sha(SOURCE/original['trace_file']) == source['source_trace_sha256']
    trace = json.loads((SOURCE/original['trace_file']).read_text())[0]
    generated = trace['generated_token_ids']
    rendered = original['message_audit']['rounds'][0]['rendered_text']
    trace_cache[source['source_id']] = (rendered, generated)
    dump(ROOT/'source_inputs'/(source['source_id']+'.json'), {'record': original, 'trace': trace})
    original_state = torch.load(SOURCE/original['prior_file'], map_location='cpu', weights_only=True)
    saved = save_state(source, 'saved_offline', original_state, original['quality_usable'])
    last, penultimate, info = replay(source, rendered, generated)
    assert info['prompt_tokens'] == trace['prompt_tokens']
    info['rendered_text'] = rendered
    info['original_generated_text'] = original['trajectory_text']
    info['source_input_token_id'] = trace['state_input_token_id']
    assert info['penultimate_input_token_id'] == trace['state_input_token_id']
    dump(ROOT/'traces'/(source['source_id']+'__unpadded.json'), info)
    pen = save_state(source, 'full_forward_penultimate', penultimate, original['quality_usable'])
    full = save_state(source, 'grpo_unpadded_last', last, original['quality_usable'])
    padded, _, pad_info = replay(source, rendered, generated, source['group_max_generated_tokens'])
    dump(ROOT/'traces'/(source['source_id']+'__padded.json'), pad_info)
    padded = save_state(source, 'grpo_group_padded_last', padded, original['quality_usable'])
    pair(source['source_id'], 'saved_vs_forward_penultimate', saved, pen)
    pair(source['source_id'], 'saved_vs_grpo_unpadded', saved, full)
    pair(source['source_id'], 'grpo_unpadded_vs_group_padded', full, padded)
    dump(ROOT/'progress.json', {'phase': 'state_replay', 'completed': n, 'total': 18})
    print('REPLAYED', n, '/18', source['source_id'], 'pad', pad_info['right_pad_count'], flush=True)

for n, source in enumerate([x for x in PLAN['records'] if x['role'] == 'selected_failure'], 1):
    rendered, old_tokens = trace_cache[source['source_id']]
    seed(source['seed'])
    inputs = processor(text=[rendered], images=[images[source['row']]], return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=True,
            temperature=1.0, output_hidden_states=True, return_dict_in_generate=True)
    new_ids = outputs.sequences[0, inputs.input_ids.shape[1]:].cpu().tolist()
    text = processor.decode(new_ids, skip_special_tokens=True)
    prefix_equal = new_ids[:len(old_tokens)] == old_tokens
    assert prefix_equal, 'Preserve mismatch evidence before running further extensions.'
    limit = len(new_ids) == 256
    usable = bool(re.search(r'<answer>(.*?)</answer>', text, re.S)) and not limit \
        and 'addCriterion' not in text and '<tool_call>' not in text
    hidden = outputs.hidden_states[-1][-1][0, -1, :].detach().cpu()
    gen = save_state(source, 'extended_offline', hidden, usable)
    del outputs
    last, pen, info = replay(source, rendered, new_ids)
    info.update(generated_text=text, original_prefix_equal=prefix_equal, quality_usable=usable,
        token_limit=limit, source_seed=source['seed'], actual_tool_executions=0,
        tool_context_origin='archived actual scene_classify result', resampled=False)
    dump(ROOT/'traces'/(source['source_id']+'__extended.json'), info)
    pen = save_state(source, 'extended_forward_penultimate', pen, usable)
    full = save_state(source, 'extended_grpo_last', last, usable)
    pair(source['source_id'], 'extended_saved_vs_forward_penultimate', gen, pen)
    pair(source['source_id'], 'extended_saved_vs_grpo_last', gen, full)
    extension.append({'source_id': source['source_id'], 'row': source['row'], 'image': source['image'],
        'old_tokens': len(old_tokens), 'new_tokens': len(new_ids), 'prefix_equal': prefix_equal,
        'quality_usable': usable, 'token_limit': limit, 'same_tokens_as_original': new_ids == old_tokens})
    dump(ROOT/'progress.json', {'phase': 'extension', 'completed': n, 'total': 6})
    print('EXTENDED', n, '/6', source['source_id'], len(new_ids), 'usable', usable, flush=True)

assert len(records) == PLAN['total_expected_states']
assert all(sha(p) == h for p, h in sources.items())
assert all(not p.requires_grad and p.grad is None for p in model.parameters())
dump(ROOT/'state_records.json', records)
dump(ROOT/'state_comparisons.json', comparisons)
dump(ROOT/'extension_records.json', extension)
dump(ROOT/'generation_verification.json', {'ok': True, 'states': len(records), 'source_files_unchanged': sources,
    'frozen_model_no_optimizer': True, 'all_states_finite_bfloat16_3584': True,
    'extension_prefix_equal_count': sum(x['prefix_equal'] for x in extension),
    'extension_quality_pass_count': sum(x['quality_usable'] for x in extension),
    'plan_sha256': sha(ROOT/'plan.json'), 'generator_sha256': sha(__file__),
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic()-started})
print('GENERATION_COMPLETE', flush=True)
