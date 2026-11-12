#!/usr/bin/env python3
"""Independent local verification of the downloaded full cache and trace archive."""
import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
settings = json.loads((args.run/'settings.json').read_text(encoding='utf-8'))
metadata = json.loads((args.run/'metadata.json').read_text(encoding='utf-8'))
assert metadata['ok'] and metadata['status'] == 'complete'
assert settings['manifest_sha256'] == sha(args.manifest)
assert settings['lora_path'] == '' and settings['tools_available'] == []
assert settings['seed_per_image'] == 0 and settings['temperature'] == 0.0
assert settings['max_new_tokens'] == 64 and settings['generation_batch_size'] == 1
expected = {(r['split'], r['row']): r for r in manifest['rows']}
records = [json.loads(s) for s in (args.run/'records.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(records) == len(expected) == 1890
assert len({(r['split'], r['row']) for r in records}) == 1890
assert {(r['split'], r['row']) for r in records} == set(expected)
rows = []
hash_groups = collections.defaultdict(list)
flags = {'token_limit_with_non_end_token': [], 'empty_text': [], 'tool_markup': [], 'addCriterion': []}
for record in records:
    item = expected[(record['split'], record['row'])]
    for key, value in item.items():
        assert record[key] == value, (record['split'], record['row'], key)
    prior_rel = f"priors/ucm_caption_karpathy_label_{record['split']}/{record['row']}.pt"
    trace_rel = f"traces/{record['split']}/{record['row']}.json"
    assert record['prior_file'] == prior_rel and record['trace_file'] == trace_rel
    path = args.run/prior_rel
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    assert tuple(tensor.shape) == (3584,) and str(tensor.dtype) == 'torch.bfloat16'
    assert bool(torch.isfinite(tensor).all()) and float(tensor.float().norm()) > 0
    assert sha(path) == record['prior_file_sha256']
    numeric_hash = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert numeric_hash == record['tensor_sha256']
    trace_path = args.run/trace_rel
    assert sha(trace_path) == record['trace_file_sha256']
    calls = json.loads(trace_path.read_text(encoding='utf-8'))
    assert len(calls) == 1
    call = calls[0]
    assert call['final_state_sha256'] == numeric_hash
    assert call['generated_text'] == record['generated_text']
    assert call['generated_tokens'] == len(call['generated_token_ids']) == record['generated_tokens']
    assert 1 <= record['generated_tokens'] <= 64
    assert call['generation_kwargs']['do_sample'] is False
    assert call['generation_kwargs']['max_new_tokens'] == 64
    assert call['generation_kwargs']['output_hidden_states'] is True
    assert record['tools_used'] == record['tools_filtered'] == 0
    anchor = manifest['smoke_tensor_anchors'].get(record['split']+':'+str(record['row']))
    if anchor:
        assert numeric_hash == anchor and record['smoke_anchor_match']
    case = {'split': record['split'], 'row': record['row'], 'image': record['image']}
    if record['generated_tokens'] == 64 and record['last_emitted_token'] not in ['<|im_end|>', '<|endoftext|>']:
        flags['token_limit_with_non_end_token'].append(case)
    if not record['generated_text'].strip():
        flags['empty_text'].append(case)
    if '<tool_call>' in record['generated_text']:
        flags['tool_markup'].append(case)
    if 'addCriterion' in record['generated_text']:
        flags['addCriterion'].append(case)
    hash_groups[numeric_hash].append(case)
    rows.append({**case, 'caption': record['caption'], 'generated_text': record['generated_text'],
                 'generated_tokens': record['generated_tokens'],
                 'last_emitted_token': record['last_emitted_token'],
                 'state_input_token': record['state_input_token'],
                 'prior_file': prior_rel, 'file_sha256': record['prior_file_sha256'],
                 'tensor_sha256': numeric_hash, 'l2_norm': float(tensor.float().norm()),
                 'verified': True})
for split, count in manifest['split_counts'].items():
    folder = args.run/'priors'/f'ucm_caption_karpathy_label_{split}'
    assert {p.name for p in folder.glob('*.pt')} == {f'{i}.pt' for i in range(count)}
with (args.out/'逐图核验清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
summary = {'verified': True, 'count': len(rows),
    'split_counts': dict(collections.Counter(r['split'] for r in records)),
    'shape': [3584], 'dtype': 'torch.bfloat16', 'all_finite_and_nonzero': True,
    'all_inputs_files_traces_and_state_hashes_match': True,
    'anchor_matches': sum(r['smoke_anchor'] and r['smoke_anchor_match'] for r in records),
    'token_length_min': min(r['generated_tokens'] for r in records),
    'token_length_max': max(r['generated_tokens'] for r in records),
    'token_length_mean': statistics.mean(r['generated_tokens'] for r in records),
    'end_token_counts': dict(collections.Counter(r['last_emitted_token'] for r in records)),
    'flags': flags, 'flag_counts': {k: len(v) for k, v in flags.items()},
    'duplicate_tensor_groups': [v for v in hash_groups.values() if len(v) > 1],
    'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in records)/1024**3,
    'generation_seconds_sum': sum(r['elapsed_seconds'] for r in records),
    'elapsed_seconds_server_attempt': metadata['elapsed_seconds_this_attempt'],
    'formal_scorer_training_started': False}
(args.out/'完整缓存核验结果.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k not in ['flags', 'duplicate_tensor_groups']}, ensure_ascii=False))
