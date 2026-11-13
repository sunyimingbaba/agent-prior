#!/usr/bin/env python3
"""Independent CPU audit of downloaded states, traces and saved score matrices."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--proxy-cache', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root = args.run
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


plan = json.loads((root/'plan.json').read_text())
assert json.loads((root/'run_result.json').read_text())['status'] == 'complete'
generation = [json.loads(line) for line in (root/'generation/records.jsonl').read_text().splitlines()]
assert len(generation) == 30
assert collections.Counter(r['role'] for r in generation) == {'candidate': 24, 'reference': 3, 'repeat': 3}
for record in generation:
    prior = root/record['prior_file']
    assert sha(prior) == record['prior_sha256']
    state = torch.load(prior, map_location='cpu', weights_only=True)
    assert list(state.shape) == [3584] and state.dtype == torch.bfloat16 and torch.isfinite(state).all()
    assert tensor_sha(state) == record['tensor_sha256']
    trace_file = root/record['trace_file']
    assert sha(trace_file) == record['trace_sha256']
    calls = json.loads(trace_file.read_text())
    audit = record['message_audit']
    assert len(audit['rounds']) == len(calls)
    assert not audit['resampled'] and record['quality_usable'] == audit['usable']
    for index, (message_round, call) in enumerate(zip(audit['rounds'], calls)):
        assert message_round['raw_text'] == call['generated_text']
        assert message_round['message_roles'] == ['system','user','assistant','user','assistant','user']+['assistant','user']*index
        assert message_round['rendered_text'].endswith('<|im_start|>assistant\n')
        assert hashlib.sha256(message_round['rendered_text'].encode()).hexdigest() == message_round['rendered_text_sha256']
        assert call['generation_kwargs']['min_new_tokens'] == 1
        assert call['generated_token_ids'][0] not in [151643,151645]
    assert len(calls) == record['rounds'] <= plan['max_rounds']
    assert calls[-1]['final_state_sha256'] == record['tensor_sha256']
    assert '\n'.join(c['generated_text'] for c in calls) == record['trajectory_text']
    for call in calls:
        assert len(call['generated_token_ids']) == call['generated_tokens'] == call['decoding_steps']
        assert 1 <= call['generated_tokens'] <= plan['max_new_tokens_per_round']
        assert call['generation_kwargs']['temperature'] == (1.0 if record['temperature'] == 0 else record['temperature'])
        assert call['generation_kwargs']['do_sample'] == (record['temperature'] > 0)
    assert record['seed'] in plan['seeds']
for row in plan['selected_rows']:
    group = [r for r in generation if r['row'] == row]
    candidate_seeds = sorted(r['seed'] for r in group if r['role'] == 'candidate')
    assert candidate_seeds == plan['seeds']
    original = next(r for r in group if r['role'] == 'candidate' and r['seed'] == 0)
    repeated = next(r for r in group if r['role'] == 'repeat')
    assert original['tensor_sha256'] == repeated['tensor_sha256']
    assert original['trajectory_text'] == repeated['trajectory_text']
    assert original['tool_events'] == repeated['tool_events']
scores = json.loads((root/'scoring/scores.json').read_text())
tensors = torch.load(root/'scoring/scoring_tensors.pt', map_location='cpu', weights_only=True)
assert len(scores) == len(tensors['record_ids']) == 33
assert [r['id'] for r in scores] == tensors['record_ids']
for tensor in tensors.values():
    if isinstance(tensor, torch.Tensor):
        assert torch.isfinite(tensor).all()
gallery = json.loads((root/'scoring/text_gallery.json').read_text())
n = gallery['unique_val_caption_count']
assert sha(args.proxy_cache) == plan['proxy_reward']['gallery_sha256']
cache = torch.load(args.proxy_cache, map_location='cpu', weights_only=True)
assert cache['image_indices'] == list(range(210))
proxy64 = tensors['proxy_features'].double() @ cache['image_features'].double().T
utility64 = tensors['image_features'].double() @ tensors['text_features'].double().T
proxy_error = float((proxy64-tensors['proxy_similarities'].double()).abs().max())
utility_error = float((utility64-tensors['actual_similarities'].double()).abs().max())
assert max(proxy_error, utility_error) < 1e-6
for i, record in enumerate(scores):
    row = record['row']
    proxy = tensors['proxy_similarities'][i]
    rank = sum(float(value) > float(proxy[row]) for value in proxy)
    assert rank == record['proxy_rank0'] and 1/(1+rank) == record['r_retrieval']
    if record['role'] != 'historical':
        format_reward = .2 if re.search(r'<answer>(.*?)</answer>', record['trajectory_text'], re.DOTALL) else 0.0
        assert format_reward == record['r_format']
        assert record['r_total'] == format_reward+1/(1+rank)
    similarities = tensors['actual_similarities'][i]
    positive = gallery['positive_gallery_columns'][str(row)]
    negative = [j for j in range(n) if j not in positive]
    best_positive = max(float(similarities[j]) for j in positive)
    best_negative = max(float(similarities[j]) for j in negative)
    rank = sum(float(value) > best_positive for value in similarities[:n])
    assert rank == record['utility_rank0'] and record['utility_rr'] == 1/(1+rank)
    assert record['utility_margin'] == best_positive-best_negative
    for label, columns in gallery['quantity_text_columns'][str(row)].items():
        margin = float(similarities[columns[0]]-similarities[columns[1]])
        assert margin == record['E1_margin_'+label]
    reference = next(r for r in scores if r['row'] == row and r['role'] == 'reference')
    for metric in ['utility_rr', 'utility_margin', 'E1_margin_A', 'E1_margin_B']:
        assert record[metric+'_delta'] == record[metric]-reference[metric]
analysis = json.loads((root/'analysis.json').read_text())
assert analysis['candidate_count'] == 24 and analysis['image_count'] == 3
for reward, metrics in analysis['aggregate'].items():
    for metric, values in metrics.items():
        assert sum(values['pair_counts'].values()) == 84
        for key, count in values['pair_counts'].items():
            assert count == sum(image['comparisons'][reward][metric]['pair_counts'][key] for image in analysis['per_image'])
result = {'ok': True, 'generation_records': 30, 'scoring_records': 33,
    'all_states_and_traces_hash_verified': True, 'all_saved_tensors_finite': True,
    'all_rewards_ranks_utilities_and_reference_deltas_recomputed': True,
    'proxy_cpu64_saved_gpu32_max_abs_dot_error': proxy_error,
    'utility_cpu64_saved_gpu32_max_abs_dot_error': utility_error,
    'plan_sha256': sha(root/'plan.json'), 'scores_sha256': sha(root/'scoring/scores.json'),
    'torch_cpu_verifier_version': torch.__version__}
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False))
