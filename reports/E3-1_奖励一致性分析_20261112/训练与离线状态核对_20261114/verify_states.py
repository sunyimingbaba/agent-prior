"""CPU reconstruction of state geometry, saved ranking scores and pair counts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import torch
torch.set_num_threads(2)

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
r = a.run
plan = json.loads((r/'plan.json').read_text())
assert json.loads((r/'run_result.json').read_text())['status'] == 'complete'
records = json.loads((r/'state_records.json').read_text())
assert len(records) == len({x['id'] for x in records}) == plan['total_expected_states'] == 90
states = {}
for x in records:
    path = r/x['state_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == x['state_file_sha256']
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert list(value.shape) == [3584] and value.dtype == torch.bfloat16 and torch.isfinite(value).all()
    assert hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == x['state_tensor_sha256']
    states[x['source_id'], x['variant']] = value
geometry = json.loads((r/'state_comparisons.json').read_text())
variants = {'saved_vs_forward_penultimate': ('saved_offline','full_forward_penultimate'),
    'saved_vs_grpo_unpadded': ('saved_offline','grpo_unpadded_last'),
    'grpo_unpadded_vs_group_padded': ('grpo_unpadded_last','grpo_group_padded_last'),
    'extended_saved_vs_forward_penultimate': ('extended_offline','extended_forward_penultimate'),
    'extended_saved_vs_grpo_last': ('extended_offline','extended_grpo_last')}
assert len(geometry) == 66
geometry_max_error = {'cosine': 0., 'max_abs_difference': 0., 'relative_l2': 0.}
for item in geometry:
    before, after = variants[item['comparison']]
    x, y = states[item['source_id'], before].float(), states[item['source_id'], after].float()
    assert item['bitwise_equal'] == torch.equal(x,y)
    for key, value in [('cosine', torch.nn.functional.cosine_similarity(x,y,dim=0)),
                       ('max_abs_difference', (x-y).abs().max()), ('relative_l2',(x-y).norm()/x.norm())]:
        error = abs(item[key]-float(value))
        geometry_max_error[key] = max(geometry_max_error[key], error)
        # Float32 reductions differ slightly across CPU kernels / torch versions.
        # State bytes and discrete saved scores/ranks below remain exact checks.
        assert error <= 1e-5*max(1., abs(item[key]))
for source in plan['records']:
    stem = source['source_id']
    original = json.loads((r/'source_inputs'/(stem+'.json')).read_text())
    trace = original['trace']
    unpadded = json.loads((r/'traces'/(stem+'__unpadded.json')).read_text())
    padded = json.loads((r/'traces'/(stem+'__padded.json')).read_text())
    assert unpadded['generated_token_ids'] == padded['generated_token_ids'] == trace['generated_token_ids']
    assert unpadded['prompt_tokens'] == padded['prompt_tokens'] == trace['prompt_tokens']
    assert unpadded['penultimate_input_token_id'] == trace['state_input_token_id']
    assert trace['state_input_sequence_position'] == trace['prompt_tokens']+trace['generated_tokens']-2
    assert unpadded['right_pad_count'] == 0
    assert padded['right_pad_count'] == source['group_max_generated_tokens']-trace['generated_tokens']
    assert unpadded['attention_all_ones'] and padded['attention_all_ones']
    assert padded['summed_logprob_token_count'] == source['group_max_generated_tokens']
    saved = states[stem,'saved_offline']
    assert hashlib.sha256(saved.view(torch.uint8).numpy().tobytes()).hexdigest() == source['source_tensor_sha256']
extensions = json.loads((r/'extension_records.json').read_text())
assert len(extensions) == 6
for item in extensions:
    stem = item['source_id']
    old = json.loads((r/'source_inputs'/(stem+'.json')).read_text())['trace']['generated_token_ids']
    trace = json.loads((r/'traces'/(stem+'__extended.json')).read_text())
    assert trace['generated_token_ids'][:len(old)] == old and item['prefix_equal']
    assert trace['actual_tool_executions'] == 0 and not trace['resampled']
    text = trace['generated_text']
    usable = bool(re.search(r'<answer>(.*?)</answer>',text,re.S)) and not trace['token_limit'] \
        and 'addCriterion' not in text and '<tool_call>' not in text
    assert usable == item['quality_usable'] == trace['quality_usable']
    assert item['new_tokens'] == len(trace['generated_token_ids']) <= 256

verification = json.loads((r/'scoring/verification.json').read_text())
for name, expected in verification['verification_input_sha256'].items():
    assert hashlib.sha256((r/'verification_inputs'/name).read_bytes()).hexdigest() == expected
gallery = json.loads((r/'verification_inputs/text_gallery.json').read_text())
cache = torch.load(r/'verification_inputs/proxy_gallery.pt',map_location='cpu',weights_only=True)
original = torch.load(r/'verification_inputs/original_scoring_tensors.pt',map_location='cpu',weights_only=True)
t = torch.load(r/'scoring/scoring_tensors.pt',map_location='cpu',weights_only=True)
assert torch.equal(t['text_features'],original['text_features'])
for value in t.values():
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
proxy = t['proxy_features'].double()@cache['image_features'].double().T
actual = t['image_features'].double()@t['text_features'].double().T
proxy_error = float((proxy-t['proxy_similarities'].double()).abs().max())
utility_error = float((actual-t['actual_similarities'].double()).abs().max())
assert max(proxy_error,utility_error) < 1e-6
scores = json.loads((r/'scoring/scores.json').read_text())
assert [x['id'] for x in scores] == t['record_ids']
rank64_proxy = rank64_utility = 0
for i, x in enumerate(scores):
    row = x['row']
    ps = t['proxy_similarities'][i]
    pr = int((ps>ps[row]).sum())
    assert pr == x['proxy_rank0'] and x['r_retrieval'] == 1/(1+pr)
    s = t['actual_similarities'][i]
    positive = gallery['positive_gallery_columns'][str(row)]
    negative = [j for j in range(gallery['unique_val_caption_count']) if j not in positive]
    best = float(s[positive].max())
    rank = int((s[:gallery['unique_val_caption_count']]>best).sum())
    assert rank == x['utility_rank0'] and x['utility_rr'] == 1/(1+rank)
    assert x['utility_margin'] == best-float(s[negative].max())
    for label, cols in gallery['quantity_text_columns'][str(row)].items():
        assert x['E1_margin_'+label] == float(s[cols[0]]-s[cols[1]])
    rank64_proxy += int((proxy[i]>proxy[i,row]).sum()) == pr
    rank64_utility += int((actual[i,:gallery['unique_val_caption_count']]>actual[i,positive].max()).sum()) == rank
index = {(x['source_id'],x['variant']):x for x in scores}
analysis = json.loads((r/'analysis.json').read_text())
for pair in analysis['per_pair']:
    x, y = index[pair['source_id'],pair['before_variant']], index[pair['source_id'],pair['after_variant']]
    for metric in ['r_retrieval','utility_rr','utility_margin','E1_margin_A','E1_margin_B']:
        assert pair[metric+'_delta'] == y[metric]-x[metric]
for name, group in analysis['comparisons'].items():
    pairs = [x for x in analysis['per_pair'] if x['comparison']==name]
    assert len(pairs) == group['records']
    assert group['proxy_rank_changed'] == sum(x['proxy_rank1_before']!=x['proxy_rank1_after'] for x in pairs)
    assert group['utility_rank_changed'] == sum(x['utility_rank1_before']!=x['utility_rank1_after'] for x in pairs)
    for metric, declared in group['mean_deltas'].items():
        assert abs(declared-sum(x[metric+'_delta'] for x in pairs)/len(pairs)) < 1e-12
for name, expected in json.loads((r/'launch_script_hashes.json').read_text()).items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest() == expected
assert hashlib.sha256((r/'source_snapshots/train_grpo.py').read_bytes()).hexdigest() == plan['train_source_sha256']
audit = {'ok':True,'states_and_scores_verified':len(scores),'state_geometry_checks':len(geometry),
    'original_sequence_replays':18,'extension_prefix_checks':6,'paired_effects_verified':len(analysis['per_pair']),
    'proxy_cpu64_gpu32_max_abs_dot_error':proxy_error,'utility_cpu64_gpu32_max_abs_dot_error':utility_error,
    'proxy_rank64_equal_count':int(rank64_proxy),'utility_rank64_equal_count':int(rank64_utility),
    'all_original_text_features_reused_bitwise':True,'torch_cpu_version':torch.__version__,
    'geometry_recompute_max_abs_error':geometry_max_error,
    'geometry_relative_to_max1_tolerance':1e-5,
    'state_file_tensor_hash_and_saved_rank_checks':'exact'}
a.out.write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
