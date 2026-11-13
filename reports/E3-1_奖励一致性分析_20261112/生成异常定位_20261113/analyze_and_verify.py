#!/usr/bin/env python3
"""Independent CPU state/logit audit and grouped summary of generation diagnoses."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--pilot', type=Path, required=True)
parser.add_argument('--first-run', type=Path)
args = parser.parse_args()
r = args.run
torch.set_num_threads(2)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def write(name, value):
    (r/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


assert json.loads((r/'run_result.json').read_text())['status'] == 'complete'
verification = json.loads((r/'verification.json').read_text())
assert verification['ok'] and verification['loaded_adapter_parameter_hashes_unchanged']
records = [json.loads(x) for x in (r/'outputs/records.jsonl').read_text().splitlines()]
assert len(records) == 42 and len({x['id'] for x in records}) == 42
calls_by_id = {}
for record in records:
    state = torch.load(r/record['state_file'], map_location='cpu', weights_only=True)
    assert list(state.shape) == [3584] and state.dtype == torch.bfloat16 and torch.isfinite(state).all()
    assert digest(state) == record['state_sha256']
    path = r/record['trace_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['trace_sha256']
    calls = json.loads(path.read_text())
    assert len(calls) == record['rounds'] and calls[-1]['state_sha256'] == record['state_sha256']
    assert record['trajectory'] == '\n'.join(c['generated_text'] for c in calls)
    calls_by_id[record['id']] = calls
logits = torch.load(r/'outputs/selected_first_step_logits.pt', map_location='cpu', weights_only=True)
assert len(logits) == 12
errors = []
for identifier, values in logits.items():
    assert values.ndim == 1 and torch.isfinite(values).all()
    probabilities = values.double().softmax(-1)
    actual = calls_by_id[identifier][0]['first_and_selected_steps'][0]['raw']['selected']
    for token, info in actual.items():
        index = int(token)
        error = abs(float(probabilities[index])-info['probability'])
        assert error < 2e-7
        assert info['rank1'] == 1+int((values > values[index]).sum())
        errors.append(error)
pilot = [json.loads(x) for x in (args.pilot/'generation/records.jsonl').read_text().splitlines()]
replays = []
for record in records:
    if record['role'] != 'native' or record['adapter'] != 'enabled' or record['variant'] != 'tool':
        continue
    old = next(x for x in pilot if x['row'] == record['row'] and x['seed'] == 0 and
        x['role'] == ('reference' if record['temperature'] == 0 else 'candidate'))
    assert record['state_sha256'] == old['tensor_sha256']
    old_calls = json.loads((args.pilot/old['trace_file']).read_text())
    assert [c['generated_token_ids'] for c in old_calls] == [c['generated_token_ids'] for c in calls_by_id[record['id']]]
    replays.append({'new': record['id'], 'prior_pilot': old['id'], 'state_and_tokens_identical': True})
assert len(replays) == 6
feedback_replays = []
for record in records:
    if record['role'] != 'fixed_feedback' or record['adapter'] != 'enabled' or record['variant'] != 'raw_unclosed':
        continue
    native = next(x for x in records if x['row'] == record['row'] and x['adapter'] == 'enabled' and
        x['role'] == 'native' and x['variant'] == 'tool' and x['temperature'] == 0)
    assert record['state_sha256'] == native['state_sha256']
    assert calls_by_id[record['id']][0]['generated_token_ids'] == calls_by_id[native['id']][-1]['generated_token_ids']
    feedback_replays.append({'row': record['row'], 'fixed_raw_feedback_matches_native_second_round': True})
grouped = []
for key in sorted({(x['role'], x['adapter'], x['variant'], x['temperature']) for x in records}):
    group = [x for x in records if (x['role'], x['adapter'], x['variant'], x['temperature']) == key]
    grouped.append({'role': key[0], 'adapter': key[1], 'variant': key[2], 'temperature': key[3],
        'count': len(group), 'empty': sum(x['empty'] for x in group),
        'addCriterion': sum(x['contains_addCriterion'] for x in group),
        'complete_answer': sum(x['has_complete_answer'] for x in group)})
eos = []
for row in [10,95,207]:
    conditions = {adapter: next(x for x in records if x['row'] == row and x['role'] == 'native' and
        x['variant'] == 'tool' and x['temperature'] == 1 and x['adapter'] == adapter) for adapter in ['enabled','disabled']}
    eos.append({'row': row, 'image': conditions['enabled']['image'],
        'enabled_raw_eos_probability': conditions['enabled']['first_step_raw_eos_probability'],
        'enabled_sampling_eos_probability': conditions['enabled']['first_step_processed_eos_probability'],
        'disabled_raw_eos_probability': conditions['disabled']['first_step_raw_eos_probability'],
        'disabled_sampling_eos_probability': conditions['disabled']['first_step_processed_eos_probability']})
first_replay = None
if args.first_run:
    first = {x['id']: x for x in [json.loads(x) for x in (args.first_run/'outputs/records.jsonl').read_text().splitlines()]}
    assert set(first) == {x['id'] for x in records}
    for record in records:
        assert record['trajectory'] == first[record['id']]['trajectory']
        assert record['state_sha256'] == first[record['id']]['state_sha256']
    first_replay = {'cases': len(records), 'all_trajectories_and_state_hashes_identical': True,
        'first_attempt_status': json.loads((args.first_run/'run_result.json').read_text())['status']}
summary = {'ok': True, 'records': 42, 'groups': grouped, 'initial_EOS': eos, 'prior_pilot_replays': replays,
    'raw_feedback_replays': feedback_replays, 'first_attempt_replay': first_replay,
    'independent_logit_vectors': len(logits), 'max_CPU_GPU_selected_probability_error': max(errors),
    'all_states_and_trace_hashes_verified': True,
    'structured_feedback_answers': [{'id': x['id'], 'answer': x['trajectory']} for x in records if x['role'] == 'fixed_feedback' and x['variant'] == 'chat_closed']}
write('诊断汇总与独立核验.json', summary)
with (r/'诊断逐项结果.csv').open('w', encoding='utf-8', newline='') as stream:
    keys = [k for k, value in records[0].items() if not isinstance(value, (list, dict))]
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
print(json.dumps({k:summary[k] for k in ['ok','records','groups','initial_EOS','max_CPU_GPU_selected_probability_error']}, ensure_ascii=False))
