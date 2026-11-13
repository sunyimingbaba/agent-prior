"""Trace fixed contexts and describe sensitivity to flagged answer failures."""
import argparse
import collections
import itertools
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--previous', type=Path, required=True)
args = parser.parse_args()
r = args.run
records = [json.loads(x) for x in (r/'generation/records.jsonl').read_text().splitlines()]
old = [json.loads(x) for x in (args.previous/'generation/records.jsonl').read_text().splitlines()]
scores = json.loads((r/'scoring/scores.json').read_text())
candidates = [x for x in scores if x['role']=='candidate']
contexts, sensitivity, failures, replays = [], [], [], []
for row in sorted({x['row'] for x in candidates}):
    group = [x for x in records if x['row']==row]
    prompts = {x['message_audit']['rounds'][0]['rendered_text_sha256'] for x in group}
    results = {x['tool_events'][0]['text'] for x in group}
    assert len(prompts)==len(results)==1
    contexts.append({'row':row,'records':len(group),'unique_contexts':len(prompts),
        'unique_tool_result_texts':len(results),'actual_tool_executions':sum(x['message_audit']['actual_tool_executions'] for x in group)})
    reference = next(x for x in group if x['role']=='reference')
    previous_reference = next(x for x in old if x['role']=='reference' and x['row']==row)
    replays.append({'row':row,'greedy_reference_state_equal_to_messages_protocol':
        reference['tensor_sha256']==previous_reference['tensor_sha256']})
    valid = [x for x in candidates if x['row']==row and x['quality_usable']]
    counts = dict.fromkeys(['concordant','discordant','reward_tie_only','utility_tie_only','both_tied'],0)
    for a,b in itertools.combinations(valid,2):
        dr,du=a['r_retrieval']-b['r_retrieval'],a['utility_rr']-b['utility_rr']
        rt,ut=dr==0,abs(du)<=1e-12
        category=('both_tied' if rt and ut else 'reward_tie_only' if rt else
                  'utility_tie_only' if ut else 'concordant' if dr*du>0 else 'discordant')
        counts[category]+=1
    sensitivity.append({'row':row,'usable_candidates':len(valid),'RR_values':sorted({x['utility_rr'] for x in valid}),
        'pair_counts':counts})
for x in candidates:
    if not x['quality_usable']:
        failures.append({k:x[k] for k in ['id','row','trajectory_text','any_token_limit','quality_stop_reason',
            'r_format','r_retrieval','utility_rank1','E1_margin_A','E1_margin_B']})
summary={'ok':True,'fixed_context_checks':contexts,'greedy_reference_replays':replays,
    'failed_candidates':failures,'sensitivity_label':'posthoc description of predefined quality_usable subset; all-candidate primary analysis preserved',
    'quality_subset_RR':sensitivity,'actual_tool_executions_all_records':sum(x['tools_used'] for x in records),
    'model_generated_initial_tool_actions':sum(x['message_audit']['model_generated_initial_tool_actions'] for x in records)}
(r/'固定上下文与截断敏感性.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
