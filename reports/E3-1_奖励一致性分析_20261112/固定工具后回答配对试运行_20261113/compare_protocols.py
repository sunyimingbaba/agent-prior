"""Compare paired seeds across original and repaired generation conditions."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--old', type=Path, required=True)
args = parser.parse_args()
r, old = args.run, args.old
plan, before = [json.loads((p/'plan.json').read_text()) for p in [r, old]]
for key in ['selected_rows','seeds','proxy_reward','utility_model_sha256','lora_weights_sha256']:
    assert plan[key] == before[key], key
galleries = [json.loads((p/'scoring/text_gallery.json').read_text()) for p in [r,old]]
for key in ['text_order','positive_gallery_columns','quantity_text_columns']:
    assert galleries[0][key] == galleries[1][key]
scores = [json.loads((p/'scoring/scores.json').read_text()) for p in [r,old]]
new = {(x['row'],x['seed']):x for x in scores[0] if x['role']=='candidate'}
previous = {(x['row'],x['seed']):x for x in scores[1] if x['role']=='candidate'}
assert set(new) == set(previous) and len(new)==24
metrics = ['r_retrieval','r_format','r_total','utility_rr','utility_margin','E1_margin_A','E1_margin_B']
paired = []
for key in sorted(new):
    a, b = previous[key], new[key]
    row = {'row':key[0],'seed':key[1],'image':b['image'],'repaired_usable':b['quality_usable'],
        'repaired_stop_reason':b['quality_stop_reason']}
    for metric in metrics:
        row[metric+'_old'], row[metric+'_repaired'] = a[metric],b[metric]
        row[metric+'_difference'] = b[metric]-a[metric]
    paired.append(row)
by_image = []
for row in plan['selected_rows']:
    group = [p for p in paired if p['row']==row]
    old_reference = next(x for x in scores[1] if x['row']==row and x['role']=='reference')
    new_reference = next(x for x in scores[0] if x['row']==row and x['role']=='reference')
    deltas = [x['utility_rr_difference'] for x in group]
    by_image.append({'row':row,'image':group[0]['image'], 'usable_candidates':sum(x['repaired_usable'] for x in group),
        'old_greedy_utility_rank1':old_reference['utility_rank1'],
        'repaired_greedy_utility_rank1':new_reference['utility_rank1'],
        'paired_RR_improved':sum(x>1e-12 for x in deltas),'paired_RR_worsened':sum(x< -1e-12 for x in deltas),
        'paired_RR_tied':sum(abs(x)<=1e-12 for x in deltas),
        'mean_paired_changes':{m:statistics.mean(x[m+'_difference'] for x in group) for m in metrics}})
analyses = [json.loads((p/'analysis.json').read_text()) for p in [r,old]]
summary = {'ok':True,'same_frozen_scorers_and_text_gallery':True,'paired_candidates':24,
    'repaired_usable_candidates':sum(x['repaired_usable'] for x in paired),
    'repaired_stop_reasons':dict(collections.Counter(x['repaired_stop_reason'] for x in paired)),
    'old_flags':analyses[1]['candidate_flags'],'repaired_flags':analyses[0]['candidate_flags'],
    'old_format_rewards':analyses[1]['candidate_format_reward_counts'],
    'repaired_format_rewards':analyses[0]['candidate_format_reward_counts'],
    'per_image':by_image, 'original_reward_utility_statistics':analyses[1]['aggregate'],
    'repaired_reward_utility_statistics':analyses[0]['aggregate']}
(r/'协议对照汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (r/'同图同种子协议对照.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(paired[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(paired)
print(json.dumps({k:summary[k] for k in ['ok','repaired_usable_candidates','repaired_stop_reasons','old_flags','repaired_flags','per_image']},ensure_ascii=False))
