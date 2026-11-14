import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
r = a.root
analysis = json.loads((r/'analysis.json').read_text())
geometry = json.loads((r/'state_comparisons.json').read_text())
raw = r/'原始运行记录'
plan = json.loads((raw/'plan.json').read_text())
scores = json.loads((raw/'scoring/scores.json').read_text())
roles = {x['source_id']: x['role'] for x in plan['records']}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={'width_ratios': [1.05, 1.2]})
categories = ['saved_vs_forward_penultimate', 'saved_vs_grpo_unpadded', 'grpo_unpadded_vs_group_padded']
colors = ['#c96f37', '#287aa5', '#58925c']
for role_i, role in enumerate(['selected_failure', 'best_quality_reward', 'greedy_reference']):
    for cat_i, cat in enumerate(categories):
        values = [x['cosine'] for x in geometry if x['comparison'] == cat and roles[x['source_id']] == role]
        offset = np.linspace(-.055, .055, len(values))+(role_i-1)*.15
        axes[0].scatter(cat_i+offset, values, s=32, color=colors[role_i], alpha=.8,
                        label=['Selected failure', 'Best quality-pass reward', 'Greedy reference'][role_i]
                        if cat_i == 0 else None)
axes[0].set_xticks(range(3), ['Cached vs.\nforward same position', 'Cached vs.\nGRPO final position', 'GRPO final vs.\nright-padded final'])
axes[0].set_ylim(0, 1.055)
axes[0].set_ylabel('Hidden-state cosine similarity')
axes[0].set_title('Same 18 archived sequences', fontsize=12)
axes[0].grid(axis='y', alpha=.18)
axes[0].legend(loc='lower left', fontsize=8, frameon=False)
index = {(x['source_id'],x['variant']):x for x in scores}
items = analysis['extensions']
x = np.arange(len(items))
for i, variant in enumerate(['saved_offline', 'extended_offline']):
    values = [index[item['source_id'],variant]['utility_rr'] for item in items]
    bars = axes[1].bar(x+(i-.5)*.34, values, .32, color=['#2475a5','#cc7336'][i],
                       label=['Original answer', 'Same seed, budget 256'][i])
    for k, (bar, value) in enumerate(zip(bars, values)):
        same = index[items[k]['source_id'],'saved_offline']['utility_rr'] == index[items[k]['source_id'],'extended_offline']['utility_rr']
        if same and i == 1:
            continue
        label_x = x[k] if same else bar.get_x()+bar.get_width()/2
        axes[1].text(label_x, value+.012, f'{value:.3f}',
                     ha='center', va='bottom', fontsize=8)
axes[1].set_xticks(x, [item['image']+(' *' if not item['quality_usable'] else '') for item in items], rotation=30, ha='right')
axes[1].set_ylim(0, 1.14)
axes[1].set_ylabel('Actual retrieval reciprocal rank (RR)')
axes[1].set_title('Five truncated answers complete with identical prefixes', fontsize=11)
axes[1].legend(frameon=False, fontsize=8, loc='upper right')
axes[1].grid(axis='y', alpha=.18)
axes[1].set_axisbelow(True)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Terminal position, padding, and generation length change the evaluated state', fontsize=14)
fig.text(.5, .025, 'Frozen weights and fixed archived tool context.  * The malformed closing tag persists at the same 15-token sequence.',
         ha='center', fontsize=9, color='.35')
fig.tight_layout(rect=[0,.055,1,.94], w_pad=2)
out = r/'figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'state_parity_effects.png', dpi=180)
fig.savefig(out/'state_parity_effects.pdf')
print('FIGURE_EXPORTED')
