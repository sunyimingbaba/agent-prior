#!/usr/bin/env python3
"""Export a standalone comparison of policy means and image-paired gains."""
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
result = json.loads((r/'选择收益完整统计.json').read_text())
diagnostic = json.loads((r/'选择结果归因诊断.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(13.4, 9.2), gridspec_kw={'width_ratios': [1, 1.08]})
colors = ['#2475a5', '#cc7336']
policies = ['greedy', 'uniform_random', 'retrieval_reward', 'total_reward', 'candidate_oracle']
labels = ['Greedy reference', 'Uniform random', 'Retrieval reward', 'Total reward', 'Candidate pool best']
y = np.arange(5)
for j, subset in enumerate(['all', 'quality_pass']):
    values = [result['groups']['all22']['subsets'][subset]['policies'][q]['utility_rr']['mean'] for q in policies]
    positions = y+(j-.5)*.31
    axes[0].barh(positions, values, height=.29, color=colors[j], alpha=.92)
    for yi, x in zip(positions, values):
        axes[0].text(x+.004, yi, f'{x:.4f}', va='center', fontsize=9)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlim(0, .445)
axes[0].set_xlabel('Image-mean expected reciprocal rank (RR)')
axes[0].set_title('22 images: exact selection expectations', fontsize=13)
axes[0].grid(axis='x', alpha=.18)
axes[0].set_axisbelow(True)
images = diagnostic['groups']['all22']['per_image']
for j, key in enumerate(['all_reward_minus_random', 'quality_reward_minus_random']):
    axes[1].scatter([x[key] for x in images], np.arange(len(images))+(j-.5)*.16,
                    color=colors[j], s=35, marker=['o', 's'][j],
                    label=['All candidates', 'Quality-pass candidates'][j], zorder=3)
axes[1].set_yticks(np.arange(len(images)), [x['image'] for x in images])
axes[1].invert_yaxis()
axes[1].axvline(0, color='.4', lw=1)
axes[1].grid(axis='x', alpha=.18)
axes[1].set_xlim(-.09, .42)
axes[1].set_xlabel('Retrieval-reward selection minus random: paired RR')
axes[1].set_title('Per-image gains use the same eligible pool', fontsize=13)
axes[1].legend(loc='lower right', frameon=False, fontsize=9)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Fixed-tool answer selection: utility and output quality', fontsize=16, y=.98)
fig.text(.5, .025, 'Reward ties and uniform random are averaged exactly. Candidate pool best uses the evaluated utility as an oracle.',
         ha='center', fontsize=10, color='.35')
fig.tight_layout(rect=[0, .045, 1, .95], w_pad=3)
out = r/'figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'selection_utility.png', dpi=180)
fig.savefig(out/'selection_utility.pdf')
print('FIGURE_EXPORTED')
