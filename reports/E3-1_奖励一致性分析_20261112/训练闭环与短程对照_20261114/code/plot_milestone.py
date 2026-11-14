import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
result=json.loads((r/'analysis.json').read_text());g=result['groups']['all22']
policies=['baseline','format_only','retrieval'];labels=['Initial','Format only','Retrieval + format']
colors=['#8a9199','#2d7faa','#d17a3c']
fig,axes=plt.subplots(1,3,figsize=(14.5,5.6),gridspec_kw={'width_ratios':[1,1.1,1.65]})
x=np.arange(3)
rr=[g['policy_means'][p]['greedy']['utility_rr']['mean'] for p in policies]
axes[0].bar(x,rr,color=colors,width=.65)
for i,v in enumerate(rr):axes[0].text(i,v+.008,f'{v:.4f}',ha='center',fontsize=10)
axes[0].set_xticks(x,labels,rotation=20,ha='right');axes[0].set_ylabel('Image-mean reciprocal rank (RR)')
axes[0].set_ylim(0,max(rr)*1.24+.01);axes[0].set_title('Greedy retrieval: same 22 images',fontsize=11)
for j,mode in enumerate(['greedy','sample_mean']):
    quality=[g['policy_means'][p][mode]['quality']['mean'] for p in policies]
    positions=x+(j-.5)*.34
    axes[1].bar(positions,quality,.32,color=['#466b87','#8db9c3'][j],label=['Greedy','4 samples / image'][j])
    for k,(xi,v) in enumerate(zip(positions,quality)):
        equal=g['policy_means'][policies[k]]['greedy']['quality']['mean']==g['policy_means'][policies[k]]['sample_mean']['quality']['mean']
        if equal and j==1:continue
        axes[1].text(x[k] if equal else xi,v+.012,f'{100*v:.0f}%',ha='center',fontsize=8)
axes[1].set_xticks(x,labels,rotation=20,ha='right');axes[1].set_ylim(0,1.13)
axes[1].set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
axes[1].set_title('Complete, protocol-valid answers',fontsize=11)
handles,legend_labels=axes[1].get_legend_handles_labels()
fig.legend(handles,legend_labels,frameon=False,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,fontsize=9)
contrasts=[('retrieval_minus_baseline','greedy','Greedy vs. initial'),
           ('retrieval_minus_format_only','greedy','Greedy vs. format only'),
           ('retrieval_minus_baseline','sample_mean','Sample mean vs. initial'),
           ('retrieval_minus_format_only','sample_mean','Sample mean vs. format only')]
for i,(name,mode,label) in enumerate(contrasts):
    v=g['paired'][name][mode]['utility_rr'];lo,hi=v['interval95']
    axes[2].errorbar(v['mean'],i,xerr=[[v['mean']-lo],[hi-v['mean']]],fmt='o',capsize=4,
                     color='#2475a5' if mode=='greedy' else '#cc7336',ms=6)
axes[2].set_yticks(range(4),[x[2] for x in contrasts]);axes[2].invert_yaxis()
axes[2].axvline(0,color='.45',lw=1);axes[2].set_xlabel('Paired RR difference; 95% image bootstrap')
axes[2].set_title('Retrieval-reward policy gain',fontsize=11)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y' if ax is not axes[2] else 'x',alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle('Corrected live-tool training: controlled short-run comparison',fontsize=15)
fig.text(.5,.015,'20 planned prompt groups per arm; actual optimizer updates are recorded separately. Frozen final retrieval evaluator.',
         ha='center',fontsize=9,color='.35')
fig.tight_layout(rect=[0,.12,1,.94],w_pad=2.5)
out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'short_training_comparison.png',dpi=180);fig.savefig(out/'short_training_comparison.pdf')
print('FIGURE_EXPORTED')
