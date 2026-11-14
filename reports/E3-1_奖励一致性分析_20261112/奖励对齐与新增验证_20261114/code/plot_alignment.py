"""Plot the prespecified new50 result and paired image bootstrap intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
g=json.loads((r/'analysis.json').read_text())['groups']['new50']
policies=['baseline','format_only','retrieval','aligned_margin']
labels=['Initial','Format only','Proxy reward','Aligned margin']
colors=['#969ca5','#6096aa','#d59a59','#447552'];x=np.arange(4)
fig,axes=plt.subplots(1,3,figsize=(15,5.7),gridspec_kw={'width_ratios':[1.1,1.1,1.65]})
rr=[g['policy_means'][p]['greedy']['utility_rr']['mean'] for p in policies]
axes[0].bar(x,rr,color=colors,width=.65)
for i,v in enumerate(rr):axes[0].text(i,v+.008,f'{v:.4f}',ha='center',fontsize=10)
axes[0].set_xticks(x,labels,rotation=25,ha='right');axes[0].set_ylim(0,max(rr)*1.24+.01)
axes[0].set_ylabel('Image-mean reciprocal rank (RR)');axes[0].set_title('Greedy retrieval: new 50 images',fontsize=11)
for j,mode in enumerate(['greedy','sample_mean']):
    vals=[g['policy_means'][p][mode]['quality']['mean'] for p in policies]
    pos=x+(j-.5)*.34
    axes[1].bar(pos,vals,.32,color=['#466b87','#8db9c3'][j],label=['Greedy','2 samples / image'][j])
    for k,(xi,v) in enumerate(zip(pos,vals)):
        same=g['policy_means'][policies[k]]['greedy']['quality']['mean']==g['policy_means'][policies[k]]['sample_mean']['quality']['mean']
        if same and j==1:continue
        other=g['policy_means'][policies[k]]['greedy' if j else 'sample_mean']['quality']['mean']
        offset=.055 if j==1 and abs(other-v)<.05 else .015
        axes[1].text(x[k] if same else xi,v+offset,f'{100*v:.0f}%',ha='center',fontsize=8)
axes[1].set_xticks(x,labels,rotation=25,ha='right');axes[1].set_ylim(0,1.13)
axes[1].set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);axes[1].set_title('Complete, protocol-valid answers',fontsize=11)
handles,legend_labels=axes[1].get_legend_handles_labels()
fig.legend(handles,legend_labels,frameon=False,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,fontsize=9)
contrasts=[('retrieval','greedy','Greedy vs. proxy'),('format_only','greedy','Greedy vs. format'),
           ('baseline','greedy','Greedy vs. initial'),('retrieval','sample_mean','Sample mean vs. proxy'),
           ('format_only','sample_mean','Sample mean vs. format')]
for i,(base,mode,label) in enumerate(contrasts):
    s=g['paired']['aligned_margin_minus_'+base][mode]['utility_rr'];lo,hi=s['interval95']
    axes[2].errorbar(s['mean'],i,xerr=[[s['mean']-lo],[hi-s['mean']]],fmt='o',capsize=4,ms=6,
                     color='#2475a5' if mode=='greedy' else '#cc7336')
axes[2].set_yticks(range(len(contrasts)),[x[2] for x in contrasts]);axes[2].invert_yaxis()
axes[2].axvline(0,color='.45',lw=1);axes[2].set_xlabel('Paired RR difference; 95% image bootstrap')
axes[2].set_title('Aligned-reward policy gain',fontsize=11)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax is axes[2] else 'y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Final-retrieval reward alignment: additional validation',fontsize=15)
fig.text(.5,.015,'Same 20 optimization images and initialization. New50 selected before evaluation. All generation failures retained.',ha='center',fontsize=9,color='.35')
fig.tight_layout(rect=[0,.12,1,.94],w_pad=2.4)
out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'reward_alignment_new50.png',dpi=180);fig.savefig(out/'reward_alignment_new50.pdf')
print('FIGURE_EXPORTED')
