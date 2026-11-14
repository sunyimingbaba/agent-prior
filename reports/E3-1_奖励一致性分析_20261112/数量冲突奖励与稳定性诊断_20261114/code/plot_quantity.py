import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'analysis.json').read_text());selectors=['greedy','format_first','quantity_quality','reverse_quality']
names=['Greedy','Format first','Quantity + quality','Reverse quantity']
fig,axes=plt.subplots(1,3,figsize=(15.5,6),gridspec_kw={'width_ratios':[1.2,1.15,1.65]})
x=np.arange(4)
for k,split in enumerate(['train','val']):
    values=[j['groups'][split]['selector_means'][s]['pairacc_B']['mean'] for s in selectors];pos=x+(k-.5)*.34
    axes[0].bar(pos,values,.32,color=['#466b87','#8db9c3'][k],label=['Train 78','Val 22'][k])
    for xi,v in zip(pos,values):axes[0].text(xi,v+.015,f'{v:.0%}',ha='center',fontsize=8)
axes[0].set_xticks(x,names,rotation=25,ha='right');axes[0].set_ylim(0,1.13);axes[0].set_ylabel('Held-out wording B pair accuracy')
axes[0].set_title('Confirmed-count candidate diagnostic',fontsize=11)
quality_modes=['sample_mean','quantity_raw','quantity_quality'];qx=np.arange(3)
for k,split in enumerate(['train','val']):
    values=[j['groups'][split]['selector_means'][s]['quality']['mean'] for s in quality_modes];pos=qx+(k-.5)*.34
    axes[1].bar(pos,values,.32,color=['#466b87','#8db9c3'][k])
    for i,(xi,v) in enumerate(zip(pos,values)):
        other=j['groups']['val' if split=='train' else 'train']['selector_means'][quality_modes[i]]['quality']['mean']
        axes[1].text(xi,v+(.055 if k==1 and abs(v-other)<.05 else .015),f'{v:.0%}',ha='center',fontsize=8)
axes[1].set_xticks(qx,['All samples','Raw quantity max','Quantity + quality'],rotation=25,ha='right')
axes[1].set_ylim(0,1.16);axes[1].set_ylabel('Protocol-valid output rate');axes[1].set_title('Generation and selection stability',fontsize=11)
contrasts=[('train','quantity_quality_minus_format_first','Train: quantity vs. format'),
           ('val','quantity_quality_minus_format_first','Val: quantity vs. format'),
           ('train','quantity_quality_minus_reverse_quality','Train: quantity vs. reverse'),
           ('val','quantity_quality_minus_reverse_quality','Val: quantity vs. reverse')]
for i,(split,key,label) in enumerate(contrasts):
    s=j['groups'][split]['paired'][key]['margin_B'];lo,hi=s['interval95']
    axes[2].errorbar(s['mean'],i,xerr=[[s['mean']-lo],[hi-s['mean']]],fmt='o',capsize=4,color='#2475a5' if split=='train' else '#cc7336')
axes[2].set_yticks(range(4),[x[2] for x in contrasts]);axes[2].invert_yaxis();axes[2].axvline(0,color='.45',lw=1)
axes[2].xaxis.set_major_locator(MaxNLocator(4));axes[2].ticklabel_format(axis='x',style='sci',scilimits=(-3,-3),useMathText=True)
axes[2].set_xlabel('Paired B-margin difference; 95% image bootstrap');axes[2].set_title('Transfer from reward wording A to B',fontsize=11)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax is axes[2] else 'y',alpha=.15);ax.set_axisbelow(True)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
fig.suptitle('Frozen policy: quantity-counterfactual reward and stability',fontsize=15)
fig.text(.5,.015,'Quantity/reverse selection uses confirmed labels offline. 100 images; 4 samples + 1 greedy each. Policy weights remain fixed.',ha='center',fontsize=9,color='.35')
fig.tight_layout(rect=[0,.12,1,.94],w_pad=2.4);out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'quantity_reward_and_stability.png',dpi=180);fig.savefig(out/'quantity_reward_and_stability.pdf');print('FIGURE_EXPORTED')
