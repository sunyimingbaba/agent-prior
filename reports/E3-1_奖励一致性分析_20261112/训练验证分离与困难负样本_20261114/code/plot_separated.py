import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'analysis.json').read_text());policies=j['policies'];means=j['policy_means']
names={'baseline':'Initial','format_only':'Format only','global_margin':'Train-gallery margin','scene_margin':'Same-scene margin'}
x=np.arange(len(policies));colors=['#969ca5','#6096aa','#447552','#cd8e45'][:len(policies)]
fig,axes=plt.subplots(1,3,figsize=(15,5.8),gridspec_kw={'width_ratios':[1.1,1.1,1.7]})
rr=[means[p]['greedy']['utility_rr']['mean'] for p in policies];axes[0].bar(x,rr,color=colors,width=.65)
for i,v in enumerate(rr):axes[0].text(i,v+.008,f'{v:.4f}',ha='center',fontsize=10)
axes[0].set_xticks(x,[names[p] for p in policies],rotation=25,ha='right');axes[0].set_ylim(0,max(rr)*1.25+.01)
axes[0].set_ylabel('Image-mean reciprocal rank (RR)');axes[0].set_title('Greedy retrieval: fresh 50 val images',fontsize=11)
for k,mode in enumerate(['greedy','sample_mean']):
    values=[means[p][mode]['quality']['mean'] for p in policies];positions=x+(k-.5)*.34
    axes[1].bar(positions,values,.32,color=['#466b87','#8db9c3'][k],label=['Greedy','2 samples / image'][k])
    for i,(xi,v) in enumerate(zip(positions,values)):
        other=means[policies[i]]['sample_mean' if k==0 else 'greedy']['quality']['mean']
        if v==other and k==1:continue
        axes[1].text(x[i] if v==other else xi,v+(.055 if k==1 and abs(v-other)<.05 else .015),f'{100*v:.0f}%',ha='center',fontsize=8)
axes[1].set_xticks(x,[names[p] for p in policies],rotation=25,ha='right');axes[1].set_ylim(0,1.13)
axes[1].set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);axes[1].set_title('Complete, protocol-valid answers',fontsize=11)
chosen=j['decision']['chosen_by_training_only_gate'];contrasts=[]
for mode in ['greedy','sample_mean']:
    for base in (['global_margin','format_only'] if chosen=='scene_margin' else ['baseline','format_only']):
        contrasts.append((chosen+'_minus_'+base,mode,('Greedy' if mode=='greedy' else 'Sample mean')+' vs. '+names[base]))
for i,(name,mode,label) in enumerate(contrasts):
    s=j['paired'][name][mode]['utility_rr'];lo,hi=s['interval95']
    axes[2].errorbar(s['mean'],i,xerr=[[s['mean']-lo],[hi-s['mean']]],fmt='o',capsize=4,color='#2475a5' if mode=='greedy' else '#cc7336')
axes[2].set_yticks(range(len(contrasts)),[x[2] for x in contrasts]);axes[2].invert_yaxis();axes[2].axvline(0,color='.45',lw=1)
axes[2].xaxis.set_major_locator(MaxNLocator(5))
axes[2].set_xlabel('Paired RR difference; 95% image bootstrap');axes[2].set_title('Prespecified primary-policy comparison',fontsize=11)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax is axes[2] else 'y',alpha=.15);ax.set_axisbelow(True)
handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
fig.suptitle('Text-purged train reward / separate validation: short-run comparison',fontsize=15)
fig.text(.5,.015,'19 training images; 1,437 training-only reward captions. Branch gate fixed before fresh validation evaluation.',ha='center',fontsize=9,color='.35')
fig.tight_layout(rect=[0,.12,1,.94],w_pad=2.4);out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'separated_reward_fresh50.png',dpi=180);fig.savefig(out/'separated_reward_fresh50.pdf');print('FIGURE_EXPORTED')
