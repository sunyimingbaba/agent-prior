import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;j=read(r/'analysis.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
cohorts=['training8','confirmation55','existing_val22'];labels=['Training 8','Confirm 52','Existing val 22'];colors=['#0072B2','#009E73']
fig,axes=plt.subplots(1,3,figsize=(15.5,5.1),gridspec_kw={'width_ratios':[1,1.15,1]});x=np.arange(3)
ax=axes[0]
for k,arm in enumerate(['before','after']):
    vals=[j['groups'][c]['macro_absolute'][arm]['sample_mean']['pairacc_B']*100 for c in cohorts];pos=x+(k-.5)*.35
    ax.bar(pos,vals,.33,color=colors[k],label=['Initial SFT-30','After 32 updates'][k])
    for xp,v in zip(pos,vals):ax.text(xp,v+1.5,f'{v:.1f}',ha='center',fontsize=8)
ax.set_xticks(x,labels,rotation=12,ha='right');ax.set_ylim(0,112);ax.set_ylabel('Count-macro B pair accuracy (%)');ax.set_title('Mean of four samples',fontsize=12);ax.legend(frameon=False,fontsize=8)
ax=axes[1]
for k,c in enumerate(cohorts):
    v=j['groups'][c]['macro']['after_minus_before__sample_mean']['pairacc_B'];m=v['mean']*100;lo,hi=np.array(v['interval95'])*100
    ax.errorbar(m,k,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#8C8C8C','#0072B2','#009E73'][k]);ax.annotate(f'{m:+.2f}',(m,k),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
ax.set_yticks(range(3),labels);ax.set_ylim(-.5,2.6);ax.axvline(0,color='.5',ls='--',lw=1);ax.set_xlabel('Final minus initial (percentage points)');ax.set_title('Paired macro-B change\n95% stratified image-bootstrap intervals',fontsize=11)
ax=axes[2];x=np.arange(2)
for k,selector in enumerate(['greedy','shared']):
    vals=[j['groups'][c]['macro_absolute']['after'][selector]['pairacc_B']*100 for c in cohorts[1:]];pos=x+(k-.5)*.35
    ax.bar(pos,vals,.33,color=['#0072B2','#E69F00'][k],label=['Own greedy state','Shared reference state'][k])
    for xp,v in zip(pos,vals):ax.text(xp,v+1.5,f'{v:.1f}',ha='center',fontsize=9)
ax.set_xticks(x,labels[1:],rotation=12,ha='right');ax.set_ylim(0,112);ax.set_ylabel('Count-macro B pair accuracy (%)');ax.set_title('After training: own versus shared',fontsize=12);ax.legend(frameon=False,fontsize=8)
for ax in axes:ax.grid(axis='x' if ax==axes[1] else 'y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Count-balanced pilot: fixed update 32, 2 training images per count, frozen downstream AGSE',fontsize=13)
fig.text(.5,.025,'Macro: equal weights for counts 1, 2, 3, 4. Confirmation set: 55 images, including 52 in macro. Existing val 22 remains supplementary. Shared reference: 102.tif.',ha='center',fontsize=8.8)
fig.tight_layout(rect=[0,.1,1,.93],w_pad=2.5);out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'balanced32_macro_and_shared_control.png',dpi=300);fig.savefig(out/'balanced32_macro_and_shared_control.pdf');print('BALANCED32_FIGURE_EXPORTED')
