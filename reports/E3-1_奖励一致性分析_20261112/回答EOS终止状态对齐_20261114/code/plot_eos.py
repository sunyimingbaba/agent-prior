import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'analysis.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
arms=['c742_strict','sft_strict','sft_fulltool_eos'];labels=['C742\nanswer-close','SFT-30\nanswer-close','SFT-30\nnative EOS']
fig,axes=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1.1,1.1,1]})
for ax,split,title in zip(axes[:2],['train','val'],['Confirmation train (63 images)','Existing val (22 images; supplementary)']):
    x=np.arange(3)
    for k,selector in enumerate(['greedy','sample_mean']):
        vals=[j['groups'][split]['means'][arm][selector]['pairacc_B']['mean']*100 for arm in arms]
        pos=x+(k-.5)*.34
        ax.bar(pos,vals,.32,color=['#0072B2','#009E73'][k],label=['Greedy','Mean of four samples'][k])
        for xi,v in zip(pos,vals):ax.text(xi,v+1.5,f'{v:.1f}',ha='center',fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylim(0,110);ax.set_ylabel('Wording B pair accuracy (%)');ax.set_title(title,fontsize=11)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
ax=axes[2]
for k,(split,label) in enumerate([('train','Train 63'),('val','Val 22')]):
    v=j['groups'][split]['paired']['sft_fulltool_eos_minus_sft_strict__sample_mean']['margin_B'];m=v['mean'];lo,hi=v['interval95']
    ax.errorbar(m,k,xerr=[[m-lo],[hi-m]],fmt='o',color=['#0072B2','#009E73'][k],capsize=4)
    ax.annotate(f'{m:+.4f}\n[{lo:+.4f}, {hi:+.4f}]',(m,k),xytext=(0,17),textcoords='offset points',ha='center',fontsize=8)
ax.axvline(0,color='.45',lw=1,ls='--');ax.set_yticks([0,1],['Train 63','Val 22']);ax.set_ylim(-.45,1.65)
ax.set_xlabel('Change in B similarity margin');ax.set_title('Native EOS minus answer-close\n95% image-bootstrap interval',fontsize=11);ax.grid(axis='x',alpha=.15)
fig.suptitle('Terminal-state alignment: frozen update-30 weights, complete real tool calls',fontsize=13)
fig.text(.5,.025,'All 425 prior trajectories are re-generated at fixed seeds; 0 optimizer updates. Validation images are excluded from adoption gates.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.09,1,.93],w_pad=2)
out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'eos_alignment_quantity.png',dpi=300);fig.savefig(out/'eos_alignment_quantity.pdf')
print('EOS_FIGURES_EXPORTED')
