import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'analysis.json').read_text());groups=['train','val'];x=np.arange(2)
fig,axes=plt.subplots(1,3,figsize=(15.5,5.7),gridspec_kw={'width_ratios':[1,1,1.7]})
for panel,metric,title,target in [(0,'strict_quality','Matched strict protocol',.9),(1,'tool_usage','Actual tool execution',.8)]:
    for k,arm in enumerate(['c742_strict','sft_strict']):
        values=[j['groups'][g]['means'][arm]['sample_mean'][metric]['mean'] for g in groups];pos=x+(k-.5)*.34
        axes[panel].bar(pos,values,.32,color=['#607c98','#4b9983'][k],label=['Before training','After 30 updates'][k])
        for xi,v in zip(pos,values):axes[panel].text(xi,v+.02,f'{v:.1%}',ha='center',fontsize=9)
    axes[panel].axhline(target,color='.45',ls='--',lw=1)
    axes[panel].set_xticks(x,['Confirm train 63','Existing val 22'],rotation=15,ha='right');axes[panel].set_ylim(0,1.16)
    axes[panel].set_ylabel('Image-mean rate across four samples');axes[panel].set_title(title,fontsize=12)
arms=['historical_legacy','c742_strict','sft_strict','sft_legacy_greedy'];xx=np.arange(4)
for k,g in enumerate(groups):
    values=[j['groups'][g]['means'][arm]['greedy']['pairacc_B']['mean'] for arm in arms];pos=xx+(k-.5)*.34
    axes[2].bar(pos,values,.32,color=['#607c98','#8ebdb6'][k],label=['Confirm train 63','Existing val 22'][k])
    for xi,v in zip(pos,values):axes[2].text(xi,v+.02,f'{v:.1%}',ha='center',fontsize=8)
axes[2].set_xticks(xx,['C742 legacy\nhistorical','C742 strict\ncontrol','SFT strict','SFT legacy\ngreedy']);axes[2].set_ylim(0,1.16)
axes[2].set_ylabel('Wording B pair accuracy');axes[2].set_title('Quantity and controller compatibility',fontsize=12)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,loc='upper left',fontsize=8);axes[2].legend(frameon=False,loc='upper left',fontsize=8)
fig.suptitle('Complete tool-call supervision: fixed prompts, temperature 1, final checkpoint at update 30',fontsize=14)
fig.text(.5,.02,'Training: 15 images, format-token loss; body labels masked. Confirmation: 63 other train images. Existing val 22 remains supplementary.',ha='center',fontsize=9,color='.35')
fig.tight_layout(rect=[0,.09,1,.94],w_pad=2.4);out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'supervision_stability_and_quantity.png',dpi=180);fig.savefig(out/'supervision_stability_and_quantity.pdf')
raw=r/'原始运行记录';updates=[json.loads(s) for s in (raw/'training/updates.jsonl').read_text().splitlines()]
fig2,ax=plt.subplots(figsize=(8,4.5));ax.semilogy([x['step'] for x in updates],[x['loss'] for x in updates],marker='o',ms=4,color='#357b77')
ax.set_xlabel('Actual optimizer updates');ax.set_ylabel('Teacher-forced format-token cross-entropy');ax.set_title('Thirty fixed supervised updates; body-token losses masked')
ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);fig2.tight_layout()
fig2.savefig(out/'format_training_loss.png',dpi=180);fig2.savefig(out/'format_training_loss.pdf');print('SFT_FIGURES_EXPORTED')
