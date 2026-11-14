import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'protocol_analysis.json').read_text());q=json.loads((r/'analysis.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,5.6),gridspec_kw={'width_ratios':[1,1.2,1.3]})
groups=['confirmation_train63','val22'];x=np.arange(2);colors=['#507593','#b27d4e']
for k,arm in enumerate(['baseline','selected']):
    values=[j['groups'][g]['means'][arm]['candidate']['quality']['mean'] for g in groups];pos=x+(k-.5)*.34
    axes[0].bar(pos,values,.32,color=colors[k],label=['Previous protocol','Selected protocol'][k])
    for xi,value in zip(pos,values):axes[0].text(xi,value+.02,f'{value:.1%}',ha='center',fontsize=9)
axes[0].axhline(.9,color='.4',ls='--',lw=1);axes[0].text(-.42,.915,'90% target',color='.4',fontsize=8)
axes[0].set_xticks(x,['Train confirmation 63','Existing val 22'],rotation=15,ha='right');axes[0].set_ylim(0,1.16)
axes[0].set_ylabel('Sampled protocol-valid rate');axes[0].set_title('Four samples per image')
pilot=j['selection']['pilot_stats'];values=[x['quality'] for x in pilot]
axes[1].bar(np.arange(3),values,color=['#bcc9d4','#87a8bb','#507593'])
for i,value in enumerate(values):axes[1].text(i,value+.02,f'{value:.1%}',ha='center',fontsize=9)
axes[1].axhline(.9,color='.4',ls='--',lw=1);axes[1].set_ylim(0,1.16)
axes[1].set_xticks(np.arange(3),['Full-tag stop\nT=1','Neutral contract\nT=1','Neutral contract\nT=0.7'])
axes[1].set_ylabel('Sampled protocol-valid rate');axes[1].set_title('Train calibration 15; selection only')
selectors=['greedy','format_first','quantity_quality'];sx=np.arange(3)
for k,split in enumerate(['train','val']):
    values=[q['groups'][split]['selector_means'][s]['pairacc_B']['mean'] for s in selectors];pos=sx+(k-.5)*.34
    axes[2].bar(pos,values,.32,color=['#507593','#8ebcb5'][k],label=['Train 78','Existing val 22'][k])
    for xi,value in zip(pos,values):axes[2].text(xi,value+.02,f'{value:.1%}',ha='center',fontsize=8)
axes[2].set_xticks(sx,['Greedy','Format first','Quantity + quality'],rotation=15,ha='right');axes[2].set_ylim(0,1.16)
axes[2].set_ylabel('Wording B pair accuracy');axes[2].set_title('Frozen-policy quantity diagnostic')
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=8,loc='upper left');axes[2].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('Protocol stability controls: fixed weights, actual tools, all failures retained',fontsize=14)
fig.text(.5,.02,'Protocol chosen using calibration 15 only. Quantity selection uses confirmed labels offline. Zero optimizer updates.',ha='center',fontsize=9,color='.35')
fig.tight_layout(rect=[0,.09,1,.94],w_pad=2.5);out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'protocol_stability.png',dpi=180);fig.savefig(out/'protocol_stability.pdf');print('PROTOCOL_FIGURE_EXPORTED')
