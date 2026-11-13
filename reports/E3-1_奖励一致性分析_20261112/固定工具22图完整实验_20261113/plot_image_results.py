"""Export a compact scientific figure of per-image reward/utility rank relations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
data=json.loads((args.run/'完整逐图统计与重采样.json').read_text());images=data['per_image']
args.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10,8),sharey=True,gridspec_kw={'wspace':.12})
y=np.arange(len(images));palette={'positive':'#3178A6','negative':'#C66535'}
for ax,metric,title in zip(axes,['utility_rr','utility_margin'],['Retrieval reciprocal rank','Best-positive minus best-negative cosine']):
    for j,image in enumerate(images):
        stats=image['subsets']['all']['r_retrieval'][metric];rho=stats['spearman']
        if rho is None:
            ax.text(0,j,'N/A',ha='center',va='center',color='#666666',fontsize=7)
        else:
            ax.barh(j,rho,height=.68,color=palette['positive'] if rho>=0 else palette['negative'])
    ax.axvline(0,color='#555555',linewidth=.7);ax.set_xlim(-1.05,1.05)
    ax.set_xticks([-1,-.5,0,.5,1]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title(title,fontsize=10,pad=10);ax.set_xlabel('Within-image Spearman correlation')
labels=[f"{x['image']}  ({x['usable']}/8 pass)"+(' *' if x['pilot_image'] else '') for x in images]
axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_ylim(len(images)-.3,-.7)
fig.suptitle('Fixed-tool answer candidates: reward vs. retrieval utility',fontsize=12,y=.97)
fig.text(.18,.025,'All 8 candidates retained per image. N/A: a constant variable.  * Previously inspected pilot image.',fontsize=8,color='#555555')
fig.subplots_adjust(left=.21,right=.98,top=.91,bottom=.09)
fig.savefig(args.out/'reward_utility_by_image.png',dpi=220)
fig.savefig(args.out/'reward_utility_by_image.pdf')
plt.close(fig)
print('FIGURE_EXPORTED')
