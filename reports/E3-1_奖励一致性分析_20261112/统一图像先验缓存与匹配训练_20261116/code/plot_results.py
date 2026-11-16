"""Local standalone figures, drawn only from the sealed numerical analysis."""
import argparse
import csv
import json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
os.environ['MPLCONFIGDIR']=str(r/'scratch/mpl_local')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
j=json.loads((r/'analysis.json').read_text());out=r/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelsize':11,'axes.titlesize':12,'savefig.dpi':250,'pdf.fonttype':42})
arms=['native','calibrated_own','calibrated_shared','learned_visual_only']
labels=['Native','Calibrated own','Calibrated shared','Learned visual']
colors=['#0072B2','#009E73','#999999','#E69F00']
proto='repository_duplicate_token_positive';strict='strict_image_correspondence'
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for arm,label,c in zip(arms,labels,colors):
    h=j['training_history'][arm];v=[h['initial']['mR']]+[x['mR'] for x in h['epochs']]
    axes[0].plot(range(11),np.array(v)*100,label=label,color=c,marker='o',ms=3,lw=1.6)
    e=h['selected']['epoch'];axes[0].scatter([e],[v[e]*100],marker='*',s=130,color=c,edgecolor='black',linewidth=.45,zorder=4)
axes[0].set(xlabel='Readout adaptation epoch',ylabel='Validation mR (%)',title='Fixed 10-epoch budget; stars show val selections',xticks=range(11),ylim=(0,100))
axes[0].grid(axis='y',alpha=.18);axes[0].legend(fontsize=9,loc='lower right')
x=np.arange(4)
for d,protocol,color,name in [(-.18,proto,'#0072B2','Repository positives'),(.18,strict,'#D55E00','Strict image positives')]:
    vals=np.array([j['test_summary'][protocol][arm]['mR'] for arm in arms])*100
    bars=axes[1].bar(x+d,vals,width=.34,color=color,label=name)
    for b,v in zip(bars,vals):axes[1].text(b.get_x()+b.get_width()/2,v+.8,f'{v:.2f}',ha='center',fontsize=8)
axes[1].set(xticks=x,xticklabels=labels,ylabel='Test mR (%)',title='Held-out test: 210 images / 1050 captions',ylim=(0,100))
axes[1].tick_params(axis='x',rotation=15);axes[1].grid(axis='y',alpha=.18);axes[1].set_axisbelow(True);axes[1].legend(fontsize=8,loc='lower right')
fig.suptitle('Image-only priors: matched readout adaptation from identical weights',fontsize=14)
for ext in ['png','pdf']:fig.savefig(out/f'matched_readout_curves_and_test.{ext}',bbox_inches='tight')
plt.close(fig)
contrasts=[('calibrated_own','native'),('calibrated_own','calibrated_shared'),('calibrated_own','learned_visual_only'),
           ('native','native_same_checkpoint_shared'),('calibrated_own','calibrated_own_same_checkpoint_shared')]
names=['Cal own minus native','Cal own minus cal shared','Cal own minus learned visual',
       'Native own minus same-weight shared','Cal own minus same-weight shared']
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for d,protocol,c,label in [(-.12,proto,'#0072B2','Repository positives'),(.12,strict,'#D55E00','Strict image positives')]:
    for i,(left,right) in enumerate(contrasts):
        row=next(x for x in j['paired_mR_intervals'] if x['protocol']==protocol and x['left']==left and x['right']==right)
        v=row['delta_mR']*100;lo,hi=np.array(row['interval95'])*100
        ax.errorbar(v,i+d,xerr=np.array([[v-lo],[hi-v]]),fmt='o',color=c,capsize=3,label=label if i==0 else None)
ax.axvline(0,color='#555555',ls='--',lw=1);ax.set(yticks=range(5),yticklabels=names,
     xlabel='Paired test mR difference (percentage points)',title='95% paired image-cluster intervals; fixed full gallery')
ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.legend(loc='best',fontsize=9)
for ext in ['png','pdf']:fig.savefig(out/f'prior_specificity_paired_intervals.{ext}',bbox_inches='tight')
plt.close(fig)
rows=[]
for protocol,table in j['test_summary'].items():
    for arm,metrics in table.items():rows.append({'protocol':protocol,'arm':arm,**{k:v*100 for k,v in metrics.items()}})
with (r/'检索结果汇总.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('MATCHED_FIGURES_AND_METRICS_CSV_WRITTEN')
