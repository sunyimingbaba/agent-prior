import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;j=read(r/'analysis.json');v=read(r/'full_validation_metrics.json');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False});out=r/'figures';out.mkdir(exist_ok=True)
fig,axs=plt.subplots(1,3,figsize=(15.7,4.9),gridspec_kw={'width_ratios':[1,1.1,1.2]});keys=['original_cached/original_cached','initial_SFT30/greedy','balanced32/greedy'];labels=['Original cache','SFT-30 EOS','Balanced-32 EOS'];colors=['#999999','#0072B2','#009E73'];vals=[j['distribution'][k]['linear_norm']['mean'] for k in keys];axs[0].bar(range(3),vals,color=colors)
for i,x in enumerate(vals):axs[0].text(i,x+2,f'{x:.1f}',ha='center')
axs[0].set_xticks(range(3),labels,rotation=17,ha='right');axs[0].set_ylabel('Mean projected pre-LayerNorm norm');axs[0].set_title('85 images per source');axs[0].set_ylim(0,145)
g=j['groups']['confirmation55']['balanced32'];arms=['baseline_own','selected_own','original_cached','learned_only'];labs=['Native','Mean shift','Original cache','Learned only'];vals=[g['groups'][a]['macro']['B_hard_correct']['mean']*100 for a in arms];axs[1].bar(range(4),vals,color=['#0072B2','#009E73','#999999','#E69F00'])
for i,x in enumerate(vals):axs[1].text(i,x+1.3,f'{x:.2f}',ha='center',fontsize=9)
axs[1].set_xticks(range(4),labs,rotation=17,ha='right');axs[1].set_ylim(0,65);axs[1].set_ylabel('Count-macro B all-count accuracy (%)');axs[1].set_title('Confirmation: 52 common-count images',fontsize=11)
for i,(name,label) in enumerate([('selected_minus_native','Calibrated minus native'),('selected_own_minus_shared','Calibrated own minus shared'),('selected_minus_learned','Calibrated minus learned only')]):
 x=g['contrasts'][name]['macro']['B_hard_correct'];m=x['mean']*100;lo,hi=np.array(x['interval95'])*100;axs[2].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#0072B2','#009E73','#D55E00'][i]);axs[2].annotate(f'{m:+.2f}',(m,i),xytext=(0,12),textcoords='offset points',ha='center')
axs[2].set_yticks(range(3),['Minus native','Own minus shared','Minus learned only']);axs[2].set_ylim(-.5,2.6);axs[2].axvline(0,color='.5',ls='--');axs[2].set_xlabel('Difference (percentage points)');axs[2].set_title('Quantity: paired 95% intervals',fontsize=11)
for i,ax in enumerate(axs):ax.grid(axis='x' if i==2 else 'y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Frozen prior readout audit: A-only calibration selection; B confirmation held separate',fontsize=13);fig.tight_layout(rect=[0,.04,1,.92],w_pad=2.5);fig.savefig(out/'injection_domain_and_quantity.png',dpi=250);fig.savefig(out/'injection_domain_and_quantity.pdf')
fig,axs=plt.subplots(1,2,figsize=(12.9,4.8),gridspec_kw={'width_ratios':[1.1,1]});arms=['native','mean_shift_auto','learned_only','mean_shift_shared102'];labs=['Native','Mean shift / own','Learned only','Mean shift / shared'];vals=[v['summary']['all210'][a]['mR']*100 for a in arms];axs[0].bar(range(4),vals,color=['#0072B2','#009E73','#E69F00','#999999'])
for i,x in enumerate(vals):axs[0].text(i,x+1.2,f'{x:.2f}',ha='center')
axs[0].set_xticks(range(4),labs,rotation=15,ha='right');axs[0].set_ylim(0,105);axs[0].set_ylabel('Mean of six recalls (%)');axs[0].set_title('Full val: 210 images',fontsize=12)
xs=[x for x in v['paired_mR_intervals'] if x['subset']=='unseen188']
for i,x in enumerate(xs):
 m=x['mean']*100;lo,hi=np.array(x['interval95'])*100;axs[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#0072B2','#E69F00','#999999'][i]);axs[1].annotate(f'{m:+.2f}',(m,i),xytext=(0,12),textcoords='offset points',ha='center')
axs[1].set_yticks(range(3),['Minus native','Minus learned only','Own minus shared']);axs[1].set_ylim(-.5,2.6);axs[1].axvline(0,color='.5',ls='--');axs[1].set_xlabel('mR difference (percentage points)');axs[1].set_title('New 188-image subset: paired 95% intervals',fontsize=11)
for i,ax in enumerate(axs):ax.grid(axis='y' if i==0 else 'x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Full retrieval extension with calibration locked before new generation',fontsize=13);fig.tight_layout(rect=[0,.03,1,.92],w_pad=2.8);fig.savefig(out/'full_val210_retrieval.png',dpi=250);fig.savefig(out/'full_val210_retrieval.pdf');print('INJECTION_AND_FULL_RETRIEVAL_FIGURES_EXPORTED')
