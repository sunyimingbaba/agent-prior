import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;j=read(r/'analysis.json');plan=read(r/'plan.json');policies=[x['name'] for x in plan['policies']];labels=['SFT-30','Balanced-32'];colors=['#0072B2','#009E73'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15.5,5.2),gridspec_kw={'width_ratios':[1,1.2,1.05]});x=np.arange(2)
for k,kind in enumerate(['own','donor']):
    vals=[j['policies'][p]['native_vs_swapped']['new_sample']['B_hard_correct'][kind]['mean']*100 for p in policies];pos=x+(k-.5)*.35;axs[0].bar(pos,vals,.33,label=['Own-image prior','Mean of 7 donor priors'][k],color=['#0072B2','#E69F00'][k])
    for xx,v in zip(pos,vals):axs[0].text(xx,v+1.2,f'{v:.2f}',ha='center',fontsize=9)
axs[0].set_xticks(x,labels);axs[0].set_ylabel('B all-count accuracy (%)');axs[0].set_ylim(0,105);axs[0].set_title('All 8 targets: count-macro mean',fontsize=11);axs[0].legend(frameon=False,fontsize=8)
ax=axs[1]
for k,p in enumerate(policies):
    for off,(key,label) in enumerate([('all','All 8 / macro'),('plane','6 airplanes / same-object')]):
        v=j['policies'][p]['native_vs_swapped']['new_sample']['B_hard_correct']['own_minus_donor'] if key=='all' else j['policies'][p]['same_object_airplanes']['new_sample']['B_hard_correct'];m=v['mean']*100;lo,hi=np.array(v['interval95'])*100;y=k*2+off
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=colors[k]);ax.annotate(f'{m:+.2f}',(m,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8)
ax.set_yticks(range(4),[f'{s}: {t}' for s in labels for t in ['All 8','6 airplanes']]);ax.set_ylim(-.5,3.8);ax.axvline(0,color='.5',ls='--',lw=1);ax.set_xlabel('Own minus swapped (percentage points)');ax.set_title('95% paired image-bootstrap intervals',fontsize=11)
ax=axs[2]
for k,pool in enumerate(['historical4','new4','new16']):
    vals=[j['policies'][p]['coverage'][pool]['qualified_B_oracle_success']['mean']*100 for p in policies];pos=x+(k-1)*.25;ax.bar(pos,vals,.23,label=['Historical 4','New prefix 4','New 16'][k],color=['#999999','#0072B2','#009E73'][k])
    for xx,v in zip(pos,vals):ax.text(xx,v+1.2,f'{v:.1f}',ha='center',fontsize=8)
ax.set_xticks(x,labels);ax.set_ylim(0,105);ax.set_ylabel('Qualified B-oracle coverage (%)');ax.set_title('Candidate coverage by image',fontsize=11);ax.legend(frameon=False,fontsize=8)
for i,ax in enumerate(axs):ax.grid(axis='x' if i==1 else 'y',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('Sampling coverage and prior permutation: 256 new trajectories, 2176 frozen scores, 0 updates',fontsize=13);fig.text(.5,.025,'Native and donor priors are scored on the same target image. New-4 is a fixed prefix of New-16. Historical-4 uses archived seeds. Six-airplane contrast averages 5 same-object donors.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.1,1,.94],w_pad=2.5);out=r/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'coverage_and_prior_permutation.png',dpi=260);fig.savefig(out/'coverage_and_prior_permutation.pdf')
rows=list(csv.DictReader((r/'donor_target_means.csv').open()));fig=plt.figure(figsize=(13.8,6.5));gs=fig.add_gridspec(1,3,width_ratios=[1,1,.035],left=.09,right=.94,bottom=.22,top=.9,wspace=.23);axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];cax=fig.add_subplot(gs[0,2]);names=plan['images'];cs={x['image']:x['actual_count'] for x in read(r/'inputs/cases.json')};tick=[n.replace('.tif','')+f' (n={cs[n]})' for n in names]
for ax,p,label in zip(axes,policies,labels):
    matrix=np.array([[float(next(x for x in rows if x['policy']==p and x['role']=='new_sample' and x['donor_image']==d and x['image']==n)['B_hard_correct']) for n in names] for d in names]);im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='equal')
    for i in range(8):
        ax.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='#E69F00',lw=2))
        for k in range(8):ax.text(k,i,f'{matrix[i,k]*100:.0f}',ha='center',va='center',fontsize=8,color='black' if matrix[i,k]>.6 else 'white')
    ax.set_xticks(range(8),tick,rotation=45,ha='right');ax.set_yticks(range(8),tick);ax.set_xlabel('Target image / true count');ax.set_title(label)
axes[0].set_ylabel('Donor image / true count');fig.colorbar(im,cax=cax,label='B all-count accuracy (fraction; mean of 16 samples)');fig.suptitle('Full donor-target grid; orange diagonal = own-image prior',fontsize=13);fig.savefig(out/'donor_target_B_accuracy.png',dpi=250);fig.savefig(out/'donor_target_B_accuracy.pdf');print('PERMUTATION_FIGURES_EXPORTED')
