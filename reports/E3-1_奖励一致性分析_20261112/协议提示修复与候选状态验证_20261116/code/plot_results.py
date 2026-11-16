"""Standalone figures for the complete, independently reconstructed state comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;j=json.loads((r/'analysis.json').read_text());d=json.loads((r/'protocol_plan.json').read_text())
assert json.loads((r/'本机评估核验.json').read_text())['ok']
out=r/'figures';out.mkdir(exist_ok=True);primary='repository_duplicate_token_positive';families=d['state_families'];labels=['Legacy / last','Legacy / answer prefill','Legacy / answer pool','Contract / last','Contract / answer prefill','Contract / answer pool'];colors=['#557A95']*3+['#A54F35']*3
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True,layout='constrained')
for ax,arm,title in zip(axes,['native','calibrated_own'],['Native readout','Calibrated own readout']):
 for i,f in enumerate(families):
  c=next(x for x in j['retrieval_contrasts'] if x['arm']==arm and x['protocol']==primary and x['contrast']==f+'/correct-minus-permutation_mean');v=c['delta_mR']*100;lo,hi=np.array(c['interval95'])*100;ax.errorbar(v,i,xerr=np.array([[v-lo],[hi-v]]),fmt='o',color=colors[i],capsize=3)
 ax.axvline(0,color='#666666',lw=.8,ls='--');ax.set_title(title);ax.set_xlabel('Correct minus 20-permutation mean mR (pp)');ax.set_yticks(range(6),labels);ax.grid(axis='x',alpha=.2)
axes[0].invert_yaxis();fig.suptitle('Frozen readouts: image–state correspondence\nNominal paired 95% intervals; val 210, fixed gallery',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(out/('state_correspondence.'+ext),dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained');y=np.arange(6)
for i,f in enumerate(families):
 s=j['state_diagnostics'][f]['val'];axes[0].barh(i,s['available'],color=colors[i]);axes[1].barh(i,s['centered_rms'],color=colors[i]);v=j['quantity']['calibrated_own'][f]['correct']['val']['B_hard_correct_macro_1to4'];axes[2].barh(i,100*v,color=colors[i])
for ax in axes:ax.set_yticks(y,labels if ax==axes[0] else ['']*6);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
axes[0].set_xlim(0,220);axes[0].set_xlabel('Defined states / 210 (fallbacks retained)');axes[1].set_xlabel('Raw state centered RMS');axes[2].set_xlim(0,100);axes[2].set_xlabel('Count B macro accuracy (%)');fig.suptitle('Coverage, raw variation and count diagnosis\nCount val 22; true counts 1–4 equally weighted',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(out/('state_coverage_and_counts.'+ext),dpi=180)
plt.close(fig);print('VERIFIED_STATE_FIGURES_WRITTEN')
