"""Readable diagnostic figures from independently verified raw-artifact analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
j=json.loads((r/'analysis.json').read_text());d=json.loads((r/'diagnostic_plan.json').read_text());ep=json.loads((r/'evaluation_plan.json').read_text());base=ep['baseline'];names=[c['name'] for c in d['conditions']]
assert json.loads((r/'本机独立核验.json').read_text())['ok'] and json.loads((r/'本机状态路径核验.json').read_text())['ok']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
out=r/'figures';out.mkdir(exist_ok=True);primary='repository_duplicate_token_positive';colors={'native':'#0072B2','calibrated_own':'#009E73'}
def label(n):
 c=next(x for x in d['conditions'] if x['name']==n)
 return ('E+' if c['example']=='with_example' else 'E-')+'/'+('RGB' if c['policy_image']=='real' else 'gray')+'/'+('F-image' if c['feedback']=='recorded_real' else 'F-constant')
contrasts=['condition/'+n for n in names[1:]]+['permutation_mean'];labels=[label(n) for n in names[1:]]+['Mean of 20 prior permutations']
fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True,layout='constrained')
for ax,arm in zip(axes,d['primary_readouts']):
 cs=[next(c for c in j['retrieval_contrasts'] if c['arm']==arm and c['protocol']==primary and c['contrast']==name) for name in contrasts]
 xs=np.array([c['delta_mR']*100 for c in cs]);lo=np.array([c['interval95'][0]*100 for c in cs]);hi=np.array([c['interval95'][1]*100 for c in cs]);y=np.arange(8)
 ax.errorbar(xs,y,xerr=np.stack([xs-lo,hi-xs]),fmt='o',capsize=4,color=colors[arm]);ax.axvline(0,color='.4',ls='--',lw=1)
 ax.set_yticks(y,labels);ax.set_ylim(7.5,-.5);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Change in validation mR (percentage points)')
 ax.set_title(('Native' if arm=='native' else 'Calibrated own')+f"; baseline mR={j['retrieval'][primary][arm][base]['mR']*100:.2f}%")
fig.suptitle('Frozen readouts: image/prompt/feedback controls and prior permutations\nPaired image-cluster 95% pointwise intervals; fixed val210 gallery',fontsize=13)
for ext in ['png','pdf']:fig.savefig(out/('frozen_prior_contrasts.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained');x=np.arange(8);short=[label(n).replace('F-image','Fi').replace('F-constant','Fc') for n in names]
ax=axes[0,0];ax.bar(x,[j['generation_states'][n]['val']['unique_state_sha256'] for n in names],color='#56B4E9');ax.set_ylabel('Unique raw Agent states');ax.set_ylim(0,220);ax.set_title('State diversity among 210 validation images')
ax=axes[0,1]
for arm in d['primary_readouts']:ax.plot(x,[100*j['readout_path'][arm][n]['gate_one_fraction'] for n in names],marker='o',label=arm.replace('_',' '),color=colors[arm])
ax.set_ylim(0,105);ax.set_ylabel('Clamped visual gate = 1 (%)');ax.set_title('Prompt mixing gate saturation');ax.legend()
ax=axes[1,0]
for arm in d['primary_readouts']:ax.plot(x,[100*j['quantity'][arm][n]['val']['B_hard_correct_counts1to4_macro'] for n in names],marker='o',label=arm.replace('_',' '),color=colors[arm])
ax.set_ylim(0,100);ax.set_ylabel('Count1-4 macro accuracy (%)');ax.set_title('Validation count subset (22 images; wording B)');ax.legend()
ax=axes[1,1]
for protocol,color,title in [('strict_image_correspondence','#D55E00','Saved top-10 prefix'),('strict_tie_averaged','#CC79A7','Uniform score-tie expectation')]:ax.plot(x,[100*j['retrieval'][protocol]['calibrated_own'][n]['mR'] for n in names],marker='o',label=title,color=color)
ax.set_ylim(0,100);ax.set_ylabel('Strict image-correspondence mR (%)');ax.set_title('Tie convention sensitivity, calibrated own readout');ax.legend()
for ax in axes.ravel():ax.set_xticks(x,short,rotation=35,ha='right');ax.grid(axis='y',alpha=.15)
fig.suptitle('Validation diagnostics: E+/E- = with/without fixed example; Fi/Fc = image-specific/constant feedback\nRetrieval backbone always receives the real image; generated states include retained failures',fontsize=12)
for ext in ['png','pdf']:fig.savefig(out/('state_path_and_count_diagnostics.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig);print('DIAGNOSTIC_FIGURES_WRITTEN')
