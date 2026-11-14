import argparse
import collections
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
j=json.loads((r/'analysis.json').read_text(encoding='utf-8'));cases=json.loads((r/'inputs/evaluation_cases.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#0072B2','#009E73'];fig,axes=plt.subplots(1,3,figsize=(16,5.3),gridspec_kw={'width_ratios':[1,1.3,1.5]})
ax=axes[0];counts=[1,2,3,4,'other'];x=np.arange(5)
for k,split in enumerate(['train','val']):
    xs=[z for z in cases if z['split']==split];v=[sum(z['actual_count']==c for z in xs)/len(xs)*100 if c!='other' else sum(z['actual_count'] not in [1,2,3,4] for z in xs)/len(xs)*100 for c in counts];pos=x+(k-.5)*.36
    ax.bar(pos,v,.34,color=colors[k],label=['Train 63','Val 22'][k])
    for xp,yp in zip(pos,v):
        if yp:ax.text(xp,yp+1,f'{yp:.1f}',ha='center',fontsize=8)
ax.set_xticks(x,['1','2','3','4','6/7']);ax.set_xlabel('User-confirmed object count');ax.set_ylabel('Share of images (%)');ax.set_ylim(0,74);ax.set_title('Composition differs across splits',fontsize=11);ax.legend(frameon=False,fontsize=8)
ax=axes[1];rows=[]
for split in ['train','val']:
    v=next(z for z in j['stratified'] if z['selector']=='sample_mean' and z['split']==split and z['dimension']=='all');rows.append((split,'Raw '+split+' '+str(v['images']),v['delta_pairacc_B'],v['delta_pairacc_B_interval95']))
for split in ['train','val']:
    v=next(z for z in j['count_standardized'] if z['selector']=='sample_mean' and z['split']==split);rows.append((split,'Count-macro '+split+' '+str(v['images']),v['delta_pairacc_B'],v['delta_pairacc_B_interval95']))
for i,(split,label,m,ci) in enumerate(rows):
    m*=100;lo,hi=np.array(ci)*100;ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o' if i<2 else 's',color=colors[split=='val'],capsize=4)
    ax.annotate(f'{m:+.2f}',(m,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
ax.set_yticks(range(4),[x[1] for x in rows]);ax.set_ylim(3.55,-.55);ax.axvline(0,color='.45',ls='--',lw=1);ax.set_xlabel('EOS minus answer-close (pp)');ax.set_title('Sample-mean B accuracy change\n95% image-bootstrap intervals',fontsize=11)
ax=axes[2];arms=['own_answer_close','own_eos','shared_answer_close','shared_eos'];x=np.arange(4)
for k,split in enumerate(['train','val']):
    values=[next(z['pairacc_B'] for z in j['control_summary'] if z['split']==split and z['arm']==arm)*100 for arm in arms];pos=x+(k-.5)*.36
    ax.bar(pos,values,.34,color=colors[k],label=['Train 63','Val 22'][k])
    for xp,yp in zip(pos,values):ax.text(xp,yp+1.5,f'{yp:.1f}',ha='center',fontsize=8)
ax.set_xticks(x,['Own\nanswer-close','Own\nEOS','Shared\nanswer-close','Shared\nEOS']);ax.set_ylim(0,110);ax.set_ylabel('Greedy B pair accuracy (%)');ax.set_title('One shared prior reproduces the direction',fontsize=11);ax.legend(frameon=False,fontsize=8)
for ax in axes:ax.grid(axis='x' if ax==axes[1] else 'y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('EOS split diagnostic: frozen AGSE, identical answers, no optimizer updates',fontsize=14)
fig.text(.5,.025,'Count-macro: equal weight for counts 1, 2, 3, 4 (train 60; existing val 22). Shared prior: fixed image 102.tif state across all 85 target images.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.09,1,.93],w_pad=2.2);out=r/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'eos_split_composition_and_shared_state.png',dpi=300);fig.savefig(out/'eos_split_composition_and_shared_state.pdf');print('DIAGNOSTIC_FIGURE_EXPORTED')
