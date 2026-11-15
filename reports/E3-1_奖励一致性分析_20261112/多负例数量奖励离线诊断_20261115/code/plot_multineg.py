import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,read
j=read(ROOT/'analysis.json');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15.8,5.2),gridspec_kw={'width_ratios':[1.1,1.25,1]});g=j['groups']['confirmation55'];sels=['uniform','pair_total','hard_total','B_oracle_quality_first'];labels=['Random draw\nexpectation','Original A\npair reward','A hardest\nnegative','B oracle\n(upper bound)']
vals=[g['summary']['after'][s]['macro']['B_hard_correct']['mean']*100 for s in sels];ax=axs[0];bars=ax.bar(np.arange(4),vals,color=['#999999','#0072B2','#009E73','#CC79A7'],width=.7)
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.2,f'{v:.2f}',ha='center',fontsize=9)
ax.set_xticks(np.arange(4),labels,rotation=12,ha='right');ax.set_ylim(0,55);ax.set_ylabel('Count-macro B all-count accuracy (%)');ax.set_title('After update 32: confirmation 52',fontsize=12)
ax=axs[1];values=[g['contrasts']['after']['hard_minus_pair']['macro']['B_hard_correct'],g['contrasts']['after']['hard_minus_uniform']['macro']['B_hard_correct'],g['own_shared']['after']['macro']['B_hard_correct']];names=['Hard minus original pair','Hard minus random expectation','Own greedy minus shared greedy']
for i,v in enumerate(values):
 m=v['mean']*100;lo,hi=np.array(v['interval95'])*100;ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#0072B2','#009E73','#D55E00'][i]);ax.annotate(f'{m:+.2f}',(m,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
ax.set_yticks(range(3),names);ax.set_ylim(-.5,2.6);ax.axvline(0,ls='--',lw=1,color='.5');ax.set_xlabel('Difference (percentage points)');ax.set_title('Paired 95% image-bootstrap intervals',fontsize=11)
ax=axs[2];cohorts=['training8','confirmation55','existing_val22'];den=[8,55,22];num=[j['groups'][c]['candidate_mechanics']['after']['B_hard_varies'] for c in cohorts];ax.bar(range(3),np.array(num)/den*100,color=['#999999','#0072B2','#E69F00'])
for i,(n,d) in enumerate(zip(num,den)):ax.text(i,n/d*100+1.4,f'{n}/{d}',ha='center')
ax.set_xticks(range(3),['Training 8','Confirm 55','Existing val 22'],rotation=15,ha='right');ax.set_ylim(0,55);ax.set_ylabel('Images with varying B decisions (%)');ax.set_title('Variation across four saved samples',fontsize=12)
for i,ax in enumerate(axs):ax.grid(axis='x' if i==1 else 'y',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('Offline multi-negative reward diagnostic: frozen candidates, zero new optimizer updates',fontsize=13)
fig.text(.5,.027,'Same-object cached count pools; A selects, B evaluates. Macro gives counts 1-4 equal weight. The +8.33-point selection gain comes from one count-4 image (2035.tif).',ha='center',fontsize=8.5)
fig.tight_layout(rect=[0,.1,1,.94],w_pad=2.6);out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'multineg_selection_and_headroom.png',dpi=250);fig.savefig(out/'multineg_selection_and_headroom.pdf');print('MULTINEG_FIGURE_EXPORTED')
