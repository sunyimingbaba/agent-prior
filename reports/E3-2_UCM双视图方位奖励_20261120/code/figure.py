"""Plot the fixed-endpoint full-gallery retrieval and same-source orientation decisions."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from shared import ROOT,read

def main():
    available=[Path('/System/Library/Fonts/STHeiti Medium.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    font=next(p for p in available if p.exists())
    fm.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':fm.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'font.size':11})
    data=read(ROOT/'results/states.json')
    turns=read(ROOT/'results/quarter_stats.json')['states']
    fig, axes=plt.subplots(1,3,figsize=(11,3.8),constrained_layout=True)
    titles=['原首句检索 mR（%）','原图与 180° 均正确（题）','四个朝向均正确（题）']
    fields=[lambda s:data[s]['native']['mR']*100,lambda s:data[s]['detail']['both'],lambda s:turns[s]['all_four']]
    for ax,title,f in zip(axes,titles,fields):
        base=f('base_00');xs=np.arange(3);ax.axhline(base,color='#515964',ls='--',lw=1.5,label='历史起点')
        for seed,color,marker,offset in [('a','#185c8d','o',-.035),('b','#be5a20','s',.035)]:
            ys=[f(seed+'_'+arm+'_16') for arm in ('single','separate','joint')]
            ax.plot(xs+offset,ys,marker=marker,ms=7,color=color,lw=1.5,label='种子 '+seed.upper())
        ax.set_xticks(xs,['单视图','分别计分','较弱侧计分'])
        ax.set_title(title,fontsize=13,pad=12)
        ax.set_xlim(-.33,2.33);ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.17)
        if '题' in title:
            from matplotlib.ticker import MaxNLocator
            ax.set_ylim(bottom=-.4,top=max(4,ax.get_ylim()[1]+1));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(loc='best',frameon=False,fontsize=8)
    fig.savefig(ROOT/'末步结果.png',dpi=180)
    plt.close(fig)
if __name__=='__main__':
    main()
