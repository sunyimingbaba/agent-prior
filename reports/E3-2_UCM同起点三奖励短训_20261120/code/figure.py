"""Summarize actual final policies; relative-photo bootstrap on the right, full fixed-gallery means on the left."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from shared import ROOT,read
def main(fontfile):
    f=FontProperties(fname=fontfile)
    data=read(ROOT/'results/summary.json')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    labels=['历史起点','原 RFPO','真实图文','真实图文 + 困难负例']
    colors=['#7e8a95','#607790','#1f756c','#b26639']
    for ax,part,key,title in zip(axs,['native','detail'],['mR','both_greedy'],['完整验证图库 · 双向平均召回','新双物体题 · 原图和旋转图均判对']):
        for i,(a,col) in enumerate(zip(['base','old','caption','hard'],colors)):
            stat=data[part][a][key];mean,lo,hi=[100*stat[k] for k in ('mean','lo','hi')]
            ax.barh(i,mean,height=.52,color=col,zorder=2)
            ax.plot([lo,hi],[i,i],color='#192d38',lw=1.3,zorder=3)
            ax.plot([lo,hi],[i,i],linestyle='none',marker='|',color='#192d38',markersize=7,zorder=3)
            ax.text(max(mean,hi)+1.5,i,f'{mean:.1f}%',va='center',fontsize=10)
        ax.set_yticks(range(4),labels=labels,fontproperties=f);ax.invert_yaxis()
        ax.set_xlim(0,100);ax.set_xlabel('百分比',fontproperties=f);ax.set_title(title,fontproperties=f,fontsize=13,pad=12)
        ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',color='#e1e6e9',alpha=.85,zorder=0)
    fig.suptitle('UCM 同起点短训后，直接用最终策略生成先验',fontproperties=f,fontsize=16)
    fig.savefig(ROOT/'三组奖励短训结果.png',dpi=170)
if __name__=='__main__':main(sys.argv[1])
