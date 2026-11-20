"""Plot empirical training trajectories without smoothing or selecting checkpoints."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from shared import ROOT,read

def main():
    for path in ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Medium.ttc']:
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams['axes.unicode_minus']=False
    x=[0,6,12,24]
    states=read(ROOT/'results/states.json'); base=states['base_00']
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.6),layout='constrained')
    colors={'off':'#506879','on':'#b15123'}
    for ax,(part,key,factor,title) in zip(axes,[('native','mR',100,'原描述检索 mR（%）'),('native','eos',1,'第一步结束（共 210 图）'),('detail','both',1,'原图和旋转图均判对（共 32 题）')]):
        for seed,marker,style in [('a','o','-'),('b','s','--')]:
            for name in ['off','on']:
                y=[factor*base[part][key]]+[factor*states[f'{seed}_{name}_{n:02d}'][part][key] for n in x[1:]]
                ax.plot(x,y,color=colors[name],ls=style,marker=marker,ms=4,alpha=.93,lw=1.8)
        ax.set_title(title,fontsize=11)
        ax.set_xticks(x);ax.set_xlabel('训练步数');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if key=='eos':ax.set_ylim(-6,215)
        if key=='both':ax.set_ylim(bottom=-.25);ax.yaxis.get_major_locator().set_params(integer=True)
    handles=[Line2D([],[],color=colors['off'],label='无格式奖励'),Line2D([],[],color=colors['on'],label='有格式奖励'),
             Line2D([],[],color='.25',marker='o',label='种子 A'),Line2D([],[],color='.25',ls='--',marker='s',label='种子 B')]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(ROOT/'训练过程.png',dpi=165,bbox_inches='tight');plt.close(fig)
if __name__=='__main__': main()
