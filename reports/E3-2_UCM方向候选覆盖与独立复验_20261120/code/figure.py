"""Display the attainable training coverage and deployed decisions from the two separate panels."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.ticker import MaxNLocator
from shared import ROOT
def rows(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig') as fp:return list(csv.DictReader(fp))
def main():
    f=next(p for p in [Path('/System/Library/Fonts/STHeiti Medium.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')] if p.exists())
    font_manager.fontManager.addfont(str(f));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(f)).get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,(a,b)=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for method,label,col,marker in [('scene','场景提示','#275e91','o'),('objects','指明两个对象','#b35120','s')]:
        z=sorted((r for r in rows('coverage_table.csv') if (r['group'],r['subset'],r['word'],r['method'])==('train','old_one_sided','A',method)),key=lambda r:int(r['k']))
        a.plot([int(r['k']) for r in z],[int(r['accessible']) for r in z],marker=marker,color=col,label=label)
    a.set_title('10 张原先单侧成功的训练图');a.set_xlabel('每个朝向的采样条数');a.set_ylabel('两侧各出现过正确候选的图数');a.set_xticks([1,2,4,8]);a.set_ylim(-.2,10.25);a.legend(frameon=False)
    src={r['method']:r for r in rows('deployment_table.csv') if r['group']=='fresh' and r['word']=='AB'}
    selected=[('scene_greedy','场景 · 贪心'),('scene_mean8','场景 · 八次平均'),('objects_greedy','对象 · 贪心'),('objects_mean8','对象 · 八次平均'),('raw_clip','原始 CLIP'),('agent_choice','Agent 二选一 · 顺序平均')]
    values=[int(src[k]['both']) for k,_ in selected];n=src['scene_greedy']['questions']
    b.barh(range(len(values)),values,color=['#275e91','#275e91','#b35120','#b35120','#7b858d','#456d57'])
    b.set_yticks(range(len(values)),[v for _,v in selected]);b.invert_yaxis();b.set_xlim(0,int(n));b.set_title('14 张新增方向验证图');b.set_xlabel('原图及半转图均对的措辞数（共 '+str(n)+' 项）')
    for j,v in enumerate(values):b.text(v+.3,j,str(v),va='center')
    for axis in (a,b):axis.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'训练候选与新图验证.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
