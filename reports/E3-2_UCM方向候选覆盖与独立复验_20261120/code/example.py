"""Show an explicitly illustrative, observed disagreement on a newly labeled source."""
import argparse,csv,io
from pathlib import Path
import pyarrow as pa
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.patches import Circle
from shared import ROOT,read
def main(arrow):
    f=next(p for p in [Path('/System/Library/Fonts/STHeiti Medium.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')] if p.exists())
    font_manager.fontManager.addfont(str(f));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(f)).get_name(),'axes.unicode_minus':False})
    r=next(r for r in read(ROOT/'inputs/fresh.json') if r['image']=='1283.tif')
    src=pa.ipc.open_file(str(arrow/'ucm_caption_karpathy_label_val.arrow')).read_all()
    assert src['path'][r['row']].as_py()==r['image']
    im=Image.open(io.BytesIO(src['image'][r['row']].as_py())).convert('RGB')
    with (ROOT/'results/matched_choice_by_source.csv').open(encoding='utf-8-sig') as fp:
        scores={s['method']:s for s in csv.DictReader(fp) if s['image']==r['image'] and s['word']=='A'}
    fig,axes=plt.subplots(1,2,figsize=(7,4.4),constrained_layout=True)
    for k,(ax,pic) in enumerate(zip(axes,(im,im.transpose(Image.Transpose.ROTATE_180)))):
        ax.imshow(pic)
        for j,((u,v),col) in enumerate(zip(r['xy'],('#ffbe00','#2edbff'))):
            x,y=u*(pic.width-1),v*(pic.height-1)
            if k:x,y=pic.width-1-x,pic.height-1-y
            ax.add_patch(Circle((x,y),10 if j==0 else 17,edgecolor=col,fill=False,lw=2))
        ax.set_title('原图 · 红车在白色屋顶左侧' if k==0 else '旋转 180° · 红车在白色屋顶右侧',fontsize=12)
        key=['raw_original','raw_rotated'][k]
        predicted=lambda method:'左侧' if float(scores[method][key])>0 else '右侧'
        ax.set_xlabel('按选项顺序平均\nAgent 选'+predicted('agent_same_state')+'     CUP 选'+predicted('cup_same_state'))
        ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(ROOT/'红车方向案例.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arrow',type=Path);main(p.parse_args().arrow)
