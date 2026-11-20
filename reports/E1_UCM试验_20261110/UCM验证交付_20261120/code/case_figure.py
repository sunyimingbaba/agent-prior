"""Create a readable natural-image contrast using real UCM RGB and archived AGSE scores."""
import csv
import io
import sys
from pathlib import Path
import pyarrow as pa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path(__file__).resolve().parents[1]
table=pa.ipc.open_file(sys.argv[1]).read_all()
images=dict(zip(table['path'].to_pylist(),table['image'].to_pylist()))
with (root/'逐条三句得分.csv').open(encoding='utf-8-sig') as f:
    rows={(r['image'],r['template']):r for r in csv.DictReader(f) if r['model']=='AGSE_archived'}
fig, axes=plt.subplots(2,2,figsize=(8,8.6),layout='constrained')
for ax,(n,t,truth,pick) in zip(axes.flat,[
    ('382','water_shore','Water on the left','right'),('387','water_shore','Water on the right','right'),
    ('1189','junction_corners','Houses at corners','houses'),('1185','junction_corners','Open ground at corners','open ground')]):
    name=n+'.tif'
    r=rows[name,t]
    ax.imshow(Image.open(io.BytesIO(images[name])).convert('RGB'),interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(f'{name}    {truth}',loc='left',fontsize=13)
    ax.set_xlabel(f'AGSE chose {pick}    |    margin {float(r["margin"]):+.4f}',fontsize=11,
                  color='#23705b' if r['correct']=='1' else '#a03a32')
fig.suptitle('Two spatial contrasts in UCM validation images',fontsize=18)
fig.savefig(root/'空间关系案例.png',dpi=145,bbox_inches='tight')
