"""Visualize the storage-tank case and the actual per-candidate relation margins."""
import io,sys
import pyarrow as pa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from shared import ROOT,read
case=next(r for r in read(ROOT/'inputs/selected.json') if r['image']=='1919.tif')
rows=[r for r in read(ROOT/'scoring/rows.json') if r['image']==case['image'] and r['mode']=='scene' and r['k']>=0]
tab=pa.ipc.open_file(sys.argv[1]).read_all()
assert tab['path'][case['row']].as_py()==case['image']
im=Image.open(io.BytesIO(tab['image'][case['row']].as_py())).convert('RGB')
fig,(left,right)=plt.subplots(1,2,figsize=(9.3,3.9),width_ratios=(.85,1.4),layout='constrained')
left.imshow(im);left.axis('off');left.set_title('1919.tif  |  UCM train',loc='left')
left.set_xlabel('The tank is right of the vertical road.')
vals=[1000*r['B'] for r in rows]
best=max(r['reward'] for r in rows)
right.bar(range(8),vals,color=['#256f5d' if x>0 else '#bb5039' for x in vals],width=.68)
right.axhline(0,color='#383838',lw=.8)
right.set_xticks(range(8),[str(i+1) for i in range(8)])
right.set_xlabel('Sampled candidate');right.set_ylabel('Correct minus incorrect cosine (× 0.001)')
right.set_title('Seven candidates rank the true relation higher',loc='left',fontsize=12)
for i,r in enumerate(rows):
    if r['reward']==best:
        right.annotate('Proxy choice',xy=(i,vals[i]),xytext=(1.0,-1.6),arrowprops=dict(arrowstyle='->',color='#7a2c21'),color='#7a2c21')
right.set_ylim(-2,4.7)
right.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'储罐案例.png',dpi=170)
