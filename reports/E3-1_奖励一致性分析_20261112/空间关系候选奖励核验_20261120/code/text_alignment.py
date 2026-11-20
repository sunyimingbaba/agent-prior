"""Describe whether the frozen text encoder gives the same direction to each phrasing contrast."""
import torch
import numpy as np
from shared import ROOT,read,dump
from analyze import save_csv
a=torch.load(ROOT/'scoring/vectors.pt',map_location='cpu',weights_only=True)
c=torch.load(ROOT/'heldout/text.pt',map_location='cpu',weights_only=True)
ci=read(ROOT/'heldout/index.json')
rows=[]
for item in read(ROOT/'inputs/selected.json'):
    if item['side']!='left' or item['image'] not in ci['columns']:continue
    vectors={}
    for w in ['A','B']:
        i,j=[a['texts'].index(v) for v in item[w]]
        vectors[w]=a['text'][i].double()-a['text'][j].double()
    i,j=ci['columns'][item['image']]
    vectors['C']=c['features'][i].double()-c['features'][j].double()
    row=dict(pair=item['pair'])
    for x,y in [('A','B'),('A','C'),('B','C')]:
        row[x+y]=float(torch.nn.functional.cosine_similarity(vectors[x],vectors[y],dim=0))
    rows.append(row)
save_csv('三套措辞的嵌入方向.csv',rows)
print({k:round(float(np.mean([r[k] for r in rows])),5) for k in ['AB','AC','BC']})
