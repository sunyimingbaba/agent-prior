"""Report the pre-score higher-confidence holdout and text-direction alignments."""
import csv
import numpy as np
import torch
from shared import ROOT,read,dump
from analyze_split import summary,csvout
def main():
    out=ROOT/'val'
    with (out/'images.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    strict=read(ROOT/'inputs/pixel_edits.json')['sensitivity_drop']
    table={}
    for mode in ['scene','original','rotated']:
        g=[r for r in rows if r['mode']==mode and r['pair'] not in strict]
        for key in ['C','D','mean']+(['RR'] if mode!='rotated' else []):
            for rule in ['AB','greedy','random']+(['old'] if mode!='rotated' else []):
                table[mode+'/'+rule+'_'+key]=summary(g,lambda r:float(r[rule+'_'+key]))
                if rule in ['random','old']:
                    table[mode+'/AB_minus_'+rule+'_'+key]=summary(g,lambda r:float(r['AB_'+key])-float(r[rule+'_'+key]))
    dump(out/'strict.json',table)
    align=[]
    for split in ('train','val'):
        ix=read(ROOT/split/'index.json')
        f=torch.load(ROOT/split/'features.pt',weights_only=True,map_location='cpu')['text'].double().numpy()
        for r in read(ROOT/'inputs/retained.json'):
            if r['split']!=split or r['side']!=0:continue
            vec={}
            for w in 'ABCD':
                p,n=[ix['texts'].index(s) for s in r[w]];v=f[p]-f[n];vec[w]=v/np.linalg.norm(v)
            align.append(dict(split=split,pair=r['pair'],image=r['image'],**{a+b:float(vec[a]@vec[b]) for a,b in ('AB','AC','AD','BC','BD','CD')}))
    csvout(ROOT/'contrast_alignment.csv',align)
if __name__=='__main__':main()
