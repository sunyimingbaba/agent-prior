"""Average seed-paired deltas on aligned validation images before the image bootstrap."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from metrics import ci

ROOT=Path(__file__).resolve().parents[1]

def aligned(directory,filename,group,metric,arm):
    with (directory/'results'/filename).open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['arm']==arm and (not group or r['group']==group)]
    grouped=defaultdict(list)
    for row in rows:
        key=row['pair'] if group in ('attribute','relation') else row['image']
        grouped[key].append(float(row[metric]))
    return {k:np.mean(v) for k,v in grouped.items()}

def main():
    result={}
    specs=[('retrieval_by_image.csv','',k) for k in ('mR','I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10')]
    specs.extend(('detail_by_case.csv',g,'correct') for g in ('quantity_A','quantity_B','attribute','relation'))
    for a,b in [('R2','R1'),('R2','R0'),('R1','R0')]:
        metrics={}
        for name,group,metric in specs:
            differences=[]
            keys=None
            for folder in (ROOT,ROOT/'repeat'):
                va=aligned(folder,name,group,metric,a)
                vb=aligned(folder,name,group,metric,b)
                assert va.keys()==vb.keys()
                if keys is None:
                    keys=sorted(va)
                assert keys==sorted(va)
                differences.append(100*np.array([va[k]-vb[k] for k in keys]))
            stats=ci(np.mean(differences,axis=0))
            stats['by_seed']=[float(v.mean()) for v in differences]
            metrics[group or metric]=stats
        result[a+'-'+b]=metrics
    (ROOT/'results/paired_two_seed.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
