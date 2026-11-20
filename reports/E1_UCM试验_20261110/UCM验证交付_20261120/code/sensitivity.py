"""Score-independent overlap sensitivity and illustrative retrieved cases."""
import csv
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))
def write(name,data):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]))
        w.writeheader();w.writerows(data)
rows=read('逐条三句得分.csv')
hits=read('画面重叠候选.csv')
scored={r['image'] for r in rows}
intersect={r[s] for r in hits for s in ['image_a','image_b'] if r[s] in scored}
out=[]
for dim in ['attribute','relation']:
    selection=[r for r in rows if r['cohort']=='extension' and r['dimension']==dim]
    bad={r['template'] for r in selection if r['image'] in intersect}
    for m in sorted({r['model'] for r in selection}):
        keep=[r for r in selection if r['model']==m and r['template'] not in bad]
        out.append(dict(dimension=dim,method=m,excluded_templates='|'.join(sorted(bad)),remaining=len(keep),
                        correct=sum(int(r['correct']) for r in keep)))
write('重叠敏感性.csv',out)
