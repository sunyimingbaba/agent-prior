"""Apply pre-score pixel audit edits, preserve reciprocal pairs, and hash locked inputs."""
import csv,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
load=lambda n:json.loads((root/n).read_bytes())
audit=load('inputs/pixel_edits.json')
cases=[]
for row in load('inputs/selected.json'):
    if row['pair'] in audit['excluded']:continue
    r=dict(row)
    for w in 'ABCD':
        vals=[]
        for text in row[w]:
            for old,new in audit['replacements'].get(row['contrast'],{}).items(): text=text.replace(old,new)
            for old,new in audit['by_pair'].get(row['pair'],{}).items():text=text.replace(old,new)
            vals.append(text)
        r[w]=vals
    cases.append(r)
for i in range(0,len(cases),2):
    a,b=cases[i:i+2];assert a['pair']==b['pair'] and all(a[w]==b[w][::-1] for w in 'ABCD')
assert len({r['rgb_sha256'] for r in cases})==len(cases)
prior=set(load('inputs/excluded.json'));assert not prior.intersection(r['image'] for r in cases)
(root/'inputs/retained.json').write_text(json.dumps(cases,indent=2))
paths=['inputs/'+p+'.json' for p in ['selected','retained','protocol','pixel_edits','excluded','pool']]+['review/像素复核.md','画面重叠筛查.json','画面重叠候选.csv']
(root/'inputs/lock.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},indent=2,ensure_ascii=False))
