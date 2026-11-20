"""Apply the visually independent prescore agreement rule and bind both queries and judgments."""
from shared import ROOT,sha,dump,read
import csv
def rows(p):
    with (ROOT/p).open() as f:return {r['pair']:r for r in csv.DictReader(f,delimiter='\t')}
votes=rows('review/decisions.tsv');questions=rows('review/questions.tsv')
cases=read(ROOT/'inputs/cases.json')
assert set(votes)==set(questions)=={r['pair'] for r in cases}
selected=[]
for r in cases:
    q=questions[r['pair']];v=votes[r['pair']]
    assert v['decision'] in ('keep','exclude')
    if v['decision']=='keep':
        grp=[x for x in cases if x['pair']==r['pair']]
        for w in ['A','B']:
            correct=next(x['image'].split('.')[0] for x in grp if x[w][0]==q[w+'1'])
            if v[w+'1_image'].split('.')[0]!=correct:break
        else:selected.append(r)
assert len(selected)>=2 and len(selected)%2==0
dump(ROOT/'inputs/selected.json',selected)
files=['实验口径.md','inputs/pool.json','inputs/excluded_e1.json','inputs/pairs.tsv','inputs/cases.json','inputs/selected.json',
       'review/questions.tsv','review/decisions.tsv','code/seal.py','code/generate.py']
dump(ROOT/'inputs/lock.json',{n:sha(ROOT/n) for n in files})
print(len(cases),len(selected))
