import hashlib,json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
old=ROOT.parent/'E3-1_UCM第二轮R3困难负样本复验_20261124'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
questions=json.loads((ROOT/'inputs/questions.json').read_text())
pools=defaultdict(lambda:defaultdict(list))
sources={}
for sub in ('','confirm'):
    p=old/sub/'inputs/panel.json'
    sources[sub]=digest(p)
    for row in json.loads(p.read_text()):
        pools[row['attribute']][row['pair']].append(row)
assert set(pools)==set(questions)
rows=[]
for attr,pairs in sorted(pools.items()):
    selected=sorted(pairs,key=lambda v:hashlib.sha256(('neutral-q-1125|'+v).encode()).digest())[1]
    a,b=sorted(pairs[selected],key=lambda z:z['image'])
    assert a['scene_query']==b['scene_query'] and a['side']!=b['side']
    assert all(a[k]==b[k][::-1] for k in 'ABC')
    for c in [a,b]:
        rows.append(dict(image=c['image'],rgb=c['rgb_sha256'],pair=selected,attribute=attr,scene=c['scene_query'],question=questions[attr]))
assert len(rows)==54 and len({c['image'] for c in rows})==54
first=ROOT.parent/'E3_UCM中性观察问题候选对照_20261125/inputs/lock.json'
original=json.loads(first.read_text())['rows']
assert all(a['attribute']==b['attribute'] and a['question']==b['question'] for a,b in zip(original,rows,strict=True))
for key in ('image','rgb','pair'):assert not {r[key] for r in original}&{r[key] for r in rows}
out=dict(first=digest(first),rows=rows,panels=sources,protocol=digest(ROOT/'实验口径.md'),questions=digest(ROOT/'inputs/questions.json'))
(ROOT/'inputs/lock.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
