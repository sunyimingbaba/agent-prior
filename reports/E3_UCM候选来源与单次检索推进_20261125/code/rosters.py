"""Freeze eligible pictures, question strings and archive identities before evaluation."""
from pathlib import Path
import json
import hashlib
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parent
Q=REP/'E3_UCM中性观察问题候选对照_20261125'
OLD=REP/'E3_UCM多数配文新图复核_20261125'
PRIOR=REP/'E3_UCM同图偏好对与独立验收_20261124'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    questions=read(Q/'inputs/questions.json')
    previous=[r for p in REP.glob('E3_UCM中性观察问题*/inputs/lock.json') for r in read(p)['rows']]
    seen_names={r['image'] for r in previous};seen_rgb={r['rgb'] for r in previous}
    assert len(seen_names)==len(seen_rgb)==108
    plan=dict(protocol=sha(ROOT/'实验口径.md'),questions=sha(Q/'inputs/questions.json'),panels={})
    for tag,base in [('first',OLD),('second',PRIOR/'reward_confirm')]:
        source=base/'inputs/panel.json'
        assert sha(source)==read(base/'inputs/lock.json')['panel']
        all_rows=read(source);tools=read(base/'records/tools.json')
        rows=[]
        for r in all_rows:
            if r['attribute'] not in questions:continue
            assert r['image'] not in seen_names and r['rgb_sha256'] not in seen_rgb and r['split']=='train'
            seen_names.add(r['image']);seen_rgb.add(r['rgb_sha256'])
            rows.append(dict(r,feedback=tools[r['image']][0],question=questions[r['attribute']]))
        assert len(rows)==(148 if tag=='first' else 98)
        assert set(Counter(r['pair'] for r in rows).values())=={2}
        dest=ROOT/'inputs'/(tag+'.json');dump(dest,rows)
        plan['panels'][tag]=dict(file=sha(dest),source=sha(source),tools=sha(base/'records/tools.json'),
            text=sha(base/'records/text.pt'),images=len(rows),excluded=len(all_rows)-len(rows))
    dump(ROOT/'inputs/lock.json',plan)
if __name__=='__main__':main()
