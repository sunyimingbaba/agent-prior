import hashlib
from collections import defaultdict
from rosters import ROOT,REP,Q,read,sha,dump
def key(text):return hashlib.sha256(('source-rfpo-1125|'+text).encode()).hexdigest()
def main():
    base=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
    questions=read(Q/'inputs/questions.json')
    groups=defaultdict(lambda:defaultdict(list))
    for part in (base,base/'confirm'):
        for c in read(part/'inputs/panel.json'):
            groups[c['attribute']][c['pair']].append(c)
    rows=[]
    for attr in sorted(groups,key=key)[:12]:
        pair=min(groups[attr],key=key)
        for c in sorted(groups[attr][pair],key=lambda r:key(r['image'])):
            rows.append(dict(image=c['image'],rgb=c['rgb_sha256'],pair=pair,attribute=attr,question=questions[attr]))
    assert len(rows)==len({r['image'] for r in rows})==24
    excluded={r['rgb_sha256'] for f in ('first','second') for r in read(ROOT/'inputs'/(f+'.json'))}
    for f in [REP/'E3_UCM多描述奖励同预算RFPO_20261124/inputs/panel.json',REP/'E3-2_UCM与RSITMD独立复验_20261124/inputs/ucm.json']:
        excluded.update(r['rgb_sha256'] for r in read(f))
    assert excluded.isdisjoint(r['rgb'] for r in rows)
    dump(ROOT/'inputs/training_roster.json',dict(rows=rows,protocol=sha(ROOT/'入围后同预算短训口径.md')))
if __name__=='__main__':main()
