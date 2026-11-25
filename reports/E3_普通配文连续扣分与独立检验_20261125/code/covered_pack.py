"""Use all eligible, non-optimization images among the original 84 natural pairs."""
from collections import Counter
import json,sys
from pathlib import Path
from compare import OUT,SOURCE,PRIOR,read,dump,sha,load
from rosters import REP
def main():
    old=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
    train={r['rgb'] for r in read(SOURCE/'inputs/training_roster.json')['rows']}
    prior={r['rgb_sha256'] for k in ('first','second','third') for r in read(SOURCE/'inputs'/(k+'.json'))}
    previous={r['rgb_sha256'] for k in ('fresh','extension') for r in read(OUT/'inputs'/(k+'.json'))}
    templ={}
    for k in ('first','second','third'):
        for r in read(SOURCE/'inputs'/(k+'.json')):templ[r['attribute'],r['side']]=r
    old_free={r['rgb'] for f in REP.glob('E3_UCM中性观察问题*/inputs/lock.json') for r in read(f)['rows']}
    sets={'saved':[],'extra':[]}
    for sub in ('','confirm'):
        rs=read(old/sub/'inputs/panel.json');fb=read(old/sub/'records/tools.json')
        for r in rs:
            rgb=r['rgb_sha256']
            if rgb in train:continue
            assert rgb not in previous|prior
            ref=templ[r['attribute'],r['side']];assert all(r[k]==ref[k] for k in 'ABC')
            sets['saved' if rgb in old_free else 'extra'].append(dict(r,D=ref['D'],E=ref['E'],feedback=fb[r['image']][0]))
    assert sorted(map(len,sets.values()))==[52,92]
    for k,rs in sets.items():
        assert set(Counter(r['pair'] for r in rs).values())=={2}
        dump(OUT/'inputs'/('covered_'+k+'.json'),rs)
    dump(OUT/'inputs/covered_lock.json',dict(protocol=sha(OUT/'已覆盖细节类型的重复检验.md'),
        source=sha(OUT/'inputs/multi_nomination.json'),files={k:sha(OUT/'inputs'/('covered_'+k+'.json')) for k in sets},rfpo=list(sorted(train))))
if __name__=='__main__':main()
