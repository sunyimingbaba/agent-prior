"""Count distinct pixel identities and document reused validation imagery across evaluation tasks."""
from collections import Counter
from rosters import ROOT,REP,read,dump,sha
from itertools import combinations

def main():
    base=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
    official=REP/'E3-2_UCM与RSITMD独立复验_20261124'
    sets={'training24':read(ROOT/'inputs/training_roster.json')['rows'],
          'route168':[r for s in ('','confirm') for r in read(base/s/'inputs/panel.json')],
          'prompt108':[r for f in sorted(REP.glob('E3_UCM中性观察问题*/inputs/lock.json')) for r in read(f)['rows']],
          **{t:read(ROOT/'inputs'/(t+'.json')) for t in ('first','second','third')},
          'details126':read(REP/'E3_UCM多描述奖励同预算RFPO_20261124/inputs/panel.json'),
          'ucm210':read(official/'inputs/ucm.json'),'rsitmd448':read(official/'inputs/rsitmd.json')}
    rgb={k:{c.get('rgb_sha256',c.get('rgb')) for c in rows} for k,rows in sets.items()}
    splits={k:dict(Counter(c.get('split','test' if k in ('ucm210','rsitmd448') else 'train') for c in rows)) for k,rows in sets.items()}
    for k,rows in sets.items():assert len(rows)==len(rgb[k]) and None not in rgb[k]
    matrix={k:{l:len(s & t) for l,t in rgb.items()} for k,s in rgb.items()}
    for a,b in combinations(('first','second','third','prompt108','training24'),2):
        if (a,b)!=('prompt108','training24'):assert not matrix[a][b],(a,b,matrix[a][b])
    assert matrix['details126']['ucm210']==106 and splits['details126']=={'val':20,'test':106}
    for a in ('training24','first','second','third','route168'):
        for b in ('details126','ucm210','rsitmd448'):assert not matrix[a][b]
    assert matrix['training24']['route168']==24
    dump(ROOT/'results/pixel_overlap.json',dict(counts={k:len(s) for k,s in rgb.items()},splits=splits,rgb_intersection=matrix))
if __name__=='__main__':main()
