from pathlib import Path
from collections import defaultdict
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parents[1]
PREV=ROOT.parent/'pair_reward'
CAL=REP/'E3_困难奖励跨图跨措辞校准_20261127'
SOURCE={'98':REP/'E3_双收益回答对的偏好训练_20261127/validation98/fresh98','114':CAL/'followup_pool'}
SIZE={'98':49,'114':57}
def read(p):return json.loads(p.read_bytes())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    data={}; pixels=set()
    for tag,path in SOURCE.items():
        z=read(path/'scoring_complete.json');seed='c' if tag=='98' else 'd'
        raw=read(path/(seed+'_complete.json'))
        assert z['sampling']==sha(path/(seed+'_complete.json')) and z['compiled']==sha(path/'inputs/manifest.json')
        groups=defaultdict(list)
        for row in read(path/'inputs/rows.json'):
            name=row['image'];f=path/'scores'/(name+'.json');old=read(f)
            assert sha(f)==z['files'][name] and old['source']==raw['files'][name]
            if old['eligible']:
                groups[row['pair']].append(row)
        groups={k:sorted(g,key=lambda r:r['side']) for k,g in sorted(groups.items()) if len(g)==2}
        assert len(groups)==SIZE[tag]
        prior=read(PREV/('逐图_'+tag+'.json'))
        assert set(r['rgb'] for r in prior)=={r['rgb_sha256'] for g in groups.values() for r in g}
        allpix={r['rgb_sha256'] for g in groups.values() for r in g};assert pixels.isdisjoint(allpix);pixels.update(allpix)
        batch={}
        for pair,rs in groups.items():
            assert [r['side'] for r in rs]==[0,1] and len({r['attribute'] for r in rs})==1
            assert all(rs[0][q]==rs[1][q][::-1] for q in 'ABDE')
            batch[pair]=[{k:r[k] for k in ('image','row','rgb_sha256','side','orig_query','captions','feedback','attribute')} for r in rs]
        data[tag]=dict(pairs=batch,proofs={'scoring_complete.json':sha(path/'scoring_complete.json'),seed+'_complete.json':sha(path/(seed+'_complete.json')),
            'inputs/rows.json':sha(path/'inputs/rows.json'),'inputs/vectors.pt':sha(path/'inputs/vectors.pt')},seed=seed)
    dump(ROOT/'inputs/groups.json',data)
    files=['事前口径.md','inputs/groups.json','code/prepare.py','code/project.py','code/score.py']
    lock=dict(files={f:sha(ROOT/f) for f in files},redistribute=sha(CAL/'code/reward.py'),paired_source=sha(PREV/'evaluate.py'),
              archived_paired_results=sha(PREV/'结果.json'),unique_pixels=len(pixels),images={t:2*v for t,v in SIZE.items()})
    f=ROOT/'inputs/lock.json'
    if f.exists():assert read(f)==lock
    else:dump(f,lock)
if __name__=='__main__':main()
