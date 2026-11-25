"""Immutable source identifiers and candidate vectors for each evaluation panel."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
REP=ROOT.parent
OLD=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
PRIOR=REP/'E3_UCM同图偏好对与独立验收_20261124'
RFPO=REP/'E3_UCM多描述奖励同预算RFPO_20261124'
QDIR={'development':REP/'E3_UCM中性观察问题候选对照_20261125','repeat':REP/'E3_UCM中性观察问题第二批复核_20261125'}
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def bank(folder):
    path=folder/'records/text.pt'
    assert sha(path)==read(folder/'records/read_complete.json')['text']
    data=load(path)
    tokens=data['native_tokens'].tolist()
    hashes=[hashlib.sha256(','.join(map(str,r)).encode()).hexdigest() for r in tokens]
    assert len(hashes)==len(set(hashes))
    vis=data['native'].double().numpy()
    assert np.allclose(np.linalg.norm(vis,axis=1),1,atol=1e-4)
    return dict(native=vis,hard=data['hard'].double().numpy(),maps=data['maps'],hashes=hashes,
        even=np.array([int(h[-1],16)%2==0 for h in hashes]))

def question_panels(tag):
    path=QDIR[tag];manifest=read(ROOT/'records/features.json')
    rows=read(path/'inputs/lock.json')['rows']
    banks=[bank(OLD/folder) for folder in ('','confirm')]
    for seed in 'ab':
        features=ROOT/'records'/(tag+'_'+seed+'.pt');meta=manifest[features.name];assert sha(features)==meta['sha256']
        proof=path/'records'/seed/'complete.json'
        assert sha(proof)==meta['records'] and sha(path/'inputs/lock.json')==meta['lock']
        cp=read(proof);assert meta['adapter']==cp['adapter']
        vec=load(features)
        for r in rows:
            nm=r['image'];b,=[x for x in banks if nm in x['maps']]
            f=path/'records'/seed/(nm+'.json');assert sha(f)==cp['records'][nm]
            z=read(f);assert z['rgb']==r['rgb'] and z['adapter']==meta['adapter']
            for mode in ('question','scene'):
                g=z['groups'][mode]
                # Append the same scene-prompt greedy anchor for both candidate sources.
                v=torch.cat((vec[nm][mode][:16],vec[nm]['scene'][-1:])).double().numpy()
                old=g['scores'][:16]+[z['groups']['scene']['scores'][-1]]
                yield r,seed,mode,b,v,np.array([s['rank'] for s in old[:16]]),'BC',old,g['retrieval']

def transfer_panel():
    sys.path.insert(0,str(REP/'E3-2_UCM有效图组定向短训_20261124/code'))
    from verify_screen import unpack
    base=PRIOR/'reward_confirm';b=bank(base);rows=read(base/'inputs/panel.json')
    prior={(v['seed'],v['image']):v['scores'] for v in read(PRIOR/'results/multi_independent_rows.json')}
    assigned={r['rgb'] for p in QDIR.values() for r in read(p/'inputs/lock.json')['rows']}
    assert len(assigned)==108 and not assigned.intersection(r['rgb_sha256'] for r in rows)
    for seed in 'ab':
        archives=[unpack(base/'records',tag) for tag in (seed,'extra_'+seed)]
        anchor_path=PRIOR/'pilot/vectors'/(seed+'_start_reward_anchor.pt')
        anchor=load(anchor_path);assert anchor['images']==[c['image'] for c in rows]
        for i,c in enumerate(rows):
            nm=c['image'];records=[z[nm] for z in archives]
            assert all(s['adapter']==anchor['adapter'] for s,_ in records)
            v=torch.cat([z['visual'] for _,z in records]+[anchor['features'][i:i+1]]).double().numpy()
            rank=np.array(sum((z['proxy_rank'] for _,z in records),[]))
            yield c,seed,'scene',b,v,rank,'CDE',None,prior[seed,nm]
