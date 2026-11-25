"""Reconstruct both rewards and calculate paired image-level effects across the two R2 policies."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'E3_UCM多描述奖励同预算RFPO_20261124'
sys.path.insert(0,str(PREV/'code'))
from exact_check import exact

def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def detail(s):return float(np.mean([s[k]>0 for k in ('B','C')]))

def summarize(group,base,k):
    sc=group['scores'][:-1];anchor=group['scores'][-1]
    rank,native,hard,keep,rr=(np.array([s[key] for s in sc]) for key in ('rank','native','A','keep','rr'))
    r2=exact(native,rank)
    floor=keep[r2==r2.max()].mean()
    dense=np.unique(native+k*hard,return_inverse=True)[1]
    r3=exact(dense+17*(keep+1e-14>=floor),rank)
    assert abs(group['floor']-floor)<1e-12
    assert np.array_equal(r2,group['retrieval']['R2']) and np.array_equal(r3,group['retrieval']['R3'])
    votes=np.array([detail(x) for x in sc]);gain=votes>detail(base);rr_gain=rr>base['rr']+1e-12
    good=gain&rr_gain
    preserved=gain&(rr>=base['rr']-1e-12)
    actions=group['actions']
    assert len(sc)==len(actions)==16 and all(1<=len(path)<=3 and all(1<=len(r)<=64 for r in path) for path in actions)
    row=dict(detail_ceiling=int(detail(base)==1),rr_ceiling=int(base['rr']>=1-1e-12),preserved=int(preserved.any()),available=int(good.any()),good_draws=int(good.sum()),detail_available=int(gain.any()),rr_available=int(rr_gain.any()),
        best_detail=float(votes.max()),rr_mean=float(rr.mean()),detail_mean=float(votes.mean()),greedy_detail=detail(anchor),greedy_rr=anchor['rr'],
        unique=len({tuple(tuple(r) for r in path) for path in actions}),with_tool=sum(len(t)>1 for t in actions),
        eligible=int(detail(base)<1 and base['rr']<1-1e-12),changed_winner=int(np.any((r2==r2.max())!=(r3==r3.max()))))
    for name,r in [('R2',r2),('R3',r3)]:
        win=r==r.max()
        row[name]=float(good[win].mean())
        row[name+'_preserved']=float(preserved[win].mean())
        row[name+'_detail']=float(votes[win].mean());row[name+'_rr']=float(rr[win].mean())
    row['increment']=row['R3']-row['R2']
    return row

def estimate(num,den,draw):
    # Arrays have one value for each seed and each image. Both sides of a pair are adjacent.
    values=[];full=num.sum()/den.sum() if den.sum() else None
    for n,d in zip(num,den):
        values.append(float(n.sum()/d.sum()) if d.sum() else None)
    n=num.sum(0).reshape(-1,2).sum(1);d=den.sum(0).reshape(-1,2).sum(1)
    bn=n[draw].sum(1);bd=d[draw].sum(1);ok=bd>0
    ci=np.quantile(bn[ok]/bd[ok],[.025,.975]).tolist() if ok.any() else [None,None]
    return dict(mean=full,interval=ci,seeds=values,resamples_defined=int(ok.sum()))

def main():
    lock=read(ROOT/'inputs/lock.json');rows=lock['rows'];n=len(rows);assert n==54
    assert all(a['pair']==b['pair'] and a['scene']==b['scene'] and a['question']==b['question'] for a,b in zip(rows[::2],rows[1::2]))
    old=ROOT.parent/'E3-1_UCM第二轮R3困难负样本复验_20261124'
    k=read(old/'inputs/continuous_lock.json')['kappa']
    provenance=read(ROOT.parent/'E3_UCM同图偏好对与独立验收_20261124/records/environment.json')['starting']
    flat=[];totals={}
    for seed in 'ab':
        folder=ROOT/'records'/seed;cp=read(folder/'complete.json')
        assert cp['adapter']==provenance[seed] and cp['lock']==sha(ROOT/'inputs/lock.json') and cp['source']==sha(ROOT/'code/run.py')
        for c in rows:
            name=c['image'];rec=folder/(name+'.json');p=folder/(name+'.pt')
            z=read(rec);assert sha(rec)==cp['records'][name] and z['states']==sha(p)
            assert z['adapter']==cp['adapter'] and z['lock']==cp['lock'] and z['rgb']==c['rgb']
            states=torch.load(p,weights_only=True)
            base=z['groups']['scene']['scores'][-1]
            for mode,query in [('scene',c['scene']),('question',c['scene']+' '+c['question'])]:
                v=states[mode];g=z['groups'][mode]
                assert v.shape==(17,3584) and v.dtype==torch.bfloat16 and v.isfinite().all() and g['query']==query
                flat.append(dict(seed=seed,mode=mode,image=name,pair=c['pair'],attribute=c['attribute'],**summarize(g,base,k)))
    for seed in 'ab':
        totals[seed]={}
        for mode in ('scene','question'):
            rs=[r for r in flat if r['seed']==seed and r['mode']==mode]
            totals[seed][mode]={key:sum(r[key] for r in rs) for key in ('available','preserved','detail_ceiling','rr_ceiling','good_draws','detail_available','rr_available','eligible','changed_winner','with_tool')}
    def mat(key,mode):
        return np.array([[r[key] for r in flat if r['seed']==s and r['mode']==mode] for s in 'ab'])
    draws=np.random.default_rng(20261125).integers(n//2,size=(20000,n//2));ones=np.ones((2,n));result={}
    for key in ('available','preserved','R2_preserved','R3_preserved','good_draws','eligible','detail_available','rr_available','best_detail','detail_mean','rr_mean','greedy_detail','greedy_rr','unique','with_tool','R2','R3','R2_detail','R3_detail','R2_rr','R3_rr'):
        a=mat(key,'scene');b=mat(key,'question')
        result[key]={m:estimate(z,ones,draws) for m,z in [('scene',a),('question',b),('change',b-a)]}
    for mode in ('scene','question'):
        has=mat('available',mode)
        result['conditional_'+mode]={key:estimate(mat(key,mode),has,draws) for key in ('R2','R3','increment')}
    transitions={}
    for seed,scene,question in zip('ab',mat('available','scene'),mat('available','question')):
        transitions[seed]=dict(both=int((scene*question).sum()),scene_only=int((scene*(1-question)).sum()),
            question_only=int((question*(1-scene)).sum()),neither=int(((1-question)*(1-scene)).sum()))
    tests=[result['available']['change'],result['conditional_question']['increment']]
    promote=all(t['interval'][0] is not None and t['interval'][0]>0 and all(x is not None and x>0 for x in t['seeds']) for t in tests)
    dump(ROOT/'results/per_image.json',flat)
    dump(ROOT/'results/summary.json',dict(pairs=n//2,images=n,sampled=2*2*n*16,greedy=2*2*n,
        counts=totals,transitions=transitions,metrics=result,advance_to_training=promote))
if __name__=='__main__':main()
