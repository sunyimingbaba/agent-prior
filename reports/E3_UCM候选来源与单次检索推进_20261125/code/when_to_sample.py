"""Separate the historical bundle comparison from the matched first-round intervention."""
from collections import Counter,defaultdict
import gzip
import io
import json
import sys
import tarfile
import numpy as np
import torch
from rosters import ROOT,OLD,REP,PRIOR,read,dump,sha
from report_scores import bank,load,image_folds
from run_stage import boot

def compare(rows):
    out={}
    for k in ('joint','opportunity','mrr','detail','conditional'):
        label=[r['pair'] for r in rows];u=list(dict.fromkeys(label));ids=[u.index(z) for z in label]
        draw=np.random.default_rng(20261125).integers(len(u),size=(20000,len(u)))
        levels={};values={}
        for arm in ('locked','free'):
            a=np.array([r[arm]['joint' if k=='conditional' else k] for r in rows])
            den=np.array([r[arm]['opportunity'] if k=='conditional' else 1 for r in rows])
            v=np.bincount(ids,weights=a)[draw].sum(1);d=np.bincount(ids,weights=den)[draw].sum(1)
            sampled=np.divide(v,d,out=np.full_like(v,np.nan),where=d>0)
            values[arm]=(a.sum()/den.sum(),sampled)
        values['change']=(values['free'][0]-values['locked'][0],values['free'][1]-values['locked'][1])
        for name,(m,s) in values.items():levels[name]=dict(mean=float(m),interval=np.nanquantile(s,[.025,.975]).tolist())
        if len({r['seed'] for r in rows})>1:
            levels['seeds']={}
            for seed in 'ab':
                s=[r for r in rows if r['seed']==seed]
                vals=[sum(r[arm]['joint' if k=='conditional' else k] for r in s)/sum(r[arm]['opportunity'] if k=='conditional' else 1 for r in s) for arm in ('locked','free')]
                levels['seeds'][seed]=vals[1]-vals[0]
        out[k]=levels
    return out

def historical(tag):
    if tag=='first':
        agg=defaultdict(list)
        for line in gzip.open(OLD/'records/folds.jsonl.gz','rt'):
            r=json.loads(line);agg[r['image'],r['seed']].append(r)
        old={}
        for key,folds in agg.items():
            z={k:np.mean([f['baseline'][field] for f in folds]) for k,field in [('joint','hit'),('mrr','mrr'),('detail','detail')]}
            old[key]=dict(z,opportunity=np.mean([bool(f['good']) for f in folds]))
    else:
        p=next(REP.glob('E3_UCM普通描述奖励门槛复核*/records/vote_transfer.json'))
        old={(r['image'],r['seed']):dict(r['baseline'],opportunity=r['opportunity']) for r in read(p)}
    rows=[dict(image=r['image'],pair=r['pair'],seed=r['seed'],locked=old[r['image'],r['seed']],free=r['scene']) for r in read(ROOT/'records'/(tag+'_images.json'))]
    dump(ROOT/'results'/('historical_timing_'+tag+'.json'),compare(rows))
    dump(ROOT/'records'/('historical_timing_'+tag+'.json'),rows)

def archive_raw(base,rows):
    proof=read(base/'archive.json');f=base/'trajectories.tar.gz';assert sha(f)==proof['sha256']
    cp=read(base/'raw/complete.json')
    with tarfile.open(f) as tf:
        assert json.loads(tf.extractfile('complete.json').read())==cp
        for c in rows:
            buf=tf.extractfile(c['image']+'.pt').read()
            import hashlib
            assert hashlib.sha256(buf).hexdigest()==cp['files'][c['image']]
            yield torch.load(io.BytesIO(buf),weights_only=True,map_location='cpu')

def controlled(tag):
    bankbase=PRIOR/('reward_confirm' if tag=='second' else 'guard');b=bank(bankbase);p=ROOT/'inputs'/(tag+'.json')
    roster=read(p);lock=read(ROOT/'inputs/prefix_lock.json');assert lock[tag]==sha(p)
    rows=[];cost={};checks=0
    for seed in 'ab':
        folder=ROOT/'records/prefix'/tag/seed;proof=read(folder/'raw/complete.json')
        proofs={k:h for sh in (0,1) for k,h in read(folder/'features'/('complete_'+str(sh)+'.json'))['files'].items()}
        assert len(proofs)==len(roster)==len(proof['files']) and proof['protocol']==lock['protocol']
        raw=archive_raw(folder,roster)
        prior_raw=archive_raw(ROOT/'records/second'/seed,roster) if tag=='second' else None
        tokens=Counter()
        for c,z in zip(roster,raw,strict=True):
            nm=c['image'];f=folder/'features'/(nm+'.pt');feat=load(f)
            assert feat['source']==proof['files'][nm] and sha(f)==proofs[nm] and z['rgb']==feat['rgb']==c['rgb_sha256']
            assert z['query']==c['scene_query'] and z['feedback']==c['feedback'] and z['adapter']==feat['adapter']==proof['adapter']
            checks+=z['exact_shared_context_batches']
            a=feat['modes']['locked'];az=z['modes']['locked'];anchor=a['visual'][-1]
            if tag=='second':
                oldf=load(ROOT/'records/second'/seed/'features'/(nm+'.pt'));old=next(prior_raw)
                q=oldf['modes']['scene'];qz=old['modes']['scene']
                assert old['image']==nm and oldf['adapter']==feat['adapter'] and qz['rngs']==z['rngs'] and torch.equal(qz['hidden'][-1],az['hidden'][-1])
            else:q=feat['modes']['free'];qz=z['modes']['free']
            assert torch.allclose(q['visual'][-1],anchor,atol=1e-5,rtol=0)
            for arm,paths in [('locked',az['paths']),('free',qz['paths'])]:
                assert len(paths)==16
                first=sum(len(t[0]) for t in paths)
                later=sum(sum(len(v) for v in t[1:]) for t in paths)
                tokens[arm+'_total_slot_tokens']+=first+later
                tokens[arm+'_sampled_tokens']+=later+(first if arm=='free' else 0)
                tokens[arm+'_later_round_slots']+=sum(len(t)>1 for t in paths)
            if len(set(b['maps'][nm]['native']))<3:continue
            out={}
            for arm,g in [('locked',a),('free',q)]:
                v=g['visual'].double().numpy();v[-1]=anchor.double().numpy()
                folds,_=image_folds(c,seed,b,v,np.asarray(g['ranks'][:16]))
                out[arm]={k:np.mean([r['baseline'][j] for r in folds]) for k,j in [('joint','hit'),('mrr','mrr'),('detail','detail')]}
                out[arm]['opportunity']=np.mean([bool(r['good']) for r in folds])
            rows.append(dict(image=nm,pair=c['pair'],seed=seed,**out))
        cost[seed]=dict(tokens)
    result=compare(rows);result['sampling_cost']=cost;result['identical_context_batch_checks']=checks
    result['images']=len(rows)//2;result['pairs']=len({r['pair'] for r in rows})
    dump(ROOT/'results'/('controlled_timing_'+tag+'.json'),result);dump(ROOT/'records'/('controlled_timing_'+tag+'.json'),rows)

def decide():
    vals=[read(ROOT/'results'/('controlled_timing_'+t+'.json')) for t in ('second','third')]
    eligible=all(r['joint']['change']['mean']>0 for r in vals) and any(r['joint']['change']['interval'][0]>0 for r in vals)
    eligible=eligible and all(r[m]['change']['interval'][1]>=0 for r in vals for m in ('mrr','detail'))
    dump(ROOT/'results/prefix_nomination.json',dict(eligible=eligible,protocol=sha(ROOT/'首轮探索复核口径.md'),sources={t:sha(ROOT/'results'/('controlled_timing_'+t+'.json')) for t in ('second','third')}))

if __name__=='__main__':
    if sys.argv[1]=='decide':decide()
    elif sys.argv[1]=='historical':historical(sys.argv[2])
    else:controlled(sys.argv[2])
