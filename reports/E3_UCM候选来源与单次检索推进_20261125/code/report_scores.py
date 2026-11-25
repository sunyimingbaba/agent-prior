"""Score both prompt sources on the same image and the same ordinary holdout."""
from collections import defaultdict,Counter
from pathlib import Path
import gzip
import json
import sys
import numpy as np
import torch
from rosters import ROOT,REP,OLD,PRIOR,read,dump,sha

sys.path.insert(0,str(OLD/'code'))
from evaluate_new import image_folds, bank, rr
from panels import load

def anchors(tag,seed):
    f=PRIOR/('confirm' if tag=='first' else 'pilot/vectors')/(seed+('_start_blind.pt' if tag=='first' else '_start_reward_anchor.pt'))
    obj=load(f)
    return obj['adapter'],dict(zip(obj['images'],obj['features']))

def greedy_score(b,c,v):
    ix=b['maps'][c['image']];ids=set(ix['native']);held=sorted(ids-{ix['native'][0]})
    s=v@b['native'].T;h=v@b['hard'].T
    allowed=~b['even'].copy();allowed[list(ids)]=False
    return dict(detail=float(np.mean([h[ix[k][0]]>h[ix[k][1]] for k in 'CDE'])),
        mrr=float(np.mean([float(rr(s[p],s[allowed])) for p in held])))

def summarize(rows,by):
    labels=[r[by] for r in rows];unique=list(dict.fromkeys(labels));idx=[unique.index(k) for k in labels]
    draws=np.random.default_rng(20261125).integers(len(unique),size=(20000,len(unique)))
    def ratio(n,d):
        x=np.bincount(idx,weights=n);y=np.bincount(idx,weights=d)
        num=x[draws].sum(1);den=y[draws].sum(1)
        return (x.sum()/y.sum() if y.sum() else None),np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    def stat(value,draw):
        return dict(mean=value,interval=np.nanquantile(draw,[.025,.975]).tolist() if np.isfinite(draw).any() else [None,None])
    out={}
    for field in ['opportunity','joint','hit_given_opportunity','mrr','detail']:
        values={};summary={};methods=['scene','mixed','question']
        if field in ('mrr','detail'):methods+=['greedy_scene','greedy_question']
        for mode in methods:
            n=[r[mode]['joint' if field=='hit_given_opportunity' else field] for r in rows]
            d=[r[mode]['opportunity'] if field=='hit_given_opportunity' else 1 for r in rows]
            val,draw=ratio(n,d);values[mode]=(val,draw);summary[mode]=stat(val,draw)
        for a,b in [('mixed','scene'),('question','scene'),('mixed','question'),('greedy_question','greedy_scene')]:
            if a in values:
                va,da=values[a];vb,db=values[b];summary[a+'_minus_'+b]=stat(va-vb if va is not None and vb is not None else None,da-db)
        out[field]=summary
    return out

def calculate(tag):
    torch.set_num_threads(1);lock=read(ROOT/'inputs/lock.json')
    assert sha(ROOT/'实验口径.md')==lock['protocol']
    p=ROOT/'inputs'/(tag+'.json');assert sha(p)==lock['panels'][tag]['file']
    panel=read(p);src=OLD if tag=='first' else PRIOR/'reward_confirm';b=bank(src)
    assert sha(src/'records/text.pt')==lock['panels'][tag]['text']
    rows=[];fold_records=[];exclusions=[];max_anchor=0.;changed_anchor=0
    for seed in 'ab':
        sig,reference=anchors(tag,seed);folder=ROOT/'records'/tag/seed
        complete=read(folder/'raw/complete.json')
        verified={}
        for i in range(2):
            cp=read(folder/'features'/('complete_'+str(i)+'.json'));assert cp['lock']==complete['lock']==sha(ROOT/'inputs/lock.json')
            assert not set(cp['files']) & set(verified)
            verified.update(cp['files'])
        assert set(complete['files'])==set(verified)=={c['image'] for c in panel}
        for c in panel:
            nm=c['image'];p=folder/'features'/(nm+'.pt');assert sha(p)==verified[nm]
            rec=load(p);assert rec['source']==complete['files'][nm] and rec['adapter']==complete['adapter']==sig and rec['rgb']==c['rgb_sha256']
            src_modes=rec['modes'];arrays={m:(v['visual'].double().numpy(),np.asarray(v['ranks'])) for m,v in src_modes.items()}
            scene,rs=arrays['scene'];question,rq=arrays['question']
            assert all(v.shape==(17,768) and r.shape==(17,) for v,r in arrays.values())
            nominal=reference[nm].double().numpy()
            err=float(np.max(abs(scene[-1]-nominal)));max_anchor=max(max_anchor,err)
            assert err<1e-4,(nm,err)
            if len(set(b['maps'][nm]['native']))<3:
                if seed=='a':exclusions.append(nm)
                continue
            greedy_s=greedy_score(b,c,scene[-1]);greedy_q=greedy_score(b,c,question[-1])
            changed_anchor+=int(greedy_s!=greedy_score(b,c,nominal))
            values={};detail_record={}
            for m,vec,rk in [('scene',scene[:16],rs[:16]),('question',question[:16],rq[:16]),
                ('mixed_1',np.vstack((scene[:8],question[8:16])),np.r_[rs[:8],rq[8:16]]),
                ('mixed_2',np.vstack((scene[8:16],question[:8])),np.r_[rs[8:16],rq[:8]])]:
                folds,_=image_folds(c,seed,b,np.vstack([vec,scene[-1]]),rk)
                values[m]={k:np.mean([r['baseline'][key] for r in folds]) for k,key in [('joint','hit'),('detail','detail'),('mrr','mrr')]}
                values[m]['opportunity']=np.mean([bool(r['good']) for r in folds])
                detail_record[m]=[{k:r[k] for k in ('held','good','slot_mrr','slot_detail','baseline')} for r in folds]
            values['mixed']={k:(values['mixed_1'][k]+values['mixed_2'][k])/2 for k in values['scene']}
            rows.append(dict(seed=seed,image=nm,pair=c['pair'],attribute=c['attribute'],greedy_scene=greedy_s,greedy_question=greedy_q,**values))
            fold_records.append(dict(seed=seed,image=nm,**detail_record))
    assert len(rows)==2*(len(panel)-len(exclusions))
    out=dict(images=len(rows)//2,excluded=exclusions,pairs=len({r['pair'] for r in rows}),max_anchor_difference=max_anchor,
             altered_historical_anchor_endpoint=changed_anchor,by_pair=summarize(rows,'pair'),by_attribute=summarize(rows,'attribute'),
             seeds={s:summarize([r for r in rows if r['seed']==s],'pair') for s in 'ab'})
    dest=ROOT/'results';dest.mkdir(exist_ok=True)
    dump(dest/(tag+'.json'),out);dump(ROOT/'records'/(tag+'_images.json'),rows)
    body=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in fold_records).encode()
    (ROOT/'records'/(tag+'_folds.jsonl.gz')).write_bytes(gzip.compress(body,mtime=0))
    return out

def nomination():
    panels=[read(ROOT/'results'/(tag+'.json'))['by_pair'] for tag in ('first','second')]
    candidates={}
    for arm in ('mixed','question'):
        k=arm+'_minus_scene'
        eligible=(panels[0]['joint'][k]['interval'][0]>0 and panels[1]['joint'][k]['mean']>0
            and all(p[m][k]['interval'][1]>=0 for p in panels for m in ('detail','mrr')))
        candidates[arm]=dict(eligible=eligible,average_joint=np.mean([p['joint'][arm]['mean'] for p in panels]))
    accepted=[k for k,v in candidates.items() if v['eligible']]
    decision=max(accepted,key=lambda k:candidates[k]['average_joint']) if accepted else None
    dump(ROOT/'results/nomination.json',dict(selected=decision,arms=candidates,protocol=sha(ROOT/'实验口径.md'),
        sources={t:sha(ROOT/'results'/(t+'.json')) for t in ('first','second')}))
if __name__=='__main__':
    if sys.argv[1]=='nominate':nomination()
    else:calculate(sys.argv[1])
