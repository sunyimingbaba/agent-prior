"""Compare whole retrieval galleries after one fixed-budget selection per image."""
from collections import Counter
import sys
import numpy as np
from rosters import ROOT,read,dump
from reward_direction import source
from gate import K,wins,rr_matrix
from run_stage import boot
from gallery_expected import expected,self_check

def choose(b,c,visual,ranks,refs,sign):
    ix=b['maps'][c['image']];ns=visual@b['native'].T
    keep=sign.copy();keep[ix['native']]=False
    negatives=ns[:,keep];rr=rr_matrix(ns[:,refs],negatives);quality=rr.sum(1)/len(refs)
    base=ns[:,refs[0]]-negatives.max(1);r2=wins(base,ranks)
    guard=quality>=quality[r2].sum()/int(r2.sum())
    hs=visual@b['hard'].T;delta=hs[:,ix['A'][0]]-hs[:,ix['A'][1]]
    dense=np.unique(base+K*delta,return_inverse=True)[1]
    return visual[wins(dense+17*guard,ranks)]

def compact(vec,weights):
    points,ix=np.unique(vec,axis=0,return_inverse=True)
    return points,np.bincount(ix,weights=weights)

def one_set(available,swapped,select=choose,detail_letters=None):
    firsts={b['maps'][c['image']]['native'][0] for c,b,s,q in available}
    prepared=[];held_cols=[];owners=[];seen=set()
    for c,b,(s,rs),(q,rq) in available:
        assert c['image'] not in seen;seen.add(c['image'])
        ix=b['maps'][c['image']]['native'];p0=ix[0];pos=set(ix);sign=~b['even'] if swapped else b['even']
        others=sorted(pos-{p0},key=b['hashes'].__getitem__)
        train=[p for p in others if p in firsts or sign[p]]
        held=[p for p in others if p not in firsts and not sign[p]]
        if not train or not held:continue
        assert not set(held)&{p0,*train}
        refs=[p0,train[0]];v={}
        for m,vec,ranks in [('scene',s[:16],rs[:16]),('question',q[:16],rq[:16]),
            ('h1',np.vstack((s[:8],q[8:16])),np.r_[rs[:8],rq[8:16]]),
            ('h2',np.vstack((s[8:16],q[:8])),np.r_[rs[8:16],rq[:8]])]:
            a=select(b,c,vec,ranks,refs,sign);v[m]=(a,np.full(len(a),1/len(a)))
        v['mixed']=(np.vstack((v['h1'][0],v['h2'][0])),np.r_[v['h1'][1]/2,v['h2'][1]/2])
        v['greedy']=(s[-1:],np.ones(1))
        held_cols+=held;owners.extend([len(prepared)]*len(held))
        prepared.append((c,refs,v))
    assert set(held_cols).isdisjoint(firsts|{p for c,refs,v in prepared for p in refs})
    assert len(prepared)>=10
    ids=np.array(held_cols);owners=np.array(owners);unique,inv=np.unique(ids,return_inverse=True)
    labels=[c['image'] for c,_,_ in prepared]
    raw={};native=available[0][1]['native']
    for arm in ('greedy','scene','question','mixed'):
        points=[];weights=[]
        for c,refs,v in prepared:
            vs,w=compact(*v[arm]);points.append(vs);weights.append(w)
        a=(np.vstack(points)@native[unique].T)[:,inv];results=expected(a,weights,owners,labels)
        if arm=='greedy':
            from single_scores import recalls
            mask=np.arange(len(labels))[:,None]==owners[None,:]
            back=np.stack(list(recalls(a.T,mask.T).values()),1)
            check=np.hstack((np.stack(list(recalls(a,mask).values()),1),np.stack([back[owners==i].mean(0) for i in range(len(labels))])))
            assert np.max(abs(results-check))<1e-12
        raw[arm]=results
    rows=[dict(image=c['image'],pair=c['pair'],attribute=c['attribute'],refs=refs,held=ids[owners==i].tolist(),
            **{a:v[i].tolist() for a,v in raw.items()}) for i,(c,refs,_) in enumerate(prepared)]
    if detail_letters:
        by_name={c['image']:b for c,b,s,q in available}
        for row,(c,_,v) in zip(rows,prepared):
            b=by_name[c['image']];ix=b['maps'][c['image']]
            row['detail']={}
            for arm in ('greedy','scene','question','mixed'):
                vv,ww=v[arm];sim=vv@b['hard'].T
                cor=np.mean([sim[:,ix[k][0]]>sim[:,ix[k][1]] for k in detail_letters],axis=0)
                row['detail'][arm]=float(ww@cor)
    return rows,dict(images=len(rows),captions=len(ids),distinct_tokens=len(set(ids)),shared_texts=sum(x>1 for x in Counter(ids).values()))

def compare(rows):
    index=[r['pair'] for r in rows]
    result={}
    for arm in ('greedy','scene','question','mixed','mixed_minus_scene','question_minus_scene','mixed_minus_question'):
        z=np.array([r[arm] if '_minus_' not in arm else np.array(r[arm.split('_minus_')[0]])-r[arm.split('_minus_')[1]] for r in rows])
        data=dict(mR=z.mean(1),I2T=z[:,:3].mean(1),T2I=z[:,3:].mean(1))
        data.update({m:z[:,i] for i,m in enumerate(('I2T1','I2T5','I2T10','T2I1','T2I5','T2I10'))})
        result[arm]={k:boot(x,np.ones(len(rows)),index) for k,x in data.items()}
    return result

def main(tag):
    err=self_check();by_seed={'a':[],'b':[]};first_hashes=None
    for c,s,b,scene,question,_ in source(tag):
        if first_hashes is None:first_hashes=b['hashes']
        assert first_hashes==b['hashes']
        by_seed[s].append((c,b,scene,question))
    result={};all_rows=[]
    for split,swapped in [('primary',False),('reversed',True)]:
        rows=[];size={}
        for s,available in by_seed.items():
            records,size[s]=one_set(available,swapped)
            rows+=[dict(r,seed=s,split=split) for r in records]
        result[split]=dict(coverage=size,metrics=compare(rows),tie_test_error=err)
        all_rows+=rows
    dump(ROOT/'results'/('gallery_'+tag+'.json'),result);dump(ROOT/'records'/('gallery_'+tag+'_images.json'),all_rows)
if __name__=='__main__':main(sys.argv[1])
