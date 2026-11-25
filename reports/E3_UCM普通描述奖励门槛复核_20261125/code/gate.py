"""Caption-reference strategies and evaluation on a strictly separate positive caption."""
import sys
from fractions import Fraction
from functools import lru_cache
import numpy as np
from panels import OLD,RFPO,read

sys.path.insert(0,str(RFPO/'code'))
from exact_check import exact
K=read(OLD/'inputs/continuous_lock.json')['kappa']
BASE='hash_two'
ALTERNATIVES=['first','central_two','diverse_two','mean_all','min_all']
NAMES=[BASE]+ALTERNATIVES
I,J=np.triu_indices(16,1)

@lru_cache(maxsize=None)
def rank_value(above,ties):
    return sum((Fraction(1,above+i+1) for i in range(ties+1)),Fraction())/(ties+1)

def rr_matrix(sim,neg):
    counts=(neg[:,:,None]>sim[:,None,:]).sum(axis=1)
    ties=(neg[:,:,None]==sim[:,None,:]).sum(axis=1)
    return np.array([rank_value(int(a),int(t)) for a,t in zip(counts.flat,ties.flat)],dtype=object).reshape(counts.shape)

def wins(order,ranks):
    scores=exact(order,ranks)
    return scores==scores.max()

def tiescore(pred,actual):
    gap=actual[I]-actual[J];mask=gap!=0
    if not mask.any():return None
    sign=np.sign(pred[I][mask]-pred[J][mask])*np.sign(gap[mask])
    return float((1+sign.mean())/2)

def prepare(b,nm,v,rank,letters,old,check):
    assert v.shape==(17,768) and rank.shape==(16,) and np.isfinite(v).all()
    ix=b['maps'][nm];ids=ix['native'];p0=ids[0]
    pos=[p0]+sorted(set(ids)-{p0},key=b['hashes'].__getitem__)
    ok=np.ones(len(b['even']),bool);ok[pos]=False
    # Dot products follow the original scorer's vector-by-vector BLAS calculation.
    s=np.stack([b['native']@h for h in v]);hard=np.stack([b['hard']@h for h in v])
    neg=[s[:,ok&portion] for portion in (b['even'],~b['even'])]
    fractions=[rr_matrix(s[:,pos],x) for x in neg]
    rr=[v.astype(float) for v in fractions]
    sim=b['native'][pos]@b['native'][pos].T
    a=hard[:,ix['A'][0]]-hard[:,ix['A'][1]]
    native=[s[:,p0]-n.max(1) for n in neg]
    detail=np.mean([hard[:,ix[l][0]]>hard[:,ix[l][1]] for l in letters],axis=0)
    reference=np.mean(rr[0][:,:2],axis=1); heldmax=rr[1][:,2:].max(1)
    assert len(pos)>=3
    r2= wins(native[0][:16],rank)
    dense=np.unique((native[0]+K*a)[:16],return_inverse=True)[1]
    guarded=wins(dense+17*(reference[:16]+1e-14>=reference[:16][r2].mean()),rank)
    err=0.
    if old is not None:
        for key,arr in [('native',native[0]),('A',a),('keep',reference),('rr',heldmax)]:
            err=max(err,float(np.max(np.abs(arr-np.array([z[key] for z in old])))))
        for l in letters:
            err=max(err,float(np.max(np.abs(hard[:,ix[l][0]]-hard[:,ix[l][1]]-np.array([z[l] for z in old])))))
        for key,w in [('R2',r2),('R3',guarded)]:assert np.array_equal(w,np.array(check[key])==max(check[key]))
    else:
        for key,w in [('R2',r2),('guard',guarded)]:
            actual=check[key]
            err=max(err,abs(float(heldmax[:16][w].mean())-actual['rr']))
            for l in letters:
                votes=hard[:16,ix[l][0]]>hard[:16,ix[l][1]]
                err=max(err,abs(float(votes[w].mean())-actual[l]))
    assert err<1e-9,(nm,err)
    return dict(rr=rr,fractions=fractions,native=native,a=a,detail=detail,rank=rank,cos=sim,pos=pos,error=err)

def evaluate(z,fold,side=0):
    rr=z['rr'][side][:16];target=z['rr'][1-side][:,fold]
    d=z['detail'];gain=(target[:16]>target[16]+1e-12)&(d[:16]>d[16])
    ref=[i for i in range(1,rr.shape[1]) if i!=fold]
    allowed=[0]+ref;sim=z['cos']
    center=max(ref,key=lambda i:np.mean([sim[i,j] for j in allowed if i!=j]))
    far=min(ref,key=lambda i:sim[0,i])
    native=z['native'][side][:16];r2=wins(native,z['rank'])
    dense=np.unique(native+K*z['a'][:16],return_inverse=True)[1]
    selections={BASE:[0,ref[0]],'first':[0],'central_two':[0,center],'diverse_two':[0,far],'mean_all':allowed,'min_all':allowed}
    result={}
    for key,cols in selections.items():
        assert fold not in cols
        values=z['fractions'][side][:16,cols]
        signal=values.min(1) if key=='min_all' else values.sum(1)/len(cols)
        passed=signal>=sum(signal[r2],Fraction())/int(r2.sum())
        w=wins(dense+17*passed,z['rank'])
        hit=tiescore(signal.astype(float),target[:16])
        truth=z['fractions'][1-side][:16,fold]
        protected=truth>=sum(truth[r2],Fraction())/int(r2.sum())
        result[key]=dict(concordance=hit,false_accept=float(np.mean(passed&~protected)),false_reject=float(np.mean((~passed)&protected)),detail=float(d[:16][w].mean()),mrr=float(target[:16][w].mean()),
            joint=float(gain[w].mean()),opportunity=float(gain.any()),blocked=float(gain.any() and not (gain&passed).any()),
            refs=[z['pos'][i] for i in cols],winners=np.flatnonzero(w).tolist(),passed=np.flatnonzero(passed).tolist())
    return result
