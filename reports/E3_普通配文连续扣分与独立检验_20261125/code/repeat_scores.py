"""Test the prespecified mean residual against single-generation baselines on additional pictures."""
import numpy as np
from head_scores import OUT,HEAD,FORM,LOCK,read,dump,load,detailed,compare,counts,summary,sha
from compare import rr

def main():
    rec=[];result={}
    bt=load(OUT/'records/covered/bank/text.pt')
    import hashlib
    hashes=[hashlib.sha256(','.join(map(str,v)).encode()).hexdigest() for v in bt['native_tokens'].tolist()]
    odd=np.array([int(h[-1],16)%2==1 for h in hashes])
    for tag in ('saved','extra'):
        rows=read(OUT/'inputs'/('covered_'+tag+'.json'));ids=[r['image'] for r in rows]
        vals={}
        for arm in ('baseline','hard','constant'):
            per={}
            for s in 'ab':
                z=load(HEAD/'test'/('covered_'+tag)/s/(arm+'.pt'));assert z['images']==ids and z['nomination']==sha(LOCK)
                v=z['scene'];d=detailed(v,bt,rows,bt['maps']).mean(1)
                sim=v.double().numpy()@bt['native'].double().numpy().T;n=[]
                for j,r in enumerate(rows):
                    pos=bt['maps'][r['image']]['native'];alt=set(pos)-{pos[0]};ok=odd.copy();ok[pos]=False
                    n.append(float(np.mean([float(rr(sim[j,k],sim[j,ok])) for k in alt])))
                    if arm=='baseline':
                        gold=load(OUT/'records/covered'/tag/s/'features'/(r['image']+'.pt'))['visual'][-1]
                        assert float((gold-v[j]).abs().max())<5e-5
                per[s]=np.column_stack([d,n])
            vals[arm]=per
        for j,r in enumerate(rows):
            rec.append(dict(image=r['image'],pair=r['pair'],tag=tag,values={a:{s:u[j].tolist() for s,u in bs.items()} for a,bs in vals.items()}))
    for tag in ('saved','extra','all'):
        rs=[r for r in rec if r['tag']==tag or tag=='all'];vs={}
        for i,k in enumerate(('detail','mrr')):
            ar={a:{s:np.array([r['values'][a][s][i] for r in rs]) for s in 'ab'} for a in ('baseline','hard','constant')}
            vs[k]=dict(levels={a:float(np.mean(list(u.values()))) for a,u in ar.items()},changes={
                a+'-'+b:compare(ar[a],ar[b],[r['pair'] for r in rs]) for a,b in [('constant','baseline'),('hard','baseline'),('hard','constant')]})
        result[tag]=dict(images=len(rs),**vs)
    gal=load(FORM/'results/rsitmd_text.pt');out={}
    for mode in ('native','alternatives'):
        vs={}
        for a in ('baseline','constant'):
            vs[a]=[]
            for s in 'ab':
                z=load(HEAD/'test/cross'/s/(a+'.pt'));assert z['nomination']==sha(LOCK);_,xs,ds=counts(z['ordinary'],gal,mode);vs[a].append(xs)
        out[mode]=dict(levels={a:summary(np.mean(xs,0),ds)['mR'] for a,xs in vs.items()},change=summary((np.array(vs['constant'])-vs['baseline']).mean(0),ds)['mR'])
    result['cross']=out
    dump(OUT/'results/constant_repeat.json',result);dump(OUT/'records/constant_repeat_rows.json',rec)
if __name__=='__main__':main()
