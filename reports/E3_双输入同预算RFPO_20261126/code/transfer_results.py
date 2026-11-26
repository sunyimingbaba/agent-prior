import numpy as np
from study import STAGE, FORM, load, read, sha, dump, counts, retrieval_summary, scalar_values


def main():
    rows=read(FORM/'inputs/rsitmd.json'); gallery=load(FORM/'results/rsitmd_text.pt')
    names=[r['image'] for r in rows]; history=read(FORM/'results/paired_results.json')['rsitmd']
    result={}; trace=[]
    for mode in ('native','alternatives'):
        data={}; labels=None
        for arm in ('retained','dualplain','dualjoint'):
            xs=[]
            for seed in 'ab':
                if arm=='retained':
                    z=load(FORM/'results'/('rsitmd_'+seed+'_R2.pt'))
                    assert z['manifest']==sha(FORM/'inputs/rsitmd.json')
                else:
                    p=STAGE/'cross'/(seed+'_'+arm)/'features.pt'
                    cp=read(p.with_name('complete.json'))
                    assert sha(p)==cp['features'] and cp['protocol']==sha(STAGE/'实验口径.md')
                    z=load(p)
                    assert z['images']==names and z['adapter']==read(STAGE/'models'/(seed+'_'+arm)/'complete.json')['final']
                ids,x,den=counts(z['features'],gallery,mode)
                if labels is None:labels,ds=ids,den
                assert np.array_equal(ids,labels) and np.array_equal(den,ds)
                if arm=='retained':
                    truth=history['all_'+mode]['levels'][seed+'_R2']['mR']
                    assert abs(100*scalar_values(x.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]-truth)<1e-9
                xs.append(x)
                trace.extend(dict(mode=mode,arm=arm,seed=seed,image=names[i],caption_count=int(d),hits=h.tolist()) for i,d,h in zip(ids,ds,x))
            data[arm]=np.stack(xs)
        result[mode]=dict(images=len(ds),captions=int(ds.sum()),levels={a:retrieval_summary(x.mean(0),ds) for a,x in data.items()},contrasts={})
        for a,b in [('dualjoint','dualplain'),('dualjoint','retained'),('dualplain','retained')]:
            diff=data[a]-data[b]
            result[mode]['contrasts'][a+'-'+b]=dict(metrics=retrieval_summary(diff.mean(0),ds),
                seed_mR=[float(scalar_values(x.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]) for x in diff])
    dump(STAGE/'results/transfer.json',result);dump(STAGE/'results/transfer_images.json',trace)


if __name__=='__main__':
    main()
