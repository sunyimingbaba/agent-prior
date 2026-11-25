"""Second reward calibration, per-fold candidate traces, and paired subgroup diagnostics."""
from collections import defaultdict
from fractions import Fraction
import gzip,json
import numpy as np
from multi_view import OUT,SOURCE,read,load,bank,sha,dump,folds,boot
from evaluate_new import image_folds
from single_scores import fixed

def indices(score,ranks):
    vals=sorted(Fraction(1,1+int(x)) for x in ranks);groups=defaultdict(list)
    for i,z in enumerate(score):groups[z].append(i)
    start=0;high=-1;win=[]
    for k in sorted(groups):
        ids=groups[k];z=sum(vals[start:start+len(ids)])/len(ids);start+=len(ids)
        if z>high:high=z;win=ids[:]
        elif z==high:win+=ids
    return sorted(win)
def main():
    lock=read(OUT/'inputs/multi_nomination.json');assert lock['weights'] is not None
    w=lock['weights'];coef=np.asarray(lock['model']['coef']);scale=np.asarray(lock['model']['scale'])
    alpha=np.array([w[2],w[0],w[1]])@coef/scale
    seen={r['attribute'] for t in ('first','second','third') for r in read(SOURCE/'inputs'/(t+'.json'))};out={};examples=[];records=[]
    b=bank(OUT);native=read(OUT/'inputs/native_index.json')['texts']
    for tag in ('fresh','extension'):
        rows=read(OUT/'inputs'/(tag+'.json'));lookup={r['image']:r for r in rows}
        fs=folds(tag)[0];base=read(OUT/'records'/('multi_'+tag+'.json'));z=defaultdict(list)
        previous={};groups=defaultdict(list)
        for f in fs:groups[f['image'],f['seed']].append(f)
        for (n,s),g in groups.items():
            c=lookup[n];ft=load(OUT/'records'/tag/s/'features'/(n+'.pt'))
            baseline,_=image_folds(c,s,b,ft['visual'].double().numpy(),np.asarray(ft['ranks'][:16]))
            assert len(baseline)==len(g)
            for f,r in zip(g,baseline):
                assert f['held']==r['held']
                # Use independently summed score components, then reassign exact rational payouts.
                order=(f['x']*alpha[None]).sum(1);ids=indices(order,f['rk']);old=r['baseline']['winners']
                if set(old)!=set(np.flatnonzero(f['old']==max(f['old']))):raise RuntimeError(n)
                vals=f['y'][ids].mean(0);ref=f['y'][old].mean(0);z[n,s].append(vals)
                rec=dict(tag=tag,image=n,pair=c['pair'],seed=s,attribute=c['attribute'],held=f['held'],held_caption=native[f['held']],
                         selected=ids,old_selected=old,good=r['good'],ordinary=f['y'][:,1].tolist(),detail=f['y'][:,2].tolist(),
                         anchor=f['anchor'].tolist(),new=vals.tolist(),old=ref.tolist())
                records.append(rec)
                if vals[0]==1 and ref[0]==0 and vals[1]>ref[1] and vals[2]>ref[2]:examples.append(rec)
        maxerr=max(float(np.max(abs(np.mean(z[r['image'],r['seed']],0)-r['means'][lock['method']]))) for r in base)
        assert maxerr<1e-12,maxerr
        out[tag]=dict(folds=sum(map(len,z.values())),baseline_replayed=len(groups),max_difference=maxerr,strata={})
        for label,sub in [('all_scene_cluster',base),('known_attribute',[r for r in base if r['attribute'] in seen]),('new_attribute',[r for r in base if r['attribute'] not in seen])]:
            if not sub:continue
            y=np.array([np.array(r['means'][lock['method']])-r['means']['old'] for r in sub])
            labels=[str(lookup[r['image']]['scene_id']) if label=='all_scene_cluster' else r['pair'] for r in sub]
            out[tag]['strata'][label]=dict(images=len(sub)//2,clusters=len(set(labels)),**{k:boot(y[:,i],np.ones(len(sub)),labels) for i,k in enumerate(('joint','mrr','detail'))})
    with gzip.open(OUT/'records/multi_folds.jsonl.gz','wt') as o:
        for r in records:o.write(json.dumps(r)+'\n')
    dump(OUT/'results/multi_audit.json',out)
    if examples:
        xs=sorted([r for r in examples if r['tag']=='fresh'],key=lambda r:(int(r['image'][:-4]),r['seed'],r['held']))
        z=xs[len(xs)//2];z['eligible_examples']=len(xs);dump(OUT/'results/example.json',z)
if __name__=='__main__':main()
