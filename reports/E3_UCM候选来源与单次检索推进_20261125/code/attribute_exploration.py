"""Paired source/reward decomposition and disjoint half-batch candidate ablation."""
from collections import defaultdict
import sys
import numpy as np
from when_to_sample import ROOT,PRIOR,read,dump,sha,load,bank,archive_raw
from verify import rr,winners
from run_stage import boot
from gate import K

def score_group(c,b,v,ranks):
    nm=c['image'];idx=b['maps'][nm]
    pos=[idx['native'][0]]+sorted(set(idx['native'])-{idx['native'][0]},key=b['hashes'].__getitem__)
    assert len(pos)>=3
    native=v@b['native'].T;hard=v@b['hard'].T
    allow=np.ones(native.shape[1],bool);allow[pos]=False
    train=native[:,allow&b['even']];val=native[:,allow&~b['even']]
    margin=native[:16,pos[0]]-train[:16].max(1);detail=np.mean([hard[:,idx[k][0]]>hard[:,idx[k][1]] for k in 'CDE'],0)
    a=hard[:16,idx['A'][0]]-hard[:16,idx['A'][1]]
    tx=[[rr(s[j],z) for j in pos] for s,z in zip(native[:16],train[:16])]
    vx=np.array([[rr(s[j],z) for j in pos] for s,z in zip(native,val)],object)
    out={}
    for slots in [np.arange(16),np.arange(8),np.arange(8,16)]:
        n=len(slots);r=ranks[slots]
        r2=slots[winners(margin[slots],r)];dense=np.unique((margin+K*a)[slots],return_inverse=True)[1]
        vals=defaultdict(list)
        for j in range(1,len(pos)):
            left=[k for k in range(len(pos)) if k!=j][:2]
            pr=[sum(x[k] for k in left) for x in tx];floor=sum(pr[i] for i in r2)/len(r2)
            gate=np.array([pr[i]>=floor for i in slots])
            r3=slots[winners(dense+(n+1)*gate,r)]
            mrr=vx[:,j].astype(float);good=(vx[:16,j]>vx[16,j])&(detail[:16]>detail[16])
            for name,ix in [('random',slots),('R2',r2),('R3',r3)]:
                for name2,y in [('joint',good),('mrr',mrr),('detail',detail)]:
                    vals[name+'_'+name2].append(float(y[ix].mean()))
        for k,x in vals.items():out.setdefault(str(n)+'_'+k,[]).append(np.mean(x))
    return {k:float(np.mean(x)) for k,x in out.items()}

def deltas(rows):
    metrics=('joint','mrr','detail');out={}
    for m in metrics:
        sigs={
            'free_R3_minus_random':lambda r:r['free']['16_R3_'+m]-r['free']['16_random_'+m],
            'free_R3_minus_R2':lambda r:r['free']['16_R3_'+m]-r['free']['16_R2_'+m],
            'free_16_minus_8_R3':lambda r:r['free']['16_R3_'+m]-r['free']['8_R3_'+m],
            'locked_16_minus_8_R3':lambda r:r['locked']['16_R3_'+m]-r['locked']['8_R3_'+m],
            'free_minus_locked_random':lambda r:r['free']['16_random_'+m]-r['locked']['16_random_'+m],
            'free_minus_locked_R3':lambda r:r['free']['16_R3_'+m]-r['locked']['16_R3_'+m],
            'selection_interaction':lambda r:(r['free']['16_R3_'+m]-r['locked']['16_R3_'+m])-(r['free']['16_random_'+m]-r['locked']['16_random_'+m])}
        vals={}
        for k,f in sigs.items():
            y=[f(r) for r in rows];z=boot(y,np.ones(len(y)),[r['pair'] for r in rows])
            vals[k]=dict(mean=z['mean'],interval=z['interval'])
        out[m]=vals
    return dict(images=len({r['image'] for r in rows}),image_seeds=len(rows),pairs=len({r['pair'] for r in rows}),metrics=out)

def main(tag):
    roster=read(ROOT/'inputs'/(tag+'.json'));b=bank(PRIOR/('guard' if tag=='third' else 'reward_confirm'))
    expected={(r['image'],r['seed']):r for r in read(ROOT/'records'/('controlled_timing_'+tag+'.json'))}
    rows=[]
    for s in 'ab':
        base=ROOT/'records/prefix'/tag/s
        for c,raw in zip(roster,archive_raw(base,roster),strict=True):
            nm=c['image']
            if (nm,s) not in expected:continue
            f=load(base/'features'/(nm+'.pt'));assert f['source']==sha_raw(raw,nm,base)
            g=f['modes'];free=g.get('free') if tag=='third' else load(ROOT/'records/second'/s/'features'/(nm+'.pt'))['modes']['scene']
            anchor=g['locked']['visual'][-1].double().numpy()
            row=dict(image=nm,seed=s,pair=c['pair'],uses_tool=bool(raw['calls']))
            for arm,mode in [('locked',g['locked']),('free',free)]:
                vv=mode['visual'].double().numpy();vv[-1]=anchor
                row[arm]=score_group(c,b,vv,np.asarray(mode['ranks'][:16]))
                for m in ('joint','mrr','detail'):assert abs(row[arm]['16_R3_'+m]-expected[nm,s][arm][m])<1e-12
            rows.append(row)
    levels={arm:{k:np.mean([r[arm][k] for r in rows]) for k in rows[0][arm]} for arm in ('locked','free')}
    result=dict(protocol=sha(ROOT/'候选数量与奖励选中核查口径.md'),all=deltas(rows),levels=levels)
    for name,flag in [('greedy_calls_tool',True),('greedy_finishes',False)]:
        sub=[r for r in rows if r['uses_tool']==flag]
        if sub:result[name]=deltas(sub)
    dump(ROOT/'results'/('attribution_'+tag+'.json'),result);dump(ROOT/'records'/('attribution_'+tag+'_images.json'),rows)
def sha_raw(raw,nm,base):
    assert raw['image']==nm
    return read(base/'raw/complete.json')['files'][nm]
if __name__=='__main__':main(sys.argv[1])
