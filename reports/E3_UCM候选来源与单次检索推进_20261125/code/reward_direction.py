"""Derivative at zero of a within-group exponential tilt, without policy training."""
import sys
from collections import defaultdict
import numpy as np
from rosters import ROOT,OLD,REP,PRIOR,read,dump
from report_scores import bank,load
from panels import question_panels
from gate import K,rr_matrix,wins
from run_stage import boot
from orders import calibrate,advantages

def read_group(c,b,v,ranks,letters):
    ix=b['maps'][c['image']];p0=ix['native'][0]
    pos=[p0]+sorted(set(ix['native'])-{p0},key=b['hashes'].__getitem__)
    ns=v@b['native'].T;hs=v@b['hard'].T;mask=np.ones(len(b['even']),bool);mask[pos]=False
    left=ns[:,mask&b['even']];right=ns[:,mask&~b['even']]
    train_rr=rr_matrix(ns[:,pos],left);test_rr=rr_matrix(ns[:,pos],right)
    margin=(ns[:,p0]-left.max(1))[:16];hard=(hs[:,ix['A'][0]]-hs[:,ix['A'][1]])[:16]
    detail=np.mean([hs[:,ix[k][0]]>hs[:,ix[k][1]] for k in letters],axis=0)
    r2=calibrate(margin,ranks);first=r2==r2.max()
    assert np.array_equal(first,wins(margin,ranks))
    floor=train_rr[:16][first].sum(0)/int(first.sum())
    dense=np.unique(margin+K*hard,return_inverse=True)[1]
    means=defaultdict(list)
    for j in range(1,len(pos)):
        cols=[k for k in range(len(pos)) if k!=j][:2]
        gate=train_rr[:16,cols].sum(1)>=floor[cols].sum()
        r3=calibrate(dense+17*gate,ranks)
        target=test_rr[:,j].astype(float)
        good=(test_rr[:16,j]>test_rr[16,j])&(detail[:16]>detail[16])
        for name,reward in [('R2',r2),('R3',r3)]:
            adv=advantages(reward)
            assert abs(adv.mean())<1e-6
            for k,y in [('joint',good),('detail',detail[:16]),('mrr',target[:16])]:
                der=np.mean(adv*(y-y.mean()))
                # Centered finite differences provide an independent numerical check.
                step=1e-5
                def tilt(t):
                    w=np.exp(t*adv);return np.dot(w,y)/w.sum()
                assert abs(der-(tilt(step)-tilt(-step))/(2*step))<2e-9
                means[name+'_'+k].append(der)
            means[name+'_nonconstant'].append(np.ptp(reward)>0)
    return {k:np.mean(vals) for k,vals in means.items()}

def source(tag):
    if tag in ('development','repeat'):
        groups=defaultdict(dict)
        for c,seed,mode,b,v,r,letters,_,_ in question_panels(tag):
            groups[c['image'],seed][mode]=(v,r)
            if len(groups[c['image'],seed])==2:
                s=groups[c['image'],seed]['scene'];q=groups[c['image'],seed]['question']
                yield c,seed,b,s,q,letters
        return
    b=bank(OLD if tag=='first' else PRIOR/'reward_confirm')
    for s in 'ab':
        for c in read(ROOT/'inputs'/(tag+'.json')):
            z=load(ROOT/'records'/tag/s/'features'/(c['image']+'.pt'))['modes']
            def get(mode):
                return z[mode]['visual'].double().numpy(), np.array(z[mode]['ranks'][:16])
            yield c,s,b,get('scene'),get('question'),'CDE'

def group_metrics(tag):
    raw=[]
    for c,seed,b,(s,rs),(q,rq),letters in source(tag):
        if len(set(b['maps'][c['image']]['native']))<3:continue
        row=dict(image=c['image'],pair=c['pair'],attribute=c['attribute'],seed=seed)
        for key,v,r in [('scene',s[:16],rs),('question',q[:16],rq),
                        ('half1',np.vstack((s[:8],q[8:16])),np.r_[rs[:8],rq[8:16]]),
                        ('half2',np.vstack((s[8:16],q[:8])),np.r_[rs[8:16],rq[:8]])]:
            row[key]=read_group(c,b,np.vstack((v,s[-1])),r,letters)
        row['mixed']={k:(row['half1'][k]+row['half2'][k])/2 for k in row['scene']}
        raw.append(row)
    dump(ROOT/'records'/('signal_'+tag+'.json'),raw)
    return raw

def summary(rows,cluster):
    out={}
    for m in ('scene','mixed','question'):
        vals={k:np.array([r[m][k] for r in rows]) for k in rows[0][m]}
        for k in ('joint','detail','mrr'):
            vals['R3_minus_R2_'+k]=vals['R3_'+k]-vals['R2_'+k]
        out[m]={k:boot(v,np.ones(len(v)),[r[cluster] for r in rows]) for k,v in vals.items()}
    return out

def main(tag):
    rows=group_metrics(tag) if tag!='old108' else [r for t in ('development','repeat') for r in read(ROOT/'records'/('signal_'+t+'.json'))]
    dump(ROOT/'results'/('signal_'+tag+'.json'),summary(rows,'attribute' if tag=='old108' else 'pair'))
if __name__=='__main__':main(sys.argv[1])
