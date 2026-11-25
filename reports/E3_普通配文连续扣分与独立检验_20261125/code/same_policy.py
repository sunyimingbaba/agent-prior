"""Use exactly the same official single-pass visual vector for both alternative-caption and hard-detail evaluation."""
from fractions import Fraction
import sys
import numpy as np
from compare import OUT,read,dump,load,boot
from single_scores import RFPO,FORM,fixed

def rr(value,negative):
    above=int((negative>value).sum());tied=1+int((negative==value).sum())
    return sum(Fraction(1,i) for i in range(above+1,above+tied+1))/tied

def main(stage):
    arms={'policy':['R2','old','new'],'exact_policy':['ordinary','joint'],
          'mined_policy':['selected_ordinary','uniform_joint','selected_joint']}[stage]
    target=arms[-1];arms+=['untrained','retained']
    primary=read(RFPO/'inputs/panel.json');official=read(FORM/'inputs/ucm.json')
    idx={r['rgb_sha256']:i for i,r in enumerate(official)}
    rows=[r for r in primary if r['rgb_sha256'] in idx];assert len(rows)==106
    gallery=load(FORM/'results/ucm_text.pt');hard=load(RFPO/'records/text.pt')
    cols=gallery['caption_order'].numpy().reshape(210,5);maps=hard['maps']['primary']
    out={};exclude=[];perphoto=[];labels=[]
    for arm in arms:
        out[arm]={}
        for s in 'ab':
            if arm in ('untrained','retained'):
                z=load(RFPO/'vectors'/(s+'_'+('R0' if arm=='untrained' else 'deployed')+'_official.pt'))
            else:
                base=OUT/stage/(s+'_'+arm);z=load(base/'evaluate/official.pt')
                assert z['adapter']==read(base/'complete.json')['final']
            assert z['images']==[r['image'] for r in official]
            v=z['features'][[idx[r['rgb_sha256']] for r in rows]]
            ordinary=fixed(v,gallery['native']);detail=fixed(v,hard['hard']);sc=[]
            for j,r in enumerate(rows):
                n=r['image'];ps=cols[idx[r['rgb_sha256']]];alt=sorted(set(ps)-{ps[0]})
                if not alt:
                    if arm==arms[0] and s=='a':exclude.append(n)
                    continue
                if arm==arms[0] and s=='a':labels.append(r['pair'])
                ok=np.ones(ordinary.shape[1],bool);ok[ps]=False
                d=sum(int(detail[j,maps[n][k][0]]>detail[j,maps[n][k][1]]) for k in 'CDE')
                ranks=[rr(ordinary[j,t],ordinary[j,ok]) for t in alt]
                sc.append((n,d,ranks))
            out[arm][s]=sc
    def avg(vals):
        x=np.array(vals).reshape(2,-1).mean(0);return boot(x,np.ones_like(x),labels)
    report=dict(shared=106,eligible=len(labels),excluded=exclude,levels={},contrasts={})
    for arm,by in out.items():
        report['levels'][arm]=dict(detail=avg([d/3 for v in by.values() for _,d,p in v]),mrr=avg([float(sum(p)/len(p)) for v in by.values() for _,d,p in v]))
        if arm==target:continue
        vals={k:[] for k in ('joint_improved','joint_worse','net','detail','mrr')}
        for s in 'ab':
            for (n,d,ps),(other,e,qs) in zip(out[target][s],by[s]):
                assert n==other and len(ps)==len(qs)
                win=sum(p>q for p,q in zip(ps,qs))/len(ps) if d>e else 0.
                loss=sum(p<q for p,q in zip(ps,qs))/len(ps) if d<e else 0.
                v=dict(joint_improved=win,joint_worse=loss,net=win-loss,detail=(d-e)/3,mrr=float((sum(ps)-sum(qs))/len(ps)))
                for k,x in v.items():vals[k].append(x)
                perphoto.append(dict(seed=s,reference=arm,image=n,**v))
        report['contrasts'][arm]={k:avg(v) for k,v in vals.items()}
    dump(OUT/'results'/(stage+'_same_output.json'),report)
    dump(OUT/'records'/(stage+'_same_output.json'),perphoto)

if __name__=='__main__':main(sys.argv[1])
