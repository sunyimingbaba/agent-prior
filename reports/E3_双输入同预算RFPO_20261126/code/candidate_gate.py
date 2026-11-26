from fractions import Fraction
import numpy as np
import torch
from study import STAGE, read, dump, load, sha, feature_path, banks, digest, rr, summary_by_seed
from strict_reward import strict_rewards
from exact_reward import payouts
from dual_scores import panel_outcomes


def ranked_safe(sc):
    ranks=[[Fraction(n,d) for n,d in row] for row in sc['rr']]
    mean=[sum(v)/len(v) for v in ranks]
    h=sc['detail'];order=[]
    for i,v in enumerate(ranks[:-1]):
        normal=mean[i]-mean[-1]; hard=h[i]-h[-1]
        strict=normal>0 and hard>0
        safe=(normal>0 and hard==0) or (hard>0 and normal==0)
        fraction=Fraction(sum(a>b for a,b in zip(v,ranks[-1])),len(v)) if strict else Fraction()
        order.append((2 if strict else 1 if safe else 0, fraction, mean[i]))
    return payouts(order)


def main():
    torch.set_num_threads(1)
    packs=banks()
    text=[(b['native'].double().numpy(),b['hard'].double().numpy(),b['maps']) for b in packs]
    even=[np.array([int(digest(','.join(map(str,t)))[-1],16)%2==0 for t in b['native_tokens'].tolist()]) for b in packs]
    logs=[]
    for s,rows in read(STAGE/'inputs/train.json').items():
        for row in rows:
            group=int(row['tag'] in ('saved','extra'))
            native,hard,maps=text[group];ix=maps[row['image']];odd=(~even[group]).copy();odd[ix['native']]=False
            alts=sorted(set(ix['native'])-{ix['native'][0]})
            for view in ('first','scene'):
                path=feature_path(row,s,view);source=load(path);v=source['visual'];assert source['rgb']==row['rgb_sha256']
                ordinary=v.double().numpy()@native.T;h=v.double().numpy()@hard.T
                # Each text comparison is made on the same archived trajectory.
                m=[[rr(q[i],q[odd]) for i in alts] for q in ordinary]
                co=np.column_stack([h[:,ix[k][0]]>h[:,ix[k][1]] for k in 'AB']).sum(1)
                judged=panel_outcomes([row]*17,v,text,even)
                for block in range(2):
                    slots=list(range(8*block,8*block+8))
                    ids=slots+[16]
                    sc=dict(rr=[[[t.numerator,t.denominator] for t in m[i]] for i in ids],detail=[int(co[i]) for i in ids])
                    regular,_=strict_rewards(sc);safe=ranked_safe(sc);picks={}
                    for name,reward in [('strict',regular['joint']),('safe',safe)]:
                        won=[slots[int(j)] for j in np.flatnonzero(reward==max(reward))]
                        z=[]
                        for w in won:
                            correct,ranks=judged[w];base,other=judged[16];change=np.array(ranks)-other
                            win=(np.mean(change>0) if sum(correct)>sum(base) else 0.)
                            loss=(np.mean(change<0) if sum(correct)<sum(base) else 0.)
                            z.append([np.mean(correct),np.mean(ranks),win,loss,win-loss])
                        picks[name]=dict(top=won,**dict(zip(('CDE','mrr','win','loss','net'),np.mean(z,axis=0))))
                    logs.append(dict(seed=s,image=row['image'],view=view,block=block,file=sha(path),**picks))
    out={}
    for view in ('first','scene'):
        rows=[r for r in logs if r['view']==view];assert len(rows)==96
        out[view]=dict(groups=len(rows),changed=sum(r['strict']['top']!=r['safe']['top'] for r in rows))
        for arm in ('strict','safe'):
            out[view][arm]={k:float(np.mean([r[arm][k] for r in rows])) for k in ('CDE','mrr','win','loss','net')}
        out[view]['net_delta']=out[view]['safe']['net']-out[view]['strict']['net']
    gate=all(z['changed']>=4 and z['net_delta']>=-1e-12 for z in out.values()) and any(z['net_delta']>1e-12 for z in out.values())
    dump(STAGE/'results/candidate_gate.json',dict(views=out,run_supplement=gate))
    dump(STAGE/'results/candidate_gate_groups.json',logs)


if __name__=='__main__':
    main()
