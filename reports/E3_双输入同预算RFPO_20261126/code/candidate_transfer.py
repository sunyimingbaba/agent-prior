"""Post-training diagnostic of the ordinary and strict rewards on the exact same archived R2 candidates."""
from fractions import Fraction
import numpy as np
import torch
from study import STAGE, read, dump, load, banks, feature_path, digest, rr, summary_by_seed, paired
from strict_reward import strict_rewards
from exact_reward import payouts


def candidate_grid(row,seed,view,reader,mask):
    path=feature_path(row,seed,view); source=load(path); assert source['rgb']==row['rgb_sha256']
    v=source['visual'].double().numpy();native,hard,maps=reader;ix=maps[row['image']]
    pos=ix['native'];alts=sorted(set(pos)-{pos[0]});train=mask.copy();test=~mask.copy();train[pos]=False;test[pos]=False
    scores=v@native.T;h=v@hard.T
    ranks=[[[rr(x[i],x[m]) for i in alts] for x in scores] for m in (train,test)]
    counts=np.column_stack([h[:,ix[k][0]]>h[:,ix[k][1]] for k in 'ABCDE'])
    return ranks,counts


def choices(grid,counts,folds):
    fit,check=grid;nt=len(fit[0]);out={a:[] for a in ('ordinary','joint','tiebreak')};changes=[];tie_changes=[]
    opportunities=0;mean_opportunities=0;hits={a:0. for a in out};examples=[]
    for held in folds:
        select=[j for j in range(nt) if j!=held];evaluate=[held] if held is not None else list(range(nt))
        if not select:continue
        for block in range(2):
            ids=list(range(block*8,(block+1)*8))+[16]
            anchor_h=int(counts[16,2:].sum())
            good={i for i in ids[:-1] if int(counts[i,2:].sum())>anchor_h and any(check[i][j]>check[16][j] for j in evaluate)}
            opportunities+=int(bool(good))
            mean_opportunities+=int(any(sum(check[i][j]-check[16][j] for j in evaluate)>0 for i in good))
            packet=dict(rr=[[[fit[i][j].numerator,fit[i][j].denominator] for j in select] for i in ids],
                        detail=[int(sum(counts[i,:2])) for i in ids])
            rewards,_=strict_rewards(packet);wins={}
            rewards['tiebreak']=payouts([(sum(fit[i][j] for j in select)/len(select),int(counts[i,:2].sum())) for i in ids[:-1]])
            for a,rs in rewards.items():
                winners=[ids[j] for j in np.flatnonzero(rs==rs.max())];wins[a]=winners
                hits[a]+=len(set(winners)&good)/len(winners)
                values=[]
                for i in winners:
                    det=int(counts[i,2:].sum());delta=det-int(counts[-1,2:].sum())
                    win=np.mean([check[i][j]>check[-1][j] for j in evaluate]) if delta>0 else 0.
                    lose=np.mean([check[i][j]<check[-1][j] for j in evaluate]) if delta<0 else 0.
                    values.append([det/3,np.mean([float(check[i][j]) for j in evaluate]),win,lose,win-lose])
                out[a].append(np.mean(values,axis=0))
            changes.append(wins['ordinary']!=wins['joint'])
            tie_changes.append(wins['ordinary']!=wins['tiebreak'])
            if held is None and changes[-1]:
                selected={}
                for a,slots in {**wins,'anchor':[16]}.items():
                    selected[a]=dict(slots=slots,AB=float(np.mean([counts[i,:2].sum() for i in slots])),
                        CDE=float(np.mean([counts[i,2:].sum() for i in slots])),
                        per_wording=counts[slots,2:].mean(0).tolist(),
                        odd_mrr=float(np.mean([float(fit[i][j]) for i in slots for j in select])),
                        even_mrr=float(np.mean([float(check[i][j]) for i in slots for j in evaluate])),
                        success=len(set(slots)&good)/len(slots))
                examples.append(dict(block=block,**selected))
    if not changes:return None
    return {a:np.mean(v,axis=0).tolist() for a,v in out.items()}|dict(changed=float(np.mean(changes)),groups=len(changes),
        changed_groups=sum(changes),tie_changes=sum(tie_changes),opportunities=opportunities,mean_opportunities=mean_opportunities,selected=hits,changed_examples=examples)


def main():
    torch.set_num_threads(1)
    packs=banks();text=[(p['native'].double().numpy(),p['hard'].double().numpy(),p['maps']) for p in packs]
    odd=[np.array([int(digest(','.join(map(str,t)))[-1],16)%2==1 for t in p['native_tokens'].tolist()]) for p in packs]
    out={};logs=[];metrics=('CDE','mrr','win','loss','net')
    for panel,filename in [('hold','holdout'),('repeat','repeat')]:
        rows=read(STAGE/'inputs'/(filename+'.json'))
        for view in ('first','scene'):
            stats={};grouped={}
            for seed in 'ab':
                for row in rows:
                    g=int(row['tag'] in ('saved','extra'))
                    grid,counts=candidate_grid(row,seed,view,text[g],odd[g])
                    for name,folds in [('all',[None]),('leave_one',range(len(grid[0][0])))]:
                        r=choices(grid,counts,folds)
                        if r is not None:
                            grouped.setdefault((seed,name),[]).append((row,r))
                            logs.append(dict(seed=seed,panel=panel,view=view,mode=name,image=row['image'],**r))
            for name in ('all','leave_one'):
                labels=[r['pair'] for r,_ in grouped['a',name]]
                assert labels==[r['pair'] for r,_ in grouped['b',name]]
                data={arm:{s:dict(zip(metrics,np.array([v[arm] for _,v in grouped[s,name]]).T)) for s in 'ab'} for arm in ('ordinary','joint','tiebreak')}
                stats[name]=dict(images=len(labels),groups=sum(v['groups'] for s in 'ab' for _,v in grouped[s,name]),
                    changed_groups=sum(v['changed_groups'] for s in 'ab' for _,v in grouped[s,name]),
                    levels={a:summary_by_seed(v,labels) for a,v in data.items()},difference=paired(data['joint'],data['ordinary'],labels))
                records=[v for s in 'ab' for _,v in grouped[s,name]]
                n=sum(v['opportunities'] for v in records)
                stats[name].update(opportunities=n,mean_opportunities=sum(v['mean_opportunities'] for v in records),
                    tiebreak_groups=sum(v['tie_changes'] for v in records),tiebreak_minus_ordinary=paired(data['tiebreak'],data['ordinary'],labels),
                    selection_given_opportunity={a:sum(v['selected'][a] for v in records)/n if n else None for a in data})
            out[panel+'_'+view]=stats
    dump(STAGE/'results/candidate_transfer.json',out);dump(STAGE/'results/candidate_transfer_images.json',logs)


if __name__=='__main__':
    main()
