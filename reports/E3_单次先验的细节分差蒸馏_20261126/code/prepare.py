from collections import defaultdict
from fractions import Fraction
import numpy as np
import torch
from local import ROOT,OLD,FORM,RFPO,read,dump,sha,load,digest,feature_path,banks,bank_for

def rr(values,neg):
    ans=[]
    for v in values:
        first=1+int((neg>v).sum());count=1+int((neg==v).sum())
        ans.append(sum(Fraction(1,i) for i in range(first,first+count))/count)
    return sum(ans)/len(ans)

def build_target(row,seed,view,pack,odd):
    path=feature_path(row,seed,view);z=load(path);assert z['rgb']==row['rgb_sha256']
    v=z['visual'].float();assert v.shape==(17,768)
    ix=pack['maps'][row['image']];pos=ix['native'];alts=sorted(set(pos)-{pos[0]});assert alts
    mask=odd.copy();mask[pos]=False;neg=np.flatnonzero(mask)
    ordinary=(v.double()@pack['native'].double().T).numpy()
    rank=[rr(o[alts],o[neg]) for o in ordinary]
    hard=torch.stack([pack['hard'][ix[k][0]]-pack['hard'][ix[k][1]] for k in 'AB']).float()
    hard_margin=v.double()@hard.double().T
    correct=(hard_margin>0).sum(1).tolist()
    ok=[i for i in range(16) if rank[i]>rank[16] and correct[i]>correct[16]]
    winner=max(ok,key=lambda i:rank[i]) if ok else 16
    top=neg[np.argsort(-ordinary[16,neg],kind='stable')[:8]]
    ordinary_dir=(pack['native'][alts,None,:]-pack['native'][None,top,:]).reshape(-1,768).float()
    initial=v[16];teacher=v[winner]
    goals=dict(base=initial,ordinary_dir=ordinary_dir,ordinary_target=torch.maximum(ordinary_dir@initial,ordinary_dir@teacher),
        hard_dir=hard,guard=hard@initial,teacher=torch.maximum(hard@initial,hard@teacher))
    return goals,dict(image=row['image'],seed=seed,view=view,file=sha(path),winner=winner,qualified=ok,ordinary=[float(rank[i]) for i in [16,winner]],
                    hard=[correct[16],correct[winner]],hard_margin=[hard_margin[i].tolist() for i in [16,winner]])

def main():
    rows=read(OLD/'inputs/real_roster.json');previous=read(OLD/'inputs/mined_roster.json');idx={r['image']:r for r in rows}
    occupied={r['pair'] for plan in previous.values() for r in plan};sets=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['pair'] not in occupied:sets[r['tag']][r['pair']].append(r)
    held=[];remaining=[]
    for tag,ps in sorted(sets.items()):
        names=sorted(ps,key=lambda k:digest('greedy-margin-v1',tag,k))
        held+=sum([ps[k] for k in names[:8]],[]);remaining += [(digest('greedy-margin-v1',tag,k),ps[k]) for k in names[8:]]
    for _,pair in sorted(remaining):
        if len(held)>=64:break
        held+=pair
    assert len(held)==len({r['rgb_sha256'] for r in held})==64
    assert not {r['pair'] for r in held}&occupied
    formal={r['rgb_sha256'] for p in (RFPO/'inputs/panel.json',FORM/'inputs/ucm.json') for r in read(p)}
    assert not formal & {r['rgb_sha256'] for r in rows}
    fields=('image','rgb_sha256','split','row','orig_query','scene_query','feedback','pair','tag')
    slim=lambda r:{k:r[k] for k in fields}
    train={s:[slim(idx[r['image']]) for r in previous[s+'_selected']] for s in 'ab'}
    for name,obj in [('train',train),('holdout',[slim(r) for r in held])]:dump(ROOT/'inputs'/(name+'.json'),obj)
    packs=banks();tokens=packs[0]['native_tokens'];assert torch.equal(tokens,packs[1]['native_tokens'])
    odd=np.array([int(digest(','.join(map(str,a)))[-1],16)%2==1 for a in tokens.tolist()])
    trace=[];hs={}
    for s in 'ab':
        target={}
        for r in train[s]:
            for view in ('first','scene'):
                key=(r['image'],view);target[key],proof=build_target(r,s,view,bank_for(r,packs),odd);trace.append(proof)
        path=ROOT/'inputs'/('targets_'+s+'.pt');torch.save(target,path);hs[s]=sha(path)
    dump(ROOT/'inputs/teacher_trace.json',trace)
    dump(ROOT/'inputs/lock.json',dict(protocol=sha(ROOT/'实验口径.md'),teacher_files=hs,trace=sha(ROOT/'inputs/teacher_trace.json'),
        train=sha(ROOT/'inputs/train.json'),holdout=sha(ROOT/'inputs/holdout.json'),old_plans=sha(OLD/'inputs/mined_roster.json'),
        starting={s:read(OLD/'mined_policy'/(s+'_selected_joint')/'complete.json')['final'] for s in 'ab'},
        implementation={p.name:sha(p) for p in [ROOT/'code'/k for k in ('model_run.py','fit.py','evaluate.py')]}))
if __name__=='__main__':main()
