import numpy as np
import torch
from local import ROOT,OLD,RFPO,FORM,read,dump,load,sha,feature_path,banks,bank_for,calculate,summary_by_seed,paired,digest
from same_policy import rr
from single_scores import fixed

ARMS=('retained','parent','ordinary','guard','teacher')

def summary(scores,labels):
    return dict(levels={a:summary_by_seed(v,labels) for a,v in scores.items()},
        teacher_minus={a:paired(scores['teacher'],v,labels) for a,v in scores.items() if a!='teacher'},
        guard_minus_retained=paired(scores['guard'],scores['retained'],labels))

def features(tag,arm,s,rows):
    if tag.startswith('hold') and arm=='retained':
        view='scene' if tag=='hold_scene' else 'first'
        return torch.stack([load(feature_path(r,s,view))['visual'][-1] for r in rows])
    if arm=='retained':z=load(RFPO/'vectors'/(s+'_deployed_'+tag+'.pt'))
    elif arm=='parent' and not tag.startswith('hold'):
        z=load(OLD/'mined_policy'/(s+'_selected_joint')/'evaluate'/(tag+'.pt'))
    else:
        d=ROOT/'evaluate'/(s+'_'+arm);f=d/(tag+'.pt')
        assert sha(f)==read(d/tag/'complete.json')['features'];z=load(f)
        expected=read(ROOT/'inputs/lock.json')['starting'][s] if arm=='parent' else read(ROOT/'models'/(s+'_'+arm)/'complete.json')['final']
        assert z['adapter']==expected
    assert z['images']==[r['image'] for r in rows]
    return z['features']

def detail_native(row,visual,pack,mask):
    ix=pack['maps'][row['image']];native=pack['native'].double().numpy();hard=pack['hard'].double().numpy()
    v=visual.double().numpy();ordinary=native@v;d=hard@v
    ps=ix['native'];neg=mask.copy();neg[ps]=False;alt=sorted(set(ps)-{ps[0]})
    ranks=[rr(ordinary[i],ordinary[neg]) for i in alt]
    det=[int(d[ix[k][0]]>d[ix[k][1]]) for k in 'CDE']
    return det,ranks

def check_models():
    lock=read(ROOT/'inputs/lock.json');trace=read(ROOT/'inputs/teacher_trace.json');parts={}
    for s in 'ab':
        plan=read(ROOT/'inputs/train.json')[s];assert len(plan)==24
        for arm in ARMS[2:]:
            p=ROOT/'models'/(s+'_'+arm);proof=read(p/'complete.json');steps=read(p/'steps.json')
            assert proof['steps']==24 and proof['lock']==sha(ROOT/'inputs/lock.json') and proof['initial']==lock['starting'][s] and proof['logs']==sha(p/'steps.json')
            assert [z['image'] for z in steps]==[r['image'] for r in plan]
            views=[v for z in steps for v in z['views'].values()]
            parts[s+'_'+arm]=dict(**proof,min_cos=min(v['cos'] for v in views),max_norm=max(r['norm'] for r in steps))
    return parts

def same_output(rows,vals):
    labels=[r['pair'] for r in rows];out={}
    for arm in ARMS[:-1]:
        diffs={}
        for s in 'ab':
            changes={k:[] for k in ('joint_improved','joint_worse','net')}
            for (d,p),(b,q) in zip(vals['teacher'][s],vals[arm][s]):
                win=sum(x>y for x,y in zip(p,q))/len(p) if sum(d)>sum(b) else 0.
                loss=sum(x<y for x,y in zip(p,q))/len(p) if sum(d)<sum(b) else 0.
                for k,v in zip(changes,[win,loss,win-loss]):changes[k].append(v)
            diffs[s]={k:np.array(q) for k,q in changes.items()}
        out[arm]=summary_by_seed(diffs,labels)
    return out

def main():
    result=dict(audit=check_models());held=read(ROOT/'inputs/holdout.json');packs=banks()
    masks=[np.array([int(digest(','.join(map(str,x)))[-1],16)%2==0 for x in pack['native_tokens'].tolist()]) for pack in packs]
    prim=read(RFPO/'inputs/panel.json');off=read(FORM/'inputs/ucm.json')
    offidx={r['rgb_sha256']:i for i,r in enumerate(off)};joint_rows=[r for r in prim if r['rgb_sha256'] in offidx];assert len(joint_rows)==106
    banks106=load(FORM/'results/ucm_text.pt');text106=load(RFPO/'records/text.pt')
    orders=banks106['caption_order'].numpy().reshape(210,5);single={a:{} for a in ARMS};per=[]
    for tag,rows in [('hold_scene',held),('hold_first',held),('primary',prim),('official',off)]:
        vals={a:{} for a in ARMS};raw={a:{} for a in ARMS};labels=[r.get('pair',r['image']) for r in rows]
        for a in ARMS:
            for s in 'ab':
                v=features(tag,a,s,rows)
                if tag.startswith('hold'):
                    pairs=[detail_native(r,vv,bank_for(r,packs),masks[int(r['tag'] in ('extra','saved'))]) for r,vv in zip(rows,v)]
                    raw[a][s]=pairs
                    vals[a][s]=dict(CDE=np.array([np.mean(ds) for ds,_ in pairs]),mrr=np.array([float(sum(p)/len(p)) for _,p in pairs]))
                else:
                    vals[a][s]=calculate(tag,v,rows)
                if tag=='official':
                    joint_idx=[offidx[r['rgb_sha256']] for r in joint_rows]
                    vis=v[joint_idx];ns=fixed(vis,banks106['native']);hs=fixed(vis,text106['hard']);pairs=[]
                    for j,(r,idx) in enumerate(zip(joint_rows,joint_idx)):
                        ps=orders[idx];al=sorted(set(ps)-{ps[0]});mask=np.ones(ns.shape[1],bool);mask[ps]=False
                        m=text106['maps']['primary'][r['image']]
                        pairs.append(([int(hs[j,m[k][0]]>hs[j,m[k][1]]) for k in 'CDE'],[rr(ns[j,col],ns[j,mask]) for col in al]))
                    single[a][s]=pairs
                for i,row in enumerate(rows):
                    per.append(dict(tag=tag,arm=a,seed=s,image=row['image'],values={k:float(vv[i]) for k,vv in vals[a][s].items()}))
        result[tag]=summary(vals,labels)
        if tag.startswith('hold'):result[tag]['joint']=same_output(rows,raw)
    vals={a:{s:{'CDE':np.array([np.mean(ds) for ds,_ in ps]),'mrr':np.array([float(sum(p)/len(p)) for _,p in ps])} for s,ps in ss.items()} for a,ss in single.items()}
    result['shared']=summary(vals,[r['pair'] for r in joint_rows]);result['shared']['joint']=same_output(joint_rows,single)
    endpoints=[result[t]['teacher_minus']['retained'][k] for t,k in [('primary','CDE'),('official','mR')]]
    result['directional']=all(min(e['seeds'])>0 for e in endpoints)
    result['promote']=result['directional'] and all(e['interval'][0]>0 for e in endpoints)
    dump(ROOT/'results/score.json',result);dump(ROOT/'results/by_image.json',per)
if __name__=='__main__':main()
