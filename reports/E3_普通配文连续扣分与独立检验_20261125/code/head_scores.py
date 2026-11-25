"""Freeze the common checkpoint using only the 122 development pictures, then assess single-pass transfer."""
import sys
from pathlib import Path
import numpy as np
import torch
from compare import OUT,SOURCE,PRIOR,read,sha,dump,load,boot
from single_scores import RFPO,FORM,calculate,fixed
from transfer_scores import counts,scalar_values,summary
HEAD=OUT/'head';LOCK=OUT/'inputs/head_nomination.json'
def creds():
    return {p.name:sha(p) for p in [OUT/'单次先验奖励监督对照.md',OUT/'code/head_data.py',OUT/'code/head_fit.py',OUT/'code/head_scores.py']}
def detailed(v,b,rows,maps):
    s=fixed(v,b['hard']);ids=np.arange(len(rows));z=[]
    for k in 'CDE':
        ix=np.array([maps[c['image']][k] for c in rows]);z.append(s[ids,ix[:,0]]>s[ids,ix[:,1]])
    return np.stack(z,1).astype(float)
def per_image_gallery(v,b,rows):
    gal={'native':b['native'],'caption_order':torch.tensor([j for r in rows for j in b['maps'][r['image']]['native']])}
    _,raw,n=counts(v,gal,'native');assert (n==5).all();return (raw/np.array([1,1,1,5,5,5])).mean(1)
def compare(a,b,labels):
    delta={s:np.array(a[s])-np.array(b[s]) for s in 'ab'};z=np.stack(list(delta.values())).mean(0)
    stat=boot(z,np.ones(len(z)),labels)
    return dict(mean=stat['mean'],interval=stat['interval'],seeds={s:float(v.mean()) for s,v in delta.items()})
def develop():
    rows=read(SOURCE/'inputs/third.json');b=load(PRIOR/'guard/records/text.pt')
    names=['base']+[arm+str(e) for arm in ('ordinary','hard') for e in (1,3,6)]
    levels={};arr={}
    for n in names:
        arr[n]={}
        for s in 'ab':
            f=HEAD/s/'base_dev.pt' if n=='base' else HEAD/s/n[:-1]/('e'+n[-1]+'_dev.pt')
            o=load(f);d=detailed(o['scene'],b,rows,b['maps']).mean(1);mr=per_image_gallery(o['ordinary'],b,rows)
            arr[n][s]=[mr,d]
        levels[n]=dict(mR=float(np.mean([v[0].mean() for v in arr[n].values()])),detail=float(np.mean([v[1].mean() for v in arr[n].values()])))
    allowed=[e for e in (1,3,6) if all(levels['hard'+str(e)][k]>levels['base'][k] for k in ('mR','detail'))]
    pick=max(allowed or [1,3,6],key=lambda e:sum(levels['hard'+str(e)].values()))
    dump(OUT/'results/head_development.json',levels)
    cps={s:{arm:read(HEAD/s/arm/'done.json')['checkpoints'][str(pick)] for arm in ('ordinary','hard')} for s in 'ab'}
    dump(LOCK,dict(epoch=pick,exploratory=not allowed,checkpoints=cps,files=creds(),development=sha(OUT/'results/head_development.json')))
    print(pick,levels,flush=True)
def external():
    lock=read(LOCK);assert lock['files']==creds();out={};data=[]
    for tag in ('fresh','extension','primary','official','cross'):
        if tag=='primary':rows=read(RFPO/'inputs/panel.json')
        elif tag in ('fresh','extension'):rows=read(OUT/'inputs'/(tag+'.json'))
        else:rows=read(FORM/'inputs'/('ucm.json' if tag=='official' else 'rsitmd.json'))
        ids=[r['image'] for r in rows];arms=['baseline','ordinary','hard'];vec={a:{} for a in arms}
        for a in arms:
            for s in 'ab':
                p=HEAD/'test'/tag/s/(a+'.pt');z=load(p);assert z['images']==ids and z['nomination']==sha(LOCK)
                vec[a][s]=z['scene' if tag in ('fresh','extension','primary') else 'ordinary']
        if tag in ('fresh','extension','primary'):
            b=load((RFPO if tag=='primary' else OUT)/'records/text.pt');maps=b['maps']['primary'] if tag=='primary' else b['maps']
            d={a:{s:detailed(vec[a][s],b,rows,maps) for s in 'ab'} for a in arms}
            acc={a:{s:x.mean(1) for s,x in vals.items()} for a,vals in d.items()}
            labels=[r['pair'] for r in rows]
            stats=dict(images=len(rows),levels={a:float(np.mean(list(acc[a].values()))) for a in arms},
                contrasts={l+'-'+r:compare(acc[l],acc[r],labels) for l,r in [('hard','baseline'),('ordinary','baseline'),('hard','ordinary')]})
            if tag=='primary':
                stats['replay']={s:int(np.count_nonzero(acc['baseline'][s]!=calculate(tag,load(RFPO/'vectors'/(s+'_deployed_primary.pt'))['features'],rows)['CDE'])) for s in 'ab'}
            else:
                mismatches={}
                for s in 'ab':
                    other=torch.stack([load(OUT/'records'/tag/s/'features'/(r['image']+'.pt'))['visual'][-1] for r in rows])
                    mismatches[s]=dict(max_embedding=float((other-vec['baseline'][s]).abs().max()),decisions=int(np.sum(detailed(other,b,rows,maps)!=d['baseline'][s])))
                stats['replay']=mismatches
            for i,c in enumerate(rows):
                data.append(dict(tag=tag,image=c['image'],pair=c['pair'],attr=c['attribute'],decisions={a:{s:d[a][s][i].tolist() for s in 'ab'} for a in arms}))
            out[tag]=stats
        else:
            dn='ucm' if tag=='official' else 'rsitmd';b=load(FORM/'results'/(dn+'_text.pt'));stats={}
            for mode in ('native','alternatives'):
                v={};basehash={}
                for a in arms:
                    v[a]=[]
                    for s in 'ab':
                        ids2,n,ds=counts(vec[a][s],b,mode);v[a].append(n)
                        if a=='baseline':
                            raw=load(FORM/'results'/(dn+'_'+s+'_R2.pt'))['features'];_,n0,_=counts(raw,b,mode)
                            basehash[s]=dict(max_embedding=float((raw-vec[a][s]).abs().max()),max_recall_diff=float(abs(n-n0).max()))
                stats[mode]=dict(images=len(ds),captions=int(ds.sum()),replay=basehash,
                    levels={a:summary(np.mean(v[a],axis=0),ds) for a in arms},
                    contrasts={l+'-'+r:summary((np.asarray(v[l])-v[r]).mean(0),ds) for l,r in [('hard','baseline'),('ordinary','baseline'),('hard','ordinary')]},
                    seed_mR={a:[float(scalar_values(z.sum(0),np.asarray(len(ds)),np.asarray(ds.sum()))[-1]) for z in v[a]] for a in arms})
                for i,j in enumerate(ids2):
                    data.append(dict(tag=tag,mode=mode,image=ids[j],n=int(ds[i]),recall={a:[v[a][k][i].tolist() for k in (0,1)] for a in arms}))
            out[tag]=stats
        print(tag,flush=True)
    dump(OUT/'results/head_single.json',out);dump(OUT/'records/head_single_rows.json',data)
if __name__=='__main__':{'develop':develop,'external':external}[sys.argv[1]]()
