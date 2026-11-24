"""Paired selection on locked candidates. Tie probabilities are exact and deterministic."""
import csv,hashlib,io,json,math,tarfile,os,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.stats import rankdata

ROOT=Path(os.environ.get('E3_PANEL_ROOT',Path(__file__).resolve().parents[1]))
RECORDS=ROOT/'records';OUT=ROOT/'results'
def read(p):return json.loads(p.read_bytes())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def tensor(data):return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
def files(arm):
    archive=RECORDS/f'{arm}.tar.gz';proof=read(RECORDS/('extra_provenance.json' if arm.startswith('extra_') else 'provenance.json'))
    assert sha(archive)==proof['archives'][arm]
    with tarfile.open(archive) as tar:
        return {m.name:tar.extractfile(m).read() for m in tar if m.isfile()}
def calibrated(order,proxy):
    # Same grouped order-statistic remapping as matched_scores in the training entry point.
    r=rankdata(order,method='min').astype(int)-1
    return np.array([np.sort(proxy)[i:i+(r==i).sum()].mean() for i in r])
def winner(x):
    return np.flatnonzero(x==np.max(x))
def boot(x,groups):
    x=np.array(x);labels=list(dict.fromkeys(groups));loc={p:i for i,p in enumerate(labels)}
    ids=np.array([loc[g] for g in groups]);n=len(labels)
    sums=np.bincount(ids,weights=x,minlength=n);cnt=np.bincount(ids,minlength=n)
    draws=np.random.default_rng(20261124).integers(n,size=(20000,n))
    low,high=np.quantile(sums[draws].sum(1)/cnt[draws].sum(1),[.025,.975])
    return dict(mean=float(x.mean()),lo=float(low),hi=float(high),groups=n)
def main(mode='main'):
    global OUT
    assert mode in ('main','extra','combined')
    if mode!='main':
        OUT=ROOT/'results'/mode;OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    panel=read(ROOT/'inputs/panel.json');lock=read(ROOT/'inputs/lock.json');assert sha(ROOT/'inputs/panel.json')==lock['panel']
    assert sha(ROOT/'实验口径.md')==lock['protocol']
    pr=read(RECORDS/'provenance.json');bankfile=RECORDS/'text.pt'
    assert sha(bankfile)==pr['text']==read(RECORDS/'read_complete.json')['text']
    text=torch.load(bankfile,map_location='cpu',weights_only=True)
    native=text['native'].double();hard=text['hard'].double()
    even=np.array([int(hashlib.sha256(','.join(map(str,v)).encode()).hexdigest()[-1],16)%2==0 for v in text['native_tokens'].tolist()])
    native_token_set={tuple(v) for v in text['native_tokens'].tolist()};assert all(tuple(v) not in native_token_set for v in text['tokens'].tolist())
    detail=[];choice=[];diagnosis=[]
    for arm in ('a','b'):
        banks=[arm] if mode=='main' else ['extra_'+arm] if mode=='extra' else [arm,'extra_'+arm]
        records={}
        for prefix in banks:records.update(files(prefix))
        sources=[key for prefix in banks for key in (prefix,prefix+'_readouts')]
        proof={source:json.loads(records[source+'/complete.json']) for source in sources}
        for case in panel:
            name=case['image']
            def entry(source):
                data=records[source+'/'+name+'.pt']
                assert hashlib.sha256(data).hexdigest()==proof[source]['files'][name]
                return tensor(data)
            gens=[];readouts=[]
            for prefix in banks:
                gen,rd=entry(prefix),entry(prefix+'_readouts')
                assert rd['source']==proof[prefix]['files'][name] and gen['adapter']==pr['adapters'][arm] and gen['rgb']==case['rgb_sha256']
                assert rd['image']==gen['image']==name and gen['query']==case['scene_query']
                gens.append(gen);readouts.append(rd)
            assert all(g['first']==gens[0]['first'] and g['tool_calls']==gens[0]['tool_calls'] for g in gens)
            assert len({g['seed'] for g in gens})==len(gens)
            counts=8*len(gens);vis=torch.cat([r['visual'] for r in readouts]).double()
            hidden=torch.cat([g['hidden'] for g in gens]);ranks=sum((r['proxy_rank'] for r in readouts),[])
            # Matrix-vector calls mirror the native worker for each candidate; keep the frozen text bank.
            sims=torch.stack([native@v for v in vis]).numpy();hs=torch.stack([hard@v for v in vis]).numpy()
            ix=text['maps'][name];p=ix['native'];pos=np.unique(p);alt=np.setdiff1d(pos,p[0]);assert len(alt)
            allowed=np.ones(len(even),dtype=bool);allowed[pos]=False
            native_margin=sims[:,p[0]]-sims[:,even&allowed].max(1)
            held=sims[:,~even&allowed];true=sims[:,alt].max(1)
            rr=np.array([np.mean(1/np.arange(1+(scores>v).sum(),2+(scores>v).sum()+(scores==v).sum())) for scores,v in zip(held,true)])
            marg={k:hs[:,ix[k][0]]-hs[:,ix[k][1]] for k in ix if k in 'ABCD'}
            A,B,C=[marg[k] for k in 'ABC']
            old=1/(1+np.asarray(ranks))
            nr,ar=rankdata(native_margin),rankdata(A)
            blend=3*nr+ar
            close=np.flatnonzero(native_margin>=native_margin.max()-.001)
            methods={'R2':winner(calibrated(native_margin,old)),'R3':winner(calibrated(blend,old)),
                     'R2_raw':winner(native_margin),'R3_raw':winner(blend),'R3_50':winner(calibrated(nr+ar,old)),
                     'R3_close_raw':close[winner(A[close])],'A_only':winner(A),'random':np.arange(counts)}
            base=dict(arm=arm,image=name,pair=case['pair'],scene=case['scene_id'],attribute=case['attribute'])
            good=B>0;both=good&(C>0)
            diagnosis.append(dict(**base,post_tool=int(gen['post_tool']),unique_hidden=len(hidden.unique(dim=0)),
               proxy_levels=len(set(ranks)),slots=counts,good_B=int(good.sum()),good_C=int((C>0).sum()),good_BC=int(both.sum()),
               r2_winners=len(methods['R2']),r3_winners=len(methods['R3']),different_set=int(set(methods['R3'])!=set(methods['R2'])),
               raw_different_set=int(set(methods['R3_raw'])!=set(methods['R2_raw'])),
               A_B_agreement=float(np.mean((A>0)==(B>0))),A_C_agreement=float(np.mean((A>0)==(C>0)))))
            for i in range(counts):
                detail.append(dict(**base,slot=i,proxy_rank=ranks[i],native_margin=native_margin[i],A=A[i],B=B[i],C=C[i],
                                   MRR=rr[i],**({'D':marg['D'][i]} if 'D' in marg else {}),R2_score=calibrated(native_margin,old)[i],R3_score=calibrated(blend,old)[i]))
            for method,ii in methods.items():
                choice.append(dict(**base,method=method,slots=','.join(map(str,ii)),B=float(good[ii].mean()),C=float((C[ii]>0).mean()),
                        both_BC=float(both[ii].mean()),**({'D':float((marg['D'][ii]>0).mean()),'BCD':float(np.mean([np.mean(marg[k][ii]>0) for k in 'BCD']))} if 'D' in marg else {}),MRR=float(rr[ii].mean()),A=float((A[ii]>0).mean()),
                        margin_B=float(B[ii].mean()),margin_C=float(C[ii].mean())))
    write_csv(OUT/'candidates.csv',detail);write_csv(OUT/'choices.csv',choice);write_csv(OUT/'coverage.csv',diagnosis)
    levels={};cov={};contrasts={};methods=list(dict.fromkeys(v['method'] for v in choice));by={}
    for arm in ('a','b'):
        diag=[r for r in diagnosis if r['arm']==arm]
        cov[arm]=dict(photos=len(diag),post_tool=sum(r['post_tool'] for r in diag),available=sum(r['good_B']>0 for r in diag),
              all_good=sum(r['good_B']==r['slots'] for r in diag),all_wrong=sum(r['good_B']==0 for r in diag),mixed=sum(0<r['good_B']<r['slots'] for r in diag),
              mixed_C=sum(0<r['good_C']<r['slots'] for r in diag),available_both=sum(r['good_BC']>0 for r in diag),
              unique_hidden_mean=float(np.mean([r['unique_hidden'] for r in diag])),flat_proxy=sum(r['proxy_levels']==1 for r in diag),
              changed=sum(r['different_set'] for r in diag),raw_changed=sum(r['raw_different_set'] for r in diag))
        levels[arm]={}
        for m in methods:
            rows=[r for r in choice if r['arm']==arm and r['method']==m]
            vals={k:np.array([r[k] for r in rows]) for k in (['A','B','C','both_BC','MRR','margin_B','margin_C']+(['D','BCD'] if 'D' in rows[0] else []))}
            by[arm,m]=vals;summary={k:float(v.mean()) for k,v in vals.items()}
            assert all(rows[i]['pair']==rows[i+1]['pair'] and rows[i]['image']!=rows[i+1]['image'] for i in range(0,len(rows),2))
            summary['pair_both_B']=float(np.prod(vals['B'].reshape(-1,2),axis=1).mean())
            mixed=np.array([0<r['good_B']<r['slots'] for r in diag])
            summary['B_when_mixed']=float(vals['B'][mixed].mean()) if mixed.any() else None
            levels[arm][m]=summary
        for k in ('B','C','MRR'):
            delta=by[arm,'R3'][k]-by[arm,'R2'][k]
            cov[arm][k+'_gains']=int((delta>1e-12).sum());cov[arm][k+'_losses']=int((delta< -1e-12).sum())
    for a,b in [('R3','R2'),('R3_raw','R2_raw'),('R3_50','R2'),('R3_close_raw','R2_raw'),('A_only','R2_raw')]:
        results={}
        for k in by['a',a]:
            diff=np.stack([by[arm,a][k]-by[arm,b][k] for arm in ('a','b')])
            avg=diff.mean(0)
            results[k]=dict(paired_images=boot(avg,[r['pair'] for r in panel]),scenes=boot(avg,[r['scene_id'] for r in panel]),
                            seed_differences=diff.mean(1).tolist())
        contrasts[a+'-'+b]=results
    dump(OUT/'summary.json',dict(photos=len(panel),pairs=len(panel)//2,slots=len(detail),coverage=cov,levels=levels,contrasts=contrasts))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'main')
