"""Deterministic full-gallery recalls with exact tie expectations; matched per-source comparisons."""
from pathlib import Path
import csv,math
from functools import lru_cache
import numpy as np
import torch
from shared import ROOT,read,dump,sha
ARMS=('base','old','caption','hard')
def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as fp:
        w=csv.DictWriter(fp,list(dict.fromkeys(k for z in rows for k in z)));w.writeheader();w.writerows(rows)
def fixed(x,y):
    a=x.double().numpy();b=y.double().numpy()
    scores=np.zeros((len(a),len(b)),dtype=np.float64)
    for i in range(a.shape[1]):scores+=a[:,i,None]*b[None,:,i]
    return scores
@lru_cache(None)
def hit(t,p,k):return 1-math.comb(t-p,k)/math.comb(t,k)
def recalls(scores,labels):
    result={}
    for k in [1,5,10]:
        boundary=np.partition(scores,-k,axis=1)[:,-k,None]
        above=scores>boundary;same=scores==boundary
        sure=(above&labels).any(1);nt=same.sum(1);cp=(same&labels).sum(1);slots=k-above.sum(1)
        result[k]=np.array([1. if ok else hit(int(n),int(p),int(s)) for ok,n,p,s in zip(sure,nt,cp,slots)])
    return result
def ci(vals):
    x=np.array(vals); sample=np.random.default_rng(20261120).integers(len(x),size=(20000,len(x)))
    a,b=np.quantile(x[sample].mean(1),[.025,.975])
    return dict(mean=float(x.mean()),lo=float(a),hi=float(b),n=len(x))
def main():
    val=read(ROOT/'inputs/validation.json');details=read(ROOT/'inputs/detail.json')
    native=[];per=[];raw=[];behavior=[];digest={};textref=None
    for arm in ARMS:
        p=ROOT/'eval'/arm
        binding=read(p/'scored.json');assert binding['vectors']==sha(p/'vectors.pt') and binding['text']==sha(p/'text.pt')
        f=torch.load(p/'vectors.pt',map_location='cpu',weights_only=True)
        t=torch.load(p/'text.pt',map_location='cpu',weights_only=True)
        if textref:
            for key in ('native','native_tokens','relation'):assert torch.equal(textref[key],t[key])
            assert textref['texts']==t['texts'] and textref['caption_order']==t['caption_order']
        textref=t
        for part in ('native','generic','detail'):
            assert torch.isfinite(f[part]['visual']).all() and (f[part]['visual'].norm(dim=1)-1).abs().max()<1e-5
            for file in sorted((p/part).glob('*.json')):
                if file.name in ('complete.json','loaded_weights.json'):continue
                z=read(file)
                assert z['sha256']==sha(file.with_suffix('.pt'))
                hs=torch.load(file.with_suffix('.pt'),weights_only=True,map_location='cpu')
                for tr,h in zip(z['traces'],hs,strict=True):
                    import hashlib
                    assert tr['hidden']==hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    key=(part,z['image'],z['view'],tr['k'])
                    if arm=='base':digest[key]=tr['hidden']
                    rounds=tr['rounds']
                    behavior.append(dict(arm=arm,part=part,image=z['image'],view=z['view'],k=tr['k'],same_hidden_as_base=tr['hidden']==digest[key],
                                  immediate_eos=len(rounds[0]['ids'])==1 and rounds[0]['ids'][0] in [151645,151643],tool=tr['used'],
                                  generated_tokens=sum(len(x['ids']) for x in rounds)))
        for part in ('native','generic'):
            pairs=f[part]['order'];assert pairs==[(r['image'],'upright',-1) for r in val]
            scores=fixed(f[part]['visual'],t['native'])[:,t['caption_order']]
            owned=np.arange(210)[:,None]==np.repeat(np.arange(210),5)[None,:]
            i2t=recalls(scores,owned);t2i=recalls(scores.T,owned.T)
            for i,r in enumerate(val):
                out=dict(arm=arm,part=part,image=r['image'],fresh=r['fresh'],block=r['block'])
                for k in (1,5,10):
                    out['I2T_R'+str(k)]=float(i2t[k][i]);out['T2I_R'+str(k)]=float(t2i[k][i*5:(i+1)*5].mean())
                out['mR']=np.mean([v for k,v in out.items() if '_R' in k]);native.append(out)
        score=fixed(f['detail']['visual'],t['relation'])
        order={tuple(id):i for i,id in enumerate(f['detail']['order'])}
        for r in details:
            m={};n=r['image'];item=dict(arm=arm,image=n,direction=r['direction'])
            for mode in ['upright','rotated']:
                for w in 'CD':
                    pos,neg=[t['texts'].index(v) for v in r[w]]
                    m[mode,w]=np.array([(score[order[n,mode,k],pos]-score[order[n,mode,k],neg])*(1 if mode=='upright' else -1) for k in range(-1,4)])
                    for k in range(-1,4):raw.append(dict(arm=arm,image=n,view=mode,wording=w,k=k,margin=m[mode,w][k+1]))
            for mode in ['upright','rotated']:
                item[mode+'_greedy']=np.mean([m[mode,w][0]>0 for w in 'CD'])
                item[mode+'_sample']=np.mean([m[mode,w][1:]>0 for w in 'CD'])
            for dec,idx in [('greedy',slice(0,1)),('sample',slice(1,None))]:
                item['both_'+dec]=np.mean([(m['upright',w][idx]>0)&(m['rotated',w][idx]>0) for w in 'CD'])
                item['avg_'+dec]=(item['upright_'+dec]+item['rotated_'+dec])/2
                item['margin_'+dec]=np.mean([m[view,w][idx] for view in ['upright','rotated'] for w in 'CD'])
            item['both_sample_independent']=np.mean([(m['upright',w][1:]>0).mean()*(m['rotated',w][1:]>0).mean() for w in 'CD'])
            per.append(item)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    csvwrite(out/'retrieval_by_source.csv',native);csvwrite(out/'detail_by_source.csv',per);csvwrite(out/'detail_margins.csv',raw);csvwrite(out/'outputs.csv',behavior)
    sm={};contrasts={}
    for part,unit,keys in [
            ('native',[r for r in native if r['part']=='native'],['mR','I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10']),
            ('generic',[r for r in native if r['part']=='generic'],['mR','I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10']),
            ('native_fresh',[r for r in native if r['part']=='native' and r['fresh']],['mR']),
            ('generic_fresh',[r for r in native if r['part']=='generic' and r['fresh']],['mR']),
            ('detail',per,[k for k in per[0] if k not in ['arm','image','direction']])]:
        by={arm:{r['image']:r for r in unit if r['arm']==arm} for arm in ARMS}
        names=list(by['base']);assert all(set(v)==set(names) for v in by.values())
        sm[part]={arm:{k:ci([by[arm][n][k] for n in names]) for k in keys} for arm in ARMS}
        contrasts[part]={arm+'-'+ref:{k:ci([by[arm][n][k]-by[ref][n][k] for n in names]) for k in keys} for arm,ref in [('old','base'),('caption','base'),('hard','base'),('caption','old'),('hard','old'),('hard','caption')]}
    decision={}
    for arm in ('caption','hard'):
        condition={}
        for part,key in [('native','mR'),('detail','both_greedy')]:
            n=sm[part][arm][key]['mean']
            condition[part+'_over_old']=n>sm[part]['old'][key]['mean']
            condition[part+'_retain_base']=n>=sm[part]['base'][key]['mean']
        decision[arm]=dict(conditions=condition,passed=bool(all(condition.values())))
    dump(out/'summary.json',sm);dump(out/'contrasts.json',contrasts);dump(out/'decision.json',decision)
    brief={}
    for arm in ARMS:
        brief[arm]={}
        for part in ('native','generic','detail'):
            g=[r for r in behavior if r['arm']==arm and r['part']==part and r['k']==-1]
            brief[arm][part]={k:float(np.mean([r[k] for r in g])) for k in ['same_hidden_as_base','immediate_eos','tool','generated_tokens']}
    dump(out/'behavior.json',brief)
if __name__=='__main__':main()
