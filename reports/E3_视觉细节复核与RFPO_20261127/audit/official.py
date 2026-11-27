"""Verify actual RFPO weights, CPU-visible trajectories, and official endpoint estimates."""
import hashlib,json,sys
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from phrases import ROOT,REP,POOL,FRESH,read,load
from candidates import native_and_bonus
FORM=REP/'E3-2_UCM与RSITMD独立复验_20261124';FINE=REP/'E3_UCM多描述奖励同预算RFPO_20261124'
spec=spec_from_file_location('extended_reference',ROOT/'extra164/audit/check.py')
ext=module_from_spec(spec);spec.loader.exec_module(ext)

def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def validate_weight(seed,arm):
    current=arm=='strict'
    source=ROOT if current else POOL
    dest=source/'models'/(seed+'_'+arm)
    asset=Path('/Users/botlv/cup-deploy-assets')/('e3-visual-detail-rfpo-20261127' if current else 'e3-aftertool-joint-20261127')/'models'/dest.name
    proof=read(dest/'complete.json');steps=read(dest/'steps.json')
    lock=ROOT/'training/strict_lock.json' if current else POOL/'training/train_lock.json'
    old=read(POOL/'training/train_lock.json')
    assert proof['lock']==sha(lock) and proof['initial']==old['historical'][seed]['R0/adapter/adapter_model.safetensors']
    assert proof['scores']==sha(dest/'steps.json') and proof['steps']==len(steps)==len(read(POOL/'scoring/locks/training.json')['rows'])==24
    assert proof['train']==sha(POOL/'scoring/locks/training.json')
    for r in steps:
        native,bonus=native_and_bonus([dict(rank=k,margin=m,hard_margins=h) for k,m,h in zip(r['proxy_rank'],r['native_margin'],r['hard'])])
        np.testing.assert_array_equal(native,r['retrieval'])
        np.testing.assert_array_equal(native+np.array(r['format']),r['r2'])
        if arm in ['r2','r3']:
            np.testing.assert_allclose(bonus,r['bonus'],atol=1e-15,rtol=0)
        train_reward=np.array(r['r2'] if arm=='r2' else r['r3'])
        np.testing.assert_array_equal((train_reward-train_reward.mean())/(train_reward.std(ddof=1)+1e-8),r['advantage'])
    model=asset/'adapter/adapter_model.safetensors'
    assert proof['final']==sha(model)
    saved=torch.load(asset/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
    assert saved['lock']==proof['lock'] and saved['arm']==arm and saved['initial']==proof['initial']
    assert saved['history']==steps
    assert all(s['seed']==(231000000 if seed=='a' else 241000000)+1000*j for j,s in enumerate(steps))
    assert [s['image'] for s in steps]==[r['image'] for r in read(POOL/'scoring/locks/training.json')['rows']]
    counters={int(z['step']) for z in saved['opt']['state'].values()};assert counters=={proof['nonzero']}
    with safe_open(str(model),framework='pt') as m:
        keys={k.replace('.default.','.'):k for k in saved['weights']}
        assert set(keys)==set(m.keys())
        for k,original in keys.items():assert torch.equal(saved['weights'][original],m.get_tensor(k))
    return {'final':proof['final'],'optimizer_updates':sorted(counters),'equal_saved_lora_tensors':len(keys),'history':proof['scores']}

def bootstrap(values,units):
    ids=list(dict.fromkeys(units));groups=np.array([units.index(u) for u in ids])
    rows=[[j for j,x in enumerate(units) if x==u] for u in ids]
    collapsed=np.mean(values[:,rows],axis=2);assert len(set(map(len,rows)))==1
    sampled=np.random.default_rng(20261125).integers(len(ids),size=(20000,len(ids)))
    means=collapsed.mean(0)
    return dict(mean=means.mean(),seeds=collapsed.mean(1).tolist(),interval=np.quantile(means[sampled].mean(1),[.025,.975]).tolist())

def main(complete):
    gallery=read(FORM/'inputs/ucm.json');fine=read(FINE/'inputs/panel.json')
    native=load(FORM/'results/ucm_text.pt');hard=load(FINE/'records/text.pt')
    extra=load(FRESH/'fresh_text.pt');words=read(FRESH/'独立措辞.json')
    phrase_features={w:v.double().numpy() for w,v in zip(extra['words'],extra['features'])}
    ids={r['rgb_sha256']:i for i,r in enumerate(gallery)};byrgb={r['rgb_sha256']:r for r in fine}
    eval_fine=[r for r in fine if r['rgb_sha256'] in ids]
    acceptable={(q['attribute'],q['side'],*q['D'],*q['E']) for q in read(FRESH/'real_roster.json')}
    eval_fg=[r for r in eval_fine if (r['attribute'],r['side'],*r['D'],*r['E']) in acceptable]
    assert len(eval_fine)==106 and len(eval_fg)==56
    measures={};assets={};reread={};units={'mR':[r['image'] for r in gallery], 'FG':[r['pair'] for r in eval_fg]}
    units['DE']=units['mrr']=[r['pair'] for r in eval_fine]
    checks=read(POOL/'evaluation/results/单次检索对照.json')
    formal=read(ROOT/'evaluation/results/单次检索对照.json') if complete else checks
    wording=read((ROOT if complete else POOL)/'results/单次新措辞复核.json')
    cols=native['caption_order'].reshape(210,5).tolist();rows=[]
    for i,r in enumerate(gallery):
        rr=byrgb.get(r['rgb_sha256'])
        d=[j for k in 'DE' for j in hard['maps']['primary'][r['image']][k]] if rr else [0,0,0,0]
        rows.append(dict(r,panel_positive_ids=cols[i],panel_de_ids=d))
    for arm in (['r2','r3','strict'] if complete else ['r2','r3']):
        vals={k:[] for k in units};target=POOL if arm!='strict' else ROOT
        for seed in 'ab':
            name=seed+'_'+arm
            assets[name]=validate_weight(seed,arm)
            folder=target/'evaluation/records'/name
            m=read(folder/'complete.json')
            assert assets[name]['final']==m['adapter'] and sha(folder/'features.pt')==m['features']
            features=load(folder/'features.pt')
            assert features['images']==[r['image'] for r in gallery] and features['adapter']==m['adapter']
            vectors=[]
            for i,row in enumerate(gallery):
                p=folder/(row['image']+'.pt')
                assert sha(p)==m['images'][row['image']]
                z=load(p)
                assert z['rgb']==row['rgb_sha256'] and z['query']==row['orig_query'] and z['adapter']==m['adapter']
                assert torch.equal(z['v'], features['features'][i])
                vectors.append(z['v'])
                if complete and arm!='strict' and i in (0,105,209):
                    new_path=ROOT/'evaluation/records'/name/p.name
                    z2=load(new_path)
                    assert all(torch.equal(z[k],z2[k]) for k in ('prior','v'))
                    assert all(z[k]==z2[k] for k in ('query','feedback','actions','calls','rgb','adapter'))
            v=torch.stack(vectors);prior=ext.reference(rows,{'captions':native['native'],'details':hard['hard']},v)
            v=v.double().numpy()
            vals['mR'].append(prior[:,2]);ix=[ids[r['rgb_sha256']] for r in eval_fine];vals['DE'].append(prior[ix,0]);vals['mrr'].append(prior[ix,1])
            fg=[np.mean([v[ids[r['rgb_sha256']]]@phrase_features[pair[r['side']]]>v[ids[r['rgb_sha256']]]@phrase_features[pair[1-r['side']]]
                         for pair in words[r['attribute']]]) for r in eval_fg]
            vals['FG'].append(fg)
        measures[arm]={k:np.asarray(x) for k,x in vals.items()}
    stats={}
    for a,b in [(a,None) for a in measures]+([('strict','r2'),('strict','r3')] if complete else []):
        out={}
        for k,vec in measures[a].items():
            stat=bootstrap(vec if b is None else vec-measures[b][k],units[k])
            if k=='FG':
                theirs=wording['levels'][a][k] if b is None else wording[a+'_minus_'+b][k]
            else:
                panel='official' if k=='mR' else 'first106'
                theirs=formal[panel]['levels'][a][k] if b is None else formal[panel]['contrasts'][a+'-'+b][k]
            for key in ('mean','seeds','interval'):np.testing.assert_allclose(stat[key],theirs[key],atol=1e-13,rtol=0)
            out[k]=stat
        stats[a if b is None else a+'-'+b]=out
    path=ROOT/'results'/('正式单次与权重独立核查.json' if complete else '历史同预算基线独立回算.json')
    path.write_text(json.dumps(dict(weights=assets,metrics=stats),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(1)
    main('--final' in sys.argv)
