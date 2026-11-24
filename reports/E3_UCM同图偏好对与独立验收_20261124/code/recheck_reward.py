"""Rebuild all reported image-level scores and the paired intervals from archived candidate vectors."""
import hashlib, sys
from collections import defaultdict
import numpy as np
import torch
from candidate_guard import ROOT, HIST, OLD, load, read, unpack, dump
from dev_metrics import sha

def choose(keys, proxy):
    ordered = sorted(range(len(keys)), key=lambda i:keys[i])
    values = np.sort(proxy)
    result = np.empty(len(proxy))
    j = 0
    while j < len(ordered):
        k = j + 1
        while k < len(ordered) and keys[ordered[k]] == keys[ordered[j]]: k += 1
        result[ordered[j:k]] = values[j:k].mean()
        j = k
    return np.flatnonzero(result == max(result))

def rr(positive, negatives):
    higher = (negatives > positive[:, None]).sum(axis=1)
    ties = (negatives == positive[:, None]).sum(axis=1)
    return np.array([sum(1/i for i in range(int(s)+1, int(s+t)+2))/(t+1) for s,t in zip(higher,ties)])

def audit(tag):
    torch.set_num_threads(2)
    base = ROOT / ('reward_confirm' if tag == 'independent' else 'guard')
    panel = read(base / 'inputs/panel.json')
    lock = read(base / 'inputs/lock.json')
    assert sha(base / 'inputs/panel.json') == lock['panel']
    prior = [read(OLD / sub / 'inputs/panel.json') for sub in ('','confirm')] + [read(ROOT / 'confirm/inputs/panel.json')]
    if tag == 'independent': prior.append(read(HIST / 'inputs/panel.json'))
    for col in ['image','rgb_sha256']:
        assert len({r[col] for r in panel}) == len(panel)
        assert all(not ({r[col] for r in p} & {r[col] for r in panel}) for p in prior)
    folder = base / 'records'
    bank = load(folder / 'text.pt')
    for name in ('provenance','extra_provenance','read_complete','extra_read_complete'):
        assert read(folder / (name+'.json'))['text'] == sha(folder/'text.pt')
    text = bank['native'].double().numpy()
    hard = bank['hard'].double().numpy()
    tokens = bank['native_tokens'].tolist()
    assert not ({tuple(row) for row in tokens} & {tuple(row) for row in bank['tokens'].tolist()})
    digest = [hashlib.sha256(','.join(str(y) for y in v).encode()).hexdigest() for v in tokens]
    split = np.array([int(v,16)%2 for v in digest])
    kappa = read(OLD/'inputs/continuous_lock.json')['kappa']
    expected = {(r['image'],r['seed']):r['scores'] for r in read(ROOT/'results'/('multi_'+tag+'_rows.json'))}
    original = read(ROOT/'results'/('multi_'+tag+'.json'))
    vals = defaultdict(list)
    below = []
    for seed in 'ab':
        group = [unpack(folder,seed),unpack(folder,'extra_'+seed)]
        for c in panel:
            name = c['image']
            x, y = [g[name] for g in group]
            assert x[0]['rgb'] == y[0]['rgb'] == c['rgb_sha256'] and x[0]['query'] == y[0]['query'] == c['scene_query']
            assert x[0]['seed'] != y[0]['seed'] and all(len(z[1]['visual']) == 8 for z in (x,y))
            vis = np.concatenate([z[1]['visual'].double().numpy() for z in (x,y)])
            pr = 1/(1+np.array(x[1]['proxy_rank']+y[1]['proxy_rank']))
            sim, hs = vis@text.T, vis@hard.T
            ix = bank['maps'][name]
            pos = ix['native']
            alt = sorted(set(pos)-{pos[0]}, key=digest.__getitem__)
            train = [pos[0]] + (alt[:1] if len(alt)>1 else [])
            hold = alt[1:] if len(alt)>1 else alt
            assert hold and not set(hold)&set(train)
            neg = ~np.isin(np.arange(len(text)),pos)
            nx, ny = sim[:,neg & (split==0)],sim[:,neg & (split==1)]
            nat = sim[:,pos[0]]-nx.max(axis=1)
            r3 = nat+kappa*(hs[:,ix['A'][0]]-hs[:,ix['A'][1]])
            first = choose(nat,pr)
            protected = np.mean([rr(sim[:,i],nx) for i in train], axis=0)
            ok = protected+1e-14 >= protected[first].mean()
            guard = choose(list(zip(ok,r3)),pr)
            below.append(np.mean(~ok[guard]))
            heldrank = rr(sim[:,hold].max(axis=1),ny)
            for method,win in [('R2',first),('R3',choose(r3,pr)),('guard',guard)]:
                obs = expected[(name,seed)][method]
                scores = {k:np.mean(hs[win,ix[k][0]]>hs[win,ix[k][1]]) for k in 'CDE'}
                scores['rr'] = heldrank[win].mean()
                assert all(abs(z-obs[k])<1e-10 for k,z in scores.items())
                vals[(method,seed)].append([np.mean([scores[k] for k in 'CDE']),scores['rr']])
    assert abs(np.mean(below)-original['winner_below_reference']) < 1e-10
    groups = defaultdict(list)
    for j,c in enumerate(panel):groups[c['pair']].append(j)
    assert all(len(g)==2 for g in groups.values())
    draws = np.random.default_rng(20261124).integers(len(groups),size=(20000,len(groups)))
    levels = {method:np.mean([vals[(method,seed)] for seed in 'ab'],axis=(0,1)).tolist() for method in ('R2','R3','guard')}
    for key, metrics in original['comparisons'].items():
        a,b = key.split('-')
        dif = np.mean([np.array(vals[(a,s)])-vals[(b,s)] for s in 'ab'],axis=0)
        pairs = np.array([dif[g].mean(axis=0) for g in groups.values()])
        ci = np.quantile(pairs[draws].mean(axis=1),[.025,.975],axis=0)
        for j,metric in enumerate(('detail','rr')):
            assert np.allclose([dif[:,j].mean(),*ci[:,j]], [metrics[metric][v] for v in ('mean','lo','hi')], atol=2e-12)
    dump(ROOT/'results'/('recheck_'+tag+'.json'),dict(images=len(panel),pairs=len(groups),candidate_positions=len(panel)*32,seed_image_groups=2*len(panel),metrics=['detail','rr'],levels=levels))

if __name__=='__main__':
    for tag in sys.argv[1:]:audit(tag)
