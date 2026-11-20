"""Independent float64 re-score and tie-aware within-image statistics on the 8 samples."""
import csv,hashlib,json,statistics
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from shared import ROOT,read,dump,sha,locked

def save_csv(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def spearman(x,y):
    if len(set(x))<2 or len(set(y))<2:return None
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
def choose(x,y):
    best=np.flatnonzero(x==np.max(x))
    return float(np.mean(y[best])),float(np.mean(y[best]>0)),len(best)
def concordance(rewards,margin):
    ds=[]
    for i in range(len(margin)):
        for j in range(i):
            if margin[i]==margin[j]:continue
            ds.append((1+np.sign(rewards[i]-rewards[j])*np.sign(margin[i]-margin[j]))/2)
    return float(np.mean(ds)) if ds else None
def stats(data,metric):
    # Resample original matched pairs as intact units, then give their images equal weight.
    by=defaultdict(list)
    for r in data:
        by[r['pair']].append(r[metric])
    blocks=[[v for v in vs if v is not None] for vs in by.values()]
    count=np.array(list(map(len,blocks))); sums=np.array(list(map(sum,blocks)))
    if count.sum()==0:return dict(mean=None,lo=None,hi=None,n=0)
    idx=np.random.default_rng(20261120).integers(len(blocks),size=(20000,len(blocks)))
    n=count[idx].sum(axis=1);s=sums[idx].sum(axis=1)
    lo,hi=np.quantile(s[n>0]/n[n>0],[.025,.975])
    return dict(mean=float(sums.sum()/count.sum()),lo=float(lo),hi=float(hi),n=int(count.sum()))
def verify(cases,rows):
    meta=read(ROOT/'scoring/binding.json')
    assert meta['rows']==sha(ROOT/'scoring/rows.json') and meta['bank']==sha(ROOT/'scoring/vectors.pt')
    assert meta['gallery']==sha(ROOT/'gallery/train.pt') and meta['lock']==sha(ROOT/'inputs/lock.json')
    f=torch.load(ROOT/'scoring/vectors.pt',map_location='cpu',weights_only=True)
    bank=torch.load(ROOT/'gallery/train.pt',map_location='cpu',weights_only=True)['image_features'].double().numpy()
    names=read(ROOT/'gallery/index.json')['names']
    assert len(set(names))==len(names)==1680 and list(f['order'])==[(r['image'],r['mode'],r['k']) for r in rows]
    assert all(torch.isfinite(f[k]).all() and torch.max(torch.abs(f[k].norm(dim=1)-1))<1e-5 for k in ['text','visual','proxy'])
    pr=f['proxy'].double().numpy()@bank.T
    sim=f['visual'].double().numpy()@f['text'].double().numpy().T
    maxerr=0.;rank_diff=[];history=[];nchecks=0;trajectory_counts=Counter()
    for i,z in enumerate(rows):
        n=z['image'];c=cases[n];mode=z['mode'];k=z['k']
        assert names[z['row']]==n==c['image'] and z['row']==c['row']
        scores=pr[i];rank=np.count_nonzero(scores>scores[z['row']])
        if int(rank)!=z['rank0']:
            band=abs(scores-scores[z['row']])<1e-6
            assert abs(int(rank)-z['rank0'])<=band.sum()-1
            rank_diff.append(dict(image=n,mode=mode,k=k,fp64=int(rank),native=z['rank0']))
        folder=ROOT/'samples'/mode/Path(n).stem
        tensor=folder/('greedy.pt' if k<0 else f'sample{k}.pt')
        rec=read(tensor.with_suffix('.json'))
        assert rec['query']==c['scene_query' if mode=='scene' else 'orig_query'] and sha(tensor)==rec['sha256']
        h=torch.load(tensor,map_location='cpu',weights_only=True)
        assert h.shape==(3584,) and torch.isfinite(h).all()
        assert hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()==z['native_tensor']==rec['vector_sha256']
        tr='\n'.join(entry['decoded'] for entry in rec['rounds'])
        assert tr==z['text'] and len(rec['events'])==z['tools'] and all(e['confidence']>=0 for e in rec['events'])
        import re
        fmt=.2 if re.search(r'<answer>.*?</answer>',tr,re.S) else 0.
        assert fmt==z['format'] and z['reward']==fmt+1/(1+z['rank0'])
        if k==-1 and mode=='original':history.append(rec['history']['bit_equal'])
        trajectory_counts[(mode,'greedy' if k<0 else 'sample')]+=1
        for w in ('A','B'):
            p,nc=c[w]
            pos=sim[i,f['texts'].index(p)];neg=sim[i,f['texts'].index(nc)]
            maxerr=max(maxerr,abs(pos-z[w+'_pos']),abs(neg-z[w+'_neg']),abs(pos-neg-z[w]))
            nchecks+=2
    assert maxerr<1e-12
    return dict(n=len(rows),cosines=nchecks,max_cos_or_margin_error=maxerr,fp64_rank_differences=rank_diff,
                greedy_archive_exact=sum(history),greedy_archive_compared=len(history),
                counts={'/'.join(k):v for k,v in trajectory_counts.items()})
def main():
    cases={r['image']:r for r in locked()}
    rows=read(ROOT/'scoring/rows.json')
    proof=verify(cases,rows)
    dump(ROOT/'核验.json',proof)
    data=[]
    for mode in ('scene','original'):
        for name,c in cases.items():
            allr=[z for z in rows if z['mode']==mode and z['image']==name]
            assert sorted(z['k'] for z in allr)==list(range(-1,8))
            ref=next(z for z in allr if z['k']==-1)
            xs=[z for z in allr if z['k']>=0]
            a,b,rew,fmt,soft=[np.array([z[k] for z in xs]) for k in ['A','B','reward','format','soft']]
            base=float(np.mean(b>0))
            item=dict(mode=mode,image=name,pair=c['pair'],block=c['block'],correct_candidates=int(np.sum(b>0)),
                      greedy_B=int(ref['B']>0),unique_states=len({z['native_tensor'] for z in xs}),
                      unique_final=len({z['final'] for z in xs}),unique_rewards=len(set(rew)),mixed=int(0<sum(b>0)<8),
                      greedy_margin=ref['B'],reward_min=float(min(rew)),reward_max=float(max(rew)),B_min=float(min(b)),B_max=float(max(b)),
                      random_B=float(b.mean()),random_correct=base,margin_span=float(np.ptp(b)),
                      correct_format_candidates=int(sum(fmt>0)),tool_candidates=sum(z['tools']>0 for z in xs),
                      immediate_eos=sum(read((ROOT/'samples'/mode/Path(name).stem/f"sample{z['k']}.json"))['rounds'][0]['ids'] in [[151645],[151643]] for z in xs),
                      rho=spearman(rew,b),rho_A=spearman(a,b),agree=concordance(rew,b),agree_A=concordance(a,b))
            for key,criterion in [('proxy',rew),('rank',soft),('format',fmt),('alt_A',a),('oracle',b)]:
                val,acc,ties=choose(criterion,b)
                item.update({key+'_B':val,key+'_correct':acc,key+'_ties':ties,key+'_gain_B':val-item['random_B'],key+'_gain_acc':acc-base})
            item['A_vs_proxy_acc']=item['alt_A_correct']-item['proxy_correct']
            item['A_vs_proxy_B']=item['alt_A_B']-item['proxy_B']
            data.append(item)
    save_csv('逐图选择.csv',data);save_csv('逐候选分数.csv',rows)
    summary={};scenes=[]
    for mode in ('scene','original'):
        group=[r for r in data if r['mode']==mode]
        metrics=['rho','rho_A','agree','agree_A','proxy_B','proxy_correct','random_correct','alt_A_correct','oracle_correct',
                 'random_B','alt_A_B','oracle_B','proxy_gain_acc','proxy_gain_B','alt_A_gain_acc','alt_A_gain_B','oracle_gain_acc',
                 'oracle_gain_B','A_vs_proxy_acc','A_vs_proxy_B','rank_correct','format_correct','greedy_B']
        summary[mode]=dict(images=len(group),pairs=len({r['pair'] for r in group}),blocks=len({r['block'] for r in group}),
                 all_wrong=sum(r['correct_candidates']==0 for r in group),all_right=sum(r['correct_candidates']==8 for r in group),
                 mixed=sum(r['mixed'] for r in group),constant_reward=sum(r['unique_rewards']==1 for r in group),
                 unique_state_counts=dict(Counter(r['unique_states'] for r in group)),
                 top_proxy_tied=sum(r['proxy_ties']>1 for r in group),format_candidates=sum(r['correct_format_candidates'] for r in group),
                 requested_tools=sum(r['tool_candidates'] for r in group),immediate_eos=sum(r['immediate_eos'] for r in group),
                 caps={k:stats(group,k) for k in metrics},
                 mixed_selection={k:sum(r[k] for r in group if r['mixed']) for k in ['proxy_correct','random_correct','alt_A_correct','oracle_correct']})
        for block in sorted({r['block'] for r in group}):
            g=[r for r in group if r['block']==block]
            scenes.append(dict(mode=mode,block=block,n=len(g),mixed=sum(r['mixed'] for r in g),
                          proxy=sum(r['proxy_correct'] for r in g),random=sum(r['random_correct'] for r in g),
                          alt_A=sum(r['alt_A_correct'] for r in g),oracle=sum(r['oracle_correct'] for r in g)))
    dump(ROOT/'选择统计.json',summary);save_csv('逐场景汇总.csv',scenes)
    print(json.dumps(proof,ensure_ascii=False))
if __name__=='__main__':main()
