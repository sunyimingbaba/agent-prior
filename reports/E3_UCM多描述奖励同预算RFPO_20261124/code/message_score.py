import re
import sys
import numpy as np
from report_io import ROOT,HIST,OLD,read,load,sha,dump
from exact_check import exact
sys.path.insert(0,str(HIST/'code'))
from score import boot

def clean(s):
    match=re.search(r'<answer>(.*?)</answer>',s,re.S)
    return bool(match and match.group(1).strip())

def values(s):
    return np.array([np.mean([s[k]>0 for k in 'BC']),s['rr']])

def main():
    lock=read(ROOT/'inputs/message_lock.json')
    archive=read(ROOT/'records/message_trial.json')
    vectors=load(ROOT/'records/message_priors.pt');ctx=load(ROOT/'records/a_contexts.pt')
    assert archive['adapter']==ctx['adapter'] and sha(ROOT/'inputs/message_lock.json')==archive['lock']
    assert sha(ROOT/'records/message_priors.pt')==archive['priors']
    names=lock['images'];assert list(archive['cases'])==names and set(vectors)==set(names)
    kappa=read(OLD/'inputs/continuous_lock.json')['kappa'];modes={};labels=[lock['pairs'][n] for n in names]
    for mode in ('raw','chat'):
        rows=[]
        for n in names:
            z=archive['cases'][n][mode];states=vectors[n][mode]
            assert states.shape==(17,3584) and states.isfinite().all() and vectors[n]['raw'][-1].equal(ctx['rows'][n]['prior'])
            assert archive['cases'][n]['raw']['feedback']==archive['cases'][n]['chat']['feedback']==ctx['rows'][n]['answer']
            assert archive['cases'][n]['raw']['first_text']==archive['cases'][n]['chat']['first_text']
            assert archive['cases'][n]['raw']['prefix']==ctx['rows'][n]['prefix'][0].tolist()
            scores=z['scores'][:-1];greedy=values(z['scores'][-1]);baseline=values(archive['cases'][n]['raw']['scores'][-1]);cands=np.stack([values(t) for t in scores])
            ranks,native,hard,keep=(np.array([s[k] for s in scores]) for k in ('rank','native','A','keep'))
            r2=exact(native,ranks)
            order=native+kappa*hard
            _,dense=np.unique(order,return_inverse=True)
            r3=exact(dense+17*(keep+1e-14>=keep[r2==r2.max()].mean()),ranks)
            assert all(np.array_equal(z['retrieval'][a],r) for a,r in [('R2',r2),('R3',r3)])
            rx,tx=(cands[r==r.max()].mean(0) for r in (r2,r3))
            draws=z['records'];assert len(draws)==16
            rows.append(dict(unique=len({tuple(r['ids']) for r in draws}),clean=sum(clean(r['text']) for r in draws),
                empty=sum(not r['text'].strip() for r in draws),artifact=sum('addCriterion' in r['text'] for r in draws),
                distinct_reward=int(np.ptp(r3)>0),joint_vs_raw=int(((cands>baseline+1e-12).all(1)).any()),
                joint_vs_own=int(((cands>greedy+1e-12).all(1)).any()),greedy_detail=greedy[0],greedy_rr=greedy[1],
                selected_detail_gain=tx[0]-rx[0],selected_rr_gain=tx[1]-rx[1]))
        modes[mode]=rows
    averages={mode:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for mode,rows in modes.items()}
    counts={mode:{k:int(sum(r[k] for r in rows)) for k in ('clean','empty','artifact','distinct_reward','joint_vs_raw','joint_vs_own')} for mode,rows in modes.items()}
    diffs={k:boot([b[k]-a[k] for a,b in zip(modes['raw'],modes['chat'])],labels) for k in averages['raw']}
    selected={mode:{k:boot([r[k] for r in rows],labels) for k in ('selected_detail_gain','selected_rr_gain')} for mode,rows in modes.items()}
    dump(ROOT/'results/message_comparison.json',dict(images=len(names),pairs=len(set(labels)),samples_per_mode=16*len(names),
        means=averages,counts=counts,chat_minus_raw=diffs,within_mode_R3_minus_R2=selected))
if __name__=='__main__':main()
