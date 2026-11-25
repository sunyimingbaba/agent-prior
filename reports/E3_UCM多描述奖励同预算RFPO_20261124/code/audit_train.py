import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from report_io import ROOT,LAST,HIST,OLD,read,steps,dump,sha,load,trainrows
sys.path.insert(0,str(OLD/'code'))
from exact_check import exact

def run(backup):
    roster=read(ROOT/'inputs/roster.json');rows=trainrows();order=roster['order'];kappa=read(OLD/'inputs/continuous_lock.json')['kappa']
    assert len(order)==len(set(order)) and len(order)+len(roster['omitted'])==len(rows)==168
    ctx={s:load(ROOT/'records'/(s+'_contexts.pt')) for s in 'ab'}
    pair_list={rows[n]['pair'] for n in order}
    assert len(pair_list)*2==len(order) and all(rows[a]['pair']==rows[b]['pair'] for a,b in zip(order[::2],order[1::2]))
    assert all(ctx[s]['rows'][n]['prefix'] is not None for s in 'ab' for n in order)
    for pair in {r['pair'] for r in rows.values()}-pair_list:
        assert any(ctx[s]['rows'][n]['prefix'] is None for s in 'ab' for n,r in rows.items() if r['pair']==pair)
    out={};checks={};detail=[]
    for seed in 'ab':
        for arm in ('R2','R3'):
            tag=seed+'_'+arm;folder=ROOT/'pilot'/tag;ss=steps(tag);proof=read(folder/'complete.json')
            assert proof['initial']==ctx[seed]['adapter']==roster['initial'][seed] and sha(backup/tag/'adapter/adapter_model.safetensors')==proof['final']
            assert proof['roster']==sha(ROOT/'inputs/roster.json') and sha(folder/'steps.jsonl')==proof['steps_sha256']
            assert [v['image'] for v in ss]==order and len(ss)==proof['steps'] and sum(v['norm']>0 for v in ss)==proof['updated']
            assert proof['numeric_lock']==sha(ROOT/'数值修正记录.md')
            assert all(sha(ROOT/'code'/n)==d for n,d in proof['sources'].items())
            for j,stage in enumerate(ss):
                nm=stage['image'];sc=stage['scores'];anc=stage['anchor']['scores'];assert len(sc)==len(stage['records'])==16 and stage['step']==j+1
                assert stage['rng']==[(1871000 if seed=='a' else 2871000)+j*2+i for i in range(2)]
                assert all(len(r['ids'])==len(r['sampled_logp'])>0 and np.isfinite(r['sampled_logp']).all() for r in stage['records'])
                for a in stage['anchor']['tools']:assert a==ctx[seed]['rows'][nm]['answer']
                assert stage['first_matches_current']==(stage['anchor']['actions'][0]==ctx[seed]['rows'][nm]['first'])
                native,hard,keep,rr,rank=(np.array([z[k] for z in sc]) for k in ('native','A','keep','rr','rank'))
                proxy=1/(1+rank);r2=exact(native,rank);limit=keep[r2==r2.max()].mean()
                r3=exact(rankdata(native+kappa*hard,method='dense')+17*(keep+1e-14>=limit),rank)
                def adv(x):
                    v=x+stage['format'];return np.zeros_like(v) if np.ptp(v)==0 else (v-v.mean())/(v.std(ddof=1)+1e-8)
                assert abs(limit-stage['floor'])<1e-12 and np.array_equal(r2,stage['retrieval']['R2']) and np.array_equal(r3,stage['retrieval']['R3'])
                assert np.allclose(adv(r2 if arm=='R2' else r3), stage['advantages'],rtol=0,atol=1e-9)
                if np.ptp(stage['retrieval'][arm])==0:assert stage['norm']==0 and np.max(np.abs(stage['advantages']))==0
                votes=np.array([np.mean([z['B']>0,z['C']>0]) for z in sc]);a0=np.mean([anc['B']>0,anc['C']>0])
                row=dict(seed=seed,arm=arm,image=nm,pair=rows[nm]['pair'],step=j+1,update=stage['norm']>0,first_same=stage['first_matches_current'],
                         reward_diff=bool(np.max(abs(adv(r2)-adv(r3)))>1e-8),winner_diff=bool(np.any((r2==r2.max())!=(r3==r3.max()))),
                         any_joint=bool(((votes>a0)&(rr>anc['rr']+1e-12)).any()),unique_actions=len({tuple(z['ids']) for z in stage['records']}))
                for name,v in [('R2',r2),('R3',r3)]:
                    win=v==v.max();d=float(votes[win].mean()-a0);m=float(rr[win].mean()-anc['rr'])
                    row[name]=dict(detail=d,rr=m,better_detail=d>1e-12,better_rr=m>1e-12,joint=d>1e-12 and m>1e-12,winners=int(win.sum()))
                detail.append(row)
            rs=[r for r in detail if r['seed']==seed and r['arm']==arm]
            lengths=[len(v['ids']) for step in ss for v in step['records']]
            out[tag]=dict(steps=len(rs),candidates=len(rs)*16,mean_sampled_second_tokens=float(np.mean(lengths)),at_token_cap=sum(n==64 for n in lengths),updates=sum(r['update'] for r in rs),different_first=sum(not r['first_same'] for r in rs),changed_advantages=sum(r['reward_diff'] for r in rs),
                changed_winners=sum(r['winner_diff'] for r in rs),joint_available=sum(r['any_joint'] for r in rs),mean_unique_answers=np.mean([r['unique_actions'] for r in rs]),max_mean_abs_logp_error=max(v['mean_abs_logp_error'] for v in ss))
            for name in ('R2','R3'):
                out[tag][name]={k:sum(r[name][k] for r in rs) for k in ('better_detail','better_rr','joint')}
                out[tag][name].update({k:np.mean([r[name][k] for r in rs]) for k in ('detail','rr','winners')})
        a,b=steps(seed+'_R2'),steps(seed+'_R3');same=0
        for v,u in zip(a,b):
            if any(v[k]!=u[k] for k in ('records','scores','anchor','retrieval')):break
            same+=1
        assert same>=1 and a[0]['anchor']['actions']==ctx[seed]['rows'][order[0]]['actions']
        checks[seed]=dict(exact_initial_sampling=True,same_consecutive_batches=same)
    dump(ROOT/'results/training.json',dict(pairs=len(pair_list),arms=out,seed_checks=checks))
    dump(ROOT/'results/training_rows.json',detail)
if __name__=='__main__':run(Path(sys.argv[1]))
