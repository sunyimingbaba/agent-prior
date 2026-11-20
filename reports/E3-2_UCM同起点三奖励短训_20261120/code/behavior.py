"""Describe training reward dispersion and deployed response patterns for interpreting the matched comparison."""
from pathlib import Path
import csv,collections
import numpy as np
from shared import ROOT,read,dump
def main():
    results={}
    for arm in ('old','caption','hard'):
        zs=read(ROOT/('policy_'+arm)/'steps.json')
        tr=[t for z in zs for t in z['traces']]
        uni=[];identical=[];nt=[]
        for z in zs:
            sequences=[tuple(tuple(q['tokens']) for q in t['rounds']) for t in z['traces']]
            uni.append(len(set(sequences)));identical.append(len(set(z['rewards'])))
            nt.append([sum(len(q['tokens']) for q in t['rounds']) for t in z['traces']])
        bystep={key:[float(v) for v in col] for key,col in [('unique_trajectories',uni),('unique_rewards',identical),
                ('mean_tokens',np.mean(nt,1)),('unclipped_norm',[z['unclipped_grad_norm'] for z in zs])]}
        def mini(g):
            return dict(samples=len(g),immediate_eos=sum(t['rounds'][0]['tokens'] in [[151645],[151643]] for t in g),
                called_tool=sum(t['used']>0 for t in g),tokens=float(np.mean([sum(len(q['tokens']) for q in t['rounds']) for t in g])))
        results[arm]=dict(overall=mini(tr),first6=mini(tr[:48]),last6=mini(tr[-48:]),by_step=bystep,
             constant_reward_groups=sum(x==1 for x in identical),gradient_clipped_groups=sum(z['unclipped_grad_norm']>1 for z in zs))
    with (ROOT/'results/outputs.csv').open(encoding='utf-8-sig') as fp:rows=list(csv.DictReader(fp))
    paired={}
    for arm in ('base','old','caption','hard'):
        for part in ('native','generic','detail'):
            selected=[r for r in rows if r['arm']==arm and r['part']==part and r['k']=='-1']
            def tokens(z):
                f=ROOT/'eval'/arm/part/(Path(z['image']).stem+'_'+z['view']+'.json')
                t=read(f)['traces'][0]
                return tuple(tuple(q['ids']) for q in t['rounds'])
            words=[tokens(r) for r in selected];cnt=collections.Counter(words)
            paired[arm+'/'+part]=dict(total=len(words),unique_responses=len(cnt),most_common=max(cnt.values()),immediate_eos=sum(r['immediate_eos']=='True' for r in selected),
                 any_tool=sum(int(r['tool'])>0 for r in selected),mean_response_tokens=float(np.mean([int(r['generated_tokens']) for r in selected])))
    dump(ROOT/'results/response_profile.json',dict(training=results,evaluation=paired))
if __name__=='__main__':main()
